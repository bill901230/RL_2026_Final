# pyright: reportMissingImports=false, reportPossiblyUnboundVariable=false, reportUntypedBaseClass=false
"""Direct controller runner for W3.T1 smoke.

verl.trainer.main_ppo creates a Ray TaskRunner actor for controller logic.  On
this node that actor stayed in PENDING_CREATION/RUNNING without entering
TaskRunner.run, so this runner keeps the controller in the driver process while
using the stock veRL worker classes, datasets, reward manager, FSDP2 actor/ref,
and vLLM rollout path.
"""

from __future__ import annotations

import os
import socket
import sys
from pathlib import Path
from pprint import pprint

import ray
from omegaconf import OmegaConf


def load_config(argv: list[str]):
    import verl.trainer

    cfg_path = Path(verl.trainer.__file__).resolve().parent / "config" / "ppo_trainer.yaml"
    config = OmegaConf.load(cfg_path)
    # OmegaConf dotlist does not use Hydra's '+' prefix; strip it for ad-hoc keys.
    overrides = [arg[1:] if arg.startswith("+") else arg for arg in argv]
    if overrides:
        config = OmegaConf.merge(config, OmegaConf.from_dotlist(overrides))
    OmegaConf.resolve(config)
    return config


def main() -> None:
    config = load_config(sys.argv[1:])

    precreate_register = os.environ.get("W3T1_PRECREATE_REGISTER", "0") == "1"
    skip_register = os.environ.get("W3T1_SKIP_REGISTER", "0") == "1"
    worker_group_prefix = "w3t1wg"
    master_addr = None
    master_port = None
    if precreate_register or skip_register:
        # Optional workaround for the register-center wait; disabled by default
        # because the stock Worker.__new__ path provides the safest distributed
        # init env when it is able to create the center itself.
        import socket as socket_lib
        if precreate_register:
            import verl.single_controller.ray.base as ray_base

            ray_base.get_random_string = lambda length: worker_group_prefix
        with socket_lib.socket() as sock:
            sock.bind(("", 0))
            master_port = str(sock.getsockname()[1])
        master_addr = os.environ.get("MY_HOST_IP") or ray._private.services.get_node_ip_address()
        os.environ["DISABLE_WORKER_INIT"] = "1"
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = master_port

    if not ray.is_initialized():
        # Avoid job-level runtime_env here: on this node Ray 2.47.1 hangs GPU
        # actor creation when ray.init(runtime_env={env_vars: ...}) is used.
        # Worker-specific env_vars are still supplied by veRL's RayWorkerGroup.
        ray.init(num_cpus=config.ray_init.num_cpus, include_dashboard=False)

    register_center_handle = None
    if precreate_register:
        from verl.single_controller.base.register_center.ray import create_worker_group_register_center

        register_center_handle = create_worker_group_register_center(
            name=f"{worker_group_prefix}_register_center",
            info={"MASTER_ADDR": master_addr, "MASTER_PORT": master_port},
        )
        print(f"Precreated register center {worker_group_prefix}_register_center at {master_addr}:{master_port}", flush=True)

    from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer, ResourcePoolManager, Role
    from verl.trainer.ppo.reward import load_reward_manager
    from verl.utils import hf_processor, hf_tokenizer
    from verl.utils.dataset.rl_dataset import collate_fn
    from verl.utils.fs import copy_to_local

    print(f"Direct TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}", flush=True)
    pprint(OmegaConf.to_container(config, resolve=True))

    local_path = copy_to_local(config.actor_rollout_ref.model.path, use_shm=config.actor_rollout_ref.model.get("use_shm", False))
    trust_remote_code = config.data.get("trust_remote_code", False)
    tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
    processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

    if config.actor_rollout_ref.actor.strategy in ["fsdp", "fsdp2"]:
        assert config.critic.strategy in ["fsdp", "fsdp2"]
        from verl.single_controller.ray import RayWorkerGroup
        from verl.single_controller.ray.base import sort_placement_group_by_node_ip
        from verl.workers.fsdp_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker, CriticWorker

        actor_rollout_cls = AsyncActorRolloutRefWorker if config.actor_rollout_ref.rollout.mode == "async" else ActorRolloutRefWorker
        ray_worker_group_cls = RayWorkerGroup
    elif config.actor_rollout_ref.actor.strategy == "megatron":
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
        from verl.workers.megatron_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker, CriticWorker

        actor_rollout_cls = AsyncActorRolloutRefWorker if config.actor_rollout_ref.rollout.mode == "async" else ActorRolloutRefWorker
        ray_worker_group_cls = NVMegatronRayWorkerGroup
    else:
        raise NotImplementedError(config.actor_rollout_ref.actor.strategy)

    if skip_register:

        class SmokeRayWorkerGroup(RayWorkerGroup):
            def _init_with_resource_pool(self, resource_pool, ray_cls_with_init, bin_pack, detached):
                use_gpu = resource_pool.use_gpu
                strategy = "STRICT_PACK" if bin_pack else "PACK"
                pgs = resource_pool.get_placement_groups(strategy=strategy, device_name=self.device_name)
                world_size = resource_pool.world_size
                self._world_size = world_size
                num_gpus = 1 / resource_pool.max_colocate_count
                self._master_addr = master_addr
                self._master_port = master_port
                rank = -1
                local_world_size = resource_pool.store[0]
                for pg_idx, pg in enumerate(sort_placement_group_by_node_ip(pgs)):
                    for local_rank in range(local_world_size):
                        rank += 1
                        env_vars = {
                            "WORLD_SIZE": str(world_size),
                            "RANK": str(rank),
                            "WG_PREFIX": self.name_prefix,
                            "WG_BACKEND": "ray",
                            "RAY_LOCAL_WORLD_SIZE": str(local_world_size),
                            "RAY_LOCAL_RANK": str(local_rank),
                            "MASTER_ADDR": str(master_addr),
                            "MASTER_PORT": str(master_port),
                            "DISABLE_WORKER_INIT": "1",
                            "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
                            "HF_HOME": os.environ.get("HF_HOME", ""),
                            "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE", ""),
                            "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE", ""),
                            "VLLM_USE_V1": os.environ.get("VLLM_USE_V1", "0"),
                            "ROCR_VISIBLE_DEVICES": "",
                            "HIP_VISIBLE_DEVICES": "",
                        }
                        name = f"{self.name_prefix}WorkerDict_{pg_idx}:{local_rank}"
                        ray_cls_with_init.update_options({"runtime_env": {"env_vars": env_vars}, "name": name})
                        if detached:
                            ray_cls_with_init.update_options({"lifetime": "detached"})
                        worker = ray_cls_with_init(
                            placement_group=pg,
                            placement_group_bundle_idx=local_rank,
                            use_gpu=use_gpu,
                            num_gpus=num_gpus,
                            device_name=self.device_name,
                        )
                        self._workers.append(worker)
                        self._worker_names.append(name)

        ray_worker_group_cls = SmokeRayWorkerGroup

    role_worker_mapping = {
        Role.ActorRollout: ray.remote(actor_rollout_cls),
        Role.Critic: ray.remote(CriticWorker),
    }
    global_pool_id = "global_pool"
    resource_pool_spec = {global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes}
    mapping = {Role.ActorRollout: global_pool_id, Role.Critic: global_pool_id}

    if config.reward_model.enable:
        if config.reward_model.strategy in ["fsdp", "fsdp2"]:
            from verl.workers.fsdp_workers import RewardModelWorker
        elif config.reward_model.strategy == "megatron":
            from verl.workers.megatron_workers import RewardModelWorker
        else:
            raise NotImplementedError(config.reward_model.strategy)
        role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
        mapping[Role.RewardModel] = global_pool_id

    if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
        role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
        mapping[Role.RefPolicy] = global_pool_id

    reward_fn = load_reward_manager(config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {}))
    val_reward_fn = load_reward_manager(config, tokenizer, num_examine=1, **config.reward_model.get("reward_kwargs", {}))
    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

    train_dataset = create_rl_dataset(config.data.train_files, config.data, tokenizer, processor)
    val_dataset = create_rl_dataset(config.data.val_files, config.data, tokenizer, processor)
    train_sampler = create_rl_sampler(config.data, train_dataset)

    trainer = RayPPOTrainer(
        config=config,
        tokenizer=tokenizer,
        processor=processor,
        role_worker_mapping=role_worker_mapping,
        resource_pool_manager=resource_pool_manager,
        ray_worker_group_cls=ray_worker_group_cls,
        reward_fn=reward_fn,
        val_reward_fn=val_reward_fn,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        collate_fn=collate_fn,
        train_sampler=train_sampler,
        device_name=config.trainer.device,
    )
    # Keep the named actor handle alive for the worker-group initialization.
    _ = register_center_handle
    trainer.init_workers()
    trainer.fit()


if __name__ == "__main__":
    main()
