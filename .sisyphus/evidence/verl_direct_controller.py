# pyright: reportMissingImports=false, reportUnknownArgumentType=false
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false

import os
import pprint
import socket
import sys
from pathlib import Path
from typing import Any


EVIDENCE_DIR = Path(__file__).resolve().parent
sys.path = [
    path
    for path in sys.path
    if path and Path(path).resolve() != EVIDENCE_DIR
]

import ray
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

import verl
from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, ResourcePoolManager, Role
from verl.trainer.ppo.reward import load_reward_manager
from verl.utils import hf_processor, hf_tokenizer
from verl.utils.dataset.rl_dataset import collate_fn
from verl.utils.fs import copy_to_local


def load_config():
    assert verl.__file__ is not None
    config_dir = Path(verl.__file__).resolve().parent / "trainer" / "config"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        return compose(config_name="ppo_trainer", overrides=sys.argv[1:])


def patch_ray_worker_group_skip_register_center() -> None:
    from verl.single_controller.ray.base import RayWorkerGroup, sort_placement_group_by_node_ip

    def init_without_register_center(
        self,
        resource_pool,
        ray_cls_with_init,
        bin_pack,
        detached,
    ):
        use_gpu = resource_pool.use_gpu
        strategy = "STRICT_PACK" if bin_pack else "PACK"
        pgs = resource_pool.get_placement_groups(
            strategy=strategy, device_name=self.device_name
        )
        world_size = resource_pool.world_size
        self._world_size = world_size
        num_gpus = 1 / resource_pool.max_colocate_count

        with socket.socket() as sock:
            sock.bind(("", 0))
            master_port = str(sock.getsockname()[1])
        master_addr = ray._private.services.get_node_ip_address()
        self._master_addr = master_addr
        self._master_port = master_port

        rank = -1
        local_world_size = resource_pool.store[0]
        for pg_idx, pg in enumerate(sort_placement_group_by_node_ip(pgs)):
            assert local_world_size <= pg.bundle_count
            for local_rank in range(local_world_size):
                rank += 1
                env_vars = {
                    "WORLD_SIZE": str(world_size),
                    "RANK": str(rank),
                    "WG_PREFIX": self.name_prefix,
                    "WG_BACKEND": "ray",
                    "RAY_LOCAL_WORLD_SIZE": str(local_world_size),
                    "RAY_LOCAL_RANK": str(local_rank),
                    "LOCAL_WORLD_SIZE": str(local_world_size),
                    "LOCAL_RANK": str(local_rank),
                    "MASTER_ADDR": master_addr,
                    "MASTER_PORT": master_port,
                    "DISABLE_WORKER_INIT": "1",
                    "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
                    "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                    "ROCR_VISIBLE_DEVICES": "",
                    "HIP_VISIBLE_DEVICES": "",
                }

                import re

                cia_name = type(ray_cls_with_init.cls).__name__
                match = re.search(r"ActorClass\(([^)]+)\)", cia_name)
                cia_name = match.group(1) if match else cia_name
                name = f"{self.name_prefix}{cia_name}_{pg_idx}:{local_rank}"

                options: dict[str, Any] = {"runtime_env": {"env_vars": env_vars}, "name": name}
                if self.profile_steps:
                    options["runtime_env"]["nsight"] = self.worker_nsight_options
                ray_cls_with_init.update_options(options)
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

    RayWorkerGroup._init_with_resource_pool = init_without_register_center


def main() -> None:
    config = load_config()
    if not ray.is_initialized():
        ray.init(
            _temp_dir=os.environ["RAY_TMPDIR"],
            include_dashboard=False,
            object_store_memory=8 * 1024**3,
            num_cpus=config.ray_init.num_cpus,
        )

    print(f"DirectController hostname: {socket.gethostname()}, PID: {os.getpid()}", flush=True)
    pprint.pp(OmegaConf.to_container(config, resolve=True))
    OmegaConf.resolve(config)

    local_path = copy_to_local(
        config.actor_rollout_ref.model.path,
        use_shm=config.actor_rollout_ref.model.get("use_shm", False),
    )
    trust_remote_code = config.data.get("trust_remote_code", False)
    tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
    processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

    if config.actor_rollout_ref.rollout.name in ["vllm"]:
        from verl.utils.vllm_utils import is_version_ge

        if config.actor_rollout_ref.model.get("lora_rank", 0) > 0:
            if not is_version_ge(pkg="vllm", minver="0.7.3"):
                raise NotImplementedError("PPO LoRA is not supported before vllm 0.7.3")

    if config.actor_rollout_ref.actor.strategy in ["fsdp", "fsdp2"]:
        from verl.single_controller.ray import RayWorkerGroup
        from verl.workers.fsdp_workers import (
            ActorRolloutRefWorker,
            AsyncActorRolloutRefWorker,
            CriticWorker,
        )

        actor_rollout_cls = (
            AsyncActorRolloutRefWorker
            if config.actor_rollout_ref.rollout.mode == "async"
            else ActorRolloutRefWorker
        )
        ray_worker_group_cls = RayWorkerGroup
    else:
        raise NotImplementedError(config.actor_rollout_ref.actor.strategy)

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
        else:
            raise NotImplementedError(config.reward_model.strategy)
        role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
        mapping[Role.RewardModel] = global_pool_id

    if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
        role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
        mapping[Role.RefPolicy] = global_pool_id

    reward_fn = load_reward_manager(
        config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {})
    )
    val_reward_fn = load_reward_manager(
        config, tokenizer, num_examine=1, **config.reward_model.get("reward_kwargs", {})
    )
    resource_pool_manager = ResourcePoolManager(
        resource_pool_spec=resource_pool_spec, mapping=mapping
    )

    train_dataset = create_rl_dataset(
        config.data.train_files, config.data, tokenizer, processor
    )
    val_dataset = create_rl_dataset(config.data.val_files, config.data, tokenizer, processor)
    train_sampler = create_rl_sampler(config.data, train_dataset)

    patch_ray_worker_group_skip_register_center()

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
    trainer.init_workers()
    trainer.fit()
    ray.shutdown()


if __name__ == "__main__":
    main()
