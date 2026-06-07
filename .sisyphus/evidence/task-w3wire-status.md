# W3 wire tiny real-CoZ GRPO status

Date: 2026-05-30

## Final verdict

PASS. Tiny real-CoZ GRPO completed with full-parameter training on the merged author base, custom CoZ reward fired, and a full-model FSDP checkpoint was saved.

This is the W4 bridge path: **no LoRA** (`lora_rank=0` by config default), `actor_rollout_ref.model.path=ckpt/VLM_LoRA/qwen2_5_vl_3b_author_merged`, FSDP2 actor/ref, vLLM rollout.

## Evidence

- Run log: `.sisyphus/evidence/task-w3wire-coz-grpo.log`
- Wrapper: `.sisyphus/evidence/task_w3wire_run_coz_grpo_direct.sh`
- Result: `rc=0`
- Reward firing: 16 `[coz_reward]` records with `raw_r_anc`, `raw_r_rep`, `raw_r_phr`, `score`
- Progress: `Training Progress: 100%|...| 1/1`
- Checkpoint folder: `checkpoints/w3wire/coz_real_tiny/global_step_1/actor`
- Saved model shards:
  - `model_world_size_2_rank_0.pt`
  - `model_world_size_2_rank_1.pt`

## Exact working command

Run wrapper:

```bash
source .venv-train/bin/activate && timeout 30 ray stop --force || true && bash .sisyphus/evidence/task_w3wire_run_coz_grpo_direct.sh
```

Direct invocation inside the wrapper:

```bash
python .sisyphus/evidence/verl_direct_controller.py \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  custom_reward_function.path="$PWD/verl_custom_reward.py" \
  custom_reward_function.name=compute_score \
  reward_model.reward_manager=batch \
  actor_rollout_ref.model.path="$PWD/ckpt/VLM_LoRA/qwen2_5_vl_3b_author_merged" \
  actor_rollout_ref.model.trust_remote_code=True \
  actor_rollout_ref.actor.strategy=fsdp2 \
  actor_rollout_ref.ref.strategy=fsdp2 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.55 \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  data.train_files=data/parquet/coz_states_train.parquet \
  data.val_files=data/parquet/coz_states_val.parquet \
  data.image_key=images \
  data.train_batch_size=4 \
  data.max_prompt_length=1536 \
  data.max_response_length=64 \
  data.shuffle=False \
  data.filter_overlong_prompts=False \
  data.truncation=error \
  data.trust_remote_code=True \
  trainer.balance_batch=False \
  actor_rollout_ref.actor.ppo_mini_batch_size=4 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.use_torch_compile=False \
  actor_rollout_ref.rollout.n=4 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.max_num_batched_tokens=4096 \
  actor_rollout_ref.rollout.max_model_len=4096 \
  actor_rollout_ref.rollout.max_num_seqs=16 \
  +actor_rollout_ref.rollout.engine_kwargs.vllm.limit_mm_per_prompt.image=1 \
  +actor_rollout_ref.rollout.engine_kwargs.vllm.limit_mm_per_prompt.video=0 \
  +actor_rollout_ref.rollout.engine_kwargs.vllm.mm_processor_kwargs.max_pixels=200704 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.use_torch_compile=False \
  trainer.n_gpus_per_node=2 \
  trainer.nnodes=1 \
  trainer.total_epochs=1 \
  trainer.total_training_steps=1 \
  trainer.save_freq=1 \
  trainer.test_freq=-1 \
  trainer.val_before_train=False \
  trainer.default_local_dir=checkpoints/w3wire/coz_real_tiny \
  trainer.resume_mode=disable \
  trainer.max_actor_ckpt_to_keep=1 \
  'trainer.logger=["console"]' \
  trainer.project_name=w3wire \
  trainer.experiment_name=coz_real_tiny \
  ray_init.num_cpus=8
```

## Note on LoRA path

The LoRA path remains blocked by veRL 0.4.1 FSDP→vLLM Qwen2.5-VL LoRA sync. Config-only LLM regex matching was verified (`252` LLM modules, `0` visual modules), but veRL still rewrote visual tower keys to `.base_layer.*`. Full-parameter training avoids that broken path.
