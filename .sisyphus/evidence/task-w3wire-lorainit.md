# W3.T4 LoRA initialization path

## Direct veRL pretrained-adapter load check

- Installed veRL 0.4.1 FSDP worker applies LoRA only by constructing a fresh `LoraConfig` in `verl/workers/fsdp_workers.py`.
- The worker passes only `task_type`, `r`, `lora_alpha`, `target_modules`, and `bias` into `LoraConfig`.
- There is no config surface in this version for loading an existing PEFT adapter into the actor before FSDP wrapping, and `exclude_modules` is not passed through by the worker even though the installed PEFT `LoraConfig` supports that parameter.
- Therefore `actor_rollout_ref.model.lora_rank=8 lora_alpha=32 target_modules=all-linear +exclude_modules='.*visual.*'` can train a fresh adapter, but cannot initialize from `ckpt/VLM_LoRA/checkpoint-10000` without patching installed veRL (forbidden).

## Chosen fallback

Fallback selected: merge the author adapter into the base Qwen2.5-VL-3B model, then let veRL train a fresh rank-8 LoRA on top of that merged policy. The KL/reference policy is the merged author behavior.

- Base: `Qwen/Qwen2.5-VL-3B-Instruct`
- Author adapter: `ckpt/VLM_LoRA/checkpoint-10000` (`r=8`, `alpha=32`, target regex from `adapter_config.json`)
- Merged model path: `ckpt/VLM_LoRA/qwen2_5_vl_3b_author_merged`
- Merge evidence log: `.sisyphus/evidence/task-w3wire-lorainit-merge.log`

Tiny veRL run should set:

```text
actor_rollout_ref.model.path=ckpt/VLM_LoRA/qwen2_5_vl_3b_author_merged
actor_rollout_ref.model.lora_rank=8
actor_rollout_ref.model.lora_alpha=32
actor_rollout_ref.model.target_modules=[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]
actor_rollout_ref.actor.use_kl_loss=True
```
