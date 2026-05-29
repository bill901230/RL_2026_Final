"""Custom GRPO trainer for the VLM prompt extractor (LoRA-only).

Per group of G completions sharing one state:
  advantage A_i = (r_i - mean) / (std + eps)            # group-normalised
  per-token ratio  rho_t = exp(logpi_new - logpi_old)
  surrogate  = min(rho_t * A, clip(rho_t, 1±eps) * A)
  loss       = -mean_t(surrogate) + beta * KL(pi || pi_ref)

Reference policy = the *initial* LoRA adapter, kept frozen as a second PEFT
adapter ("reference") on the same base model (no extra full model in memory).
Only the trainable "policy" adapter receives gradients.
"""
import os

import torch
import torch.nn.functional as F
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    get_scheduler,
)
from peft import PeftModel, LoraConfig

try:
    import wandb
except ImportError:
    wandb = None


def _load_policy(cfg, device):
    """Return (model, processor, has_ref_adapter)."""
    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        cfg["model"]["policy_id"], torch_dtype=torch.float16, device_map=device
    )
    processor = AutoProcessor.from_pretrained(cfg["model"]["policy_id"])
    lcfg = cfg["lora"]
    # trainable adapter is named "default" so save_pretrained writes a flat dir
    # that inference_coz.py can load directly via --vlm_lora_path.
    if lcfg.get("continue_from"):
        # trainable adapter = the shipped checkpoint, kept training
        model = PeftModel.from_pretrained(
            base, lcfg["continue_from"], adapter_name="default", is_trainable=True
        )
        # frozen reference adapter = a second copy of the same init weights
        model.load_adapter(lcfg["continue_from"], adapter_name="reference")
        has_ref = True
    else:
        peft_cfg = LoraConfig(
            r=lcfg["r"], lora_alpha=lcfg["alpha"], lora_dropout=lcfg["dropout"],
            target_modules=lcfg["target_modules"], task_type="CAUSAL_LM",
        )
        model = PeftModel(base, peft_cfg, adapter_name="default")
        has_ref = False  # reference == base (adapter disabled)
    model.set_adapter("default")
    if model.generation_config.pad_token_id is None:
        model.generation_config.pad_token_id = processor.tokenizer.pad_token_id
    return model, processor, has_ref


class GRPOTrainer:
    def __init__(self, cfg, rewards, rollout_cls, dataset, sr_backbone):
        self.cfg = cfg
        self.device = cfg["device_policy"]
        self.model, self.processor, self.has_ref = _load_policy(cfg, self.device)
        self.pad_id = self.processor.tokenizer.pad_token_id

        self.rewards = rewards
        self.sr = sr_backbone
        self.dataset = dataset
        self.rollout = rollout_cls(
            cfg, self.model, self.processor, sr_backbone, rewards, self.device
        )

        opt = cfg["optim"]
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            params, lr=opt["lr"],
            betas=(opt["adam_beta1"], opt["adam_beta2"]),
            weight_decay=opt["weight_decay"], eps=opt["adam_epsilon"],
        )
        self.lr_scheduler = get_scheduler(
            opt["lr_scheduler"], optimizer=self.optimizer,
            num_warmup_steps=opt["warmup_steps"], num_training_steps=opt["max_steps"],
        )
        self.global_step = 0
        n_train = sum(p.numel() for p in params)
        print(f"[GRPO] trainable LoRA params: {n_train/1e6:.2f}M | ref_adapter={self.has_ref}")

        self.log_cfg = cfg["logging"]
        if self.log_cfg["wandb"] and wandb is not None:
            wandb.init(project=self.log_cfg["project"],
                       name=self.log_cfg.get("run_name"), config=cfg)

    # ---- log-prob utilities -------------------------------------------------
    def _token_logprobs(self, seq_ids, prompt_len, pixel_values, image_grid_thw):
        """Per-token log-probs of the completion. Returns (logp(T,), mask(T,))."""
        out = self.model(
            input_ids=seq_ids,
            attention_mask=torch.ones_like(seq_ids),
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        )
        logits = out.logits[:, :-1, :]            # predict token t+1 from t
        targets = seq_ids[:, 1:]                   # (1, L-1)
        logp_all = F.log_softmax(logits.float(), dim=-1)
        tok_logp = logp_all.gather(-1, targets.unsqueeze(-1)).squeeze(-1)[0]  # (L-1,)
        idx = torch.arange(targets.shape[1], device=seq_ids.device)
        comp = idx >= (prompt_len - 1)             # completion target positions
        nonpad = targets[0] != self.pad_id
        mask = (comp & nonpad).float()
        return tok_logp, mask

    @torch.no_grad()
    def _ref_logprobs(self, seq_ids, prompt_len, pixel_values, image_grid_thw):
        if self.has_ref:
            self.model.set_adapter("reference")
            logp, _ = self._token_logprobs(seq_ids, prompt_len, pixel_values, image_grid_thw)
            self.model.set_adapter("default")
        else:
            with self.model.disable_adapter():
                logp, _ = self._token_logprobs(seq_ids, prompt_len, pixel_values, image_grid_thw)
        return logp

    # ---- one optimizer step over a list of Groups ---------------------------
    def _step(self, groups):
        beta = self.cfg["grpo"]["kl_beta"]
        clip_eps = self.cfg["grpo"]["clip_eps"]
        adv_eps = self.cfg["grpo"]["adv_eps"]

        self.optimizer.zero_grad(set_to_none=True)
        total_loss = total_kl = 0.0
        n_comp = sum(len(g.completions) for g in groups) or 1
        reward_log = {}

        for g in groups:
            rewards = torch.tensor([c.reward for c in g.completions], dtype=torch.float32)
            adv = (rewards - rewards.mean()) / (rewards.std(unbiased=False) + adv_eps)
            pv = g.pixel_values.to(self.device)
            grid = g.image_grid_thw.to(self.device)

            for c, a in zip(g.completions, adv.tolist()):
                seq = c.seq_ids.to(self.device)
                with torch.no_grad():
                    old_logp, mask = self._token_logprobs(seq, c.prompt_len, pv, grid)
                    ref_logp = self._ref_logprobs(seq, c.prompt_len, pv, grid)
                new_logp, mask = self._token_logprobs(seq, c.prompt_len, pv, grid)

                ratio = torch.exp(new_logp - old_logp)
                surr1 = ratio * a
                surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * a
                pg = -torch.min(surr1, surr2)
                # k3 KL estimator (non-negative)
                diff = ref_logp - new_logp
                kl = torch.exp(diff) - diff - 1.0
                per_tok = pg + beta * kl
                denom = mask.sum().clamp(min=1.0)
                loss = (per_tok * mask).sum() / denom / n_comp
                loss.backward()
                total_loss += loss.item() * n_comp
                total_kl += ((kl * mask).sum() / denom).item()

            for k in (g.completions[0].raw if g.completions else {}):
                vals = [c.raw[k] for c in g.completions]
                reward_log.setdefault(f"reward/{k}", []).extend(vals)
            reward_log.setdefault("reward/total", []).extend(
                [c.reward for c in g.completions]
            )

        gn = torch.nn.utils.clip_grad_norm_(
            [p for p in self.model.parameters() if p.requires_grad],
            self.cfg["optim"]["max_grad_norm"],
        )
        self.optimizer.step()
        self.lr_scheduler.step()
        self.global_step += 1

        logs = {k: sum(v) / len(v) for k, v in reward_log.items()}
        logs.update({
            "loss/policy": total_loss / n_comp,
            "loss/kl": total_kl / n_comp,
            "grad_norm": float(gn),
            "lr": self.lr_scheduler.get_last_lr()[0],
        })
        return logs

    # ---- main loop ----------------------------------------------------------
    def train(self):
        opt = self.cfg["optim"]
        roll = self.cfg["rollout"]
        grad_accum = opt["grad_accum"]
        data_iter = _cycle(self.dataset)

        while self.global_step < opt["max_steps"]:
            groups = []
            for _ in range(roll["images_per_step"] * grad_accum):
                item = next(data_iter)
                groups.extend(self.rollout.run_episode(item["image"], item["path"]))
            if not groups:
                continue
            logs = self._step(groups)

            if self.global_step % self.log_cfg["log_every"] == 0:
                print(f"step {self.global_step} | " +
                      " | ".join(f"{k}={v:.4f}" for k, v in logs.items()))
                if self.log_cfg["wandb"] and wandb is not None:
                    wandb.log(logs, step=self.global_step)
            if self.global_step % self.log_cfg["sample_table_every"] == 0:
                self._log_samples(groups)
            if self.global_step % self.log_cfg["ckpt_every"] == 0:
                self.save_checkpoint()
            if (self.cfg["eval"]["enabled"]
                    and self.global_step % self.log_cfg["eval_every"] == 0):
                self.run_eval()

        self.save_checkpoint(final=True)
        if self.log_cfg["wandb"] and wandb is not None:
            wandb.finish()

    def _log_samples(self, groups):
        if not (self.log_cfg["wandb"] and wandb is not None):
            return
        table = wandb.Table(columns=["scale", "prompt", "reward"])
        for g in groups[:4]:
            for c in g.completions:
                table.add_data(g.scale, c.text, round(c.reward, 4))
        wandb.log({"samples": table}, step=self.global_step)

    def save_checkpoint(self, final=False):
        self.model.set_adapter("default")
        tag = "final" if final else f"checkpoint-{self.global_step}"
        out = os.path.join(self.cfg["output_dir"], tag)
        os.makedirs(out, exist_ok=True)
        self.model.save_pretrained(out, selected_adapters=["default"])
        print(f"[GRPO] saved adapter -> {out}")

    def run_eval(self):
        try:
            from train.evaluate import evaluate_adapter
        except Exception as e:
            print(f"[GRPO] eval skipped: {e}")
            return
        metrics = evaluate_adapter(self.cfg, self.model, self.processor, self.sr)
        print(f"[GRPO] eval @ {self.global_step}: {metrics}")
        if self.log_cfg["wandb"] and wandb is not None:
            wandb.log({f"eval/{k}": v for k, v in metrics.items()},
                      step=self.global_step)


def _cycle(dataset):
    while True:
        order = torch.randperm(len(dataset)).tolist()
        for i in order:
            yield dataset[i]
