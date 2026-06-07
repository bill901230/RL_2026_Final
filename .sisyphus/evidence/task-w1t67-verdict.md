# W1.T6 + W1.T7 verdict

Status: PASS for `train/grpo/rollout.py` and `train/grpo/trainer.py`.

## Initial inspection quotes

`train/grpo/trainer.py` pre-fix R1 quote:

```python
160:             for c, a in zip(g.completions, adv.tolist()):
161:                 seq = c.seq_ids.to(self.device)
162:                 with torch.no_grad():
163:                     old_logp, mask = self._token_logprobs(seq, c.prompt_len, pv, grid)
164:                     ref_logp = self._ref_logprobs(seq, c.prompt_len, pv, grid)
165:                 new_logp, mask = self._token_logprobs(seq, c.prompt_len, pv, grid)
166:
167:                 ratio = torch.exp(new_logp - old_logp)
168:                 surr1 = ratio * a
169:                 surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * a
170:                 pg = -torch.min(surr1, surr2)
```

`train/grpo/rollout.py` pre-fix R2 quote:

```python
154:                 for seq, text in samples:
155:                     total, raw = self.rewards.compute(text or "", ctx)
156:                     group.completions.append(
157:                         Completion(seq, prompt_len, total, raw, text)
158:                     )
159:                 groups.append(group)
160:                 # advance with the best prompt of the group
161:                 best = max(group.completions, key=lambda c: c.reward)
162:                 chosen_prompt = best.text or ""
```

## Fix summary

- R1 fixed: `Completion.old_logp` is cached at rollout sample time (`rollout.py:32`, `rollout.py:107-116`) and consumed in trainer update (`trainer.py:189-196`). This restores a live clipped surrogate matching `train/README.md:24` (`trainer.py   GRPO loss (clipped surrogate + KL to frozen init adapter)`).
- R1 telemetry added: ratio mean/std and clamp-bind fraction (`trainer.py:207-214`, `trainer.py:231-245`).
- R2 fixed: rollout accepts `advance_policy={best,sampled}`, defaults to `sampled`, and keeps `best` available (`rollout.py:43`, `rollout.py:65-67`, `rollout.py:185-189`).

## Validation

- Targeted pytest: `.sisyphus/evidence/task-w1t67-pytest.txt` — 8 passed.
- R1 telemetry/no-op proof: `.sisyphus/evidence/task-w1t7-telemetry.txt`.
- R2 advance-default evidence: `.sisyphus/evidence/task-w1t6-advance.txt`.
- LSP diagnostics: no errors on changed files after fixes.
