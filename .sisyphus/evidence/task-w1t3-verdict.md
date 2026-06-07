# W1.T3 Verdict

Verdict: **PASS for R_crit after fix**; **R_phr has a documented default-filler gap for the rewards owner**.

## Critic status
- `train/grpo/critic.py:50-65` now parses signed numeric ratings, falls back to `0.0` for unparsable output such as `"banana"`, and clamps to finite `[0, 1]`.
- `train/grpo/critic.py:70-75` now clamps clipscore output to finite `[0, 1]`.
- No public API signature changes and no 7B VLM load in tests.

## R_phr finding for rewards owner
- `train/grpo/rewards.py:104-108` lowercases the prompt and applies exact substring matching against configured fillers.
- Default fillers in `train/configs/grpo_default.yaml:76-87` include `"the image shows"`, but not a broader pattern such as `"image shows"`.
- Therefore `"the first image shows..."` is not penalized by the default config even though it appears to be in the intended phrase-exclusion contract. This is documented as an expected xfail in `tests/test_critic_phr.py:115-137`. I did **not** edit `rewards.py`.

## Validation evidence
- Pytest: `.sisyphus/evidence/task-w1t3-pytest.txt` (`source activate.sh && pytest tests/test_critic_phr.py -q` → `10 passed, 1 xfailed`).
- Unparsable fallback: `.sisyphus/evidence/task-w1t3-fallback.txt`.
- LSP diagnostics: error-clean for `train/grpo/critic.py` and `tests/test_critic_phr.py`.
