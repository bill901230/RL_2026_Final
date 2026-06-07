# W1.T8 Verdict

- PASS: `tests/test_env_data.py` validates `FrozenSRBackbone.render` with a stubbed `OSEDiff_SD3_TEST` and never loads the real SR model (`tests/test_env_data.py:26-101`; source contract `train/grpo/sr_env.py:41-49`).
- PASS: `ZoomBaseImageDataset` reads `data/manifests/valid.txt` and yields RGB 512x512 PIL images with `image`/`path` keys (`tests/test_env_data.py:104-118`; implementation `train/dataset/zoom_dataset.py:52-66`).
- PASS: `train/evaluate.py` standalone build loads the base VLM from `cfg["model"]["policy_id"]` before applying an adapter, so the stale adapter `base_model_name_or_path` is not used as the base model (`train/evaluate.py:65-71`; config `train/configs/grpo_default.yaml:14`).
- De-hardcode: `_common.sh` now defaults `PY=${PY:-python}` (`scripts/train/experiments/_common.sh:15`); config uses local manifests and local checkpoint paths (`train/configs/grpo_default.yaml:17-18`, `:27`, `:121`, `:125`). README examples under `scripts/train/` were also made path-neutral to satisfy the no-foreign-path assertion.
- R_phr blacklist: added `first image`, `second image`, `third image`, `the first image`, `the second image` without removing existing fillers (`train/configs/grpo_default.yaml:76-91`).

Evidence:
- Pytest: `.sisyphus/evidence/task-w1t8-pytest.txt`
- Foreign path grep: `.sisyphus/evidence/task-w1t8-paths.txt`
- Blacklist grep count: `.sisyphus/evidence/task-w1t8-blacklist.txt`
