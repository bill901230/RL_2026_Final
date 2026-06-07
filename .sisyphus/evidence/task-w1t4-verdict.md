# W1.T4 verdict: PASS

Status: PASS.

Read refs: pre-fix `train/grpo/metrics.py` had `HIGHER_BETTER` at lines 13-18 and `IQAMetrics` at lines 21-37. Final refs: map at lines 16-21; input validation/raw scoring/oriented `score_all` at lines 42-70.

Bug/fix:
- Fixed the documented tensor contract: HWC, integer, non-finite, or out-of-[0,1] inputs now fail before pyiqa instead of being silently clamped (`metrics.py:42-53`).
- Fixed uniform direction for aggregate wrappers: `score()` remains raw pyiqa-native, while `score_all()` applies the `HIGHER_BETTER` map so NIQE is negated and all returned aggregate scores are bigger==better (`metrics.py:55-70`).

Map agreement: PASS. `test_map_matches_pyiqa` confirms `HIGHER_BETTER[name] == (not pyiqa.create_metric(name).lower_better)` for NIQE/MUSIQ/MANIQA/CLIPIQA; live pyiqa reports NIQE as lower-better and the other three as higher-better.

Verification:
- `lsp_diagnostics`: no diagnostics for `train/grpo/metrics.py` or `tests/test_metrics.py`.
- pytest evidence: `.sisyphus/evidence/task-w1t4-pytest.txt`
- mapcheck evidence: `.sisyphus/evidence/task-w1t4-mapcheck.txt`
