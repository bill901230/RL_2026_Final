def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    """Tiny deterministic rule reward for W3.T1 stack smoke.

    Keep scoring cheap and dependency-free: reward any non-empty generated response,
    with a bonus if it happens to include the requested ground-truth token.
    """
    text = (solution_str or "").strip().lower()
    target = str(ground_truth or "").strip().lower()
    if not text:
        return {"score": 0.0, "nonempty": 0.0, "contains_target": 0.0}
    contains = 1.0 if target and target in text else 0.0
    return {"score": 0.2 + 0.8 * contains, "nonempty": 1.0, "contains_target": contains}
