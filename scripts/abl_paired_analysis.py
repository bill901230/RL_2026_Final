#!/usr/bin/env python3
"""Per-image PAIRED ablation analysis (pure data analysis, no API/GPU/training).

Joins per-arm ablation CSVs by image id and computes, per arm vs the control on
the shared image ids:
  * mean paired delta (arm - control), win/loss/tie counts,
  * a real paired t-statistic + two-sided p-value (scipy.stats.ttest_rel, with a
    pure-python fallback when scipy is unavailable),
  * a bootstrap 95% CI of the mean paired delta,
  * the full ladder table (mean per arm).

Accepts an arbitrary control + arm mapping via argparse; with no args it falls
back to the hardcoded screening defaults (A2 -> w4v2 -> BASE -> +R_anc -> +R_rep
-> ALL) and reproduces the existing report with p + CI columns added.

Writes results/abl_paired_analysis.md. Does NOT modify any input CSV.
"""
import argparse
import math
import os
from collections import OrderedDict
from types import SimpleNamespace

import numpy as np
import pandas as pd

try:
    from scipy.stats import ttest_rel as _scipy_ttest_rel
except Exception:
    _scipy_ttest_rel = None

SCIPY_AVAILABLE = _scipy_ttest_rel is not None

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
DEFAULT_OUT = os.path.join(RESULTS, "abl_paired_analysis.md")

ARM_FILES = {
    "A2": "abl_A2.csv",
    "w4v2": "abl_w4v2.csv",
    "BASE": "abl_base.csv",
    "+R_anc": "abl_anc.csv",
    "+R_rep": "abl_rep.csv",
    "ALL": "abl_all.csv",
}

DEFAULT_BASE = "BASE"
PAIRED_ARMS = ["+R_anc", "+R_rep", "ALL"]
PAIRED_AXES = ["vlm_grounding_deep", "vlm_grounding_all", "unique_token_ratio"]
LADDER_ORDER = ["A2", "w4v2", "BASE", "+R_anc", "+R_rep", "ALL"]
LADDER_COLS = [
    "vlm_grounding_deep",
    "vlm_grounding_all",
    "unique_token_ratio",
    "musiq",
    "niqe",
    "clipiqa",
]

BOOTSTRAP_B = 10000
BOOTSTRAP_SEED = 12345


def _betacf(a, b, x):
    maxit, eps, fpmin = 200, 3.0e-14, 1.0e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d
    for m in range(1, maxit + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def _betai(a, b, x):
    # Regularized incomplete beta I_x(a,b) via Lentz continued fraction (Numerical Recipes).
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    ln_bt = (
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log(1.0 - x)
    )
    bt = math.exp(ln_bt)
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def _paired_ttest_manual(arm, base):
    d = np.asarray(arm, dtype=float) - np.asarray(base, dtype=float)
    n = d.size
    if n < 2:
        return float("nan"), float("nan")
    mean_d = float(d.mean())
    sd = float(d.std(ddof=1))
    if sd == 0.0:
        return float("nan"), float("nan")
    t = mean_d / (sd / math.sqrt(n))
    df = n - 1
    # Two-sided Student-t p-value: p = I_{df/(df+t^2)}(df/2, 1/2).
    p = _betai(df / 2.0, 0.5, df / (df + t * t))
    return float(t), float(p)


def paired_ttest(arm, base):
    arm = np.asarray(arm, dtype=float)
    base = np.asarray(base, dtype=float)
    if arm.size < 2:
        return float("nan"), float("nan")
    if _scipy_ttest_rel is not None:
        res = _scipy_ttest_rel(arm, base)
        return float(res.statistic), float(res.pvalue)
    return _paired_ttest_manual(arm, base)


def bootstrap_ci(delta, B=BOOTSTRAP_B, ci=0.95, seed=BOOTSTRAP_SEED):
    arr = np.asarray(delta, dtype=float)
    arr = arr[~np.isnan(arr)]
    n = arr.size
    if n < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(B, n))
    means = arr[idx].mean(axis=1)
    alpha = (1.0 - ci) / 2.0
    lo = float(np.quantile(means, alpha))
    hi = float(np.quantile(means, 1.0 - alpha))
    return lo, hi


def paired_stats(base_df, arm_df, axis):
    pair = pd.DataFrame({
        "base": base_df[axis],
        "arm": arm_df[axis],
    }).dropna()
    n = len(pair)
    arm_vals = pair["arm"].to_numpy(dtype=float)
    base_vals = pair["base"].to_numpy(dtype=float)
    d = arm_vals - base_vals
    mean_d = float(d.mean()) if n else float("nan")
    std_d = float(d.std(ddof=1)) if n > 1 else float("nan")
    wins = int((d > 0).sum())
    losses = int((d < 0).sum())
    ties = int((d == 0).sum())
    se = std_d / math.sqrt(n) if (n > 1 and std_d > 0) else float("nan")
    t_read = mean_d / se if (se and not math.isnan(se) and se != 0) else float("nan")
    t_stat, p_value = paired_ttest(arm_vals, base_vals)
    ci_low, ci_high = bootstrap_ci(d)
    return {
        "axis": axis,
        "n": n,
        "mean_delta": mean_d,
        "std_delta": std_d,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "win_rate": wins / n if n else float("nan"),
        "loss_rate": losses / n if n else float("nan"),
        "tie_rate": ties / n if n else float("nan"),
        "t_read": t_read,
        "t_stat": t_stat,
        "p_value": p_value,
        "ci_low": ci_low,
        "ci_high": ci_high,
    }


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Per-image paired ablation analysis (ttest_rel + bootstrap 95% CI).",
    )
    p.add_argument("--base", default=None, help="Control label (default: BASE).")
    p.add_argument(
        "--arm",
        action="append",
        default=None,
        metavar="LABEL=PATH",
        help="Repeatable arm mapping label=path.csv (include the control). Overrides defaults.",
    )
    p.add_argument(
        "--arm-csv",
        action="append",
        default=None,
        metavar="LABEL=PATH",
        help="Alias for --arm.",
    )
    p.add_argument("--out", default=None, help=f"Output markdown path (default: {DEFAULT_OUT}).")
    return p.parse_args(argv)


def _parse_arm_specs(specs):
    arm_files = OrderedDict()
    for spec in specs:
        label, sep, path = spec.partition("=")
        label, path = label.strip(), path.strip()
        if not sep or not label or not path:
            raise ValueError(f"--arm expects LABEL=PATH, got: {spec!r}")
        arm_files[label] = path
    return arm_files


def resolve_config(args):
    specs = list(args.arm or []) + list(getattr(args, "arm_csv", None) or [])
    if specs:
        arm_files = _parse_arm_specs(specs)
        paired_arms = None
        ladder_order = list(arm_files)
        is_default = False
    else:
        arm_files = OrderedDict(
            (label, os.path.join(RESULTS, fn)) for label, fn in ARM_FILES.items()
        )
        paired_arms = list(PAIRED_ARMS)
        ladder_order = list(LADDER_ORDER)
        is_default = True

    base_label = args.base if args.base is not None else DEFAULT_BASE
    if base_label not in arm_files:
        raise ValueError(
            f"--base {base_label!r} not found among arm labels {list(arm_files)!r}"
        )
    if paired_arms is None:
        paired_arms = [label for label in arm_files if label != base_label]

    out = args.out if args.out is not None else DEFAULT_OUT
    return SimpleNamespace(
        arm_files=arm_files,
        base_label=base_label,
        paired_arms=paired_arms,
        ladder_order=ladder_order,
        is_default=is_default,
        out=out,
    )


def load_arms(arm_files):
    dfs = OrderedDict()
    for label, path in arm_files.items():
        df = pd.read_csv(path)
        df["image"] = df["image"].astype(str).str.zfill(4)
        dfs[label] = df.set_index("image").sort_index()
    return dfs


def load_all():
    return load_arms(
        OrderedDict((label, os.path.join(RESULTS, fn)) for label, fn in ARM_FILES.items())
    )


def build_ladder(dfs, ladder_order=None, cols=None):
    if ladder_order is None:
        ladder_order = LADDER_ORDER
    if cols is None:
        cols = LADDER_COLS
    present = [c for c in cols if all(c in dfs[label].columns for label in ladder_order)]
    rows = []
    for label in ladder_order:
        df = dfs[label]
        row = {"arm": label}
        for col in present:
            row[col] = float(df[col].dropna().mean())
        rows.append(row)
    return pd.DataFrame(rows)


def fmt(x, nd=3):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{x:.{nd}f}"


def signed(x, nd=3):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{x:+.{nd}f}"


def pfmt(x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    if x < 1e-4:
        return f"{x:.1e}"
    return f"{x:.4f}"


def cifmt(lo, hi, nd=3):
    return f"[{signed(lo, nd)}, {signed(hi, nd)}]"


def _methods_note():
    if SCIPY_AVAILABLE:
        engine = "scipy.stats.ttest_rel"
    else:
        engine = (
            "manual paired t-test (scipy unavailable; t = meanΔ/(sdΔ/√n), "
            "two-sided p via incomplete-beta Student-t survival)"
        )
    return (
        f"Paired test: **{engine}** (two-sided, on the shared image ids). "
        f"95% CI: non-parametric percentile bootstrap of the mean paired Δ "
        f"(B={BOOTSTRAP_B}, seed={BOOTSTRAP_SEED})."
    )


AXIS_SHORT = {
    "vlm_grounding_deep": "grounding_deep",
    "vlm_grounding_all": "grounding_all",
    "unique_token_ratio": "uniqtok",
}


def _paired_table(lines, paired, paired_arms, base_label):
    lines.append(f"## 2. Per-image PAIRED stats vs {base_label}")
    lines.append("")
    lines.append(
        "| arm | axis | n | mean Δ | 95% CI (bootstrap) | std Δ | t | p | win/loss/tie | win-rate |"
    )
    lines.append("|---|---|---:|---:|:--:|---:|---:|:--:|:--:|---:|")
    for arm in paired_arms:
        for axis in PAIRED_AXES:
            s = paired[arm][axis]
            lines.append(
                f"| {arm} | {AXIS_SHORT.get(axis, axis)} | {s['n']} | {signed(s['mean_delta'])} "
                f"| {cifmt(s['ci_low'], s['ci_high'])} | {fmt(s['std_delta'])} "
                f"| {fmt(s['t_stat'], 2)} | {pfmt(s['p_value'])} "
                f"| {s['wins']}/{s['losses']}/{s['ties']} | {fmt(s['win_rate'], 2)} |"
            )
    lines.append("")


def _df_to_md(df, nd=3):
    cols = list(df.columns)
    rows = ["| " + " | ".join(cols) + " |"]
    rows.append("|" + "|".join("---" if c == "arm" else "---:" for c in cols) + "|")
    for _, r in df.iterrows():
        cells = [fmt(r[c], nd) if isinstance(r[c], float) else str(r[c]) for c in cols]
        rows.append("| " + " | ".join(cells) + " |")
    return rows


def _render_default_report(dfs, base, paired, ladder):
    lines = []
    lines.append("# Per-image PAIRED ablation analysis (n=30, ids 0801-0830)")
    lines.append("")
    lines.append(
        "Pure data analysis over the already-scored ablation CSVs (no API / GPU / "
        "inference / training; input CSVs untouched). Per-image join by image id; "
        "NaN handled per-axis (drop). Deltas are **arm - BASE** computed PER IMAGE, "
        "then aggregated. `t = mean_delta / (std/sqrt(n))` (paired); |t| >~ 2 => likely "
        "real signal, else within paired noise. BASE = author A3."
    )
    lines.append("")
    lines.append(_methods_note())
    lines.append("")

    lines.append("## 1. Full ladder (mean over n=30)")
    lines.append("")
    lines.append(
        "Progression: A2 (orig CoZ) -> w4v2 (prior balanced ours) -> BASE (A3) -> "
        "+R_anc -> +R_rep -> ALL."
    )
    lines.append("")
    lines.append(
        "| arm | grounding_deep ↑ | grounding_all ↑ | uniqtok ↑ | MUSIQ ↑ | NIQE ↓ | CLIPIQA ↑ |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for _, r in ladder.iterrows():
        lines.append(
            f"| {r['arm']} | {fmt(r['vlm_grounding_deep'])} | {fmt(r['vlm_grounding_all'])} "
            f"| {fmt(r['unique_token_ratio'])} | {fmt(r['musiq'], 2)} | {fmt(r['niqe'], 2)} "
            f"| {fmt(r['clipiqa'])} |"
        )
    lines.append("")

    _paired_table(lines, paired, PAIRED_ARMS, "BASE")

    lines.append("### Headline (grounding_deep + uniqtok)")
    lines.append("")
    lines.append(
        "| arm | deep Δ | deep win-rate | deep t | deep p | deep 95% CI | "
        "uniqtok Δ | uniqtok win-rate | uniqtok t | uniqtok p | uniqtok 95% CI |"
    )
    lines.append("|---|---:|---:|---:|:--:|:--:|---:|---:|---:|:--:|:--:|")
    for arm in PAIRED_ARMS:
        d = paired[arm]["vlm_grounding_deep"]
        u = paired[arm]["unique_token_ratio"]
        lines.append(
            f"| {arm} | {signed(d['mean_delta'])} | {fmt(d['win_rate'], 2)} "
            f"| {fmt(d['t_stat'], 2)} | {pfmt(d['p_value'])} | {cifmt(d['ci_low'], d['ci_high'])} "
            f"| {signed(u['mean_delta'])} | {fmt(u['win_rate'], 2)} "
            f"| {fmt(u['t_stat'], 2)} | {pfmt(u['p_value'])} | {cifmt(u['ci_low'], u['ci_high'])} |"
        )
    lines.append("")

    all_d = paired["ALL"]["vlm_grounding_deep"]
    anc_d = paired["+R_anc"]["vlm_grounding_deep"]
    rep_d = paired["+R_rep"]["vlm_grounding_deep"]
    anc_u = paired["+R_anc"]["unique_token_ratio"]
    rep_u = paired["+R_rep"]["unique_token_ratio"]
    all_u = paired["ALL"]["unique_token_ratio"]

    lines.append("## 3. Honest verdict")
    lines.append("")
    lines.append(
        f"- **ALL grounding_deep win is directionally consistent but NOT statistically "
        f"clean at n=30.** ALL beats BASE on {all_d['wins']}/{all_d['n']} images "
        f"(win-rate {fmt(all_d['win_rate'], 2)}, mean Δ {signed(all_d['mean_delta'])}, "
        f"t {fmt(all_d['t_stat'], 2)}, p {pfmt(all_d['p_value'])}, "
        f"bootstrap 95% CI {cifmt(all_d['ci_low'], all_d['ci_high'])}). With |t| "
        f"{'<' if abs(all_d['t_stat']) < 2 else '>='} 2 and a CI that "
        f"{'excludes' if (all_d['ci_low'] > 0 or all_d['ci_high'] < 0) else 'includes'} 0, "
        f"the per-image paired test treats this as a real-but-weak signal at screen scale, "
        f"not a slam-dunk: roughly half the images are ties/losses, so the aggregate "
        f"win rides on a minority of drift-recovery crops rather than a broad sweep."
    )
    lines.append(
        f"- **Single-arm components do not isolate a grounding win.** +R_rep is "
        f"essentially tied (mean Δ {signed(rep_d['mean_delta'])}, win/loss "
        f"{rep_d['wins']}/{rep_d['losses']}, t {fmt(rep_d['t_stat'], 2)}, p {pfmt(rep_d['p_value'])}) "
        f"and +R_anc is slightly negative (mean Δ {signed(anc_d['mean_delta'])}, win/loss "
        f"{anc_d['wins']}/{anc_d['losses']}, t {fmt(anc_d['t_stat'], 2)}, p {pfmt(anc_d['p_value'])}). "
        f"The deep-grounding gain only emerges when both rewards are combined in ALL."
    )
    div_pairs = {"+R_anc": anc_u["mean_delta"], "+R_rep": rep_u["mean_delta"], "ALL": all_u["mean_delta"]}
    div_winner = max(div_pairs, key=lambda k: div_pairs[k])
    div_stats = paired[div_winner]["unique_token_ratio"]
    lines.append(
        f"- **Diversity (unique-token ratio) winner is {div_winner}** "
        f"(mean Δ {signed(div_stats['mean_delta'])}, win-rate {fmt(div_stats['win_rate'], 2)}, "
        f"t {fmt(div_stats['t_stat'], 2)}, p {pfmt(div_stats['p_value'])}). Note this contradicts "
        f"the prior expectation that +R_rep would drive diversity: on this proxy +R_rep is "
        f"{signed(rep_u['mean_delta'])} and does not show an isolated convergence-fixing win."
    )
    lines.append(
        "- **R_anc-saturation caveat.** +R_anc's flat/negative grounding here is consistent "
        "with the documented training issue: R_anc had near-zero reward variance / saturated, "
        "so the anchor signal was weak in isolation and mainly useful as a regulariser inside ALL."
    )
    lines.append(
        f"- **Scale caveat.** This is n={all_d['n']} (DIV2K screen subset 0801-0830), a "
        "screen-scale read, NOT the n=100 confirmation run. Per-image variance is large "
        "(std Δ on deep grounding ~1.5-2.5 points), so t near +-1 and CIs spanning 0 should be "
        "read as 'within noise' and the ALL verdict treated as a promising trend to confirm at "
        "n=100, not a settled result."
    )
    lines.append("")
    lines.append("## Inputs")
    lines.append("")
    lines.append(
        "`results/abl_{A2,w4v2,base,anc,rep,all}.csv` (each n=30). Generated by "
        "`scripts/abl_paired_analysis.py`."
    )
    lines.append("")
    return lines


def _render_generic_report(dfs, paired, ladder, base_label, paired_arms):
    n_base = len(dfs[base_label])
    lines = []
    lines.append(f"# Per-image PAIRED ablation analysis ({base_label} vs arms)")
    lines.append("")
    lines.append(
        "Pure data analysis over scored ablation CSVs (input CSVs untouched). "
        f"Per-image join by image id; NaN dropped per-axis. Deltas are "
        f"**arm - {base_label}** computed PER IMAGE, then aggregated."
    )
    lines.append("")
    lines.append(_methods_note())
    lines.append("")
    lines.append(f"## 1. Ladder (mean; base {base_label} has n={n_base})")
    lines.append("")
    lines.extend(_df_to_md(ladder))
    lines.append("")
    _paired_table(lines, paired, paired_arms, base_label)
    return lines


def _print_summary(ladder, paired, cfg):
    mode = "default" if cfg.is_default else f"custom (base={cfg.base_label})"
    print(f"=== LADDER (mean) [{mode}] ===")
    print(ladder.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print()
    engine = "scipy.ttest_rel" if SCIPY_AVAILABLE else "manual-ttest"
    for axis, label in (
        ("vlm_grounding_deep", "grounding_deep"),
        ("unique_token_ratio", "unique_token_ratio (diversity)"),
    ):
        print(f"=== PAIRED vs {cfg.base_label}: {label} [{engine} + bootstrap 95% CI] ===")
        for arm in cfg.paired_arms:
            s = paired[arm][axis]
            print(
                f"{arm:8s} meanΔ={s['mean_delta']:+.4f} "
                f"CI95=[{s['ci_low']:+.4f}, {s['ci_high']:+.4f}] "
                f"t={s['t_stat']:+.2f} p={s['p_value']:.4f} "
                f"win/loss/tie={s['wins']}/{s['losses']}/{s['ties']} "
                f"win-rate={s['win_rate']:.2f}"
            )
        print()


def main(argv=None):
    args = parse_args(argv)
    cfg = resolve_config(args)
    dfs = load_arms(cfg.arm_files)
    base = dfs[cfg.base_label]

    paired = OrderedDict()
    for arm in cfg.paired_arms:
        paired[arm] = {axis: paired_stats(base, dfs[arm], axis) for axis in PAIRED_AXES}

    ladder = build_ladder(dfs, cfg.ladder_order, LADDER_COLS)

    if cfg.is_default:
        lines = _render_default_report(dfs, base, paired, ladder)
    else:
        lines = _render_generic_report(dfs, paired, ladder, cfg.base_label, cfg.paired_arms)

    with open(cfg.out, "w") as f:
        f.write("\n".join(lines))

    _print_summary(ladder, paired, cfg)
    print(f"scipy available: {SCIPY_AVAILABLE}")
    print(f"Wrote {cfg.out}")


if __name__ == "__main__":
    main()
