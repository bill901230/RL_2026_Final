#!/usr/bin/env python3
# pyright: reportAny=false, reportDeprecated=false, reportImplicitStringConcatenation=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnusedCallResult=false
"""Validate that the API VLM judge separates CoZ ablation arms.

Primary metric: prompt<->image grounding over saved per-scale prompts/images.
Also records deepest-scale absolute quality and deepest pairwise quality win-rates
for ours vs A3/A2.  Uses sorted first-N common sample IDs for a fixed split.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from pathlib import Path
from typing import Iterable, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train.grpo.vlm_judge import DEFAULT_BROWSER_UA, VlmJudge  # noqa: E402


PRIMARY_BASE_URL = "https://intern-vl3-8b.seanmamasde.me/v1"
PRIMARY_MODEL = "InternVL3-8B"

ARMS: dict[str, Path] = {
    "A2": ROOT / "results" / "A2_full_sr" / "per-sample",
    "A3": ROOT / "results" / "A3_full_sr" / "per-sample",
    "ours": ROOT / "results" / "w4v2_full_sr" / "per-sample",
}

GROUNDING_COLS = [f"grounding_s{i}" for i in range(4)]
PRIMARY_COLUMNS = [
    "sample_id",
    "arm",
    *GROUNDING_COLS,
    "mean_grounding_all",
    "mean_grounding_deep",
    "quality_deep",
    "pairwise_quality_ours_vs_A3_winner",
    "pairwise_quality_ours_vs_A2_winner",
]

ALT_COLUMNS = [
    "sample_id",
    "scale_prompt_index",
    "image_arm",
    "comparison",
    "winner",
]


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    text = str(value).strip()
    return text == "" or text.lower() in {"nan", "none", "null"}


def _cell_float(value: object) -> float | None:
    if _is_missing(value):
        return None
    try:
        parsed = float(str(value))
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def _fmt_cell(value: float | None) -> str:
    return "" if value is None else f"{value:.6g}"


def _mean(values: Iterable[float | None]) -> float | None:
    finite = [value for value in values if value is not None and math.isfinite(value)]
    if not finite:
        return None
    return sum(finite) / len(finite)


def _format_mean(value: float | None) -> str:
    return "nan" if value is None else f"{value:.3f}"


def _load_rows(out_path: Path) -> dict[tuple[str, str], dict[str, str]]:
    if not out_path.is_file():
        return {}
    rows: dict[tuple[str, str], dict[str, str]] = {}
    with out_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            sample_id = raw.get("sample_id", "")
            arm = raw.get("arm", "")
            if sample_id and arm:
                row = {column: raw.get(column, "") for column in PRIMARY_COLUMNS}
                rows[(sample_id, arm)] = row
    return rows


def _ordered_rows(rows_by_key: dict[tuple[str, str], dict[str, str]], sample_ids: Sequence[str]) -> list[dict[str, str]]:
    ordered: list[dict[str, str]] = []
    for sample_id in sample_ids:
        for arm in ARMS:
            key = (sample_id, arm)
            row = rows_by_key.get(key)
            if row is not None:
                ordered.append(row)
    return ordered


def _write_primary(out_path: Path, rows_by_key: dict[tuple[str, str], dict[str, str]], sample_ids: Sequence[str]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PRIMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(_ordered_rows(rows_by_key, sample_ids))


def _sample_ids(root: Path) -> set[str]:
    return {path.name for path in root.iterdir() if path.is_dir() and (path / "0.png").is_file()}


def fixed_common_ids(n: int) -> list[str]:
    common = set.intersection(*(_sample_ids(root) for root in ARMS.values()))
    ids = sorted(common)[:n]
    if len(ids) < n:
        raise SystemExit(f"only found {len(ids)} common IDs, need {n}")
    return ids


def _prompt_text(sample_dir: Path, prompt_index: int) -> str:
    return (sample_dir / "txt" / f"{prompt_index}.txt").read_text(encoding="utf-8", errors="replace").strip()


def _image_path(sample_dir: Path, image_index: int) -> Path:
    return sample_dir / f"{image_index}.png"


def _ensure_row(rows_by_key: dict[tuple[str, str], dict[str, str]], sample_id: str, arm: str) -> dict[str, str]:
    key = (sample_id, arm)
    if key not in rows_by_key:
        rows_by_key[key] = {column: "" for column in PRIMARY_COLUMNS}
        rows_by_key[key]["sample_id"] = sample_id
        rows_by_key[key]["arm"] = arm
    return rows_by_key[key]


def _refresh_means(row: dict[str, str]) -> None:
    grounding = [_cell_float(row[column]) for column in GROUNDING_COLS]
    row["mean_grounding_all"] = _fmt_cell(_mean(grounding))
    row["mean_grounding_deep"] = _fmt_cell(_mean(grounding[2:4]))


def run_primary(judge: VlmJudge, sample_ids: Sequence[str], out_path: Path) -> dict[tuple[str, str], dict[str, str]]:
    rows_by_key = _load_rows(out_path)
    for index, sample_id in enumerate(sample_ids, start=1):
        print(f"[{index}/{len(sample_ids)}] scoring {sample_id}", flush=True)
        for arm, root in ARMS.items():
            sample_dir = root / sample_id
            row = _ensure_row(rows_by_key, sample_id, arm)

            for prompt_index in range(4):
                column = f"grounding_s{prompt_index}"
                if _is_missing(row[column]):
                    score = judge.grounding(
                        _prompt_text(sample_dir, prompt_index),
                        _image_path(sample_dir, prompt_index + 1),
                    )
                    row[column] = _fmt_cell(score)
                    _refresh_means(row)

            if _is_missing(row["quality_deep"]):
                row["quality_deep"] = _fmt_cell(judge.quality(_image_path(sample_dir, 4)))

        ours_row = _ensure_row(rows_by_key, sample_id, "ours")
        for other in ("A3", "A2"):
            column = f"pairwise_quality_ours_vs_{other}_winner"
            if _is_missing(ours_row[column]):
                winner = judge.pairwise(
                    _image_path(ARMS["ours"] / sample_id, 4),
                    _image_path(ARMS[other] / sample_id, 4),
                )
                ours_row[column] = "" if winner is None else ("ours" if winner == "A" else other)

        _write_primary(out_path, rows_by_key, sample_ids)
    return rows_by_key


def _arm_summary(rows_by_key: dict[tuple[str, str], dict[str, str]], sample_ids: Sequence[str], arm: str) -> dict[str, float | int | None]:
    all_scores: list[float | None] = []
    deep_scores: list[float | None] = []
    quality_scores: list[float | None] = []
    for sample_id in sample_ids:
        row = rows_by_key[(sample_id, arm)]
        all_scores.append(_cell_float(row["mean_grounding_all"]))
        deep_scores.append(_cell_float(row["mean_grounding_deep"]))
        quality_scores.append(_cell_float(row["quality_deep"]))
    return {
        "grounding_all": _mean(all_scores),
        "grounding_deep": _mean(deep_scores),
        "quality": _mean(quality_scores),
        "n_grounding": sum(value is not None for value in all_scores),
        "n_quality": sum(value is not None for value in quality_scores),
    }


def _quality_win_rate(rows_by_key: dict[tuple[str, str], dict[str, str]], sample_ids: Sequence[str], other: str) -> tuple[int, int, float | None]:
    column = f"pairwise_quality_ours_vs_{other}_winner"
    winners = [rows_by_key[(sample_id, "ours")].get(column, "") for sample_id in sample_ids]
    valid = [winner for winner in winners if not _is_missing(winner)]
    if not valid:
        return 0, 0, None
    wins = sum(winner == "ours" for winner in valid)
    return wins, len(valid), wins / len(valid)


def print_primary_summary(rows_by_key: dict[tuple[str, str], dict[str, str]], sample_ids: Sequence[str]) -> dict[str, dict[str, float | int | None]]:
    summaries = {arm: _arm_summary(rows_by_key, sample_ids, arm) for arm in ARMS}
    print("\nFixed IDs:", ",".join(sample_ids))
    print("\narm | mean_grounding_all | mean_grounding_deep | mean_quality | n_grounding | n_quality")
    print("--- | ---: | ---: | ---: | ---: | ---:")
    for arm in ("A2", "A3", "ours"):
        summary = summaries[arm]
        print(
            f"{arm} | {_format_mean(summary['grounding_all'])} | "
            f"{_format_mean(summary['grounding_deep'])} | "
            f"{_format_mean(summary['quality'])} | "
            f"{summary['n_grounding']} | {summary['n_quality']}"
        )
    print("\nPairwise quality win-rates at deepest scale:")
    for other in ("A3", "A2"):
        wins, valid, rate = _quality_win_rate(rows_by_key, sample_ids, other)
        print(f"ours vs {other}: {wins}/{valid} = {_format_mean(rate)}")
    return summaries


def grounding_looks_discriminative(summaries: dict[str, dict[str, float | int | None]]) -> bool:
    values = [summaries[arm]["grounding_deep"] for arm in ("A2", "A3", "ours")]
    deep = [value for value in values if isinstance(value, float) and math.isfinite(value)]
    if len(deep) < 3:
        return False
    spread = max(deep) - min(deep)
    # Half a judge point is large relative to integer 1-10 ratings and enough to
    # avoid auto-running the more expensive prompt-pairwise fallback.
    return spread >= 0.5


def _load_alt_rows(out_path: Path) -> dict[tuple[str, int, str, str], dict[str, str]]:
    if not out_path.is_file():
        return {}
    rows: dict[tuple[str, int, str, str], dict[str, str]] = {}
    with out_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            try:
                scale = int(raw.get("scale_prompt_index", ""))
            except ValueError:
                continue
            key = (
                raw.get("sample_id", ""),
                scale,
                raw.get("image_arm", ""),
                raw.get("comparison", ""),
            )
            if all(key):
                rows[key] = {column: raw.get(column, "") for column in ALT_COLUMNS}
    return rows


def _write_alt(out_path: Path, rows_by_key: dict[tuple[str, int, str, str], dict[str, str]]) -> None:
    ordered = sorted(rows_by_key.values(), key=lambda row: (row["comparison"], row["sample_id"], row["scale_prompt_index"], row["image_arm"]))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ALT_COLUMNS)
        writer.writeheader()
        writer.writerows(ordered)


def run_alt_prompt_pairwise(judge: VlmJudge, sample_ids: Sequence[str], out_path: Path) -> dict[tuple[str, int, str, str], dict[str, str]]:
    """Fallback: pairwise prompt faithfulness at deep scales, symmetric refs."""

    rows_by_key = _load_alt_rows(out_path)
    for sample_index, sample_id in enumerate(sample_ids, start=1):
        print(f"[alt {sample_index}/{len(sample_ids)}] prompt-pairwise {sample_id}", flush=True)
        for other in ("A3", "A2"):
            comparison = f"ours_vs_{other}"
            for prompt_index in (2, 3):
                ours_prompt = _prompt_text(ARMS["ours"] / sample_id, prompt_index)
                other_prompt = _prompt_text(ARMS[other] / sample_id, prompt_index)
                for image_arm in ("ours", other):
                    key = (sample_id, prompt_index, image_arm, comparison)
                    row = rows_by_key.get(key)
                    if row is None:
                        row = {
                            "sample_id": sample_id,
                            "scale_prompt_index": str(prompt_index),
                            "image_arm": image_arm,
                            "comparison": comparison,
                            "winner": "",
                        }
                        rows_by_key[key] = row
                    if _is_missing(row["winner"]):
                        winner = judge.prompt_pairwise_grounding(
                            ours_prompt,
                            other_prompt,
                            _image_path(ARMS[image_arm] / sample_id, prompt_index + 1),
                        )
                        row["winner"] = "" if winner is None else ("ours" if winner == "A" else other)
        _write_alt(out_path, rows_by_key)
    return rows_by_key


def print_alt_summary(rows_by_key: dict[tuple[str, int, str, str], dict[str, str]]) -> None:
    print("\nAlternative prompt-pairwise grounding win-rates (deep scales, symmetric image refs):")
    for other in ("A3", "A2"):
        comparison = f"ours_vs_{other}"
        rows = [row for row in rows_by_key.values() if row["comparison"] == comparison and not _is_missing(row["winner"])]
        if not rows:
            print(f"ours vs {other}: 0/0 = nan")
            continue
        wins = sum(row["winner"] == "ours" for row in rows)
        print(f"ours vs {other}: {wins}/{len(rows)} = {wins / len(rows):.3f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate VLM judge discrimination on CoZ ablation outputs.")
    parser.add_argument("--n", type=int, default=20, help="first-N sorted common IDs")
    parser.add_argument("--out", type=Path, default=ROOT / ".sisyphus" / "evidence" / "judge_validate.csv")
    parser.add_argument("--alt-out", type=Path, default=ROOT / ".sisyphus" / "evidence" / "judge_validate_prompt_pairwise.csv")
    parser.add_argument("--base-url", default=os.getenv("VLM_JUDGE_BASE_URL", PRIMARY_BASE_URL))
    parser.add_argument("--api-key", default=os.getenv("VLM_JUDGE_API_KEY"))
    parser.add_argument("--model", default=os.getenv("VLM_JUDGE_MODEL", PRIMARY_MODEL))
    parser.add_argument("--ua", default=os.getenv("VLM_JUDGE_UA", DEFAULT_BROWSER_UA))
    parser.add_argument("--timeout", type=float, default=float(os.getenv("VLM_JUDGE_TIMEOUT", "45")))
    parser.add_argument("--retries", type=int, default=int(os.getenv("VLM_JUDGE_RETRIES", "1")))
    parser.add_argument("--transport", choices=("auto", "urllib", "curl"), default=os.getenv("VLM_JUDGE_TRANSPORT", "auto"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--run-alt", action="store_true", help="force the prompt-pairwise grounding fallback")
    parser.add_argument("--no-alt-if-needed", action="store_true", help="do not auto-run fallback when scalar grounding has <0.5 spread")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.api_key:
        raise SystemExit("Set VLM_JUDGE_API_KEY or pass --api-key; keys are intentionally not hardcoded.")

    judge = VlmJudge(
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        ua=args.ua,
        timeout=args.timeout,
        retries=args.retries,
        transport=args.transport,
        seed=args.seed,
        verbose=args.verbose,
    )
    sample_ids = fixed_common_ids(args.n)
    rows_by_key = run_primary(judge, sample_ids, args.out)
    summaries = print_primary_summary(rows_by_key, sample_ids)
    print(f"\nRaw primary CSV: {args.out}")

    should_run_alt = args.run_alt or (not args.no_alt_if_needed and not grounding_looks_discriminative(summaries))
    if should_run_alt:
        alt_rows = run_alt_prompt_pairwise(judge, sample_ids, args.alt_out)
        print_alt_summary(alt_rows)
        print(f"Alternative CSV: {args.alt_out}")


if __name__ == "__main__":
    main()
