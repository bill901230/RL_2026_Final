import csv
import math
import sys
from pathlib import Path
from statistics import mean

ROOT = Path("/work/seanma0627/RL_2026_Final")
RES = ROOT / "results"
AXES = ["niqe", "musiq", "maniqa", "clipiqa", "consistency", "unique_token_ratio"]


def col_means(csv_path: Path) -> tuple[dict[str, float], int, dict[str, int]]:
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    means: dict[str, float] = {}
    finite_n: dict[str, int] = {}
    for a in AXES:
        vals = [float(r[a]) for r in rows]
        finite = [v for v in vals if math.isfinite(v)]
        means[a] = mean(finite) if finite else math.nan
        finite_n[a] = len(finite)
    return means, len(rows), finite_n


def w4v2_3seed_means() -> dict[str, float]:
    src = {r["axis"]: float(r["w4v2_3seed_mean"]) for r in
           csv.DictReader((RES / "aggregate_full_3seed.csv").open(encoding="utf-8"))}
    return {a: src[a] for a in AXES}


def main() -> None:
    arms = {
        "A0_nn_interp": RES / "A0_full.csv",
        "A1_directsr_null": RES / "A1_full.csv",
        "A2_original_coz": RES / "A2_full.csv",
        "A3_author": RES / "A3_full.csv",
    }
    means: dict[str, dict[str, float]] = {}
    for name, path in arms.items():
        m, n, finite_n = col_means(path)
        means[name] = m
        if n != 100:
            sys.exit(f"FATAL: {path} has {n} rows, expected 100")
        notes = "".join(f"  [{a}: {finite_n[a]}/100 finite]" for a in AXES if finite_n[a] != 100)
        print(f"{name}: n={n} rows OK{notes}")
    means["ours_w4v2_3seed"] = w4v2_3seed_means()

    out = RES / "baseline_ladder_n100.csv"
    cols = ["A0_nn_interp", "A1_directsr_null", "A2_original_coz",
            "A3_author", "ours_w4v2_3seed"]
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["axis", *cols, "ours_minus_A2_origcoz", "ours_minus_A3_author", "n"])
        for a in AXES:
            row = [a] + [f"{means[c][a]:.6f}" for c in cols]
            row.append(f"{means['ours_w4v2_3seed'][a] - means['A2_original_coz'][a]:.6f}")
            row.append(f"{means['ours_w4v2_3seed'][a] - means['A3_author'][a]:.6f}")
            row.append(100)
            w.writerow(row)
    print(f"wrote {out}")
    print(out.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
