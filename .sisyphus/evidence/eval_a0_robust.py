import math
import sys
from pathlib import Path

ROOT = Path("/work/seanma0627/RL_2026_Final")
sys.path.insert(0, str(ROOT))

from evaluate import (  # noqa: E402
    CoZEvaluator,
    discover_samples,
    load_image_tensor,
    load_pil_rgb,
    unique_token_ratio,
    write_rows,
)


def score_sample_guarded(ev: CoZEvaluator, sample) -> dict:
    deepest = load_image_tensor(sample.deepest_path)
    musiq = float(ev.iqa.score("musiq", deepest).reshape(-1)[0].item())
    maniqa = float(ev.iqa.score("maniqa", deepest).reshape(-1)[0].item())
    clipiqa = float(ev.iqa.score("clipiqa", deepest).reshape(-1)[0].item())
    try:
        raw_niqe = float(ev.iqa.score("niqe", deepest).reshape(-1)[0].item())
        niqe = -raw_niqe
    except Exception:
        niqe = math.nan

    sr_image = load_pil_rgb(sample.deepest_path)
    scale0_image = load_pil_rgb(sample.scale0_path)
    feats = ev._embed_image([sr_image, scale0_image])
    consistency = float(ev._cosine(feats[:1], feats[1:2]).reshape(-1)[0].item())
    consistency = min(1.0, max(0.0, consistency))

    return {
        "niqe": niqe,
        "musiq": musiq,
        "maniqa": maniqa,
        "clipiqa": clipiqa,
        "consistency": consistency,
        "unique_token_ratio": unique_token_ratio(sample.prompt_paths),
    }


def main() -> None:
    samples = discover_samples(ROOT / "results" / "A0_full_sr")
    ev = CoZEvaluator()
    rows = []
    niqe_fail = 0
    for s in samples:
        sc = score_sample_guarded(ev, s)
        if math.isnan(sc["niqe"]):
            niqe_fail += 1
        rows.append({"arm": "A0_full", "seed": 0, "image": s.name, **sc})
    write_rows(rows, ROOT / "results" / "A0_full.csv")
    print(f"wrote {len(rows)} rows -> results/A0_full.csv ; niqe ill-conditioned: {niqe_fail}/{len(rows)}")


if __name__ == "__main__":
    main()
