# pyright: reportMissingTypeArgument=false
from pathlib import Path

from datasets import Dataset


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "geo3k"
IMG_DIR = ROOT / "data" / "eval_subset"


def image_uri(name: str) -> dict:
    return {"image": f"file://{(IMG_DIR / name).resolve()}"}


def make_row(index: int, image_names: list[str], split: str) -> dict:
    placeholders = "\n".join("<image>" for _ in image_names)
    question = (
        f"{placeholders}\n"
        "Smoke-test prompt: describe the visible image content in one short sentence, "
        "then finish with the token smoke."
    )
    return {
        "prompt": [{"role": "user", "content": question}],
        "images": [image_uri(name) for name in image_names],
        "data_source": "w3t1_smoke",
        "reward_model": {"style": "rule", "ground_truth": "smoke"},
        "extra_info": {"index": index, "split": split, "image_count": len(image_names)},
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    train_specs = [
        ["0801.png"],
        ["0802.png"],
        ["0803.png", "0804.png"],
        ["0805.png"],
        ["0806.png"],
        ["0807.png", "0808.png"],
        ["0809.png"],
        ["0810.png"],
    ]
    test_specs = [["0811.png"], ["0812.png", "0813.png"]]

    train = [make_row(i, names, "train") for i, names in enumerate(train_specs)]
    test = [make_row(i, names, "test") for i, names in enumerate(test_specs)]

    Dataset.from_list(train).to_parquet(str(OUT / "train.parquet"))
    Dataset.from_list(test).to_parquet(str(OUT / "test.parquet"))
    print(f"wrote {OUT / 'train.parquet'} rows={len(train)}")
    print(f"wrote {OUT / 'test.parquet'} rows={len(test)}")


if __name__ == "__main__":
    main()
