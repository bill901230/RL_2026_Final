#!/usr/bin/env bash
# Download DIV2K (HR) train (800) + valid (100) images and build txt path lists
# used by the GRPO VLM trainer / evaluator.
#
# Usage:
#   bash scripts/download_div2k.sh [DATA_ROOT]
#
# DATA_ROOT defaults to ./datasets/DIV2K . After running, the following are created:
#   $DATA_ROOT/DIV2K_train_HR/*.png        (800 images)
#   $DATA_ROOT/DIV2K_valid_HR/*.png        (100 images)
#   train_utils/dataset_paths/DIV2K_TRAIN.txt   (absolute paths, one per line)
#   train_utils/dataset_paths/DIV2K_VALID.txt
#
# The trainer/evaluator resize+center-crop to 512x512 on the fly (reuse of
# inference_coz.resize_and_center_crop), so we keep the raw HR images here.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${1:-$REPO_ROOT/datasets/DIV2K}"
PATHS_DIR="$REPO_ROOT/train_utils/dataset_paths"

BASE_URL="https://data.vision.ee.ethz.ch/cvl/DIV2K"
TRAIN_ZIP="DIV2K_train_HR.zip"
VALID_ZIP="DIV2K_valid_HR.zip"

mkdir -p "$DATA_ROOT" "$PATHS_DIR"

download_and_unzip () {
    local zip_name="$1"
    local out_dir="$2"
    local zip_path="$DATA_ROOT/$zip_name"

    if [ -d "$out_dir" ] && [ "$(find "$out_dir" -name '*.png' | head -n1)" != "" ]; then
        echo "[skip] $out_dir already populated."
        return
    fi
    if [ ! -f "$zip_path" ]; then
        echo "[download] $BASE_URL/$zip_name"
        # -c resumes partial downloads; DIV2K HR zips are several GB each.
        wget -c -O "$zip_path" "$BASE_URL/$zip_name"
    fi
    echo "[unzip] $zip_name -> $DATA_ROOT"
    unzip -q -o "$zip_path" -d "$DATA_ROOT"
}

write_list () {
    local img_dir="$1"
    local list_path="$2"
    # absolute paths, sorted, one per line
    find "$img_dir" -maxdepth 1 -name '*.png' | sort > "$list_path"
    echo "[list] $(wc -l < "$list_path") images -> $list_path"
}

download_and_unzip "$TRAIN_ZIP" "$DATA_ROOT/DIV2K_train_HR"
download_and_unzip "$VALID_ZIP" "$DATA_ROOT/DIV2K_valid_HR"

write_list "$DATA_ROOT/DIV2K_train_HR" "$PATHS_DIR/DIV2K_TRAIN.txt"
write_list "$DATA_ROOT/DIV2K_valid_HR" "$PATHS_DIR/DIV2K_VALID.txt"

echo "Done. Point grpo_default.yaml dataset.train_txt / eval.valid_txt at the lists above."
