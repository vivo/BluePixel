#!/bin/bash
set -euo pipefail

# =====================================================================
# BluePixel 4-step inference (single GPU, FP32 or INT8)
#
# FP32 model:
#   bash run_infer.sh
#
# INT8 model (vae/text_encoder/tokenizer/scheduler must come from the
# fp32 release; transformer is read from the int8 directory):
#   MODEL_DIR=/path/to/..._int8 BASE_MODEL_DIR=/path/to/BluePixel-Turbo \
#       bash run_infer.sh
#
# Quick start uses assets/test_examples.jsonl (3 samples).
# =====================================================================

MODEL_DIR="${MODEL_DIR:-/path/to/BluePixel-Turbo}"
BASE_MODEL_DIR="${BASE_MODEL_DIR:-}"

JSONL="${JSONL:-assets/test_examples.jsonl}"
OUT_DIR="${OUT_DIR:-./output/infer}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
BUCKET_AREA="${BUCKET_AREA:-16777216}"
STEPS="${STEPS:-4}"
CPS_ETA="${CPS_ETA:-0.9}"
SEED="${SEED:-42}"
SEEDS="${SEEDS:-}"
MU_MODE="${MU_MODE:-training}"
DTYPE="${DTYPE:-bf16}"
INT8_COMPUTE_DTYPE="${INT8_COMPUTE_DTYPE:-fp32}"
VAE_TILING="${VAE_TILING:-0}"
START_IDX="${START_IDX:-0}"
NUM_SAMPLES="${NUM_SAMPLES:-}"
LIMIT="${LIMIT:-}"
INSTRUCTION_FIELD="${INSTRUCTION_FIELD:-edit_prompt}"
IMAGE_FIELD="${IMAGE_FIELD:-image_name}"

echo "===== BluePixel inference ====="
echo "MODEL_DIR         : ${MODEL_DIR}"
echo "BASE_MODEL_DIR    : ${BASE_MODEL_DIR:-<same as MODEL_DIR>}"
echo "JSONL             : ${JSONL}"
echo "OUT_DIR           : ${OUT_DIR}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "BUCKET_AREA       : ${BUCKET_AREA}"
echo "STEPS / ETA       : ${STEPS} / ${CPS_ETA}"
echo "MU_MODE           : ${MU_MODE}"
echo "INT8_COMPUTE_DTYPE: ${INT8_COMPUTE_DTYPE}"
echo "==============================="

ARGS=(
    --model-dir "${MODEL_DIR}"
    --jsonl "${JSONL}"
    --out-dir "${OUT_DIR}"
    --bucket-area "${BUCKET_AREA}"
    --steps "${STEPS}"
    --cps-eta "${CPS_ETA}"
    --seed "${SEED}"
    --mu-mode "${MU_MODE}"
    --dtype "${DTYPE}"
    --int8-compute-dtype "${INT8_COMPUTE_DTYPE}"
    --instruction-field "${INSTRUCTION_FIELD}"
    --image-field "${IMAGE_FIELD}"
    --start-idx "${START_IDX}"
)

[ -n "${BASE_MODEL_DIR}" ] && ARGS+=(--base-model-dir "${BASE_MODEL_DIR}")
[ -n "${SEEDS}" ] && ARGS+=(--seeds "${SEEDS}")
[ -n "${NUM_SAMPLES}" ] && ARGS+=(--num-samples "${NUM_SAMPLES}")
[ -n "${LIMIT}" ] && ARGS+=(--limit "${LIMIT}")
[ "${VAE_TILING}" = "1" ] && ARGS+=(--vae-tiling)

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python vins_infer.py "${ARGS[@]}"

echo ""
echo "Done. Output: ${OUT_DIR}"
