#!/bin/bash
set -euo pipefail

# =====================================================================
# VINS-Bench v2 推理脚本 (BluePixel 4 步蒸馏模型)
# 支持断点续传：重复运行会自动跳过已完成的任务
#
# 快速测试：配置好 MODEL_DIR 后直接运行 `bash run_vins_bench_v2.sh`，
# 默认使用仓库自带的 assets/test_examples.jsonl（3 条测试样例）
# =====================================================================

# 1. 环境与路径配置
MODEL_DIR="/path/to/BluePixel-Turbo"
JSONL="assets/test_examples.jsonl"
OUT_DIR="./output/vins_bench_v2"

# 2. 运行参数配置
CUDA_VISIBLE_DEVICES=0
TARGET_PIXELS=16777216
STEPS=4
GUIDANCE=3.0
SEED=42
SEEDS=""
SHIFT=0
MU_MODE="empirical"
VAE_TILING=0
LORA_SCALE=""
INSTRUCTION_FIELD="edit_prompt"
LIMIT=""
LOCK_TIMEOUT=1800

echo "===== 启动 VINS-Bench v2 推理 ====="
echo "模型目录: ${MODEL_DIR}"
echo "数据文件: ${JSONL}"
echo "输出目录: ${OUT_DIR}"
echo "使用显卡: ${CUDA_VISIBLE_DEVICES}"
echo "指令字段: ${INSTRUCTION_FIELD}"
echo "================================="

# 3. 构造公共参数
ARGS=(
    --model-dir "${MODEL_DIR}"
    --jsonl "${JSONL}"
    --out-dir "${OUT_DIR}"
    --device-id 0
    --target-pixels "${TARGET_PIXELS}"
    --steps "${STEPS}"
    --guidance "${GUIDANCE}"
    --seed "${SEED}"
    --shift "${SHIFT}"
    --mu-mode "${MU_MODE}"
    --instruction-field "${INSTRUCTION_FIELD}"
    --lock-timeout "${LOCK_TIMEOUT}"
)

[ -n "${SEEDS}" ] && ARGS+=(--seeds "${SEEDS}")
[ -n "${LIMIT}" ] && ARGS+=(--limit "${LIMIT}")
[ -n "${LORA_SCALE}" ] && ARGS+=(--lora-scale "${LORA_SCALE}")
[ "${VAE_TILING}" = "1" ] && ARGS+=(--vae-tiling)

# 4. 启动推理
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python vins_infer.py "${ARGS[@]}"

echo ""
echo "✅ 推理完成！输出目录: ${OUT_DIR}"
