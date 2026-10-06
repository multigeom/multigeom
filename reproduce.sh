#!/usr/bin/env bash
# Stages 1-4 for one (encoder, area) configuration.
# Default: (Inception-V3, V1). Override via $ENCODER and $AREA env vars.
#
#   bash reproduce.sh                     # Inception, V1
#   ENCODER=resnet50 AREA=V4 bash reproduce.sh
#   ENCODER=onoff    AREA=V1 bash reproduce.sh
#
# Cost: ~30 A100-hours, almost all of it Stage 3 (200 diffusion epochs).
# Stages 5-6 and the figures run once all six configurations are done
# (see README.md).

set -euo pipefail

ENCODER="${ENCODER:-inception}"
AREA="${AREA:-V1}"
RUN_ROOT="${RUN_ROOT:-runs}"
DEVICE="${DEVICE:-cuda:0}"

ENC_DIR="${RUN_ROOT}/encoders/${ENCODER}/${AREA}"
RESP_DIR="${RUN_ROOT}/responses/${ENCODER}/${AREA}"
DIFF_DIR="${RUN_ROOT}/diffusion/${ENCODER}/${AREA}"
GX_DIR="${RUN_ROOT}/per_image/${ENCODER}/${AREA}"

echo "=== Reproducing ${ENCODER} × ${AREA} pipeline → ${RUN_ROOT} ==="

echo "--- Stage 1: train encoder readout on Papale (${AREA}) ---"
python stages/01_train_encoder.py \
    --config "configs/${ENCODER}.yaml" \
    --area "${AREA}" \
    --device "${DEVICE}" \
    --out-dir "${ENC_DIR}"

echo "--- Stage 2: precompute TinyImageNet responses ---"
python stages/02_precompute_responses.py \
    --config "configs/${ENCODER}.yaml" \
    --area "${AREA}" \
    --encoder-ckpt "${ENC_DIR}/best_model.pt" \
    --device "${DEVICE}" \
    --out-dir "${RESP_DIR}"

echo "--- Stage 3: train conditional DDPM on TinyImageNet ---"
python stages/03_train_diffusion.py \
    --config configs/diffusion.yaml \
    --responses "${RESP_DIR}/train_responses.pt" \
    --device "${DEVICE}" \
    --out-dir "${DIFF_DIR}"

echo "--- Stage 4: estimate G(x) (conditional-pair) and J(x) (closed form) ---"
python stages/04_estimate_gx.py \
    --config configs/estimator.yaml \
    --encoder-cfg "configs/${ENCODER}.yaml" \
    --encoder-ckpt "${ENC_DIR}/best_model.pt" \
    --diffusion-ckpt "${DIFF_DIR}/final_model.pt" \
    --area "${AREA}" \
    --device "${DEVICE}" \
    --out-dir "${GX_DIR}"

echo "--- Next, once all six configurations are done: ---"
echo "    python stages/05_compute_metrics.py --runs-dir ${RUN_ROOT} --out-dir ${RUN_ROOT}/metrics"

echo "=== Done: ${ENCODER} × ${AREA} ==="
