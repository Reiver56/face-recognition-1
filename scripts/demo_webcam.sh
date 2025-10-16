#!/usr/bin/env bash
set -e

CFG=configs/arcface_openvino.json
IDX=data/index/lfw_arcface_raw255_bgr.npz
DET=intel/face-detection-retail-0004/FP32/face-detection-retail-0004.xml
LMK=intel/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009.xml
SAVE=data/index/lfw_plus_me.npz
ENR=data/enroll
NAME=Matteo

echo "[INFO] Starting webcam demo ..."
python demo/webcam_identify_ui.py \
  --config "$CFG" \
  --index "$IDX" \
  --det-xml "$DET" \
  --lmk-xml "$LMK" \
  --tau 0.30 --topk 5 --source 0 \
  --enroll-dir "$ENR" --enroll-name "$NAME" \
  --index-save "$SAVE"

echo "[DONE] Demo finished."
