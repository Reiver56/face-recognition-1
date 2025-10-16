#!/usr/bin/env bash
set -e

CFG=configs/arcface_openvino.json
ROOT=data/aligned/lfw
OUT=data/index/lfw_arcface_raw255_bgr.npz

echo "[INFO] Building gallery from $ROOT ..."
python tools/build_gallery.py \
  --config "$CFG" \
  --root "$ROOT" \
  --out "$OUT"

echo "[DONE] Gallery saved to $OUT"
