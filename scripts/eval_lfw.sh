#!/usr/bin/env bash
set -e

XML=public/face-recognition-resnet100-arcface-onnx/FP32/face-recognition-resnet100-arcface-onnx.xml

echo "[INFO] Running LFW verification evaluation ..."
python src/eval_verify_openvino.py \
  --xml "$XML" \
  --pairs 60000 \
  --swaprb 0 \
  --preproc raw255 \
  --split lfw

echo "[DONE] Evaluation complete."
