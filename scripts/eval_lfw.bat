@echo off
REM ==========================================
REM  Evaluate LFW - Face Recognition Project
REM ==========================================

set XML=public\face-recognition-resnet100-arcface-onnx\FP32\face-recognition-resnet100-arcface-onnx.xml

echo [INFO] Running LFW verification evaluation ...
python src\eval_verify_openvino.py ^
  --config "configs\arcface_openvino.json" ^
  --root "data\aligned\lfw" ^
  --pairs "data\lfw\pairs_auto.txt" ^
  --preproc raw255 ^
  --save-dir "figs"

echo [DONE] Evaluation complete.
pause
