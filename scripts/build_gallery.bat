@echo off
REM ==========================================
REM  Build Gallery - Face Recognition Project
REM ==========================================

set CFG=configs\arcface_openvino.json
set ROOT=data\aligned\lfw
set OUT=data\index\lfw_arcface_raw255_bgr.npz

echo [INFO] Building gallery from %ROOT% ...
python tools\build_gallery.py ^
  --config %CFG% ^
  --root %ROOT% ^
  --out %OUT%

echo [DONE] Gallery saved to %OUT%
pause
