@echo off
echo ==============================================================
echo 🌕 Starting Lunar Crater Dual Training Pipeline (CUDA GPU)
echo ==============================================================
set PYTHON_EXE="C:\Users\mukes\AppData\Local\Programs\Python\Python312\python.exe"

%PYTHON_EXE% train_all.py --data "D:/moon/moon" --epochs-invariance 20 --epochs-yolo 30 --batch-yolo 16 --batch-invariance 64

pause
