@echo off
echo ==============================================================
echo 🌙 Starting Lunar Crater Detection & Graph Matching App (CUDA)
echo ==============================================================
set PYTHON_EXE="C:\Users\mukes\AppData\Local\Programs\Python\Python312\python.exe"

%PYTHON_EXE% -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

pause
