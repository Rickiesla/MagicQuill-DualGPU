@echo off
cd /d "%~dp0"
set USE_DIRECTML=1
venv\Scripts\python.exe gradio_run.py
pause
