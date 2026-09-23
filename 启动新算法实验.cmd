@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\pythonw.exe" (
  echo Python environment missing. Please check the project setup.
  pause
  exit /b 1
)
start "" ".venv\Scripts\pythonw.exe" -m apps.desktop --gaze-experiment
