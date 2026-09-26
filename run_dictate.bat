@echo off
rem Double-clickable launcher for OpenVoice. Starts the app with no console
rem window, using the virtualenv in this folder.
cd /d "%~dp0"
if not exist ".venv\Scripts\pythonw.exe" (
  echo OpenVoice is not set up yet.
  echo   python -m venv .venv
  echo   .venv\Scripts\pip install -r requirements.txt
  pause
  exit /b 1
)
start "" ".venv\Scripts\pythonw.exe" "app.py"
