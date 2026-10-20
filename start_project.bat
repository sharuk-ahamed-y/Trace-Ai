@echo off
cd /d "%~dp0"
echo ==================================================
echo   TRACEAI — Image Provenance Laboratory Launcher
echo ==================================================
if exist .venv\Scripts\python.exe (
    set "PY_BIN=.venv\Scripts\python.exe"
) else if exist C:\Python312\python.exe (
    set "PY_BIN=C:\Python312\python.exe"
) else (
    set "PY_BIN=python"
)

echo Starting TRACE AI via %PY_BIN% ...
"%PY_BIN%" start_project.py
pause
