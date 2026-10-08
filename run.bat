@echo off
rem Double-click to start TEDO AI Research Lab. The first start installs what it needs (run.py).
cd /d "%~dp0"
where py >nul 2>nul
if %errorlevel%==0 (
    py -3 run.py %*
) else (
    python run.py %*
)
if errorlevel 1 pause
