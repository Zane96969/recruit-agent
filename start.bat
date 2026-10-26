@echo off
chcp 65001 >nul
cd /d "%~dp0"

where python >nul 2>nul
if %errorlevel%==0 (
    python run.py %*
) else (
    py run.py %*
)

if errorlevel 1 pause
