@echo off
setlocal
cd /d "%~dp0"
title VN30 Guarded Daily Strategy

py .\scripts\run_daily_strategy.py
set "workflow_exit_code=%ERRORLEVEL%"

echo.
if "%workflow_exit_code%"=="0" (
    echo SUCCESS: Daily strategy preview completed.
) else (
    echo FAILED: Workflow stopped at a safety gate.
)

echo No real orders were submitted.
pause
exit /b %workflow_exit_code%
