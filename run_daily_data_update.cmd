@echo off
setlocal
cd /d "%~dp0"
title VN30 Daily Data Update

echo Starting the guarded VN30 daily data update...
echo.

py .\scripts\run_daily_data_update.py
set "update_exit_code=%ERRORLEVEL%"

echo.
if "%update_exit_code%"=="0" (
    echo SUCCESS: Daily market data was updated and validated.
) else (
    echo FAILED: No validated daily update was completed. Review the error above.
)

echo.
echo This launcher does not place paper orders or real orders.
pause
exit /b %update_exit_code%
