@echo off
echo ===================================================
echo   Starting Diabetes ML Clinical Decision Web Portal
echo ===================================================
echo.
echo Opening Web Portal at http://localhost:5000 ...
start "" "http://localhost:5000"
py -3.10 server.py
pause
