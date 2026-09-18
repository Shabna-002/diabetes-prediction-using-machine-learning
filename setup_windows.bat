@echo off
set PYTHON_CMD=python
py -3.10 --version >nul 2>&1 && set PYTHON_CMD=py -3.10
echo Using %PYTHON_CMD% ...
%PYTHON_CMD% -m pip install -r requirements.txt
echo.
echo Training benchmark models...
%PYTHON_CMD% train_model.py
echo.
echo Starting application...
%PYTHON_CMD% app.py
pause
