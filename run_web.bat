@echo off
setlocal

set PYTHON_EXE=C:/Users/reddy/AppData/Local/Programs/Python/Python310/python.exe
cd /d %~dp0
%PYTHON_EXE% web\app.py

endlocal
