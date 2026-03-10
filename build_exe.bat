@echo off
setlocal

set PYTHON_EXE=C:/Users/reddy/AppData/Local/Programs/Python/Python310/python.exe

%PYTHON_EXE% -m pip install pyinstaller
%PYTHON_EXE% -m PyInstaller --noconfirm --onefile --windowed --name AI_Face_Monitor motion.py

echo Build completed. Check the dist folder.
endlocal
