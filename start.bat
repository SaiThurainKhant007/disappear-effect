@echo off
rem Sets up a local Python environment on first run, then starts the effect.
rem Any arguments are passed through, e.g.  start.bat --camera 1
setlocal
cd /d "%~dp0"

rem The environment lives at a short fixed path: pip breaks on Windows when its
rem files end up longer than 260 characters, which happens in deep folders.
set "VENV=%LOCALAPPDATA%\disappear\venv"
set "PY=%VENV%\Scripts\python.exe"

if not exist "%PY%" (
  echo Creating the Python environment in %VENV% ...
  py -3.12 -m venv "%VENV%" 2>nul || python -m venv "%VENV%"
  if not exist "%PY%" (
    echo Could not create it. Install Python 3.12 first:  winget install -e --id Python.Python.3.12
    pause
    exit /b 1
  )
)

echo Installing / checking packages...
"%PY%" -m pip install --disable-pip-version-check -q -r requirements.txt
if errorlevel 1 (
  echo Package install failed, see the messages above.
  pause
  exit /b 1
)

"%PY%" disappear.py %*
pause
