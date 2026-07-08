@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
set "PYTHON_EXE=D:\anaconda3\envs\x-anylabeling\python.exe"
set "APP_PATH=%SCRIPT_DIR%anylabeling\app.py"

if not exist "%PYTHON_EXE%" (
    echo Python environment not found: %PYTHON_EXE%
    exit /b 1
)

if not exist "%APP_PATH%" (
    echo Application entrypoint not found: %APP_PATH%
    exit /b 1
)

set "PYTHONPATH=%SCRIPT_DIR%"
pushd "%SCRIPT_DIR%"

if "%~1"=="" (
    "%PYTHON_EXE%" "%APP_PATH%" --no-auto-update-check
) else (
    "%PYTHON_EXE%" "%APP_PATH%" %*
)

set "EXIT_CODE=%ERRORLEVEL%"
popd
exit /b %EXIT_CODE%
