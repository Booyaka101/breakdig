@echo off
setlocal
cd /d "%~dp0"
call :install
set rc=%errorlevel%
rem Double-clicked, the window would close before anyone could read the result. Explorer runs
rem cmd /c ""path" ", with that trailing space; a shell runs it without, and needs no pause.
set "cl=%cmdcmdline:"=_%"
if "%cl:~-3%"=="_ _" pause
exit /b %rc%

:install
if not exist breakdig-*.whl (
    echo The breakdig wheel is not next to install.bat. Unzip the whole folder first, then run install.bat from there.
    exit /b 1
)
py -3.11 --version >nul 2>nul || (
    echo Python 3.11 is required. Install it with: winget install Python.Python.3.11
    exit /b 1
)
where ffmpeg >nul 2>nul || echo warning: ffmpeg is not on PATH. Install it with: winget install Gyan.FFmpeg

if not exist .venv (
    py -3.11 -m venv .venv || exit /b 1
)
for %%w in (breakdig-*.whl) do (
    .venv\Scripts\python -m pip install "%%w" || exit /b 1
)

rem pip resolves torch to the CPU build on Windows; swap in the CUDA build if there is an NVIDIA driver.
where nvidia-smi >nul 2>nul || goto :done
.venv\Scripts\python -c "import sys, torch; sys.exit(0 if torch.version.cuda else 1)" 2>nul && goto :done
rem The cu130 build needs driver 580 or newer.
set cuda=cu130
for /f "tokens=1 delims=." %%v in ('nvidia-smi --query-gpu^=driver_version --format^=csv^,noheader') do (
    if %%v LSS 580 set cuda=cu126
)
.venv\Scripts\python -m pip install --force-reinstall --no-deps torch torchvision torchaudio --index-url https://download.pytorch.org/whl/%cuda% || exit /b 1

:done
echo.
echo Installed. Try:  "%~dp0breakdig.bat" index "%USERPROFILE%\Music"
echo            then: "%~dp0breakdig.bat" ui
echo Add "%~dp0" to PATH to type just "breakdig".
exit /b 0
