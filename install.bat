@echo off
rem SREAvatar install (Windows): conda env -> pip requirements -> CUDA extensions built for this PC's GPU.
rem Needs conda, git, Visual Studio Build Tools (C++) and CUDA Toolkit 12.8.
rem Env name: %SREAVATAR_ENV% (default "sreavatar"); created with Python 3.10 if missing.
rem Works from Anaconda Prompt, plain cmd or a double-click: conda is looked up in the usual install dirs
rem when it is not on PATH.
setlocal

rem --- conda -----------------------------------------------------------------
where conda >nul 2>&1 && goto :conda_ok
for %%d in ("%USERPROFILE%\anaconda3" "%USERPROFILE%\miniconda3" "%USERPROFILE%\miniforge3"
            "%LOCALAPPDATA%\anaconda3" "%LOCALAPPDATA%\miniconda3"
            "%ProgramData%\anaconda3" "%ProgramData%\miniconda3" "%ProgramData%\miniforge3") do (
    if exist "%%~d\condabin\conda.bat" (
        echo found conda: %%~d
        set "PATH=%%~d\condabin;%PATH%"
        goto :conda_ok
    )
)
echo conda not found. Install Anaconda/Miniconda, or run this from an Anaconda Prompt.
goto :fail
:conda_ok

if "%SREAVATAR_ENV%"=="" set SREAVATAR_ENV=sreavatar
call conda activate %SREAVATAR_ENV% 2>nul || (
    echo creating conda env "%SREAVATAR_ENV%"
    call conda create -n %SREAVATAR_ENV% python=3.10 -y || (echo conda create failed & goto :fail)
    call conda activate %SREAVATAR_ENV% || (echo conda activate failed & goto :fail)
)
echo using conda env: %CONDA_DEFAULT_ENV%

python -m pip install -r "%~dp0requirements.txt" || (echo pip install failed & goto :fail)
call "%~dp0build_cuda_ext_win.bat" || (echo CUDA extension build failed & goto :fail)

echo.
echo install done. Run the viewer with:  conda activate %SREAVATAR_ENV% ^& cd main ^& python viewer.py
pause
endlocal
exit /b 0

:fail
rem keep the window open when started by double-click
pause
endlocal
exit /b 1