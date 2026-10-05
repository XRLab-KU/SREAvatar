@echo off
rem Build SREAvatar's CUDA extensions for this PC's GPU arch.
rem Loads the MSVC environment (nvcc needs cl.exe on PATH), then hands off to
rem build_cuda_ext_win.py. Extra args pass through, e.g.:
rem     build_cuda_ext_win.bat --steps pytorch3d
rem     build_cuda_ext_win.bat --arch "8.9;12.0+PTX"
rem     build_cuda_ext_win.bat --verify-only
rem Env selection: already-activated conda env > %SREAVATAR_ENV% > "sreavatar".
rem MSVC toolset: set %SREAVATAR_VCVARS_VER% (e.g. 14.29) if CUDA rejects the default one.
setlocal

if not "%CONDA_DEFAULT_ENV%"=="" if not "%CONDA_DEFAULT_ENV%"=="base" goto :env_ok
where conda >nul 2>&1 && goto :conda_ok
for %%d in ("%USERPROFILE%\anaconda3" "%USERPROFILE%\miniconda3" "%USERPROFILE%\miniforge3"
            "%LOCALAPPDATA%\anaconda3" "%LOCALAPPDATA%\miniconda3"
            "%ProgramData%\anaconda3" "%ProgramData%\miniconda3" "%ProgramData%\miniforge3") do (
    if exist "%%~d\condabin\conda.bat" (
        set "PATH=%%~d\condabin;%PATH%"
        goto :conda_ok
    )
)
echo conda not found. Install Anaconda/Miniconda, or run this from an Anaconda Prompt.
exit /b 1
:conda_ok
if "%SREAVATAR_ENV%"=="" set SREAVATAR_ENV=sreavatar
call conda activate %SREAVATAR_ENV% || (echo conda env "%SREAVATAR_ENV%" not found. Create it first: conda create -n sreavatar python=3.10 -y & exit /b 1)
:env_ok
echo using conda env: %CONDA_DEFAULT_ENV%

rem --- MSVC ------------------------------------------------------------------
rem cl.exe on PATH alone is not enough: without INCLUDE/LIB it can't find cstddef etc.
if defined INCLUDE where cl.exe >nul 2>&1 && goto :msvc_ok
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" (echo vswhere.exe not found - is Visual Studio Build Tools installed? & exit /b 1)
for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -products * -latest -property installationPath`) do set "VSPATH=%%i"
if not exist "%VSPATH%\VC\Auxiliary\Build\vcvars64.bat" (echo vcvars64.bat not found under "%VSPATH%" & exit /b 1)
if "%SREAVATAR_VCVARS_VER%"=="" (
    echo loading MSVC: "%VSPATH%\VC\Auxiliary\Build\vcvars64.bat"
    call "%VSPATH%\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1 || (echo vcvars64.bat failed & exit /b 1)
) else (
    echo loading MSVC toolset %SREAVATAR_VCVARS_VER%: "%VSPATH%\VC\Auxiliary\Build\vcvars64.bat"
    call "%VSPATH%\VC\Auxiliary\Build\vcvars64.bat" -vcvars_ver=%SREAVATAR_VCVARS_VER% >nul 2>&1 || (echo vcvars64.bat failed & exit /b 1)
)
:msvc_ok
for /f "tokens=*" %%c in ('where cl.exe') do (set "CLPATH=%%c" & goto :cl_done)
:cl_done
echo using cl.exe: %CLPATH%

rem --- CUDA ------------------------------------------------------------------
rem torch here is cu128, so pin the 12.8 toolkit ahead of any newer nvcc on PATH.
if "%CUDA_HOME%"=="" set "CUDA_HOME=%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
set "CUDA_PATH=%CUDA_HOME%"
set "PATH=%CUDA_HOME%\bin;%PATH%"
if not exist "%CUDA_HOME%\bin\nvcc.exe" (echo nvcc.exe not found under "%CUDA_HOME%" - install CUDA Toolkit 12.8 & exit /b 1)
echo using nvcc:   %CUDA_HOME%\bin\nvcc.exe

python "%~dp0build_cuda_ext_win.py" %*
endlocal