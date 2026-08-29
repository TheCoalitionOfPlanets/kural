@echo off
rem  One-shot setup for a fresh clone: builds the three virtual environments,
rem  installs every package, downloads the model weights, and verifies it.
rem
rem    setup.bat
rem
rem  Safe to re-run. Every step checks for its own result first, so an
rem  interrupted run resumes where it stopped and a finished run is a no-op.
rem  Nothing here retries on failure - a step that fails stops the script and
rem  says why, because a setup that silently loops is worse than one that stops.
rem
rem  The POSIX twin of this file is setup.sh. Both delegate the downloading and
rem  the verifying to tools\fetch_models.py and tools\verify_setup.py, so the
rem  two shells cannot drift about which weights land where.

setlocal EnableExtensions
set "ROOT=%~dp0"
cd /d "%ROOT%"

set "SKIP_VENVS=0"
set "SKIP_MODELS=0"
set "WITH_SET_B=0"
set "CPU_ONLY=0"

rem ------------------------------------------------------------- arguments --
rem Every branch shifts before looping back, so the parse always terminates.

:parse
if "%~1"=="" goto parsed
if /i "%~1"=="--skip-venvs"  ( set "SKIP_VENVS=1"  & shift & goto parse )
if /i "%~1"=="--skip-models" ( set "SKIP_MODELS=1" & shift & goto parse )
if /i "%~1"=="--with-set-b"  ( set "WITH_SET_B=1"  & shift & goto parse )
if /i "%~1"=="--cpu"         ( set "CPU_ONLY=1"    & shift & goto parse )
if /i "%~1"=="-h"            goto usage
if /i "%~1"=="--help"        goto usage
echo Unknown option: %~1  ^(try --help^)
exit /b 2
:parsed

rem ------------------------------------------------------------- preflight --

echo.
echo ==^> Preflight

if "%SKIP_VENVS%"=="1" goto skip_preflight

rem Each environment pins a different interpreter, so each is resolved
rem separately and by exact version. An override that points at the wrong
rem version is a hard error: falling back silently would build the environment
rem on the wrong Python, and that would not surface until a model failed to
rem load.
call :find_python 3.14 "%PYTHON_314%" PY314
if errorlevel 2 exit /b 1
if errorlevel 1 goto no_314
echo     ok   3.14  %PY314%

call :find_python 3.12 "%PYTHON_312%" PY312
if errorlevel 2 exit /b 1
if errorlevel 1 goto no_312
echo     ok   3.12  %PY312%

rem webrtcvad is a C extension with no prebuilt wheel for recent Pythons, so
rem pip compiles it. Without the MSVC build tools that fails deep inside the
rem install behind a wall of compiler output; saying so here is far cheaper.
where cl.exe >nul 2>&1
if errorlevel 1 (
    echo     warn No MSVC compiler on PATH. webrtcvad builds from source and
    echo          will fail. Install "Build Tools for Visual Studio" with the
    echo          C++ workload, then run this from a Developer Command Prompt.
) else (
    echo     ok   MSVC compiler present
)

:skip_preflight

rem ----------------------------------------------------------------- venvs --
rem The exact builds the pipeline was developed against. The two stacks
rem genuinely differ, which is the reason they are separate environments.

set "TORCH_STT=torch==2.12.1 torchvision==0.27.1"
set "TORCH_STT_INDEX=https://download.pytorch.org/whl/cu132"
set "TORCH_LLM=torch==2.13.0 torchvision==0.28.0"
set "TORCH_LLM_INDEX=https://download.pytorch.org/whl/cu130"
if "%CPU_ONLY%"=="1" set "TORCH_STT_INDEX=https://download.pytorch.org/whl/cpu"
if "%CPU_ONLY%"=="1" set "TORCH_LLM_INDEX=https://download.pytorch.org/whl/cpu"

if "%SKIP_VENVS%"=="1" (
    echo.
    echo ==^> Environments
    echo     skip --skip-venvs
    goto models
)

echo.
echo ==^> Root environment - orchestrator, capture, playback, STT [3.14]
set "MV_DIR=%ROOT%venv"
set "MV_INTERP=%PY314%"
set "MV_REQS=%ROOT%requirements\stt.txt"
set "MV_TORCH=%TORCH_STT%"
set "MV_INDEX=%TORCH_STT_INDEX%"
call :make_venv
if errorlevel 1 exit /b 1

echo.
echo ==^> reasoning\venv - Gemma 3 4B [3.12]
set "MV_DIR=%ROOT%reasoning\venv"
set "MV_INTERP=%PY312%"
set "MV_REQS=%ROOT%requirements\llm.txt"
set "MV_TORCH=%TORCH_LLM%"
set "MV_INDEX=%TORCH_LLM_INDEX%"
call :make_venv
if errorlevel 1 exit /b 1

echo.
echo ==^> tts\venv - Indic-Mio + MioCodec [3.12]
rem Indic-Mio is a causal LM like the reasoning stack, so it shares that cu130
rem build rather than needing one of its own.
set "MV_DIR=%ROOT%tts\venv"
set "MV_INTERP=%PY312%"
set "MV_REQS=%ROOT%requirements\tts.txt"
set "MV_TORCH=%TORCH_LLM%"
set "MV_INDEX=%TORCH_LLM_INDEX%"
call :make_venv
if errorlevel 1 exit /b 1

rem ---------------------------------------------------------------- models --

:models
set "ROOT_PY=%ROOT%venv\Scripts\python.exe"

if "%SKIP_MODELS%"=="1" (
    echo.
    echo ==^> Models
    echo     skip --skip-models
    goto verify
)

if not exist "%ROOT_PY%" goto no_root_env

set "FETCH_ARGS="
if "%WITH_SET_B%"=="1" set "FETCH_ARGS=--set-b"
if not "%MMS_TTS_LANGS%"=="" set "FETCH_ARGS=%FETCH_ARGS% --langs %MMS_TTS_LANGS%"

"%ROOT_PY%" "%ROOT%tools\fetch_models.py" %FETCH_ARGS%
if errorlevel 1 goto fetch_failed

rem ---------------------------------------------------------------- verify --

:verify
if not exist "%ROOT_PY%" goto nothing_to_verify

"%ROOT_PY%" "%ROOT%tools\verify_setup.py"
if errorlevel 1 goto problems

echo.
echo ==^> Setup complete
echo.
echo     Start the pipeline - speak, and it answers out loud:
echo.
echo       venv\Scripts\python.exe pipeline\run_realtime.py
echo.
echo     Check the microphone and voice detection alone first. It starts
echo     instantly and loads no models:
echo.
echo       venv\Scripts\python.exe pipeline\run_realtime.py --capture-only
echo.
echo     Or serve it to a browser instead of the local microphone:
echo.
echo       venv\Scripts\python.exe -m pipeline.server
echo.
if "%WITH_SET_B%"=="1" goto done
echo     Set A only - Indian languages and English. For Spanish, Russian,
echo     Japanese and the rest, re-run with --with-set-b and set stt.lid,
echo     stt.whisper and tts.mms_tts to enabled in
echo     pipeline\config\realtime.yaml.
echo.

:done
exit /b 0

rem =========================================================== subroutines ==

rem find_python <version> <override> <output variable>
rem
rem Sets the named variable to a plain interpreter path, which callers quote
rem like any other path. Returns 1 when nothing matches, 2 when an override
rem points at the wrong version.
:find_python
set "FP_WANT=%~1"
set "FP_OVER=%~2"
set "FP_OUT=%~3"
set "FP_PATH="
rem Prints the interpreter's own path when the version matches, and exits
rem non-zero printing nothing when it does not - so one probe both tests and
rem resolves. No carets: inside set "..." they would be stored literally and
rem end up in the Python source.
set "FP_CHECK=import sys; assert '.'.join(map(str, sys.version_info[:2])) == '%FP_WANT%'; print(sys.executable)"

rem An override is already a full path, so it is probed directly rather than
rem through `for /f`: a quoted path inside for /f meets cmd's rule about
rem stripping the outer quotes, and a Program Files install is exactly the
rem case that would break. The bare commands below have no spaces, so they
rem are safe there.
if "%FP_OVER%"=="" goto fp_search
"%FP_OVER%" -c "%FP_CHECK%" >nul 2>&1
if errorlevel 1 goto fp_bad_override
set "%FP_OUT%=%FP_OVER%"
exit /b 0

:fp_bad_override
echo.
echo   error  The override does not point at a Python %FP_WANT%:
echo          %FP_OVER%
echo.
exit /b 2

:fp_search
rem The py launcher is the reliable way to pick a version on Windows.
for /f "delims=" %%P in ('py -%FP_WANT% -c "%FP_CHECK%" 2^>nul') do set "FP_PATH=%%P"
if defined FP_PATH goto fp_found

for /f "delims=" %%P in ('python -c "%FP_CHECK%" 2^>nul') do set "FP_PATH=%%P"
if not defined FP_PATH exit /b 1

:fp_found
set "%FP_OUT%=%FP_PATH%"
exit /b 0

rem make_venv - reads MV_DIR, MV_INTERP, MV_REQS, MV_TORCH, MV_INDEX.
rem
rem Passed as globals rather than as arguments so that a path containing
rem spaces survives without a second round of quoting.
:make_venv
if exist "%MV_DIR%\" goto mv_exists
echo     creating %MV_DIR%
"%MV_INTERP%" -m venv "%MV_DIR%"
if errorlevel 1 goto mv_fail_create
goto mv_ready

:mv_exists
echo     skip %MV_DIR% exists

:mv_ready
set "MV_PY=%MV_DIR%\Scripts\python.exe"
if not exist "%MV_PY%" goto mv_fail_interp

echo     upgrading pip
"%MV_PY%" -m pip install --upgrade --quiet pip setuptools wheel
if errorlevel 1 goto mv_fail_pip

rem torch first and from its own index, so the plain-PyPI resolver in the next
rem step cannot pull a CPU build over the CUDA one.
echo     installing torch
"%MV_PY%" -m pip install --index-url %MV_INDEX% %MV_TORCH%
if errorlevel 1 goto mv_fail_torch

echo     installing %MV_REQS%
"%MV_PY%" -m pip install -r "%MV_REQS%"
if errorlevel 1 goto mv_fail_reqs

echo     ok   %MV_DIR%
exit /b 0

:mv_fail_create
echo.
echo   error  Could not create the environment at %MV_DIR%
exit /b 1

:mv_fail_interp
echo.
echo   error  No interpreter inside %MV_DIR%
echo          Delete the directory and re-run.
exit /b 1

:mv_fail_pip
echo.
echo   error  Could not upgrade pip in %MV_DIR%
exit /b 1

:mv_fail_torch
echo.
echo   error  torch install failed for %MV_DIR%
exit /b 1

:mv_fail_reqs
echo.
echo   error  Requirements install failed for %MV_DIR%
echo          If the error above mentions webrtcvad or a missing compiler,
echo          install "Build Tools for Visual Studio" with the C++ workload.
exit /b 1

rem =============================================================== failures ==

:no_314
echo.
echo   error  Python 3.14 not found - it hosts the orchestrator and STT.
echo          Install it, or point at it:  set PYTHON_314=C:\path\to\python.exe
echo.
exit /b 1

:no_312
echo.
echo   error  Python 3.12 not found - it hosts the LLM and TTS.
echo          Install it, or point at it:  set PYTHON_312=C:\path\to\python.exe
echo.
exit /b 1

:no_root_env
echo.
echo   error  The root environment does not exist yet.
echo          Run without --skip-venvs first.
echo.
exit /b 1

:fetch_failed
echo.
echo   error  Model download failed. Fix the problem above and re-run;
echo          finished downloads are skipped.
echo.
exit /b 1

:nothing_to_verify
echo.
echo ==^> Verifying
echo     warn The root environment does not exist; nothing to verify.
exit /b 1

:problems
echo.
echo ==^> Setup finished with problems
echo     Fix the items above and re-run; completed steps are skipped.
exit /b 1

rem =================================================================== help ==

:usage
echo One-shot setup for a fresh clone: builds the three virtual environments,
echo installs every package, downloads the model weights, and verifies it.
echo.
echo   setup.bat
echo.
echo Safe to re-run. Every step checks for its own result first, so an
echo interrupted run resumes where it stopped and a finished run is a no-op.
echo.
echo WHY THREE ENVIRONMENTS
echo   The model stacks pin incompatible builds: STT wants Python 3.14 with
echo   cu132 torch, while the LLM and TTS want Python 3.12 with cu130. They
echo   cannot share an interpreter, which is why each stage is a subprocess.
echo.
echo     venv\             Python 3.14   orchestrator, capture, playback, STT
echo     reasoning\venv\   Python 3.12   Gemma 3 4B
echo     tts\venv\         Python 3.12   Indic-Mio + MioCodec
echo.
echo BEFORE RUNNING YOU NEED
echo   * Python 3.14 and 3.12 installed, reachable through the py launcher
echo     or through PYTHON_314 / PYTHON_312
echo   * Build Tools for Visual Studio with the C++ workload - webrtcvad
echo     builds from source
echo   * ~25 GB free disk, and an NVIDIA GPU with 12 GB VRAM to run it
echo   * A Hugging Face token, because two of the models are gated:
echo       google/gemma-3-4b-it        accept the license on the model page
echo       ARTPARK-IISc/SraVaani-1.0   request access on the model page
echo     Then run  hf auth login,  or  set HF_TOKEN=hf_xxx
echo.
echo OPTIONS
echo   --skip-venvs    Reuse the existing environments; install nothing.
echo   --skip-models   Build the environments only; download no weights.
echo   --with-set-b    Also fetch the international stack: language router,
echo                   Whisper large-v3, MMS-TTS voices. About 6 GB more.
echo                   Suspended in the config by default, so it is opt-in.
echo                   Choose the voices with  set MMS_TTS_LANGS=spa,fra,jpn
echo   --cpu           Install CPU torch. The pipeline will not run, since
echo                   every stage sets require_cuda, but it installs.
echo   -h, --help      Show this help.
exit /b 0
