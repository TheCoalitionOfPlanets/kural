@echo off
setlocal EnableExtensions EnableDelayedExpansion
rem
rem One-shot setup for a fresh clone: builds the three venvs, installs every
rem package, and downloads every model the real-time pipeline needs.
rem
rem   download.bat
rem
rem Safe to re-run. Every step checks for its own result first, so an interrupted
rem run resumes instead of starting over, and a finished run is a no-op.
rem
rem WHY THREE VENVS
rem The three model stacks pin incompatible dependencies - STT wants Python 3.14
rem with a cu132 torch, and both the LLM and TTS want 3.12 with cu130 but are
rem kept in separate venvs regardless (see pipeline\README.md). They cannot
rem share one interpreter, so each stage runs as a subprocess in its own venv.
rem
rem BEFORE RUNNING YOU NEED
rem   * Python 3.14 and Python 3.12 installed (the py launcher finds them)
rem   * ~27 GB free disk, and an NVIDIA GPU with 12 GB VRAM to actually run it
rem   * A Hugging Face account and token, because two of the models are gated:
rem       - google/gemma-3-4b-it       accept the Gemma license on the model page
rem       - ARTPARK-IISc/SraVaani-1.0  request access on the model page
rem     Then either run `hf auth login`, or set HF_TOKEN=hf_xxx.
rem
rem OPTIONAL - INTERNATIONAL LANGUAGES
rem   The local models are Indic: they hear and speak the scheduled Indian
rem   languages plus English, and nothing else. Spanish, Russian, Japanese and
rem   the rest are routed to ElevenLabs instead. To enable that, set an API
rem   key before running the pipeline:
rem       set ELEVENLABS_API_KEY=sk_xxx
rem   Without it everything still works - Indic and English turns are unaffected
rem   - and international turns are reported as unavailable rather than
rem   transcribed into gibberish by a model that cannot hear them.
rem
rem OPTIONS
rem   --skip-venvs    Reuse existing venvs; do not create or install.
rem   --skip-models   Set up venvs only; download no model weights.
rem   --cpu           Install CPU torch everywhere. The pipeline will not run
rem                   (STT, the LLM, and TTS all set require_cuda), but this
rem                   makes the repo installable on a machine with no NVIDIA GPU.
rem   -h, --help      Show this help.

rem cmd has no `set -e`: every fallible command is followed by an explicit
rem errorlevel check, and the failure paths all funnel through :die.

set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"
cd /d "%ROOT%" || exit /b 1

set "SKIP_VENVS=0"
set "SKIP_MODELS=0"
set "CPU_ONLY=0"

:parse_args
if "%~1"=="" goto args_done
if /i "%~1"=="--skip-venvs"  ( set "SKIP_VENVS=1"  & shift & goto parse_args )
if /i "%~1"=="--skip-models" ( set "SKIP_MODELS=1" & shift & goto parse_args )
if /i "%~1"=="--cpu"         ( set "CPU_ONLY=1"    & shift & goto parse_args )
if /i "%~1"=="-h"            ( call :usage & exit /b 0 )
if /i "%~1"=="--help"        ( call :usage & exit /b 0 )
echo Unknown option: %~1 (try --help) 1>&2
exit /b 2
:args_done

rem ---------------------------------------------------------------- output ----
rem
rem ANSI colours, which every supported Windows 10/11 console understands. ESC
rem is built with a for/f rather than a literal byte so the file stays plain
rem ASCII and survives being opened in an editor.
for /f %%E in ('echo prompt $E ^| cmd') do set "ESC=%%E"
set "BOLD=%ESC%[1m"
set "DIM=%ESC%[2m"
set "RED=%ESC%[31m"
set "GREEN=%ESC%[32m"
set "YELLOW=%ESC%[33m"
set "RESET=%ESC%[0m"

set "FAILED=0"

rem ------------------------------------------------------- python versions ----
rem
rem The venvs pin different interpreters, so each is resolved separately. On
rem Windows the `py` launcher is the reliable way to pick a version. Override
rem either by setting PYTHON_314 or PYTHON_312 to an absolute path first.

if not "%SKIP_VENVS%"=="1" goto build_venvs
call :step "Virtual environments"
call :skip "--skip-venvs"
goto models
:build_venvs

call :step "Locating interpreters"
call :find_python 3.14 PYTHON_314 PY314
if errorlevel 1 exit /b 1
call :find_python 3.12 PYTHON_312 PY312
if errorlevel 1 exit /b 1
call :ok "3.14: !PY314!"
call :ok "3.12: !PY312!"

rem ------------------------------------------------------------------ venvs ---
rem
rem CUDA wheel indexes. These are the exact builds the pipeline was developed
rem against; the two stacks genuinely differ, which is why they are separate
rem venvs in the first place.
set "TORCH_STT=torch==2.12.1 torchvision==0.27.1"
set "TORCH_STT_INDEX=https://download.pytorch.org/whl/cu132"
set "TORCH_LLM=torch==2.13.0 torchvision==0.28.0"
set "TORCH_LLM_INDEX=https://download.pytorch.org/whl/cu130"
if "%CPU_ONLY%"=="1" (
  set "TORCH_STT_INDEX=https://download.pytorch.org/whl/cpu"
  set "TORCH_LLM_INDEX=https://download.pytorch.org/whl/cpu"
)

call :step "Root venv - orchestrator + SraVaani STT (Python 3.14)"
call :make_venv "%ROOT%\venv" "!PY314!" "%ROOT%\requirements\stt.txt" "!TORCH_STT!" "!TORCH_STT_INDEX!"
if errorlevel 1 exit /b 1

call :step "reasoning\venv - Gemma 3 4B (Python 3.12)"
call :make_venv "%ROOT%\reasoning\venv" "!PY312!" "%ROOT%\requirements\llm.txt" "!TORCH_LLM!" "!TORCH_LLM_INDEX!"
if errorlevel 1 exit /b 1

call :step "tts\venv - Indic-Mio (Python 3.12)"
rem Indic-Mio is a causal LM like the reasoning stack, so it shares the same
rem cu130 torch build rather than needing a build of its own.
call :make_venv "%ROOT%\tts\venv" "!PY312!" "%ROOT%\requirements\tts.txt" "!TORCH_LLM!" "!TORCH_LLM_INDEX!"
if errorlevel 1 exit /b 1

rem ----------------------------------------------------------------- models ---
:models

call :venv_python "%ROOT%\venv" HF_PY
if errorlevel 1 set "HF_PY="

if not "%SKIP_MODELS%"=="1" goto fetch_models
call :step "Models"
call :skip "--skip-models"
goto verify
:fetch_models

rem Each step is written flat - test, jump past on a hit - rather than as an
rem if/else. A `call` and an `exit /b` inside a parenthesized block do not
rem compose in cmd: the block is parsed as one command before it runs, and the
rem early exit unwinds through the wrong scope. Labels have neither problem.

rem SraVaani STT - a ~900 MB FP16 TorchScript archive, plus the remote-code
rem modules that trust_remote_code=True loads alongside it.
call :step "STT model - ARTPARK-IISc/SraVaani-1.0 (~900 MB)"
if exist "%ROOT%\stt\models\model-asr.fp16.ts" (
  call :skip "stt\models already populated"
  goto stt_done
)
call :hf_download "ARTPARK-IISc/SraVaani-1.0" "%ROOT%\stt\models" "SraVaani STT"
if errorlevel 1 exit /b 1
:stt_done

rem MMS-LID - the language gate in front of SraVaani. It decides, from the
rem waveform alone, whether a turn is one the local Indic models can handle or
rem one that belongs to ElevenLabs. Not gated, and not required: without it the
rem pipeline routes everything locally, exactly as it did before.
call :step "Language ID - facebook/mms-lid-126 (~1.2 GB)"
call :has_weights "%ROOT%\stt\models\mms-lid-126"
if not errorlevel 1 (
  call :skip "stt\models\mms-lid-126 already populated"
  goto lid_done
)
call :hf_download "facebook/mms-lid-126" "%ROOT%\stt\models\mms-lid-126" "MMS language ID" "*.json,*.safetensors"
if errorlevel 1 exit /b 1
rem Older repos ship only a .bin, and asking for both formats up front would
rem download the weights twice on the repos that carry both.
if exist "%ROOT%\stt\models\mms-lid-126\*.safetensors" goto lid_done
call :hf_download "facebook/mms-lid-126" "%ROOT%\stt\models\mms-lid-126" "MMS language ID (pytorch weights)" "*.json,*.bin"
if errorlevel 1 exit /b 1
:lid_done

rem Gemma 3 4B IT - ~8.6 GB of bf16 safetensors on disk, quantized to 4-bit
rem NF4 at load time so all three models fit in 12 GB.
call :step "LLM - google/gemma-3-4b-it (~8.6 GB)"
if exist "%ROOT%\reasoning\models\gemma-3-4b-it\model.safetensors.index.json" (
  call :skip "reasoning\models\gemma-3-4b-it already populated"
  goto llm_done
)
rem Only what transformers loads: this skips any GGUF/ONNX variants the repo
rem may carry, which the worker never touches.
call :hf_download "google/gemma-3-4b-it" "%ROOT%\reasoning\models\gemma-3-4b-it" "Gemma 3 4B IT" "*.json,*.safetensors,*.model,*.txt,*.md"
if errorlevel 1 exit /b 1
:llm_done

rem Indic-Mio - a ~1.2 GB bf16 causal LM, plus the MioCodec decoder it needs
rem to turn generated audio tokens into a waveform. Neither repo is gated.
call :step "TTS model - SPRINGLab/Indic-Mio (~1.2 GB)"
if exist "%ROOT%\tts\models\Indic-Mio\model.safetensors" (
  call :skip "tts\models\Indic-Mio already populated"
  goto tts_done
)
call :hf_download "SPRINGLab/Indic-Mio" "%ROOT%\tts\models\Indic-Mio" "Indic-Mio TTS"
if errorlevel 1 exit /b 1
:tts_done

call :step "TTS codec - Aratako/MioCodec-25Hz-24kHz"
call :dir_nonempty "%ROOT%\tts\models\MioCodec-25Hz-24kHz"
if not errorlevel 1 (
  call :skip "tts\models\MioCodec-25Hz-24kHz already populated"
  goto codec_done
)
call :hf_download "Aratako/MioCodec-25Hz-24kHz" "%ROOT%\tts\models\MioCodec-25Hz-24kHz" "MioCodec"
if errorlevel 1 exit /b 1
:codec_done

rem ------------------------------------------------------------------ check ---
:verify

call :step "Verifying"

call :check_import "%ROOT%\venv" "root venv  (torch, transformers, sounddevice)" "torch, transformers, sounddevice, yaml"
call :check_import "%ROOT%\reasoning\venv" "reasoning  (torch, transformers, bitsandbytes)" "torch, transformers, bitsandbytes"
call :check_import "%ROOT%\tts\venv" "tts        (torch, transformers, miocodec)" "torch, transformers, miocodec"

if "%SKIP_MODELS%"=="1" goto cuda_check

call :require_file "%ROOT%\stt\models\model-asr.fp16.ts" "STT weights"
call :require_file "%ROOT%\reasoning\models\gemma-3-4b-it\model.safetensors.index.json" "LLM weights"
call :require_file "%ROOT%\tts\models\Indic-Mio\model.safetensors" "TTS weights"

rem Optional, so a miss is a warning that does not fail the run: the pipeline
rem is fully functional for Indic and English without it.
call :has_weights "%ROOT%\stt\models\mms-lid-126"
if not errorlevel 1 (
  call :ok "language ID weights"
) else (
  call :warn "language ID weights missing - international languages will be routed"
  call :warn "  to the local Indic models, which cannot transcribe them"
)

if defined ELEVENLABS_API_KEY (
  call :ok "ELEVENLABS_API_KEY is set"
) else (
  call :warn "ELEVENLABS_API_KEY is not set - Spanish, Russian, Japanese and other"
  call :warn "  non-Indic languages will have no ear and no voice. Indic and"
  call :warn "  English are unaffected."
)

rem CUDA is what the pipeline actually requires at run time - stt.require_cuda
rem and llm.require_cuda are both true - so report it here rather than let the
rem first run fail on it.
:cuda_check
if "%CPU_ONLY%"=="1" goto report
if not defined HF_PY goto report
"!HF_PY!" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" >nul 2>&1
if errorlevel 1 (
  call :warn "CUDA not available - the pipeline needs it (stt.require_cuda, llm.require_cuda)."
) else (
  for /f "delims=" %%D in ('"!HF_PY!" -c "import torch;print(torch.cuda.get_device_name(0))" 2^>nul') do call :ok "CUDA available: %%D"
)

:report
if not "%FAILED%"=="0" goto report_failed
call :step "Setup complete"
echo     Run the pipeline:
echo.
echo       venv\Scripts\python.exe pipeline\run_realtime.py
echo.
echo     Check mic capture and VAD alone first - it starts instantly:
echo.
echo       venv\Scripts\python.exe pipeline\run_realtime.py --capture-only
echo.
exit /b 0

:report_failed
call :step "Setup finished with warnings"
call :info "Re-run 'download.bat' once the items above are fixed; completed steps are skipped."
exit /b 1

rem ============================================================ subroutines ===

:usage
rem The comment block at the top is the help text, so the two cannot drift
rem apart. Lines 4..42 of this file, with the "rem " prefix stripped.
for /f "tokens=1* delims=:" %%A in ('findstr /n ".*" "%~f0"') do (
  if %%A geq 4 if %%A leq 42 (
    set "line=%%B"
    if "!line!"=="rem" (echo.) else (echo !line:~4!)
  )
)
exit /b 0

:step
echo.
echo %BOLD%==^> %~1%RESET%
exit /b 0

:info
echo     %~1
exit /b 0

:skip
echo     %DIM%skip%RESET% %~1
exit /b 0

:ok
echo     %GREEN%ok%RESET%   %~1
exit /b 0

:warn
echo     %YELLOW%warn%RESET% %~1 1>&2
exit /b 0

:die
echo.
echo %RED%error%RESET% %~1 1>&2
exit /b 1

rem find_python <want> <override-var-name> <out-var-name>
rem An explicit override wins, and is a hard error if it is wrong: silently
rem falling back would build the venv on the wrong interpreter.
:find_python
set "_want=%~1"
set "_ovr="
if defined %~2 call set "_ovr=%%%~2%%"
if defined _ovr (
  "!_ovr!" -c "import sys;assert '.'.join(map(str,sys.version_info[:2]))=='%_want%'" >nul 2>&1
  if errorlevel 1 call :die "The override for Python %_want% does not point at a Python %_want%: !_ovr!" & exit /b 1
  set "%~3=!_ovr!"
  exit /b 0
)
rem The py launcher first - on Windows it is the reliable way to pick a version.
py -%_want% -c "" >nul 2>&1
if not errorlevel 1 (
  set "%~3=py -%_want%"
  exit /b 0
)
for %%C in (python%_want%.exe python3.exe python.exe) do (
  where %%C >nul 2>&1
  if not errorlevel 1 (
    %%C -c "import sys;assert '.'.join(map(str,sys.version_info[:2]))=='%_want%'" >nul 2>&1
    if not errorlevel 1 (
      for /f "delims=" %%P in ('where %%C') do (
        set "%~3=%%P"
        exit /b 0
      )
    )
  )
)
call :die "Python %_want% not found. Install it, or set PYTHON_3%_want:.=% to the full path of python.exe"
exit /b 1

rem venv_python <venv-dir> <out-var-name> - covers both layouts, as the shell
rem version does: a venv built elsewhere may carry bin\ rather than Scripts\.
:venv_python
if exist "%~1\Scripts\python.exe" ( set "%~2=%~1\Scripts\python.exe" & exit /b 0 )
if exist "%~1\bin\python.exe"     ( set "%~2=%~1\bin\python.exe"     & exit /b 0 )
set "%~2="
exit /b 1

rem require_file <path> <label> - present is ok, missing is a warning that
rem fails the run. FAILED survives the call because there is no setlocal here.
:require_file
if exist "%~1" (
  call :ok "%~2"
  exit /b 0
)
call :warn "%~2 missing"
set "FAILED=1"
exit /b 1

rem has_weights <dir> - 0 if the dir holds .safetensors or .bin weights
:has_weights
if not exist "%~1" exit /b 1
if exist "%~1\*.safetensors" exit /b 0
if exist "%~1\*.bin" exit /b 0
exit /b 1

rem dir_nonempty <dir> - 0 if the dir exists and holds anything at all
:dir_nonempty
if not exist "%~1" exit /b 1
for /f %%F in ('dir /b /a "%~1" 2^>nul ^| find /c /v ""') do if %%F gtr 0 exit /b 0
exit /b 1

rem make_venv <dir> <interpreter> <requirements> [torch spec] [torch index]
:make_venv
set "_dir=%~1"
set "_interp=%~2"
set "_reqs=%~3"
set "_torch=%~4"
set "_index=%~5"

rem No labels inside a subroutine: `goto` resolves against the whole file, not
rem the called scope, so a label here is reachable from the main body too and
rem re-enters it. Each conditional step is delegated to its own :sub instead.
if exist "%_dir%" call :skip "%_dir% exists"
if not exist "%_dir%" call :create_venv "%_dir%" "%_interp%"
if errorlevel 1 exit /b 1

call :venv_python "%_dir%" _py
if errorlevel 1 call :die "No interpreter inside %_dir% - delete it and re-run." & exit /b 1

call :info "upgrading pip"
"!_py!" -m pip install --quiet --upgrade pip setuptools wheel

rem torch first, and from its own index, so the plain-PyPI resolver in the next
rem step cannot pull a CPU build over the CUDA one.
if not "%_torch%"=="" call :install_torch "!_py!" "%_index%" "%_torch%" "%_dir%"
if errorlevel 1 exit /b 1

for %%R in ("%_reqs%") do call :info "installing %%~nxR"
"!_py!" -m pip install --quiet -r "%_reqs%"
if errorlevel 1 call :die "Requirements install failed for %_dir%" & exit /b 1
call :ok "%_dir% ready"
exit /b 0

rem create_venv <dir> <interpreter>
:create_venv
call :info "creating %~1"
rem The interpreter may be "py -3.12", so it is deliberately unquoted here.
%~2 -m venv "%~1"
if errorlevel 1 call :die "Could not create the venv at %~1" & exit /b 1
exit /b 0

rem install_torch <python> <index> <spec> <dir>
:install_torch
for %%I in ("%~2") do call :info "installing torch (%%~nxI)"
"%~1" -m pip install --quiet --index-url "%~2" %~3
if errorlevel 1 call :die "torch install failed for %~4" & exit /b 1
exit /b 0

rem hf_download <repo_id> <dest_dir> <human name> [comma-separated allow patterns]
:hf_download
set "_repo=%~1"
set "_dest=%~2"
set "_name=%~3"
set "_allow=%~4"
if "%_allow%"=="" set "_allow=None"
if not defined HF_PY call :die "No root venv interpreter; run without --skip-venvs first." & exit /b 1
call :info "fetching %_repo%"
rem cmd has no heredoc, so the helper is written to a temp file and deleted
rem after - the same script the shell version pipes in on stdin.
set "_hfpy=%TEMP%\kural_hf_download_%RANDOM%.py"
call :write_hf_helper "%_hfpy%"
"!HF_PY!" "%_hfpy%" "%_repo%" "%_dest%" "%_allow%"
set "_rc=!errorlevel!"
del /q "%_hfpy%" >nul 2>&1
if not "%_rc%"=="0" call :die "Download failed: %_repo%" & exit /b 1
call :ok "%_name%"
exit /b 0

rem write_hf_helper <path> - snapshot_download, with the two authentication
rem failures reported as instructions rather than as a traceback.
:write_hf_helper
> "%~1" (
  echo import os
  echo import sys
  echo.
  echo from huggingface_hub import snapshot_download
  echo from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError
  echo.
  echo repo, dest, allow = sys.argv[1], sys.argv[2], sys.argv[3]
  echo patterns = None if allow == "None" else allow.split^(","^)
  echo token = os.environ.get^("HF_TOKEN"^) or os.environ.get^("HUGGING_FACE_HUB_TOKEN"^)
  echo.
  echo try:
  echo     snapshot_download^(
  echo         repo_id=repo,
  echo         local_dir=dest,
  echo         allow_patterns=patterns,
  echo         token=token,
  echo         max_workers=4,
  echo     ^)
  echo except GatedRepoError:
  echo     sys.exit^(
  echo         f"\n  {repo} is GATED.\n"
  echo         f"  1. Open https://huggingface.co/{repo} and accept the license "
  echo         f"or request access.\n"
  echo         f"  2. Authenticate:  hf auth login   ^(or set HF_TOKEN=hf_xxx^)\n"
  echo         f"  3. Re-run download.bat - finished steps are skipped.\n"
  echo     ^)
  echo except RepositoryNotFoundError:
  echo     sys.exit^(
  echo         f"\n  {repo} not found. If it is private, authenticate first:\n"
  echo         f"  hf auth login   ^(or set HF_TOKEN=hf_xxx^)\n"
  echo     ^)
)
exit /b 0

rem check_import <venv> <label> <import statement>
:check_import
call :venv_python "%~1" _cpy
if errorlevel 1 call :warn "%~2: venv missing" & set "FAILED=1" & exit /b 1
"!_cpy!" -c "import %~3" >nul 2>&1
if errorlevel 1 call :warn "%~2: import failed" & set "FAILED=1" & exit /b 1
call :ok "%~2"
exit /b 0
