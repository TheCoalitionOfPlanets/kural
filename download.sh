#!/usr/bin/env bash
#
# One-shot setup for a fresh clone: builds the three venvs, installs every
# package, and downloads every model the real-time pipeline needs.
#
#   bash download.sh
#
# Safe to re-run. Every step checks for its own result first, so an interrupted
# run resumes instead of starting over, and a finished run is a no-op.
#
# WHY THREE VENVS
# The three model stacks pin incompatible dependencies — STT wants Python 3.14
# with a cu132 torch, and both the LLM and TTS want 3.12 with cu130 but are
# kept in separate venvs regardless (see pipeline/README.md). They cannot
# share one interpreter, so each stage runs as a subprocess in its own venv.
#
# BEFORE RUNNING YOU NEED
#   * Python 3.14 and Python 3.12 on PATH (see PYTHON VERSIONS below)
#   * curl or wget
#   * ~25 GB free disk, and an NVIDIA GPU with 12 GB VRAM to actually run it
#   * A Hugging Face account and token, because two of the models are gated:
#       - google/gemma-3-4b-it       accept the Gemma license on the model page
#       - ARTPARK-IISc/SraVaani-1.0  request access on the model page
#     Then either run `hf auth login`, or export HF_TOKEN=hf_xxx.
#
# OPTIONS
#   --skip-venvs    Reuse existing venvs; do not create or install.
#   --skip-models   Set up venvs only; download no model weights.
#   --cpu           Install CPU torch everywhere. The pipeline will not run
#                   (STT, the LLM, and TTS all set require_cuda), but this
#                   makes the repo installable on a machine with no NVIDIA GPU.
#   -h, --help      Show this help.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

SKIP_VENVS=0
SKIP_MODELS=0
CPU_ONLY=0

usage() {
  # The comment block above is the help text, so the two cannot drift apart.
  sed -n '3,32p' "$0" | sed 's/^#\{1,2\} \{0,1\}//'
}

while [ $# -gt 0 ]; do
  case "$1" in
    --skip-venvs)  SKIP_VENVS=1 ;;
    --skip-models) SKIP_MODELS=1 ;;
    --cpu)         CPU_ONLY=1 ;;
    -h|--help)     usage; exit 0 ;;
    *) echo "Unknown option: $1 (try --help)" >&2; exit 2 ;;
  esac
  shift
done

# ---------------------------------------------------------------- output ----

BOLD=""; DIM=""; RED=""; GREEN=""; YELLOW=""; RESET=""
if [ -t 1 ]; then
  BOLD=$'\033[1m'; DIM=$'\033[2m'; RED=$'\033[31m'
  GREEN=$'\033[32m'; YELLOW=$'\033[33m'; RESET=$'\033[0m'
fi

step() { printf '\n%s==> %s%s\n' "$BOLD" "$*" "$RESET"; }
info() { printf '    %s\n' "$*"; }
skip() { printf '    %sskip%s %s\n' "$DIM" "$RESET" "$*"; }
ok()   { printf '    %sok%s   %s\n' "$GREEN" "$RESET" "$*"; }
warn() { printf '    %swarn%s %s\n' "$YELLOW" "$RESET" "$*" >&2; }
die()  { printf '\n%serror%s %s\n' "$RED" "$RESET" "$*" >&2; exit 1; }

# ------------------------------------------------------- python versions ----
#
# The venvs pin different interpreters, so each is resolved separately. On
# Windows the `py` launcher is the reliable way to pick a version; elsewhere it
# is python3.14 / python3.12 on PATH. Override either by exporting PYTHON_314
# or PYTHON_312 to an absolute path before running.

find_python() {
  local want="$1" override="$2" c
  # An explicit override wins, and is a hard error if it is wrong: silently
  # falling back would build the venv on the wrong interpreter.
  if [ -n "$override" ]; then
    if "$override" -c "import sys;assert '.'.join(map(str,sys.version_info[:2]))=='$want'" 2>/dev/null; then
      echo "$override"; return 0
    fi
    die "The override for Python $want does not point at a Python $want: $override"
  fi
  for c in "python$want" python3 python; do
    if command -v "$c" >/dev/null 2>&1 &&
       "$c" -c "import sys;assert '.'.join(map(str,sys.version_info[:2]))=='$want'" 2>/dev/null; then
      command -v "$c"; return 0
    fi
  done
  if command -v py >/dev/null 2>&1 && py "-$want" -c "" 2>/dev/null; then
    echo "py -$want"; return 0
  fi
  return 1
}

# A venv's interpreter, covering both the Windows and POSIX layouts.
venv_python() {
  local venv="$1"
  if   [ -x "$venv/Scripts/python.exe" ]; then echo "$venv/Scripts/python.exe"
  elif [ -x "$venv/bin/python" ];         then echo "$venv/bin/python"
  else return 1; fi
}

# ------------------------------------------------------------------ venvs ---
#
# CUDA wheel indexes. These are the exact builds the pipeline was developed
# against; the two stacks genuinely differ, which is why they are separate
# venvs in the first place.
TORCH_STT="torch==2.12.1 torchvision==0.27.1"
TORCH_STT_INDEX="https://download.pytorch.org/whl/cu132"
TORCH_LLM="torch==2.13.0 torchvision==0.28.0"
TORCH_LLM_INDEX="https://download.pytorch.org/whl/cu130"
if [ "$CPU_ONLY" -eq 1 ]; then
  TORCH_STT_INDEX="https://download.pytorch.org/whl/cpu"
  TORCH_LLM_INDEX="https://download.pytorch.org/whl/cpu"
fi

# make_venv <dir> <interpreter> <requirements> [torch spec] [torch index]
make_venv() {
  local dir="$1" interp="$2" reqs="$3" torch_spec="${4:-}" torch_index="${5:-}"
  local py

  if [ ! -d "$dir" ]; then
    info "creating $dir"
    # $interp may be "py -3.12", so it is deliberately unquoted here.
    # shellcheck disable=SC2086
    $interp -m venv "$dir" || die "Could not create the venv at $dir"
  else
    skip "$dir exists"
  fi

  py="$(venv_python "$dir")" || die "No interpreter inside $dir — delete it and re-run."

  info "upgrading pip"
  "$py" -m pip install --quiet --upgrade pip setuptools wheel

  # torch first, and from its own index, so the plain-PyPI resolver in the next
  # step cannot pull a CPU build over the CUDA one.
  if [ -n "$torch_spec" ]; then
    info "installing torch (${torch_index##*/})"
    # shellcheck disable=SC2086
    "$py" -m pip install --quiet --index-url "$torch_index" $torch_spec \
      || die "torch install failed for $dir"
  fi

  info "installing ${reqs##*/}"
  "$py" -m pip install --quiet -r "$reqs" || die "Requirements install failed for $dir"
  ok "$dir ready"
}

if [ "$SKIP_VENVS" -eq 1 ]; then
  step "Virtual environments"
  skip "--skip-venvs"
else
  step "Locating interpreters"
  PY314="$(find_python 3.14 "${PYTHON_314:-}")" \
    || die "Python 3.14 not found (needed for venv/). Install it, or export PYTHON_314=/path/to/python3.14"
  PY312="$(find_python 3.12 "${PYTHON_312:-}")" \
    || die "Python 3.12 not found (needed for reasoning/venv and tts/venv). Install it, or export PYTHON_312=/path/to/python3.12"
  ok "3.14: $PY314"
  ok "3.12: $PY312"

  step "Root venv — orchestrator + SraVaani STT (Python 3.14)"
  make_venv "$ROOT/venv" "$PY314" "$ROOT/requirements/stt.txt" "$TORCH_STT" "$TORCH_STT_INDEX"

  step "reasoning/venv — Gemma 3 4B (Python 3.12)"
  make_venv "$ROOT/reasoning/venv" "$PY312" "$ROOT/requirements/llm.txt" "$TORCH_LLM" "$TORCH_LLM_INDEX"

  step "tts/venv — Indic-Mio (Python 3.12)"
  # Indic-Mio is a causal LM like the reasoning stack, so it shares the same
  # cu130 torch build rather than needing a build of its own.
  make_venv "$ROOT/tts/venv" "$PY312" "$ROOT/requirements/tts.txt" "$TORCH_LLM" "$TORCH_LLM_INDEX"
fi

# ----------------------------------------------------------------- models ---

HF_PY="$(venv_python "$ROOT/venv" 2>/dev/null || true)"

# hf_download <repo_id> <dest_dir> <human name> [comma-separated allow patterns]
hf_download() {
  local repo="$1" dest="$2" name="$3" allow="${4:-None}"
  [ -n "$HF_PY" ] || die "No root venv interpreter; run without --skip-venvs first."
  info "fetching $repo"
  "$HF_PY" - "$repo" "$dest" "$allow" <<'PY'
import os
import sys

from huggingface_hub import snapshot_download
from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

repo, dest, allow = sys.argv[1], sys.argv[2], sys.argv[3]
patterns = None if allow == "None" else allow.split(",")
token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

try:
    snapshot_download(
        repo_id=repo,
        local_dir=dest,
        allow_patterns=patterns,
        token=token,
        max_workers=4,
    )
except GatedRepoError:
    sys.exit(
        f"\n  {repo} is GATED.\n"
        f"  1. Open https://huggingface.co/{repo} and accept the license "
        f"or request access.\n"
        f"  2. Authenticate:  hf auth login   (or export HF_TOKEN=hf_xxx)\n"
        f"  3. Re-run download.sh — finished steps are skipped.\n"
    )
except RepositoryNotFoundError:
    sys.exit(
        f"\n  {repo} not found. If it is private, authenticate first:\n"
        f"  hf auth login   (or export HF_TOKEN=hf_xxx)\n"
    )
PY
  ok "$name"
}

if [ "$SKIP_MODELS" -eq 1 ]; then
  step "Models"
  skip "--skip-models"
else
  # SraVaani STT — a ~900 MB FP16 TorchScript archive, plus the remote-code
  # modules that trust_remote_code=True loads alongside it.
  step "STT model — ARTPARK-IISc/SraVaani-1.0 (~900 MB)"
  if [ -f "$ROOT/stt/models/model-asr.fp16.ts" ]; then
    skip "stt/models already populated"
  else
    hf_download "ARTPARK-IISc/SraVaani-1.0" "$ROOT/stt/models" "SraVaani STT"
  fi

  # Gemma 3 4B IT — ~8.6 GB of bf16 safetensors on disk, quantized to 4-bit
  # NF4 at load time so all three models fit in 12 GB.
  step "LLM — google/gemma-3-4b-it (~8.6 GB)"
  if [ -f "$ROOT/reasoning/models/gemma-3-4b-it/model.safetensors.index.json" ]; then
    skip "reasoning/models/gemma-3-4b-it already populated"
  else
    # Only what transformers loads: this skips any GGUF/ONNX variants the repo
    # may carry, which the worker never touches.
    hf_download "google/gemma-3-4b-it" "$ROOT/reasoning/models/gemma-3-4b-it" \
      "Gemma 3 4B IT" "*.json,*.safetensors,*.model,*.txt,*.md"
  fi

  # Indic-Mio — a ~1.2 GB bf16 causal LM, plus the MioCodec decoder it needs
  # to turn generated audio tokens into a waveform. Neither repo is gated.
  step "TTS model — SPRINGLab/Indic-Mio (~1.2 GB)"
  if [ -f "$ROOT/tts/models/Indic-Mio/model.safetensors" ]; then
    skip "tts/models/Indic-Mio already populated"
  else
    hf_download "SPRINGLab/Indic-Mio" "$ROOT/tts/models/Indic-Mio" "Indic-Mio TTS"
  fi

  step "TTS codec — Aratako/MioCodec-25Hz-24kHz"
  if [ -d "$ROOT/tts/models/MioCodec-25Hz-24kHz" ] \
      && [ -n "$(ls -A "$ROOT/tts/models/MioCodec-25Hz-24kHz" 2>/dev/null)" ]; then
    skip "tts/models/MioCodec-25Hz-24kHz already populated"
  else
    hf_download "Aratako/MioCodec-25Hz-24kHz" "$ROOT/tts/models/MioCodec-25Hz-24kHz" "MioCodec"
  fi
fi

# ------------------------------------------------------------------ check ---

step "Verifying"

# check_import <venv> <label> <import statement>
check_import() {
  local venv="$1" label="$2" imports="$3" py
  py="$(venv_python "$venv" 2>/dev/null || true)"
  if [ -z "$py" ]; then warn "$label: venv missing"; return 1; fi
  if "$py" -c "import $imports" 2>/dev/null; then ok "$label"; else
    warn "$label: import failed"; return 1
  fi
}

FAILED=0
check_import "$ROOT/venv" "root venv  (torch, transformers, sounddevice)" \
  "torch, transformers, sounddevice, yaml" || FAILED=1
check_import "$ROOT/reasoning/venv" "reasoning  (torch, transformers, bitsandbytes)" \
  "torch, transformers, bitsandbytes" || FAILED=1
check_import "$ROOT/tts/venv" "tts        (torch, transformers, miocodec)" \
  "torch, transformers, miocodec" || FAILED=1

if [ "$SKIP_MODELS" -eq 0 ]; then
  if [ -f "$ROOT/stt/models/model-asr.fp16.ts" ]; then ok "STT weights"
  else warn "STT weights missing"; FAILED=1; fi

  if [ -f "$ROOT/reasoning/models/gemma-3-4b-it/model.safetensors.index.json" ]; then ok "LLM weights"
  else warn "LLM weights missing"; FAILED=1; fi

  if [ -f "$ROOT/tts/models/Indic-Mio/model.safetensors" ]; then ok "TTS weights"
  else warn "TTS weights missing"; FAILED=1; fi
fi

# CUDA is what the pipeline actually requires at run time — stt.require_cuda
# and llm.require_cuda are both true — so report it here rather than let the
# first run fail on it.
if [ "$CPU_ONLY" -eq 0 ] && [ -n "$HF_PY" ]; then
  if "$HF_PY" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    ok "CUDA available: $("$HF_PY" -c "import torch;print(torch.cuda.get_device_name(0))" 2>/dev/null)"
  else
    warn "CUDA not available — the pipeline needs it (stt.require_cuda, llm.require_cuda)."
  fi
fi

if [ "$FAILED" -eq 0 ]; then
  step "Setup complete"
  cat <<'EOM'
    Run the pipeline:

      venv/Scripts/python.exe pipeline/run_realtime.py     # Windows
      venv/bin/python pipeline/run_realtime.py             # Linux/macOS

    Check mic capture and VAD alone first — it starts instantly:

      venv/Scripts/python.exe pipeline/run_realtime.py --capture-only
EOM
else
  step "Setup finished with warnings"
  info "Re-run 'bash download.sh' once the items above are fixed; completed steps are skipped."
  exit 1
fi
