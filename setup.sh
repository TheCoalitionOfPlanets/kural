#!/usr/bin/env bash
#
#   bash setup.sh
#
#     venv/            Python 3.14   orchestrator, capture, playback, STT
#     reasoning/venv/  Python 3.12   Gemma 3 4B
#     tts/venv/        Python 3.12   Indic-Mio + MioCodec
#
# OPTIONS
#   --skip-venvs    Reuse the existing environments; install nothing.
#   --skip-models   Build the environments only; download no weights.
#   --with-set-b    Also fetch the international stack (language router,
#                   Whisper large-v3, MMS-TTS voices) — about 6 GB more.
#                   Suspended in the config by default, so it is opt-in.
#                   Choose the voices with MMS_TTS_LANGS=spa,fra,jpn
#   --cpu           Install CPU torch. The pipeline will not run (every stage
#                   sets require_cuda) but the repo becomes installable.
#   -h, --help      Show this help.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

SKIP_VENVS=0
SKIP_MODELS=0
WITH_SET_B=0
CPU_ONLY=0

usage() {
  # The comment block above *is* the help text. Printed by walking from the
  # line after the shebang to the first line that is not a comment, so editing
  # the header cannot leave the help behind at a stale line range.
  awk 'NR == 1 { next } /^#/ { sub(/^# ?/, ""); print; next } { exit }' \
    "${BASH_SOURCE[0]}"
}

while [ $# -gt 0 ]; do
  case "$1" in
    --skip-venvs)  SKIP_VENVS=1 ;;
    --skip-models) SKIP_MODELS=1 ;;
    --with-set-b)  WITH_SET_B=1 ;;
    --cpu)         CPU_ONLY=1 ;;
    -h|--help)     usage; exit 0 ;;
    *) printf 'Unknown option: %s (try --help)\n' "$1" >&2; exit 2 ;;
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
warn() { printf '    %swarn%s %s\n' "$YELLOW" "$RESET" "$*"; }
die()  { printf '\n%serror%s %s\n\n' "$RED" "$RESET" "$*" >&2; exit 1; }

# ------------------------------------------------------------ interpreters --
#
# Each environment pins a different interpreter, so each is resolved separately
# and by exact version. An override is a hard error when it points somewhere
# else: falling back silently would build the environment on the wrong Python,
# and that would not surface until a model failed to load.

find_python() {
  local want="$1" override="$2" candidate
  local check="import sys; assert '.'.join(map(str, sys.version_info[:2])) == '$want'"

  if [ -n "$override" ]; then
    if [ -x "$override" ] && "$override" -c "$check" 2>/dev/null; then
      printf '%s\n' "$override"
      return 0
    fi
    die "The override does not point at a Python $want: $override"
  fi

  for candidate in "python$want" python3 python; do
    if command -v "$candidate" >/dev/null 2>&1 &&
       "$candidate" -c "$check" 2>/dev/null; then
      command -v "$candidate"
      return 0
    fi
  done
  return 1
}

# The interpreter inside an environment, whichever layout it was built with.
venv_python() {
  if   [ -x "$1/bin/python" ];         then printf '%s\n' "$1/bin/python"
  elif [ -x "$1/Scripts/python.exe" ]; then printf '%s\n' "$1/Scripts/python.exe"
  else return 1
  fi
}

# ------------------------------------------------------------------ preflight

step "Preflight"

if [ "$SKIP_VENVS" -eq 0 ]; then
  PY314="$(find_python 3.14 "${PYTHON_314:-}")" || die \
    "Python 3.14 not found — it hosts the orchestrator and STT.
  Install it, or point at it:  export PYTHON_314=/path/to/python3.14"
  PY312="$(find_python 3.12 "${PYTHON_312:-}")" || die \
    "Python 3.12 not found — it hosts the LLM and TTS.
  Install it, or point at it:  export PYTHON_312=/path/to/python3.12"
  ok "3.14  $PY314"
  ok "3.12  $PY312"

  # webrtcvad is a C extension with no prebuilt wheel for recent Pythons, so
  # pip compiles it. Without a toolchain that fails deep inside the install
  # behind a wall of compiler output; saying so here is far cheaper.
  if command -v cc >/dev/null 2>&1 || command -v gcc >/dev/null 2>&1; then
    ok "C compiler present (webrtcvad builds from source)"
  else
    warn "No C compiler found. webrtcvad builds from source and will fail."
    warn "  Debian/Ubuntu:  sudo apt install build-essential python3.14-dev"
  fi
fi

# Advisory only: the real footprint depends on which optional models are
# fetched, and a wrong guess should not block a setup that would have fit.
if command -v df >/dev/null 2>&1; then
  FREE_GB="$(df -Pk "$ROOT" 2>/dev/null | awk 'NR==2 {printf "%d", $4 / 1048576}')"
  if [ -n "${FREE_GB:-}" ] && [ "$FREE_GB" -lt 25 ]; then
    warn "${FREE_GB} GB free; about 25 GB is needed (more with --with-set-b)."
  else
    ok "${FREE_GB:-?} GB free"
  fi
fi

# ------------------------------------------------------------------ venvs ---
#
# The exact builds the pipeline was developed against. The two stacks genuinely
# differ, which is the reason they are separate environments at all.

TORCH_STT="torch==2.12.1 torchvision==0.27.1"
TORCH_STT_INDEX="https://download.pytorch.org/whl/cu132"
TORCH_LLM="torch==2.13.0 torchvision==0.28.0"
TORCH_LLM_INDEX="https://download.pytorch.org/whl/cu130"
if [ "$CPU_ONLY" -eq 1 ]; then
  TORCH_STT_INDEX="https://download.pytorch.org/whl/cpu"
  TORCH_LLM_INDEX="https://download.pytorch.org/whl/cpu"
fi

# make_venv <dir> <interpreter> <requirements> <torch spec> <torch index>
make_venv() {
  local dir="$1" interp="$2" reqs="$3" torch_spec="$4" torch_index="$5"
  local py

  if [ -d "$dir" ]; then
    skip "$dir exists"
  else
    info "creating $dir"
    "$interp" -m venv "$dir" || die "Could not create the environment at $dir"
  fi

  py="$(venv_python "$dir")" || die \
    "No interpreter inside $dir. Delete the directory and re-run."

  info "upgrading pip"
  "$py" -m pip install --upgrade --quiet pip setuptools wheel \
    || die "Could not upgrade pip in $dir"

  # torch first and from its own index, so the plain-PyPI resolver in the next
  # step cannot pull a CPU build over the CUDA one.
  info "installing torch (${torch_index##*/})"
  # Unquoted on purpose: the spec names two packages.
  # shellcheck disable=SC2086
  "$py" -m pip install --index-url "$torch_index" $torch_spec \
    || die "torch install failed for $dir"

  info "installing $(basename "$reqs")"
  "$py" -m pip install -r "$reqs" || die "Requirements install failed for $dir"
  ok "$dir"
}

if [ "$SKIP_VENVS" -eq 1 ]; then
  step "Environments"
  skip "--skip-venvs"
else
  step "Root environment — orchestrator, capture, playback, STT (3.14)"
  make_venv "$ROOT/venv" "$PY314" "$ROOT/requirements/stt.txt" \
    "$TORCH_STT" "$TORCH_STT_INDEX"

  step "reasoning/venv — Gemma 3 4B (3.12)"
  make_venv "$ROOT/reasoning/venv" "$PY312" "$ROOT/requirements/llm.txt" \
    "$TORCH_LLM" "$TORCH_LLM_INDEX"

  step "tts/venv — Indic-Mio + MioCodec (3.12)"
  # Indic-Mio is a causal LM like the reasoning stack, so it shares that cu130
  # build rather than needing one of its own.
  make_venv "$ROOT/tts/venv" "$PY312" "$ROOT/requirements/tts.txt" \
    "$TORCH_LLM" "$TORCH_LLM_INDEX"
fi

# ------------------------------------------------------------------ models --
#
# Downloading and verifying both live in tools/, shared with setup.bat: bash
# and batch disagree about quoting badly enough that two copies of this would
# drift, and what they would drift about is which weights land where.

ROOT_PY="$(venv_python "$ROOT/venv" 2>/dev/null || true)"

if [ "$SKIP_MODELS" -eq 1 ]; then
  step "Models"
  skip "--skip-models"
else
  [ -n "$ROOT_PY" ] || die \
    "The root environment does not exist yet. Run without --skip-venvs first."

  FETCH_ARGS=()
  [ "$WITH_SET_B" -eq 1 ] && FETCH_ARGS+=(--set-b)
  [ -n "${MMS_TTS_LANGS:-}" ] && FETCH_ARGS+=(--langs "$MMS_TTS_LANGS")

  "$ROOT_PY" "$ROOT/tools/fetch_models.py" "${FETCH_ARGS[@]+"${FETCH_ARGS[@]}"}" \
    || die "Model download failed. Fix the problem above and re-run; finished
  downloads are skipped."
fi

# ------------------------------------------------------------------ verify --

if [ -z "$ROOT_PY" ]; then
  step "Verifying"
  warn "The root environment does not exist; nothing to verify."
  exit 1
fi

if ! "$ROOT_PY" "$ROOT/tools/verify_setup.py"; then
  step "Setup finished with problems"
  info "Fix the items above and re-run; completed steps are skipped."
  exit 1
fi

# -------------------------------------------------------------------- done --

step "Setup complete"
cat <<'EOM'
    Start the pipeline — speak, and it answers out loud:

      venv/bin/python pipeline/run_realtime.py

    Check the microphone and voice detection alone first. It starts instantly
    and loads no models:

      venv/bin/python pipeline/run_realtime.py --capture-only

    Or serve it to a browser instead of the local microphone:

      venv/bin/python -m pipeline.server
EOM

if [ "$WITH_SET_B" -eq 0 ]; then
  printf '\n'
  info "Set A only — Indian languages and English. For Spanish, Russian,"
  info "Japanese and the rest, re-run with --with-set-b and set stt.lid,"
  info "stt.whisper and tts.mms_tts to enabled in pipeline/config/realtime.yaml."
fi
printf '\n'
