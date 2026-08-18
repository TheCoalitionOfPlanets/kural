#!/usr/bin/env bash
# Download Piper voices for the pipeline's language set: English plus the
# Indic languages that have a native Piper voice (Tamil, Hindi, Malayalam,
# Telugu, Urdu, Bengali, Marathi, Nepali) and a core international set.
#
# Tamil is not in the official rhasspy/piper-voices repo — it comes from a
# community repo (see TAMIL_VOICES below), which is why it is fetched
# separately.
#
# Kannada has no Piper voice at all, so it is built here by converting the
# SYSPIN (IISc Bangalore, MIT) Coqui VITS checkpoint — see
# tts/convert_coqui_to_piper.py.
#
# Languages with no usable voice — Gujarati, Punjabi, Odia, Assamese, Maithili,
# Konkani, Manipuri, Sanskrit, Santali — are not spoken at all. The TTS worker
# reports `no_voice` and the reply is shown as text only, which is deliberate:
# reading them aloud with a wrong-language voice mispronounces every word.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BASE="https://huggingface.co/rhasspy/piper-voices/resolve/main"

download() {
  local url="$1"
  local out="$2"
  if command -v curl >/dev/null 2>&1; then
    curl -L --fail --progress-bar -o "$out" "$url"
  elif command -v wget >/dev/null 2>&1; then
    wget -q --show-progress -O "$out" "$url"
  else
    echo "Need curl or wget installed." >&2
    exit 1
  fi
}

# voice_key  hf_subpath  export_dir
VOICES=(
  "en_US-lessac-medium en/en_US/lessac/medium en_lessac_medium"
  "hi_IN-priyamvada-medium hi/hi_IN/priyamvada/medium hi_official_v1"
  "ml_IN-meera-medium ml/ml_IN/meera/medium ml_meera"
  "te_IN-maya-medium te/te_IN/maya/medium te_maya"
  "ur_PK-fasih-medium ur/ur_PK/fasih/medium ur_fasih"
  "bn_BD-google-medium bn/bn_BD/google/medium bn_google"
  "mr_IN-google-medium mr/mr_IN/google/medium mr_google"
  "ne_NP-google-medium ne/ne_NP/google/medium ne_google"
  "es_ES-davefx-medium es/es_ES/davefx/medium es_davefx"
  "fr_FR-siwis-medium fr/fr_FR/siwis/medium fr_siwis"
  "de_DE-thorsten-medium de/de_DE/thorsten/medium de_thorsten"
  "zh_CN-huayan-medium zh/zh_CN/huayan/medium zh_huayan"
  "ru_RU-irina-medium ru/ru_RU/irina/medium ru_irina"
  "ar_JO-kareem-medium ar/ar_JO/kareem/medium ar_kareem"
  "pt_BR-faber-medium pt/pt_BR/faber/medium pt_faber"
  "it_IT-paola-medium it/it_IT/paola/medium it_paola"
  "ko_KR-kss-medium ko/ko_KR/kss/medium ko_kss"
)

mkdir -p "$ROOT/tts/models/upstream"

for entry in "${VOICES[@]}"; do
  read -r key subpath export_dir <<<"$entry"
  echo "Downloading ${key}…"
  download "$BASE/$subpath/${key}.onnx" "$ROOT/tts/models/upstream/${key}.onnx"
  download "$BASE/$subpath/${key}.onnx.json" "$ROOT/tts/models/upstream/${key}.onnx.json"
  mkdir -p "$ROOT/tts/models/export/${export_dir}"
  cp "$ROOT/tts/models/upstream/${key}.onnx" "$ROOT/tts/models/export/${export_dir}/voice.onnx"
  cp "$ROOT/tts/models/upstream/${key}.onnx.json" "$ROOT/tts/models/export/${export_dir}/voice.onnx.json"
done

# Tamil, from a community repo (Apache-2.0). Standard Piper format —
# espeak_voice "ta", 22050 Hz — so it loads exactly like the official voices.
# Laid out one directory per voice rather than flat, hence the separate loop.
#
# ValluvarNeural is the default: it is trained far longer than HemaLatha
# (step 1388260 vs epoch 14) and is the steadier of the two.
TAMIL_BASE="https://huggingface.co/Jeyaram-K/piper-tamil-voices/resolve/main"
TAMIL_VOICES=(
  "ta_IN-ValluvarNeural-medium ta_valluvar"
  "ta_IN-HemaLatha-medium ta_hemalatha"
)

for entry in "${TAMIL_VOICES[@]}"; do
  read -r key export_dir <<<"$entry"
  echo "Downloading ${key}…"
  download "$TAMIL_BASE/${key}/${key}.onnx" "$ROOT/tts/models/upstream/${key}.onnx"
  download "$TAMIL_BASE/${key}/${key}.onnx.json" "$ROOT/tts/models/upstream/${key}.onnx.json"
  mkdir -p "$ROOT/tts/models/export/${export_dir}"
  cp "$ROOT/tts/models/upstream/${key}.onnx" "$ROOT/tts/models/export/${export_dir}/voice.onnx"
  cp "$ROOT/tts/models/upstream/${key}.onnx.json" "$ROOT/tts/models/export/${export_dir}/voice.onnx.json"
done

# Kannada has no Piper voice in any repo, official or community. The SYSPIN
# project (IISc Bangalore, MIT) publishes a Coqui VITS checkpoint for it, which
# converts cleanly because Piper is VITS too and this model is character-based.
# The conversion needs torch, so it is skipped if the export already exists or
# if the tts venv cannot import torch.
KN_DIR="$ROOT/tts/models/export/kn_syspin"
KN_PY="$ROOT/tts/venv/Scripts/python.exe"
[ -x "$KN_PY" ] || KN_PY="$ROOT/tts/venv/bin/python"

if [ -f "$KN_DIR/voice.onnx" ]; then
  echo "Kannada voice already converted; skipping."
elif ! "$KN_PY" -c "import torch" >/dev/null 2>&1; then
  echo "Skipping Kannada: torch is not installed in the tts venv." >&2
  echo "  Install it, then re-run this script to build the Kannada voice." >&2
else
  echo "Building Kannada voice from the SYSPIN Coqui checkpoint…"
  KN_SRC="$ROOT/tts/models/upstream/syspin_kannada"
  mkdir -p "$KN_SRC"
  KN_BASE="https://huggingface.co/SYSPIN/vits_Kannada_Female/resolve/main"
  [ -f "$KN_SRC/best_model.pth" ] || download "$KN_BASE/best_model.pth" "$KN_SRC/best_model.pth"
  [ -f "$KN_SRC/config.json" ]    || download "$KN_BASE/config.json"    "$KN_SRC/config.json"
  "$KN_PY" "$ROOT/tts/convert_coqui_to_piper.py" \
      --checkpoint "$KN_SRC/best_model.pth" \
      --config "$KN_SRC/config.json" \
      --output-dir "$KN_DIR" \
      --espeak-voice kn
fi

echo "Done. Exported voices:"
for entry in "${VOICES[@]}"; do
  read -r key subpath export_dir <<<"$entry"
  echo "  $export_dir: $ROOT/tts/models/export/${export_dir}/voice.onnx"
done
for entry in "${TAMIL_VOICES[@]}"; do
  read -r key export_dir <<<"$entry"
  echo "  $export_dir: $ROOT/tts/models/export/${export_dir}/voice.onnx"
done
if [ -f "$KN_DIR/voice.onnx" ]; then echo "  kn_syspin: $KN_DIR/voice.onnx"; fi
