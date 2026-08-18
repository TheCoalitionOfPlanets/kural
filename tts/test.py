"""Standalone Piper smoke test — synthesis outside the pipeline.

    tts\\venv\\Scripts\\python.exe tts\\test.py            # default (english)
    tts\\venv\\Scripts\\python.exe tts\\test.py hindi      # a specific language

Uses the same voice directory the pipeline uses, so if this sounds right the
pipeline will too. Piper runs on CPU, so this needs no GPU.
"""
import sys
import wave
from pathlib import Path

from piper import PiperVoice

HERE = Path(__file__).resolve().parent
VOICES_DIR = HERE / "models" / "export"

# Mirrors tts.voices in pipeline/config/realtime.yaml. Languages with no Piper
# voice upstream (Tamil, Kannada, Gujarati, Punjabi, Odia, ...) are absent
# here for the same reason they are absent there.
VOICES = {
    "english": "en_lessac_medium",
    "hindi": "hi_official_v1",
    "malayalam": "ml_meera",
    "telugu": "te_maya",
    "urdu": "ur_fasih",
    "bengali": "bn_google",
    "marathi": "mr_google",
    "nepali": "ne_google",
    "spanish": "es_davefx",
    "french": "fr_siwis",
    "german": "de_thorsten",
    "chinese": "zh_huayan",
    "russian": "ru_irina",
    "arabic": "ar_kareem",
    "portuguese": "pt_faber",
    "italian": "it_paola",
    "korean": "ko_kss",
}

PROMPTS = {
    "english": "The meaning of life is found in the journey, not the destination.",
    "hindi": "जीवन का अर्थ यात्रा में है, मंज़िल में नहीं।",
    "malayalam": "ജീവിതത്തിന്റെ അർത്ഥം യാത്രയിലാണ്, ലക്ഷ്യത്തിലല്ല.",
    "telugu": "జీవితం యొక్క అర్థం ప్రయాణంలో ఉంది, గమ్యంలో కాదు.",
    "spanish": "El sentido de la vida está en el viaje, no en el destino.",
    "french": "Le sens de la vie est dans le voyage, pas dans la destination.",
    "german": "Der Sinn des Lebens liegt in der Reise, nicht im Ziel.",
}


def main():
    lang = (sys.argv[1] if len(sys.argv) > 1 else "english").lower()

    print("=" * 60)
    print("PIPER TEST")
    print("=" * 60)

    if lang not in VOICES:
        print(f"No Piper voice for {lang!r}.", file=sys.stderr)
        print("Available:", ", ".join(sorted(VOICES)), file=sys.stderr)
        return 1

    voice_path = VOICES_DIR / VOICES[lang] / "voice.onnx"
    if not voice_path.is_file():
        print(f"Voice not found at {voice_path}", file=sys.stderr)
        print("Run: bash tts/download_voices.sh", file=sys.stderr)
        return 1

    text = PROMPTS.get(lang, PROMPTS["english"])
    output = HERE / f"output_{lang}.wav"

    print("Language:", lang)
    print("Voice:", voice_path)
    print("\nLoading voice...")
    voice = PiperVoice.load(str(voice_path))
    print("Voice loaded successfully!")

    print("\nGenerating speech...")
    rate = voice.config.sample_rate
    frames = []
    for chunk in voice.synthesize(text):
        frames.append(chunk.audio_int16_bytes)
        rate = getattr(chunk, "sample_rate", rate)

    with wave.open(str(output), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(rate)
        fh.writeframes(b"".join(frames))

    total = sum(len(f) for f in frames) // 2
    print("\nDone!")
    print("Output:", output)
    print("Sample rate:", rate)
    print("Duration: %.2fs" % (total / rate))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
