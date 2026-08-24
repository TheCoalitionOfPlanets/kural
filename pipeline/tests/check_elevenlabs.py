"""LIVE check — does the ElevenLabs path actually work with your key?

    venv/bin/python pipeline/tests/check_elevenlabs.py

Unlike every other file in this directory, this one **makes real API calls and
spends real credit**. `test_elevenlabs.py` replaces `urlopen` and proves the
wire format; only this proves the account, the key, the voice id and the two
models are real and reachable.

It is a round trip, so one run covers both halves with no audio fixture:

    text -> [ElevenLabs TTS] -> audio -> [ElevenLabs Scribe] -> text + language

If Scribe hears Spanish in audio that ElevenLabs itself just spoke, then the
voice, the ear and the language reporting the router depends on are all good.

Run it before a demo. Everything else in the pipeline can be healthy and this
can still be the thing that is broken, because nothing offline can detect a
key that was revoked or a voice id that does not exist on your account.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "pipeline"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from realtime.elevenlabs import ElevenLabs, ElevenLabsError, pcm16_to_wav  # noqa: E402
from realtime.languages import language_from_code, route_for  # noqa: E402

# Spanish, because it is unambiguous to Scribe and is the case the whole
# international path exists for.
PHRASE = "Hola, ¿cómo estás? Espero que tengas un buen día."
EXPECT = "spanish"


def load_config():
    import yaml

    cfg = yaml.safe_load(
        (ROOT / "pipeline/config/realtime.yaml").read_text("utf-8"))
    return cfg.get("elevenlabs") or {}


def main():
    cfg = load_config()
    env_name = cfg.get("api_key_env", "ELEVENLABS_API_KEY")
    key = os.environ.get(env_name, "")

    print(f"config      elevenlabs.enabled = {cfg.get('enabled', True)}")
    print(f"            tts_model = {cfg.get('tts_model')}")
    print(f"            stt_model = {cfg.get('stt_model')}")
    print(f"            voice_id  = {cfg.get('voice_id')}")
    print(f"            output    = {cfg.get('output_format')}")

    if not key:
        print(f"\n  ${env_name} is not set. International turns have no ear and")
        print("  no voice until it is:\n")
        print(f"      export {env_name}=sk_...\n")
        return 1
    print(f"key         ${env_name} is set ({len(key)} chars)\n")

    try:
        client = ElevenLabs(
            key,
            stt_model=cfg.get("stt_model"),
            tts_model=cfg.get("tts_model"),
            voice_id=cfg.get("voice_id"),
            output_format=cfg.get("output_format"),
            voice_settings=cfg.get("voice_settings"),
            timeout_s=cfg.get("timeout_s", 20),
            retries=cfg.get("retries", 1),
        )
    except ElevenLabsError as exc:
        print(f"  FAIL  client: {exc}")
        return 1

    # -- the voice --------------------------------------------------------
    print(f"speaking    {PHRASE!r}")
    try:
        pcm, rate = client.synthesize(PHRASE)
    except ElevenLabsError as exc:
        print(f"  FAIL  text-to-speech: {exc}")
        if exc.status == 401:
            print("        The key was rejected. Check it is current and has")
            print("        text-to-speech permission.")
        elif exc.status == 404:
            print(f"        voice_id {cfg.get('voice_id')!r} is not on this")
            print("        account. Pick one from your ElevenLabs voice list.")
        return 1

    seconds = len(pcm) / 2 / rate
    print(f"  OK    {len(pcm)} bytes of PCM, {rate} Hz, {seconds:.2f}s\n")

    # -- the ear ----------------------------------------------------------
    print("listening   sending that audio back to Scribe")
    try:
        heard = client.transcribe(pcm16_to_wav(pcm, rate))
    except ElevenLabsError as exc:
        print(f"  FAIL  speech-to-text: {exc}")
        if exc.status == 401:
            print("        The key works for TTS but not STT — check that the")
            print("        key has speech-to-text permission enabled.")
        return 1

    lang = language_from_code(heard.get("language_code"))
    print(f"  OK    text: {heard['text']!r}")
    print(f"        language: {heard.get('language_code')} -> {lang} "
          f"({heard.get('probability')})")
    print(f"        routes to: {route_for(lang)}\n")

    ok = True
    if lang != EXPECT:
        # Not necessarily broken — but the router keys off exactly this value,
        # so a surprise here is worth seeing before a demo rather than during.
        print(f"  WARN  expected {EXPECT!r}, got {lang!r}. The router uses this")
        print("        value to pick the voice, so check the mapping in")
        print("        realtime/languages.py if this looks wrong.")
        ok = False
    if route_for(lang) != "international":
        print(f"  FAIL  {lang!r} routes locally — it would never reach ElevenLabs.")
        ok = False

    if ok:
        print("ElevenLabs is connected: the voice speaks, the ear hears, and")
        print("the language it reports routes to the international path.")
        print("\nOne thing this cannot check: whether the local language gate")
        print("(stt/models/mms-lid-126) is downloaded. Without it every turn")
        print("routes to the Indic models and none of the above is ever used.")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
