"""Tests for language detection and the reply-language directive.

    venv\\Scripts\\python.exe pipeline\\tests\\test_languages.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from pipeline.realtime.languages import (  # noqa: E402
    ENGLISH,
    SUPPORTED,
    detect_language,
    language_directive,
)

failures = []


def check(name, cond):
    if cond:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}")
        failures.append(name)


def expect(text, lang):
    got = detect_language(text)
    check(f"{text[:38]!r} -> {lang}", got == lang)


print("native script — unambiguous")
expect("வணக்கம், எப்படி இருக்கீங்க?", "tamil")
expect("நான் நன்றாக இருக்கிறேன்", "tamil")
expect("नमस्ते, आप कैसे हैं?", "hindi")
expect("मैं ठीक हूँ धन्यवाद", "hindi")
expect("ఎలా ఉన్నారు?", "telugu")
expect("ನಮಸ್ಕಾರ ಹೇಗಿದ್ದೀರಿ", "kannada")
expect("നമസ്കാരം സുഖമാണോ", "malayalam")
expect("নমস্কার কেমন আছেন", "bengali")
expect("નમસ્તે કેમ છો", "gujarati")

print("\nenglish")
expect("What is the capital of France?", ENGLISH)
expect("Hello, how are you doing today?", ENGLISH)
expect("Set a reminder for six in the evening", ENGLISH)
expect("", ENGLISH)
expect("   ", ENGLISH)
expect("12345 !!! ???", ENGLISH)

print("\nromanized indic")
expect("enna panra machan", "tamil")
expect("epdi irukka", "tamil")
expect("naan nalla iruken", "tamil")
expect("vanakkam nanba", "tamil")
expect("kya kar rahe ho", "hindi")
expect("mujhe ek baat batao", "hindi")
expect("aap kaise hain", "hindi")
expect("nenu bagunnanu ela unnaru", "telugu")

print("\ncode-switching stays english")
# A borrowed word or two is not a language change.
expect("Send it to my machan", ENGLISH)
expect("The meeting is at four", ENGLISH)
expect("Can you do that for me", ENGLISH)

print("\nmixed script — majority meaning-bearing wins")
expect("வணக்கம் Google Meet ல சேருங்க", "tamil")
expect("मुझे WhatsApp पर भेज दो", "hindi")

print("\nEnglish sentences that share romanized tokens")
# 'do', 'ek', 'bol', 'sab', 'wala' etc. must not drag plain English away.
expect("I have to do two things", ENGLISH)
expect("Do you want tea", ENGLISH)
expect("No worries at all", ENGLISH)

print("\ndirective")
d_ta = language_directive("tamil")
check("names the language", "Tamil" in d_ta)
check("demands native script", "script" in d_ta.lower())
check("forbids english", "not reply in english" in d_ta.lower()
      or "do not reply in english" in d_ta.lower())
check("forbids latin letters", "latin" in d_ta.lower())

d_en = language_directive(ENGLISH)
check("english directive names english", "English" in d_en)
check("english directive has no script demand", "native" not in d_en.lower())

check("handles none", isinstance(language_directive(None), str))
check("handles empty", isinstance(language_directive(""), str))

print("\nenglish is not stolen by romanized markers")
# Regression: "the" was a Hindi marker (थे) and "do" (दो), "no", "da"/"di"
# were markers elsewhere. Two occurrences of "the" were enough to classify a
# 34-word English sentence as Hindi and reply in the wrong language entirely.
expect("here doing this project under the guidance of krishnam and balamurti "
       "sir at a career development center of chennai instruct of technology "
       "coming under the atonomous university of anna university located in "
       "tamil nadu", ENGLISH)
expect("what do you think about the new plan and how do we do it", ENGLISH)
expect("no i do not think the answer is correct", ENGLISH)
expect("the sari and the dosa were the best part of the trip", ENGLISH)

# The guard that keeps it fixed: no marker set may contain a common English
# word, since English is the fallback and has no markers to outvote them.
from pipeline.realtime.languages import (  # noqa: E402
    _ENGLISH_COLLISIONS,
    _ROMANIZED,
)

for _lang, _markers in _ROMANIZED.items():
    _clash = sorted(_markers & _ENGLISH_COLLISIONS)
    check(f"{_lang} has no english collisions", not _clash)
    if _clash:
        print(f"        collides on: {_clash}")

# Removing collisions must not have gutted the sets.
for _lang in ("tamil", "hindi", "telugu"):
    check(f"{_lang} still has enough markers", len(_ROMANIZED[_lang]) >= 20)

print("\nsupported set")
for lang in ("tamil", "hindi", "telugu", "kannada", "malayalam", ENGLISH):
    check(f"{lang} supported", lang in SUPPORTED)

print("\nspanish dropped — no longer a TTS-supported language")
check("spanish not in supported set", "spanish" not in SUPPORTED)
check("hola como estas no longer detected as spanish",
      detect_language("hola como estas") != "spanish")

print()
if failures:
    print(f"{len(failures)} FAILED: {failures}")
    raise SystemExit(1)
print("all language tests passed")
