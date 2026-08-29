"""Tests for language detection, stack routing, and the reply directive.

    venv\\Scripts\\python.exe pipeline\\tests\\test_languages.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from pipeline.realtime.languages import (  # noqa: E402
    ENGLISH,
    ROUTE_INTERNATIONAL,
    ROUTE_LOCAL,
    SUPPORTED,
    RouteGate,
    detect_language,
    has_mms_voice,
    mms_voice_code,
    language_directive,
    language_from_code,
    route_for,
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

print("\nnon-Indic scripts — they route abroad, so they must be named")
expect("Привет, как дела?", "russian")
expect("こんにちは、元気ですか", "japanese")
expect("안녕하세요 잘 지내세요", "korean")
expect("你好，最近怎么样", "chinese")
expect("Γεια σου, τι κάνεις;", "greek")
expect("มาลาสวัสดี", "thai")

# Han is shared, so a Japanese sentence lands in both buckets and can lose the
# majority vote to its own kanji. Kana settle it.
expect("私は日本語を話します", "japanese")

# Urdu and Arabic share a Unicode block and route to different stacks: Urdu is
# one of Indic-Mio's languages, Arabic is not.
expect("آپ کیسے ہیں؟", "urdu")
expect("مرحبا كيف حالك اليوم", "arabic")

print("\nrouting — which stack owns the turn")
for _lang in ("tamil", "hindi", "telugu", "kannada", "malayalam", "marathi",
              "assamese", "sanskrit", ENGLISH):
    check(f"{_lang} -> local", route_for(_lang) == ROUTE_LOCAL)

# Urdu and Kashmiri are the split case, and the reason route_for takes a stage
# at all: SraVaani's model card excludes both, Indic-Mio speaks both. So each
# stage answers separately — the ear leaves, the voice stays home — and the
# stage-less answer is "not fully local", which is what it should be.
for _lang in ("urdu", "kashmiri"):
    check(f"{_lang} is not heard locally",
          route_for(_lang, stage="stt") == ROUTE_INTERNATIONAL)
    check(f"{_lang} is spoken locally",
          route_for(_lang, stage="tts") == ROUTE_LOCAL)

for _lang in ("spanish", "russian", "japanese", "arabic", "korean", "chinese",
              "french", "german", "sinhala"):
    check(f"{_lang} -> set B", route_for(_lang) == ROUTE_INTERNATIONAL)

# No language established yet is not a reason to load the Set B stack.
check("unknown language stays local", route_for(None) == ROUTE_LOCAL)
check("empty language stays local", route_for("") == ROUTE_LOCAL)
# Anything the tables have never heard of has no local voice by definition.
check("unrecognized language routes abroad",
      route_for("klingon") == ROUTE_INTERNATIONAL)

print("\nlanguage codes — ISO 639-3 from LID, 639-1 from anywhere else")
for _code, _name in (("spa", "spanish"), ("rus", "russian"), ("jpn", "japanese"),
                     ("tam", "tamil"), ("eng", ENGLISH), ("mar", "marathi"),
                     ("ory", "odia"), ("cmn", "chinese"),
                     ("es", "spanish"), ("ja", "japanese"), ("ta", "tamil")):
    check(f"{_code} -> {_name}", language_from_code(_code) == _name)

# Regional tags arrive as "es-ES" / "spa-Latn"; only the primary subtag names
# the language.
check("es-ES -> spanish", language_from_code("es-ES") == "spanish")
check("spa-Latn -> spanish", language_from_code("spa-Latn") == "spanish")
check("no code -> nothing", language_from_code(None) is None)
# An unmapped code must survive rather than be silently called English: it is
# not in LOCAL, so it still routes to the stack that might handle it.
check("unknown code kept as-is", language_from_code("qqq") == "qqq")

print("\ntts coverage — heard is not the same as speakable")
check("spanish has an mms voice", has_mms_voice("spanish"))
check("japanese has an mms voice", has_mms_voice("japanese"))
# The code is what the worker actually needs: checkpoints are per-language
# directories named by ISO 639-3, so "can it be spoken" and "which one loads"
# are the same question.
check("spanish maps to its checkpoint", mms_voice_code("spanish") == "spa")
# MMS ships Mandarin as `cmn`, not `zho`, and Filipino under Tagalog — the two
# codes most likely to be written from memory and be wrong.
check("chinese maps to cmn", mms_voice_code("chinese") == "cmn")
check("filipino maps to tgl", mms_voice_code("filipino") == "tgl")
# Whisper transcribes these; no checkpoint is configured for them, so the reply
# is shown as text instead of mispronounced.
check("klingon has no mms voice", not has_mms_voice("klingon"))
check("no language has no voice", mms_voice_code(None) is None)

print("\nreply directive")
# Naming the script is right for Tamil and Japanese, and actively wrong for
# Spanish — "do not use Latin letters for Spanish words" can only be obeyed by
# producing nonsense.
check("spanish directive names the language",
      "Spanish" in language_directive("spanish"))
check("spanish directive does not demand a non-Latin script",
      "Latin letters" not in language_directive("spanish"))
check("japanese directive demands the native script",
      "native Japanese script" in language_directive("japanese"))
check("tamil directive demands the native script",
      "native Tamil script" in language_directive("tamil"))

print("\nroute gate — the policy in front of the language-ID model")
clock = {"t": 1000.0}


def gate(**kw):
    kw.setdefault("min_confidence", 0.55)
    kw.setdefault("min_audio_s", 0.7)
    kw.setdefault("sticky_ttl_s", 60)
    return RouteGate(clock=lambda: clock["t"], **kw)


g = gate()
check("confident spanish routes abroad",
      g.decide("spanish", 0.9, 3.0) == (ROUTE_INTERNATIONAL, "spanish"))
check("confident tamil stays local",
      g.decide("tamil", 0.9, 3.0) == (ROUTE_LOCAL, "tamil"))

# Below the bar the gate declines to commit, and declining means local: a bad
# local route costs one transcript, a bad international route costs money.
check("unconfident spanish stays local",
      gate().decide("spanish", 0.3, 3.0) == (ROUTE_LOCAL, None))
check("unconfident tamil commits to nothing",
      gate().decide("tamil", 0.3, 3.0) == (ROUTE_LOCAL, None))

# A fragment is a coin flip however confident the model claims to be.
check("short utterance stays local however confident",
      gate().decide("spanish", 0.99, 0.4) == (ROUTE_LOCAL, None))
check("no prediction stays local",
      gate().decide(None, 0.99, 3.0) == (ROUTE_LOCAL, None))

print("\nroute gate — hysteresis only ever confirms, never overrules")
g = gate()
g.observe("spanish")
check("shaky spanish is confirmed by the previous turn",
      g.decide("spanish", 0.3, 3.0) == (ROUTE_INTERNATIONAL, "spanish"))
# The whole point: switching back to an Indian language must land on the very
# next sentence, not once the window lapses.
check("confident tamil wins immediately over sticky spanish",
      g.decide("tamil", 0.9, 3.0) == (ROUTE_LOCAL, "tamil"))
check("shaky russian is not confirmed by sticky spanish",
      g.decide("russian", 0.3, 3.0) == (ROUTE_LOCAL, None))

# A turn that resolved to a local language ends the streak.
g.observe("tamil")
check("a local turn closes the window",
      g.decide("spanish", 0.3, 3.0) == (ROUTE_LOCAL, None))

# But an abstention must not. It is precisely the case the window exists to
# cover, so closing it there would defeat the whole mechanism on the turns it
# was built for.
g = gate()
g.observe("spanish")
g.observe(None)
check("an abstention leaves the window open",
      g.decide("spanish", 0.3, 3.0) == (ROUTE_INTERNATIONAL, "spanish"))

# A clip too short to classify never reaches the model at all.
check("short clips are not worth a forward pass",
      not gate().should_identify(0.4))
check("long enough clips are", gate().should_identify(1.5))

g = gate()
g.observe("spanish")
clock["t"] += 61
check("the window expires",
      g.decide("spanish", 0.3, 3.0) == (ROUTE_LOCAL, None))

g = gate(sticky_ttl_s=0)
g.observe("spanish")
check("sticky_ttl_s: 0 disables hysteresis",
      g.decide("spanish", 0.3, 3.0) == (ROUTE_LOCAL, None))

print()
if failures:
    print(f"{len(failures)} FAILED: {failures}")
    raise SystemExit(1)
print("all language tests passed")
