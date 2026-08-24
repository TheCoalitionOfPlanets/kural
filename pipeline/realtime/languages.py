"""Language detection, reply-language policy, and stack routing.

The reply must be in the same language the user spoke. The prompt asks for
that, but a 4B model ignores the instruction under load — it was observed
correctly identifying "epdi irukka" as Tamil and answering in English anyway.
So the language is decided here, in code, and stated to the model as a
per-turn directive rather than left to it to infer.

Two detection paths, because STT output arrives two ways:

* **Native script** — unambiguous. Unicode blocks map one-to-one onto scripts,
  so a single Tamil character is proof of Tamil.
* **Romanized** ("enna panra", "kya kar rahe ho") — ambiguous by nature, since
  it is Latin text. Scored against per-language function-word sets: the words
  a speaker cannot avoid using.

This module is also where the **stack routing** decision lives. The local
models are Indic by construction — SraVaani hears the scheduled Indian
languages plus English, Indic-Mio speaks the same set — so a language outside
that set has neither a local ear nor a local voice, and is served by ElevenLabs
instead. `route_for()` is the one place that decision is made; STT, the LLM and
TTS all consult it rather than each keeping their own list.

Imported by workers running in three different venvs, so it stays stdlib-only.
"""
import re
import time
import unicodedata

ENGLISH = "english"

# Unicode ranges are the strongest signal available: a script identifies its
# language directly, with no statistics involved.
_SCRIPT_RANGES = [
    ("tamil", 0x0B80, 0x0BFF),
    ("telugu", 0x0C00, 0x0C7F),
    ("kannada", 0x0C80, 0x0CFF),
    ("malayalam", 0x0D00, 0x0D7F),
    ("gujarati", 0x0A80, 0x0AFF),
    ("odia", 0x0B00, 0x0B7F),
    ("punjabi", 0x0A00, 0x0A7F),
    ("sinhala", 0x0D80, 0x0DFF),    # not one of Indic-Mio's languages; no voice
    # Devanagari and Bengali each carry several languages; the majority case is
    # named, since script alone cannot separate them. Indic-Mio speaks all of
    # these natively (Marathi, Sanskrit, Nepali, Konkani, Maithili, Dogri,
    # Assamese, ...), so bucketing them under "hindi"/"bengali" no longer
    # costs a wrong-language voice — TTS simply speaks the majority language.
    ("hindi", 0x0900, 0x097F),      # also Marathi, Sanskrit, Nepali, Konkani
    ("bengali", 0x0980, 0x09FF),    # also Assamese
    # Perso-Arabic is shared. Arabic is the default and Urdu is promoted out
    # of it by its own letters — see _PERSO_ARABIC_URDU below.
    ("arabic", 0x0600, 0x06FF),

    # -- non-Indic scripts, for the ElevenLabs path ------------------------
    # These exist so a reply whose language was never established upstream
    # (no LID answer, no Scribe answer) still routes somewhere sane instead of
    # falling through to English and being read aloud by the wrong voice.
    ("russian", 0x0400, 0x04FF),    # Cyrillic; also Ukrainian and Bulgarian
    ("greek", 0x0370, 0x03FF),
    ("hebrew", 0x0590, 0x05FF),
    ("thai", 0x0E00, 0x0E7F),
    ("korean", 0xAC00, 0xD7AF),     # Hangul syllables
    ("korean", 0x1100, 0x11FF),     # Hangul jamo
    ("japanese", 0x3040, 0x309F),   # Hiragana — Japanese-only, unlike Han
    ("japanese", 0x30A0, 0x30FF),   # Katakana
    ("chinese", 0x4E00, 0x9FFF),    # Han: shared, so kana breaks the tie
    ("chinese", 0x3400, 0x4DBF),    # Han extension A
]

# 0x0600-0x06FF carries both Urdu and Arabic, and only Urdu uses these letters
# (ٹ ڈ ڑ ں ھ ہ ے ...). The split matters because the two route to
# different stacks: Urdu is one of Indic-Mio's languages, Arabic is not. Any one
# of these is proof of Urdu; real Urdu text is dense with ہ and ے in
# particular, so a sentence never lacks them all.
_PERSO_ARABIC_URDU = frozenset(
    "\u0679\u0688\u0691\u06BA\u06BE\u06C1\u06C2\u06C3\u06D2\u06D3"
)


# Function words: grammatical machinery a speaker cannot write around, so they
# appear in nearly every real utterance of that language. Content words are
# deliberately excluded — they are what code-switching borrows.
_ROMANIZED = {
    "tamil": {
        "enna", "epdi", "eppadi", "irukka", "irukku", "irukkinga", "iruken",
        "panra", "panren", "panni", "pannu", "seiya", "vanakkam", "nandri",
        "naan", "nee", "neenga", "avan", "aval", "avanga", "namma", "unga",
        "illa", "illai", "aama", "ama", "sari", "seri", "romba", "konjam",
        "ipo", "ippo", "appo", "yaaru", "yaar", "edhu", "ethu", "engo", "enga",
        "machan", "machi", "thala", "solla", "sollu", "sonna",
        "venum", "vendam", "mudiyum", "mudiyathu", "theriyuma", "theriyala",
        "poi", "poga", "vaa", "vanga", "kudu", "kodu", "paaru", "paru",
    },
    "hindi": {
        "kya", "kaise", "kaisa", "kaisi", "kar", "kare", "karo", "karna",
        "raha", "rahi", "rahe", "ho", "hai", "hain", "tha", "thi",
        "main", "mera", "meri", "tum", "tera", "aap", "apka", "hum", "hamara",
        "nahi", "nahin", "haan", "haa", "acha", "accha", "theek", "thik",
        "bohot", "bahut", "thoda", "kyun", "kyon", "kab", "kahan", "kaun",
        "yaar", "bhai", "namaste", "dhanyavad", "shukriya", "chalo", "batao",
        "bata", "suno", "dekho", "kuch", "sab", "abhi", "phir", "lekin",
        "mujhe", "mujhko", "tujhe", "usko", "unko", "humko", "hume", "aapko",
        "ek", "baat", "bol", "bolo", "jana", "jaana", "aana", "milta",
        "chahiye", "hoga", "hogi", "sakta", "sakte", "sakti", "wala", "wali",
    },
    "telugu": {
        "enti", "ela", "unnav", "unnaru", "unnanu", "chesthunnav", "chesthunna",
        "cheyyi", "cheyi", "nenu", "nuvvu", "meeru", "manam", "vaadu", "aame",
        "ledu", "kadu", "avunu", "sare", "chala", "koncham", "ippudu", "appudu",
        "evaru", "emiti", "ekkada", "eppudu", "namaskaram", "dhanyavadalu",
        "bagunnara", "baagunnanu", "cheppu", "cheppandi", "randi", "vellandi",
    },
    "kannada": {
        "enu", "hege", "iddiya", "iddira", "iddini", "madtidiya", "madu",
        "nanu", "neenu", "neevu", "namma", "avanu", "avalu", "illa", "houdu",
        "sari", "tumba", "swalpa", "yaaru", "elli", "yavaga", "namaskara",
        "dhanyavada", "heli", "hELi", "banni", "hogi", "gottu", "gottilla",
    },
    "malayalam": {
        "entha", "engane", "undo", "undu", "cheyyunnu", "cheyyu", "njan",
        "nee", "ningal", "avan", "aval", "illa", "alla", "athe", "sari",
        "orupad", "kurach", "aara", "evide", "eppo", "namaskaram", "nandi",
        "parayu", "para", "vaa", "വരൂ", "ariyilla", "ariyam",
    },
}

# Markers that are also ordinary English words are worse than useless: they
# fire on English sentences, and English is the fallback that has no markers of
# its own to outvote them. Two occurrences of "the" were enough to classify a
# 34-word English sentence as Hindi, so any marker colliding with this list is
# dropped at import rather than trusted to review.
#
# Only genuine collisions belong here — words that are common in English AND
# were plausible markers elsewhere ("the" थे, "do" दो, "no", "da"/"di").
_ENGLISH_COLLISIONS = frozenset({
    "the", "do", "no", "da", "di", "a", "an", "and", "or", "if", "is", "are",
    "was", "were", "be", "am", "he", "it", "we", "you", "they", "i", "me",
    "my", "to", "of", "in", "on", "at", "by", "for", "with", "from", "as",
    "not", "yes", "so", "too", "up", "out", "one", "two", "can", "will",
    "what", "when", "who", "how", "all", "any", "some", "more", "like",
    "see", "go", "get", "make", "take", "time", "way", "man", "men", "sir",
    "ok", "okay", "hi", "hello", "bye", "please", "thanks", "thank", "sari",
})

_ROMANIZED = {
    lang: frozenset(markers) - _ENGLISH_COLLISIONS
    for lang, markers in _ROMANIZED.items()
}

# Bare Latin text with none of the markers above is treated as English. English
# has no romanized-marker set of its own: it is the fallback, so adding one
# would only create ties.
SUPPORTED = frozenset({ENGLISH} | set(_ROMANIZED) | {n for n, _, _ in _SCRIPT_RANGES})

# --------------------------------------------------------------- routing ---
#
# Which stack owns which language. The local models are Indic by construction:
# SraVaani hears the scheduled Indian languages plus English, Indic-Mio speaks
# the same set from one set of weights. Everything else — Spanish, Russian,
# Japanese, Arabic — has no local ear and no local voice, so both STT and TTS
# for it are served by ElevenLabs.
#
# The set below is therefore the definition of "handled locally", and its
# complement is the definition of "international". There is no separate list of
# international languages to keep in sync: anything not named here is one.

ROUTE_LOCAL = "local"
ROUTE_INTERNATIONAL = "international"

LOCAL = frozenset({
    ENGLISH,
    # Indic-Mio's own language list (model-card frontmatter), which is also
    # SraVaani's. The script detector cannot separate most of these — every
    # Devanagari language reads as "hindi" — but audio-level LID names them
    # directly, so they have to be recognized here or a Marathi turn would be
    # routed abroad.
    "hindi", "bengali", "marathi", "telugu", "kannada", "tamil", "malayalam",
    "gujarati", "punjabi", "odia", "urdu", "nepali", "assamese", "sanskrit",
    "konkani", "maithili", "dogri", "bodo", "santali", "sindhi", "manipuri",
    "kashmiri",
})

# What ElevenLabs' multilingual TTS actually speaks. Scribe hears far more
# languages than this, so a turn can be transcribed perfectly and still have no
# voice — Sinhala and Vietnamese are the common cases. Those get the same
# text-only treatment the local stack already gives a missing voice, rather
# than being read aloud by a model guessing at the script.
ELEVEN_TTS_LANGUAGES = frozenset({
    "english", "japanese", "chinese", "german", "hindi", "french", "korean",
    "portuguese", "italian", "spanish", "indonesian", "dutch", "turkish",
    "filipino", "polish", "swedish", "bulgarian", "romanian", "arabic",
    "czech", "greek", "finnish", "croatian", "malay", "slovak", "danish",
    "tamil", "ukrainian", "russian",
})

# ISO 639-3, because that is what the LID model emits and what ElevenLabs'
# speech-to-text accepts as a hint. Only the mapping to this pipeline's own
# language names is needed; unmapped codes are handled by name-from-code
# returning the bare code, which still routes correctly (it is not in LOCAL).
_ISO639_3 = {
    # -- local: English plus the scheduled Indian languages -----------------
    "eng": ENGLISH,
    "hin": "hindi", "ben": "bengali", "mar": "marathi", "tel": "telugu",
    "kan": "kannada", "tam": "tamil", "mal": "malayalam", "guj": "gujarati",
    "pan": "punjabi", "ory": "odia", "ori": "odia", "urd": "urdu",
    "npi": "nepali", "nep": "nepali", "asm": "assamese", "san": "sanskrit",
    "kok": "konkani", "gom": "konkani", "mai": "maithili", "doi": "dogri",
    "brx": "bodo", "sat": "santali", "snd": "sindhi", "mni": "manipuri",
    "kas": "kashmiri",
    # -- international ------------------------------------------------------
    "spa": "spanish", "fra": "french", "fre": "french", "deu": "german",
    "ger": "german", "ita": "italian", "por": "portuguese", "rus": "russian",
    "jpn": "japanese", "kor": "korean", "nld": "dutch", "dut": "dutch",
    "tur": "turkish", "pol": "polish", "swe": "swedish", "bul": "bulgarian",
    "ron": "romanian", "rum": "romanian", "ara": "arabic", "arb": "arabic",
    "ces": "czech", "cze": "czech", "ell": "greek", "gre": "greek",
    "fin": "finnish", "hrv": "croatian", "slk": "slovak", "slo": "slovak",
    "dan": "danish", "ukr": "ukrainian", "ind": "indonesian",
    "zsm": "malay", "msa": "malay", "zlm": "malay",
    "fil": "filipino", "tgl": "filipino",
    "cmn": "chinese", "zho": "chinese", "chi": "chinese", "yue": "chinese",
    "vie": "vietnamese", "tha": "thai", "heb": "hebrew", "hun": "hungarian",
    "nob": "norwegian", "nor": "norwegian", "sin": "sinhala", "fas": "persian",
    "pes": "persian", "per": "persian", "swa": "swahili", "afr": "afrikaans",
}

# ISO 639-1 is what most of the world's language tags look like, and Scribe
# accepts either. Two-letter codes are folded in so a code arriving from
# somewhere other than the LID model still resolves.
_ISO639_1 = {
    "en": ENGLISH, "hi": "hindi", "bn": "bengali", "mr": "marathi",
    "te": "telugu", "kn": "kannada", "ta": "tamil", "ml": "malayalam",
    "gu": "gujarati", "pa": "punjabi", "or": "odia", "ur": "urdu",
    "ne": "nepali", "as": "assamese", "sa": "sanskrit", "sd": "sindhi",
    "ks": "kashmiri", "si": "sinhala",
    "es": "spanish", "fr": "french", "de": "german", "it": "italian",
    "pt": "portuguese", "ru": "russian", "ja": "japanese", "ko": "korean",
    "nl": "dutch", "tr": "turkish", "pl": "polish", "sv": "swedish",
    "bg": "bulgarian", "ro": "romanian", "ar": "arabic", "cs": "czech",
    "el": "greek", "fi": "finnish", "hr": "croatian", "sk": "slovak",
    "da": "danish", "uk": "ukrainian", "id": "indonesian", "ms": "malay",
    "tl": "filipino", "zh": "chinese", "vi": "vietnamese", "th": "thai",
    "he": "hebrew", "hu": "hungarian", "no": "norwegian", "fa": "persian",
    "sw": "swahili", "af": "afrikaans",
}

# Languages written in Latin script. The reply directive tells the model to use
# the language's *native* script, which is right for Tamil and Japanese and
# actively wrong for Spanish — "do not use Latin letters for Spanish words"
# is an instruction the model can only obey by producing nonsense.
_LATIN_SCRIPT = frozenset({
    ENGLISH, "spanish", "french", "german", "italian", "portuguese", "dutch",
    "turkish", "polish", "swedish", "romanian", "czech", "finnish",
    "croatian", "slovak", "danish", "indonesian", "malay", "filipino",
    "vietnamese", "hungarian", "norwegian", "swahili", "afrikaans",
})


def language_from_code(code):
    """ISO 639-1/639-3 -> this pipeline's language name.

    An unknown code is returned as-is rather than dropped. It will not be in
    LOCAL, so it routes to ElevenLabs — which is the right answer for a
    language nobody here has heard of, and better than silently calling it
    English and handing it to a stack that cannot say it.
    """
    if not code:
        return None
    key = str(code).strip().lower().replace("_", "-")
    # Tags arrive as "es", "es-ES", "spa-Latn"; only the primary subtag names
    # the language.
    key = key.split("-")[0]
    if key in _ISO639_3:
        return _ISO639_3[key]
    if key in _ISO639_1:
        return _ISO639_1[key]
    return key or None


def route_for(lang):
    """Which stack should handle this language.

    The default is local: a turn whose language was never established is
    overwhelmingly English or Indic here, and guessing local costs a bad
    transcript while guessing international costs an API call on every
    ambiguous turn.
    """
    if not lang:
        return ROUTE_LOCAL
    return ROUTE_LOCAL if str(lang).strip().lower() in LOCAL \
        else ROUTE_INTERNATIONAL


def is_international(lang):
    return route_for(lang) == ROUTE_INTERNATIONAL


def has_eleven_voice(lang):
    """Whether ElevenLabs' multilingual TTS can speak this language."""
    return str(lang or "").strip().lower() in ELEVEN_TTS_LANGUAGES


class RouteGate:
    """Turns a language-ID prediction into a routing decision.

    Deliberately separate from the model that produces the prediction: the
    policy here — how confident is confident enough, how short is too short,
    how long a language keeps the benefit of the doubt — is the part that gets
    tuned and the part that can be wrong, and none of it needs a GPU to reason
    about. The worker owns the forward pass; this owns what to do with it.

    Every default leans local. A wrong local route costs one bad transcript; a
    wrong international route costs an API call, and on a mostly-Indic pipeline
    the ambiguous turns are mostly Indic.
    """

    def __init__(self, min_confidence=0.55, min_audio_s=0.7, sticky_ttl_s=60.0,
                 clock=time.time):
        self.min_confidence = float(min_confidence)
        self.min_audio_s = float(min_audio_s)
        self.sticky_ttl_s = float(sticky_ttl_s)
        self._clock = clock
        self._sticky = None
        self._sticky_at = 0.0

    @property
    def sticky(self):
        """The language recent turns settled on, or None once it has expired."""
        if self._sticky is None or not self.sticky_ttl_s:
            return None
        if (self._clock() - self._sticky_at) > self.sticky_ttl_s:
            return None
        return self._sticky

    def should_identify(self, duration_s):
        """Whether this utterance is even worth running the model on.

        LID on a fragment is a coin flip, and short turns ("yes", "mm", "wait")
        are overwhelmingly in the language already being spoken. Asked before
        the forward pass rather than after, so a clip that cannot produce a
        usable answer does not cost one.
        """
        return duration_s >= self.min_audio_s

    def decide(self, lang, confidence, duration_s):
        """(route, language) for one utterance.

        The language is None whenever the gate declined to commit, which puts
        everything downstream back on transcript-based detection — the
        behaviour from before any of this existed.
        """
        if not lang or not self.should_identify(duration_s):
            return ROUTE_LOCAL, None

        confident = confidence >= self.min_confidence
        # Hysteresis, and only in this direction: it can confirm a shaky
        # prediction that already agrees with the last turn, never overrule a
        # confident one. So a conversation in Spanish survives one mumbled
        # sentence, and switching back to Tamil still lands on the very next
        # sentence rather than waiting for the window to lapse.
        if not confident and lang == self.sticky:
            confident = True
        if not confident:
            return ROUTE_LOCAL, None
        return route_for(lang), lang

    def observe(self, lang):
        """Record what the turn actually turned out to be.

        Called with the *final* language, which on the international path comes
        from Scribe rather than from LID — so the window tracks what was really
        being spoken, not what was guessed before transcription.

        `None` means nothing was established, and leaves the window alone. That
        distinction is the difference between hysteresis that works and
        hysteresis that does not: an abstention is exactly the case the window
        exists to cover, so treating it as a language change would close the
        window on the turns it was built for. Only a turn that actually
        resolved to a local language closes it — and that is a confident
        prediction, which is what makes switching back immediate.
        """
        if not lang:
            return
        if is_international(lang):
            self._sticky, self._sticky_at = lang, self._clock()
        else:
            self._sticky, self._sticky_at = None, 0.0

_WORD = re.compile(r"[a-z]+")

# A couple of romanized markers inside an otherwise English sentence is
# code-switching ("send it to my machan"), not a language change. Requiring a
# real share of the sentence keeps those in English.
_MIN_ROMANIZED_HITS = 2
_MIN_ROMANIZED_SHARE = 0.30


def _script_of(ch):
    for name, lo, hi in _SCRIPT_RANGES:
        if lo <= ord(ch) <= hi:
            return name
    return None


def detect_language(text):
    """Return the language name to reply in.

    Native script wins outright when present — it is unambiguous, and mixing a
    little Latin into an Indic sentence (brand names, numerals) is normal.
    """
    if not text or not text.strip():
        return ENGLISH

    # Count characters per script, ignoring punctuation/whitespace/digits.
    counts = {}
    latin = 0
    for ch in text:
        if ch.isspace() or unicodedata.category(ch).startswith(("P", "N", "Z")):
            continue
        script = _script_of(ch)
        if script:
            counts[script] = counts.get(script, 0) + 1
        elif "a" <= ch.lower() <= "z":
            latin += 1

    if counts:
        # Han characters are shared, so a Japanese sentence lands in both
        # buckets and can lose the majority vote to its own kanji. Kana are
        # Japanese-only, so their presence settles it before counting.
        if "japanese" in counts and "chinese" in counts:
            counts["japanese"] += counts.pop("chinese")

        # Majority meaning-bearing script, per the prompt's code-mixing rule.
        winner = max(counts.items(), key=lambda kv: kv[1])[0]

        # Urdu and Arabic share a block and route to different stacks, so the
        # Urdu-only letters decide between them.
        if winner == "arabic" and any(ch in _PERSO_ARABIC_URDU for ch in text):
            return "urdu"
        return winner

    if not latin:
        return ENGLISH

    words = _WORD.findall(text.lower())
    if not words:
        return ENGLISH

    best, best_hits = ENGLISH, 0
    for lang, markers in _ROMANIZED.items():
        hits = sum(1 for w in words if w in markers)
        if hits > best_hits:
            best, best_hits = lang, hits

    if best_hits >= _MIN_ROMANIZED_HITS or (
        best_hits and best_hits / len(words) >= _MIN_ROMANIZED_SHARE
    ):
        return best
    return ENGLISH


def language_directive(lang):
    """The per-turn instruction appended to the system prompt.

    Placed last because position beats emphasis: a rule at the end of the
    prompt is followed far more reliably than one marked "HIGHEST PRIORITY"
    in the middle.
    """
    name = (lang or ENGLISH).strip() or ENGLISH
    if name == ENGLISH:
        return ("The user just spoke in English. Reply in English only, "
                "not any other language.")

    label = name.capitalize()
    base = (f"The user just spoke in {label}. Reply ONLY in {label}. "
            f"Do not reply in English.")
    if name in _LATIN_SCRIPT:
        # Spanish and its neighbours are *written* in Latin script, so the
        # native-script clause below would be an instruction to garble them.
        # Naming the language is the whole requirement here.
        return base
    return (
        f"{base} Write it in the native {label} script. "
        f"Do not use Latin letters for {label} words."
    )
