"""Language detection and reply-language policy.

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
"""
import re
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
    ("urdu", 0x0600, 0x06FF),       # Perso-Arabic
]

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
        # Majority meaning-bearing script, per the prompt's code-mixing rule.
        return max(counts.items(), key=lambda kv: kv[1])[0]

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
    return (
        f"The user just spoke in {name.capitalize()}. "
        f"Reply ONLY in {name.capitalize()}, written in the native "
        f"{name.capitalize()} script. Do not reply in English. "
        f"Do not use Latin letters for {name.capitalize()} words."
    )
