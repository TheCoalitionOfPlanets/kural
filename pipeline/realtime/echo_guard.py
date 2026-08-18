"""Text-level echo detection.

With speakers open the mic hears the assistant's own reply, STT transcribes it,
and the model answers itself — a runaway loop. The acoustic layers (quieter
output, muting capture during playback) reduce how much bleed exists but none
of them can be made airtight.

This layer is the only one that *identifies* echo rather than reducing its
odds: if the mic heard "I can help with that" while the assistant was saying
"I can help with that", that is bleed, not a human.

It is not a human-vs-bot classifier. The same sentence is valid from either
speaker — the only signal is overlap with what the assistant is currently
saying.
"""
import re
import threading
import unicodedata

# Below this, common words ("yes", "ok", "sure") would false-positive against
# any long reply that happens to contain them. Missing a suppression is
# cheaper than swallowing a real user turn.
MIN_ECHO_WORDS = 3

_SPACE = re.compile(r"\s+")


def _norm(text):
    """Lowercase, strip punctuation, split into words.

    STT punctuates and capitalizes differently from the raw reply text, so a
    literal comparison would miss nearly every real echo.

    Punctuation is removed by Unicode category rather than by `[^\\w\\s]`:
    `\\w` excludes Indic combining vowel marks (category Mn/Mc), so a regex
    approach splits "வணக்கம்" into fragments and no Tamil echo ever matches.
    """
    if not text:
        return []
    cleaned = "".join(
        " " if unicodedata.category(ch).startswith("P") or
        unicodedata.category(ch) in ("Sm", "Sk", "So", "Sc")
        else ch
        for ch in text.lower()
    )
    return _SPACE.sub(" ", cleaned).strip().split()


def is_echo_of(transcript, spoken, threshold=0.6):
    """True when `transcript` looks like bleed from `spoken`.

    Scores *containment* — the fraction of transcript words that appear in what
    was spoken — not similarity. The mic catches a short fragment of a much
    longer reply, so symmetric measures (SequenceMatcher, Jaccard) are dominated
    by the length mismatch and score real echoes near zero.

    `spoken` may be a single string or an iterable of recent chunks.
    """
    words = _norm(transcript)
    if len(words) < MIN_ECHO_WORDS:
        return False

    if isinstance(spoken, str):
        spoken = [spoken]
    vocab = set()
    for chunk in spoken:
        vocab.update(_norm(chunk))
    if not vocab:
        return False

    overlap = sum(1 for w in words if w in vocab)
    return (overlap / len(words)) >= threshold


class RecentSpeech:
    """Thread-safe rolling window of what the assistant recently said aloud.

    A window rather than just the newest utterance because playback lags
    synthesis — at the moment bleed is captured, the audible sentence may be
    any of the last few.
    """

    def __init__(self, maxlen=6):
        self._items = []
        self._maxlen = int(maxlen)
        self._lock = threading.Lock()

    def add(self, text):
        text = (text or "").strip()
        if not text:
            return
        with self._lock:
            self._items.append(text)
            del self._items[: -self._maxlen]

    def snapshot(self):
        with self._lock:
            return list(self._items)

    def clear(self):
        with self._lock:
            self._items.clear()

    def is_echo(self, transcript, threshold=0.6):
        return is_echo_of(transcript, self.snapshot(), threshold)

    def __len__(self):
        with self._lock:
            return len(self._items)
