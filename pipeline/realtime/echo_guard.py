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
import time
import unicodedata

# Below this, common words ("yes", "ok", "sure") would false-positive against
# any long reply that happens to contain them. Missing a suppression is
# cheaper than swallowing a real user turn.
MIN_ECHO_WORDS = 3

# An entry older than this is no longer audible and cannot be bleeding. Keeping
# it in the comparison set only invents false positives against later user
# speech that happens to reuse the same words.
DEFAULT_TTL_S = 30.0

# Length of the contiguous word run that counts as bleed on its own. The mic
# catches a fragment of a reply, and a fragment is a *run*, not a scattered bag
# of words: five words in the same order is far past coincidence, which set
# containment cannot distinguish.
NGRAM_N = 5

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


def _longest_run(words, spoken_words):
    """Longest run of `words` appearing consecutively inside `spoken_words`."""
    if not words or not spoken_words:
        return 0
    positions = {}
    for i, w in enumerate(spoken_words):
        positions.setdefault(w, []).append(i)

    best = 0
    for i, w in enumerate(words):
        for start in positions.get(w, ()):
            run = 0
            while (i + run < len(words) and start + run < len(spoken_words)
                   and words[i + run] == spoken_words[start + run]):
                run += 1
            best = max(best, run)
    return best


def is_echo_of(transcript, spoken, threshold=0.6, ngram=NGRAM_N):
    """True when `transcript` looks like bleed from `spoken`.

    Each candidate is scored *separately* rather than against a merged bag of
    words from all of them. Pooling inflates the vocabulary until unrelated user
    speech clears the threshold on common words alone.

    Two independent signals, either sufficient:

    1. **Containment** — the fraction of transcript words present in one reply.
       The mic catches a short fragment of a much longer reply, so symmetric
       measures (SequenceMatcher, Jaccard) are dominated by the length mismatch
       and score real echoes near zero.
    2. **Contiguous run** — a shared run of NGRAM_N words in the same order.
       Word order is evidence that set containment throws away: a long verbatim
       run is bleed regardless of what fraction of the transcript it covers.

    `spoken` may be a single string or an iterable of recent chunks.
    """
    words = _norm(transcript)
    if len(words) < MIN_ECHO_WORDS:
        return False

    if isinstance(spoken, str):
        spoken = [spoken]

    for chunk in spoken:
        chunk_words = _norm(chunk)
        if not chunk_words:
            continue

        vocab = set(chunk_words)
        overlap = sum(1 for w in words if w in vocab)
        if (overlap / len(words)) >= threshold:
            return True

        if ngram > 0 and _longest_run(words, chunk_words) >= ngram:
            return True

    return False


class RecentSpeech:
    """Thread-safe cache of what the assistant recently said aloud.

    A window rather than just the newest utterance because playback lags
    synthesis — at the moment bleed is captured, the audible sentence may be
    any of the last few.

    Entries carry a timestamp and expire: a reply that finished playing long ago
    is not audible and cannot be bleeding, so leaving it in the comparison set
    only blocks later user speech that reuses the same words.
    """

    def __init__(self, maxlen=6, ttl_s=DEFAULT_TTL_S, ngram=NGRAM_N):
        self._items = []          # list of (monotonic_ts, text)
        self._maxlen = int(maxlen)
        self._ttl_s = float(ttl_s)
        self._ngram = int(ngram)
        self._lock = threading.Lock()

    def _prune(self, now):
        if self._ttl_s > 0:
            cutoff = now - self._ttl_s
            self._items = [it for it in self._items if it[0] >= cutoff]
        del self._items[: -self._maxlen]

    def add(self, text):
        """Record text the assistant is about to say.

        Idempotent: the same reply is recorded at generation and again at
        synthesis, and duplicates would otherwise consume two of the `maxlen`
        slots and evict a still-audible earlier reply.
        """
        text = (text or "").strip()
        if not text:
            return
        now = time.monotonic()
        with self._lock:
            for i, (_ts, existing) in enumerate(self._items):
                if existing == text:
                    self._items[i] = (now, existing)
                    break
            else:
                self._items.append((now, text))
            self._prune(now)

    # Playback calls this when audio actually starts, to restamp the entry as
    # audible now. Text is recorded at generation, but the room does not hear it
    # until synthesis, queue wait and playback have elapsed — long enough for a
    # reply to expire while still coming out of the speakers.
    touch = add

    def snapshot(self):
        now = time.monotonic()
        with self._lock:
            self._prune(now)
            return [text for _ts, text in self._items]

    def clear(self):
        with self._lock:
            self._items.clear()

    def is_echo(self, transcript, threshold=0.6):
        return is_echo_of(transcript, self.snapshot(), threshold, self._ngram)

    def __len__(self):
        return len(self.snapshot())
