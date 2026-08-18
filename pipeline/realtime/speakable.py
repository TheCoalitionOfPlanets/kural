"""Backstops for prompt rules that matter acoustically.

A 4B quantized model ignores instructions under load. Every rule whose
violation is *audible* gets enforced in code rather than trusted to the prompt:
the prompt asks, this decides.
"""
import re

# Emoji and pictographs. Gemma sneaks these in despite the prompt, and TTS
# either reads them as words or emits noise.
_EMOJI = re.compile(
    "["
    "\U0001F300-\U0001FAFF"  # symbols, pictographs, emoticons, extended-A
    "\U00002600-\U000027BF"  # misc symbols, dingbats
    "\U0001F000-\U0001F0FF"  # mahjong, dominoes, cards
    "\U00002190-\U000021FF"  # arrows
    "\U0000FE00-\U0000FE0F"  # variation selectors
    "\U0001F1E6-\U0001F1FF"  # regional indicators (flags)
    "\U00002B00-\U00002BFF"
    "]+",
    flags=re.UNICODE,
)

# A leaked reasoning span is fatal twice over: it burns the token budget before
# any speakable text appears, and whatever escapes gets read aloud.
_THINK = re.compile(r"<think>.*?</think>", flags=re.DOTALL | re.IGNORECASE)
_THINK_OPEN = re.compile(r"<think>.*", flags=re.DOTALL | re.IGNORECASE)

_CODE_FENCE = re.compile(r"```[\s\S]*?```")
# Markdown emphasis/heading/bullet markers that TTS would read as punctuation
# noise or awkward pauses.
_MD_HEADING = re.compile(r"^\s{0,3}#{1,6}\s*", flags=re.MULTILINE)
_MD_BULLET = re.compile(r"^\s{0,3}[-*+]\s+", flags=re.MULTILINE)
_MD_NUMBER = re.compile(r"^\s{0,3}\d+[.)]\s+", flags=re.MULTILINE)
_MD_EMPHASIS = re.compile(r"(\*{1,3}|_{1,3}|`+)")
_BLOCKQUOTE = re.compile(r"^\s{0,3}>\s?", flags=re.MULTILINE)

_WS = re.compile(r"[ \t]+")
_BLANKS = re.compile(r"\n{3,}")


def strip_think(text):
    """Remove reasoning spans, closed or unterminated."""
    text = _THINK.sub("", text)
    return _THINK_OPEN.sub("", text)


def strip_unspeakable(text, keep_code=False):
    """Make model output safe to hand to a TTS voice.

    Removes what a voice cannot pronounce (emoji, markdown syntax) while
    leaving the words themselves intact. Pass keep_code=True when the user
    explicitly asked for code.
    """
    if not text:
        return ""

    text = strip_think(text)
    if not keep_code:
        text = _CODE_FENCE.sub(" ", text)
    text = _EMOJI.sub("", text)
    text = _MD_HEADING.sub("", text)
    text = _MD_BULLET.sub("", text)
    text = _MD_NUMBER.sub("", text)
    text = _BLOCKQUOTE.sub("", text)
    text = _MD_EMPHASIS.sub("", text)

    text = _WS.sub(" ", text)
    text = _BLANKS.sub("\n\n", text)
    return text.strip()
