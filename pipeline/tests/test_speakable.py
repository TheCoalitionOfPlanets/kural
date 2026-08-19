"""Tests for the speech-output backstops.

    venv\\Scripts\\python.exe pipeline\\tests\\test_speakable.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pipeline.realtime.speakable import (  # noqa: E402
    strip_think,
    strip_unspeakable,
)

failures = []


def check(name, cond):
    if cond:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}")
        failures.append(name)


print("strip_think")
check("removes closed span",
      strip_think("<think>hmm let me see</think>The answer is four.").strip()
      == "The answer is four.")
check("removes unterminated span",
      strip_think("Sure.<think>still thinking and never closed").strip() == "Sure.")
check("case insensitive",
      "<THINK>" not in strip_think("<THINK>x</THINK>ok"))
check("leaves normal text", strip_think("no tags here") == "no tags here")

print("\nstrip_unspeakable — emoji")
check("strips emoji", strip_unspeakable("Sure thing! 👍") == "Sure thing!")
check("strips multiple emoji",
      strip_unspeakable("Done ✅ 🎉 all set") == "Done all set")
check("strips flags", "🇮🇳" not in strip_unspeakable("India 🇮🇳 is large"))
check("keeps words around emoji",
      strip_unspeakable("🔥 hot today") == "hot today")

print("\nstrip_unspeakable — markdown")
check("strips bold", strip_unspeakable("This is **important** news")
      == "This is important news")
check("strips italics", strip_unspeakable("An _emphasis_ here")
      == "An emphasis here")
check("strips headings", strip_unspeakable("## Summary\nAll good")
      == "Summary\nAll good")
check("strips bullets",
      strip_unspeakable("- first\n- second") == "first\nsecond")
check("strips numbered lists",
      strip_unspeakable("1. first\n2. second") == "first\nsecond")
check("strips inline code", strip_unspeakable("Run `ls` now") == "Run ls now")
check("strips blockquote", strip_unspeakable("> quoted line") == "quoted line")
check("strips code fence",
      "print" not in strip_unspeakable("Here:\n```py\nprint(1)\n```\nDone"))
check("keeps code when asked",
      "print" in strip_unspeakable("```py\nprint(1)\n```", keep_code=True))

print("\nstrip_unspeakable — whitespace")
check("collapses runs of spaces",
      strip_unspeakable("too    many     spaces") == "too many spaces")
check("caps blank lines",
      strip_unspeakable("a\n\n\n\n\nb") == "a\n\nb")
check("trims ends", strip_unspeakable("   padded   ") == "padded")

print("\nstrip_unspeakable — leaves speech intact")
plain = "The meeting is at four in the afternoon. I will remind you at three."
check("plain sentences untouched", strip_unspeakable(plain) == plain)
check("keeps tamil", strip_unspeakable("வணக்கம் நண்பா") == "வணக்கம் நண்பா")
check("keeps hyphens and apostrophes",
      strip_unspeakable("It's a well-known fact") == "It's a well-known fact")
check("keeps sentence punctuation",
      strip_unspeakable("Really? Yes! Fine.") == "Really? Yes! Fine.")

print("\nstrip_unspeakable — edge cases")
check("empty string", strip_unspeakable("") == "")
check("none", strip_unspeakable(None) == "")
check("only emoji becomes empty", strip_unspeakable("🎉🎉") == "")
check("combined think and markdown",
      strip_unspeakable("<think>x</think>## Hi\n**bold** 🎉").strip() == "Hi\nbold")

print()
if failures:
    print(f"{len(failures)} FAILED: {failures}")
    raise SystemExit(1)
print("all speakable tests passed")
