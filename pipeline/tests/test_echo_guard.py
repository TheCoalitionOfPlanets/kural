"""Unit tests for the echo guard.

    venv\\Scripts\\python.exe pipeline\\tests\\test_echo_guard.py
"""
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pipeline.realtime.echo_guard import (  # noqa: E402
    MIN_ECHO_WORDS,
    RecentSpeech,
    _norm,
    is_echo_of,
)

failures = []


def check(name, cond):
    if cond:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}")
        failures.append(name)


print("_norm")
check("lowercases", _norm("Hello World") == ["hello", "world"])
check("strips punctuation", _norm("I can help, sure!") == ["i", "can", "help", "sure"])
check("collapses whitespace", _norm("a   b\n\tc") == ["a", "b", "c"])
check("empty input", _norm("") == [])
check("none input", _norm(None) == [])
check("keeps non-latin script", _norm("வணக்கம் நண்பா") == ["வணக்கம்", "நண்பா"])

print("\nis_echo_of — real echoes")
reply = "Paris is the capital of France. It is a beautiful and historic city."
check("verbatim fragment", is_echo_of("paris is the capital of france", reply))
check("different punctuation/case",
      is_echo_of("Paris is the capital of France!", reply))
check("mid-reply fragment", is_echo_of("a beautiful and historic city", reply))
check("fragment of long reply scores high",
      is_echo_of("it is a beautiful city", reply))

print("\nis_echo_of — real speech must pass through")
check("unrelated question", not is_echo_of("what time is the meeting", reply))
check("new topic sharing a word",
      not is_echo_of("book me a flight to tokyo tomorrow", reply))
check("empty spoken window", not is_echo_of("paris is the capital", ""))
check("empty transcript", not is_echo_of("", reply))

print("\nis_echo_of — short-transcript guard")
check("two words never flagged", not is_echo_of("the capital", reply))
check("single word never flagged", not is_echo_of("paris", reply))
check("min words is 3", MIN_ECHO_WORDS == 3)
check("exactly 3 words can flag", is_echo_of("is the capital", reply))

print("\nis_echo_of — threshold")
half = "paris is the zebra umbrella"  # 3/5 words overlap
check("0.6 threshold accepts 60% overlap", is_echo_of(half, reply, threshold=0.6))
check("0.75 threshold rejects 60% overlap",
      not is_echo_of(half, reply, threshold=0.75))
check("0.5 threshold is more aggressive", is_echo_of(half, reply, threshold=0.5))

print("\nis_echo_of — accepts a list of chunks")
chunks = ["Paris is the capital.", "It is beautiful and historic."]
check("matches across chunk list", is_echo_of("it is beautiful and historic", chunks))
check("matches first chunk", is_echo_of("paris is the capital", chunks))

print("\nRecentSpeech")
rs = RecentSpeech(maxlen=3)
check("starts empty", len(rs) == 0)
rs.add("one two three")
check("add grows", len(rs) == 1)
rs.add("")
rs.add("   ")
check("ignores blank", len(rs) == 1)
rs.add(None)
check("ignores none", len(rs) == 1)
for t in ["four five six", "seven eight nine", "ten eleven twelve"]:
    rs.add(t)
check("respects maxlen", len(rs) == 3)
check("evicts oldest", "one two three" not in rs.snapshot())
check("keeps newest", "ten eleven twelve" in rs.snapshot())

check("is_echo matches window", rs.is_echo("seven eight nine"))
check("is_echo rejects unrelated", not rs.is_echo("completely different words here"))

snap = rs.snapshot()
snap.append("mutation")
check("snapshot is a copy", len(rs) == 3)

rs.clear()
check("clear empties", len(rs) == 0)
check("empty window flags nothing", not rs.is_echo("seven eight nine"))

print("\nRecentSpeech — thread safety")
rs2 = RecentSpeech(maxlen=50)
errors = []


def hammer(n):
    try:
        for i in range(200):
            rs2.add(f"worker {n} line {i}")
            rs2.snapshot()
            rs2.is_echo("worker line something else entirely")
    except Exception as exc:  # a race would surface as a list mutation error
        errors.append(exc)


threads = [threading.Thread(target=hammer, args=(i,)) for i in range(8)]
for t in threads:
    t.start()
for t in threads:
    t.join()
check("concurrent access raises nothing", not errors)
check("bounded under concurrency", len(rs2) == 50)

print()
if failures:
    print(f"{len(failures)} FAILED: {failures}")
    raise SystemExit(1)
print("all echo guard tests passed")
