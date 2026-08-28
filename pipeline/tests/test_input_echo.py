r"""Unit tests for the input-side echo guard.

    venv\Scripts\python.exe pipeline	ests	est_input_echo.py
"""
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pipeline.realtime.echo_guard import (  # noqa: E402
    INPUT_ECHO_THRESHOLD,
    RecentInput,
    _similarity,
    is_input_echo_of,
)

failures = []


def check(name, cond):
    if cond:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}")
        failures.append(name)


print("_similarity")
check("identical is 1.0", _similarity(["a", "b", "c"], ["a", "b", "c"]) == 1.0)
check("disjoint is 0.0", _similarity(["a", "b"], ["c", "d"]) == 0.0)
check("empty is 0.0", _similarity([], ["a"]) == 0.0)
check("symmetric",
      _similarity(["a", "b", "c"], ["a", "b"]) ==
      _similarity(["a", "b"], ["a", "b", "c"]))
# The property that separates this from containment: a short text fully inside
# a long one must NOT score as a match, or every reply fragment is an "input".
check("subset of a longer text scores below threshold",
      _similarity(["a", "b"], ["a", "b", "c", "d", "e", "f"])
      < INPUT_ECHO_THRESHOLD)
check("duplicated word counts only as often as both sides have it",
      _similarity(["a", "a", "a"], ["a"]) == 0.5)

print("\nis_input_echo_of — real re-feeds")
turn = "book me a flight to tokyo tomorrow morning"
check("verbatim re-feed", is_input_echo_of(turn, turn))
check("different case and punctuation",
      is_input_echo_of("Book me a flight to Tokyo tomorrow morning!", turn))
check("one word of stt jitter",
      is_input_echo_of("book me a flight to tokyo tomorrow mornings", turn))
check("dropped filler word",
      is_input_echo_of("book me flight to tokyo tomorrow morning", turn))
check("matches across a list of earlier turns",
      is_input_echo_of(turn, ["what time is it", "unrelated thing", turn]))

print("\nis_input_echo_of — real speech must pass through")
check("unrelated turn", not is_input_echo_of("what time is the meeting", turn))
check("same topic, different request",
      not is_input_echo_of("cancel my flight to tokyo", turn))
check("empty previous", not is_input_echo_of(turn, ""))
check("empty transcript", not is_input_echo_of("", turn))
check("no previous turns at all", not is_input_echo_of(turn, []))
# The asymmetry check that matters: a fragment of an earlier turn is a user
# being cut off, not a re-feed of the whole turn.
check("fragment of an earlier turn is not a re-feed",
      not is_input_echo_of("book me a flight", turn))

print("\nis_input_echo_of — short-transcript guard")
check("single word never flagged", not is_input_echo_of("yes", "yes"))
check("two words can flag", is_input_echo_of("yes please", "yes please"))

print("\nis_input_echo_of — threshold")
near = "book me a flight to tokyo tomorrow evening"
check("0.85 accepts one-word jitter", is_input_echo_of(near, turn, 0.85))
check("1.0 demands an exact match",
      not is_input_echo_of(near, turn, threshold=1.0))
check("1.0 still accepts a verbatim repeat",
      is_input_echo_of(turn, turn, threshold=1.0))

print("\nis_input_echo_of — non-latin script")
tamil = "எனக்கு நாளை டோக்கியோ விமானம் வேண்டும்"
check("verbatim tamil re-feed", is_input_echo_of(tamil, tamil))
check("unrelated tamil passes",
      not is_input_echo_of("இப்போது மணி என்ன", tamil))

print("\nRecentInput")
ri = RecentInput(maxlen=3)
check("starts empty", len(ri) == 0)
check("empty window flags nothing", not ri.is_echo("book me a flight to tokyo"))
ri.add(turn)
check("records a turn", len(ri) == 1)
check("flags the re-feed", ri.is_echo(turn))
check("passes an unrelated turn", not ri.is_echo("what time is the meeting"))
ri.add("")
ri.add(None)
check("ignores empty and none", len(ri) == 1)
ri.add(turn)
check("re-adding the same text does not duplicate", len(ri) == 1)
for i in range(5):
    ri.add(f"turn number {i} with distinct words")
check("bounded to maxlen", len(ri) == 3)
check("oldest evicted", not ri.is_echo(turn))
ri.clear()
check("clear empties", len(ri) == 0)

print("\nRecentInput — expiry")
expiring = RecentInput(maxlen=3, ttl_s=0.05)
expiring.add(turn)
check("fresh entry flags", expiring.is_echo(turn))
time.sleep(0.1)
check("expired entry no longer flags", not expiring.is_echo(turn))
check("expired entry pruned", len(expiring) == 0)

forever = RecentInput(maxlen=3, ttl_s=0)
forever.add(turn)
time.sleep(0.05)
check("ttl_s=0 disables expiry", forever.is_echo(turn))

print("\nRecentInput — per-instance threshold")
strict = RecentInput(maxlen=3, threshold=1.0)
strict.add(turn)
check("strict instance rejects jitter", not strict.is_echo(near))
check("strict instance accepts verbatim", strict.is_echo(turn))
check("call-site threshold overrides the instance", strict.is_echo(near, 0.85))

print("\nRecentInput — thread safety")
ri2 = RecentInput(maxlen=50, ttl_s=0)
errors = []


def hammer(n):
    try:
        for i in range(200):
            ri2.add(f"worker {n} line {i}")
            ri2.snapshot()
            ri2.is_echo("worker line something else entirely")
    except Exception as exc:  # a race would surface as a list mutation error
        errors.append(exc)


threads = [threading.Thread(target=hammer, args=(i,)) for i in range(8)]
for t in threads:
    t.start()
for t in threads:
    t.join()
check("concurrent access raises nothing", not errors)
check("bounded under concurrency", len(ri2) == 50)

print()
if failures:
    print(f"{len(failures)} FAILED: {failures}")
    raise SystemExit(1)
print("all input echo tests passed")
