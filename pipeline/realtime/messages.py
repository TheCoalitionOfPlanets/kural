"""Queue message types.

Every item carries `utt_id` and `t_captured` so one utterance can be traced
end-to-end and its per-stage latency reported.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class Utterance:
    """Raw speech segment closed by the VAD endpointer."""
    utt_id: str
    pcm: np.ndarray        # float32 mono in [-1, 1]
    sample_rate: int
    duration_s: float
    t_captured: float      # perf_counter at VAD close
    # True when this utterance provisionally interrupted playback. The STT stage
    # owes a verdict on it: real speech confirms the interrupt, an echo reverses
    # it. See interrupt.py.
    barge_in: bool = False


@dataclass
class Sentence:
    """What the user said, per STT."""
    utt_id: str
    text: str
    t_captured: float
    # The language the STT stage identified, from the waveform rather than from
    # the words. Carried instead of re-detected because the text alone cannot
    # recover it: a Spanish transcript is Latin script with no markers, and
    # detect_language() would call it English. None when nothing identified it,
    # which puts the LLM worker back on transcript-based detection.
    lang: Optional[str] = None
    # "sravaani" or "elevenlabs" — for the console, not for any decision.
    backend: Optional[str] = None
    t_stt_done: float = 0.0
    barge_in: bool = False


@dataclass
class Reply:
    """What the assistant decided to say back."""
    utt_id: str
    text: str
    prompt: str            # kept for the latency log
    t_captured: float
    # TTS needs to know which language the reply is in — it is what chooses
    # between the local voice and ElevenLabs (languages.route_for). The LLM
    # stage already decides this to enforce the reply language, so it is
    # carried forward rather than re-detected downstream.
    lang: Optional[str] = None
    t_stt_done: float = 0.0
    t_llm_done: float = 0.0


@dataclass
class WavJob:
    """Synthesized audio ready for the speaker."""
    utt_id: str
    wav_path: Path
    sample_rate: int
    text: str
    t_captured: float
    t_stt_done: float = 0.0
    t_llm_done: float = 0.0
    t_tts_done: float = 0.0


# Pushed through each queue in order to unwind the pipeline on shutdown.
SENTINEL = object()
