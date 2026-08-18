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


@dataclass
class Sentence:
    """What the user said, per STT."""
    utt_id: str
    text: str
    t_captured: float
    t_stt_done: float = 0.0


@dataclass
class Reply:
    """What the assistant decided to say back."""
    utt_id: str
    text: str
    prompt: str            # kept for the latency log
    t_captured: float
    # A Piper voice is per-language, so TTS needs to know which language the
    # reply is in. The LLM stage already decides this to enforce the reply
    # language, so it is carried forward rather than re-detected downstream.
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
