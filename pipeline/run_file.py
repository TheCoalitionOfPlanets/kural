"""Run the full pipeline over a WAV file instead of the microphone.

Segments the file with the project's own VAD, so utterances are endpointed the
same way live capture would do it, then runs each one through STT -> LLM -> TTS
and plays the reply on the speakers. Reports per-stage latency.

    venv/Scripts/python.exe pipeline/run_file.py audio.wav
    venv/Scripts/python.exe pipeline/run_file.py audio.wav --no-play
    venv/Scripts/python.exe pipeline/run_file.py audio.wav --whole-file
    venv/Scripts/python.exe pipeline/run_file.py audio.wav --max-new-tokens 48
"""
import argparse
import pathlib
import sys
import tempfile
import time

import numpy as np
import soundfile as sf
import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.realtime.audio_out import Player          # noqa: E402
from pipeline.realtime.capture import make_vad          # noqa: E402
from pipeline.realtime.proc import WorkerProcess        # noqa: E402

DEFAULT_CONFIG = ROOT / "pipeline/config/realtime.yaml"


def load_mono(path, target_sr):
    """File -> mono float32 at the capture rate."""
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        n = int(round(len(audio) * target_sr / sr))
        audio = np.interp(
            np.linspace(0, len(audio) - 1, n), np.arange(len(audio)), audio
        ).astype(np.float32)
    return audio, sr


def segment(audio, cap):
    """Split into utterances with the pipeline's VAD and endpointing rules."""
    sr = int(cap["sample_rate"])
    frame_ms = int(cap["frame_ms"])
    frame = sr * frame_ms // 1000
    v = cap["vad"]

    vad = make_vad(v, sr, frame_ms)
    frames = [audio[i:i + frame] for i in range(0, len(audio) - frame + 1, frame)]
    if not frames:
        return []

    # Live capture calibrates on the first second, which is room tone because
    # the user has not started talking yet. A file has no such runway — it
    # often opens mid-word, and calibrating on that sets the noise floor from
    # speech, pushing the threshold above every frame in the file. Calibrate
    # on the quietest frames instead: wherever the pauses actually are.
    levels = np.array([float(np.sqrt(np.mean(np.square(f))) + 1e-12) for f in frames])
    cal_n = max(10, min(len(frames),
                        int(float(v.get("calibration_s", 1.0)) * 1000) // frame_ms))
    quietest = np.argsort(levels)[:cal_n]
    vad.calibrate([frames[i] for i in quietest])

    sil_n = int(v["silence_ms"]) // frame_ms
    min_n = int(v["min_utterance_ms"]) // frame_ms
    max_n = int(v["max_utterance_ms"]) // frame_ms
    pre_n = int(v["pre_roll_ms"]) // frame_ms

    utts, buf, pre, in_speech, sil = [], [], [], False, 0
    for f in frames:
        speech = vad.is_speech(f)
        if not in_speech:
            pre.append(f)
            if pre_n:
                pre[:] = pre[-pre_n:]
            if speech:
                in_speech, buf, sil = True, list(pre), 0
                buf.append(f)
        else:
            buf.append(f)
            sil = 0 if speech else sil + 1
            if sil >= sil_n or len(buf) >= max_n:
                if len(buf) >= min_n:
                    utts.append(np.concatenate(buf))
                in_speech, buf, pre = False, [], []
    if in_speech and len(buf) >= min_n:
        utts.append(np.concatenate(buf))
    return utts


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("wav", help="input audio file (any rate/channels)")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--no-play", action="store_true",
                    help="synthesize but do not use the speakers")
    ap.add_argument("--whole-file", action="store_true",
                    help="treat the file as one utterance, skipping the VAD")
    ap.add_argument("--max-new-tokens", type=int,
                    help="override llm.max_new_tokens (shorter = lower latency)")
    ap.add_argument("--keep-wavs", action="store_true",
                    help="keep the synthesized replies and print their paths")
    args = ap.parse_args()

    cfg = yaml.safe_load(pathlib.Path(args.config).read_text(encoding="utf-8"))
    if args.max_new_tokens:
        cfg["llm"]["max_new_tokens"] = args.max_new_tokens

    sr = int(cfg["capture"]["sample_rate"])
    audio, orig_sr = load_mono(args.wav, sr)
    print(f"{args.wav} -> {len(audio)/sr:.2f}s mono @ {sr}Hz (from {orig_sr}Hz)")

    utts = [audio] if args.whole_file else segment(audio, cfg["capture"])
    if not utts:
        sys.exit("VAD found no speech. Try --whole-file, or lower "
                 "capture.vad.min_threshold in the config.")
    print(f"{len(utts)} utterance(s): " + ", ".join(f"{len(u)/sr:.1f}s" for u in utts))

    outdir = pathlib.Path(tempfile.mkdtemp(prefix="kural_file_"))
    workers, rows = {}, []

    def spawn(key):
        section = dict(cfg[key])
        for pk in ("prompt_file", "voices_dir"):
            if section.get(pk):
                section[pk] = str(ROOT / section[pk])
        wp = WorkerProcess(name=key, python=ROOT / section["python"],
                           script=ROOT / section["worker"], config=section,
                           cwd=ROOT, on_log=lambda n, m: None)
        info = wp.start(timeout_s=int(cfg["runtime"].get("startup_timeout_s", 300)))
        workers[key] = wp
        print(f"  {key:4} ready in {info.get('load_s')}s ({info.get('vram_gb')} GB VRAM)")

    print("\nStarting workers...")
    try:
        for key in ("stt", "llm", "tts"):
            spawn(key)
    except Exception as exc:
        for w in workers.values():
            w.stop()
        sys.exit(f"Startup failed: {exc}")

    player = None if args.no_play else Player(cfg.get("playback", {}).get("device"))
    print()

    try:
        for i, pcm in enumerate(utts, 1):
            dur = len(pcm) / sr
            print("=" * 68)
            print(f"UTTERANCE {i}  ({dur:.2f}s of speech)")
            npy = outdir / f"u{i}.npy"
            np.save(npy, pcm)

            t0 = time.time()
            r = workers["stt"].run({"utt_id": f"u{i}", "pcm_path": str(npy)})
            t_stt = time.time() - t0
            if not r.get("ok"):
                print(f"  STT failed: {r}")
                continue
            text = r["text"]
            print(f"  [STT {t_stt*1000:7.0f} ms]  heard : {text}")
            if not text.strip():
                print("  (empty transcript, skipping)")
                continue

            t0 = time.time()
            r = workers["llm"].run({"utt_id": f"u{i}", "text": text})
            t_llm = time.time() - t0
            if not r.get("ok"):
                print(f"  LLM failed: {r}")
                continue
            reply, lang = r["text"], r.get("lang", "english")
            print(f"  [LLM {t_llm*1000:7.0f} ms]  reply : {reply}")

            wav_out = outdir / f"u{i}_reply.wav"
            t0 = time.time()
            r = workers["tts"].run({"utt_id": f"u{i}", "text": reply,
                                    "lang": lang, "wav_path": str(wav_out)})
            t_tts = time.time() - t0
            if not r.get("ok"):
                print(f"  TTS failed: {r}")
                continue
            audio_s = r["audio_s"]
            print(f"  [TTS {t_tts*1000:7.0f} ms]  {audio_s:.2f}s of audio @ {r['sample_rate']}Hz")

            total = t_stt + t_llm + t_tts
            print(f"  --> time to first sound: {total*1000:.0f} ms   (RTF {total/dur:.2f})")
            if args.keep_wavs:
                print(f"      {wav_out}")
            if player is not None:
                played, done = player.play(wav_out)
                print(f"  played {played:.2f}s ({'complete' if done else 'cut short'})")
            print()
            rows.append((i, dur, t_stt, t_llm, t_tts, total, audio_s))
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        for w in workers.values():
            w.stop()

    if not rows:
        return
    print("=" * 68)
    print("LATENCY SUMMARY  (ms)")
    print(f"{'utt':>4} {'speech_s':>9} {'STT':>8} {'LLM':>8} {'TTS':>8} "
          f"{'TOTAL':>9} {'reply_s':>8} {'RTF':>6}")
    for i, d, s, l, t, tot, out in rows:
        print(f"{i:>4} {d:>9.2f} {s*1000:>8.0f} {l*1000:>8.0f} {t*1000:>8.0f} "
              f"{tot*1000:>9.0f} {out:>8.2f} {tot/d:>6.2f}")
    n = len(rows)
    print(f"{'mean':>4} {sum(r[1] for r in rows)/n:>9.2f} "
          f"{sum(r[2] for r in rows)/n*1000:>8.0f} {sum(r[3] for r in rows)/n*1000:>8.0f} "
          f"{sum(r[4] for r in rows)/n*1000:>8.0f} {sum(r[5] for r in rows)/n*1000:>9.0f} "
          f"{sum(r[6] for r in rows)/n:>8.2f} {sum(r[5]/r[1] for r in rows)/n:>6.2f}")
    if args.keep_wavs:
        print(f"\nreplies in {outdir}")


if __name__ == "__main__":
    main()
