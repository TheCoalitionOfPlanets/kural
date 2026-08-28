/* Microphone -> 16 kHz int16 frames, on the audio thread.
 *
 * The server's VAD is frame-size sensitive: webrtcvad accepts only exact
 * 10/20/30 ms frames, and the energy gate's calibrated threshold assumes a
 * fixed frame length. A worklet is handed 128 samples at a time at whatever
 * rate the AudioContext is running, which is neither 16 kHz nor a divisor of a
 * 20 ms frame. So both conversions happen here, before anything crosses to the
 * main thread:
 *
 *     128 @ ctx rate  ->  linear resample to 16 kHz  ->  320-sample frames
 *
 * Doing it on the audio thread also means the main thread never touches raw
 * audio — it only forwards finished frames to the socket. Layout jank cannot
 * drop microphone data.
 */

const TARGET_RATE = 16000;

class CaptureProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const opts = options.processorOptions || {};
    this.targetRate = opts.targetRate || TARGET_RATE;
    this.frameSamples = opts.frameSamples || Math.round(this.targetRate * 0.02);

    // `sampleRate` is a global in worklet scope: the context's real rate,
    // which is not always the one that was asked for.
    this.ratio = sampleRate / this.targetRate;

    this.frame = new Int16Array(this.frameSamples);
    this.filled = 0;

    // Resampling has to be continuous across process() calls, so the read
    // position and the last sample of the previous block both carry over.
    // Restarting the phase every block would put a click at every 128 samples.
    this.pos = 0;
    this.carry = new Float32Array(0);

    this.muted = false;
    this.port.onmessage = (event) => {
      const data = event.data;
      if (data && data.type === "mute") this.muted = !!data.value;
    };
  }

  process(inputs) {
    const channel = inputs[0] && inputs[0][0];
    if (!channel || channel.length === 0) return true;

    // Muting sends silence rather than sending nothing. The server's capture
    // thread is driven by frames arriving: starve it and the noise floor
    // calibration, the level meter and the whole endpointing state machine
    // simply stop, which looks like a hung app rather than a muted mic.
    const block = this.muted ? new Float32Array(channel.length) : channel;

    let buf;
    if (this.carry.length) {
      buf = new Float32Array(this.carry.length + block.length);
      buf.set(this.carry, 0);
      buf.set(block, this.carry.length);
    } else {
      buf = block;
    }

    // Linear interpolation. At 48k -> 16k this is a plain 3:1 decimation with
    // no aliasing worth worrying about for speech, and it costs a fraction of
    // what a windowed-sinc would on the audio thread.
    let pos = this.pos;
    while (pos + 1 < buf.length) {
      const i = pos | 0;
      const frac = pos - i;
      const sample = buf[i] + (buf[i + 1] - buf[i]) * frac;

      // int16 is what the server reads off the wire; clamping here means a
      // loud speaker cannot wrap around into a burst of noise.
      const clamped = sample > 1 ? 1 : sample < -1 ? -1 : sample;
      this.frame[this.filled++] = (clamped * 32767) | 0;

      if (this.filled === this.frameSamples) {
        // Transferred, not copied: the buffer is handed over and a fresh one
        // allocated, so no audio-thread memory is shared with the main thread.
        const out = this.frame;
        this.frame = new Int16Array(this.frameSamples);
        this.filled = 0;
        this.port.postMessage(out.buffer, [out.buffer]);
      }
      pos += this.ratio;
    }

    // Keep the tail the next block needs to interpolate against.
    const consumed = pos | 0;
    this.carry = buf.slice(consumed);
    this.pos = pos - consumed;
    return true;
  }
}

registerProcessor("capture-processor", CaptureProcessor);
