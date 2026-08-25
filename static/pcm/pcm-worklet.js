// Captures microphone audio as int16 PCM and hands it to the main thread
// in fixed-size blocks.
//
// This replaces MediaRecorder. MediaRecorder emits a WebM/Opus container
// whose header exists only in the first chunk of a session, which forces
// the server to re-decode the entire call from byte 0 on every poll --
// O(T^2) per call, plus an ffmpeg process spawn every 500ms. Raw PCM has
// no container, so the server appends and slices instead of decoding.
//
// Runs on the audio render thread, so it must stay allocation-light:
// one reusable buffer, and a copy handed off only when a block is full.

const BLOCK_SAMPLES = 2048; // 128ms at 16kHz -- small enough for responsive
                            // turn detection, large enough that we are not
                            // sending a WebSocket frame every 8ms.

class PcmCapture extends AudioWorkletProcessor {
  constructor() {
    super();
    this._buf = new Int16Array(BLOCK_SAMPLES);
    this._n = 0;
  }

  process(inputs) {
    const input = inputs[0];
    // No connected input yet (or the graph is being torn down). Returning
    // true keeps the processor alive rather than ending the stream.
    if (!input || !input[0]) return true;

    const channel = input[0];
    for (let i = 0; i < channel.length; i++) {
      // Clamp before scaling: values outside [-1, 1] are legal in Web
      // Audio and would wrap around as int16 without this.
      const s = Math.max(-1, Math.min(1, channel[i]));
      this._buf[this._n++] = s < 0 ? s * 0x8000 : s * 0x7fff;

      if (this._n === BLOCK_SAMPLES) {
        const block = this._buf.slice();
        // Transfer the buffer rather than copying it across the thread
        // boundary; `block` is a fresh copy so the reusable one is safe.
        this.port.postMessage(block.buffer, [block.buffer]);
        this._n = 0;
      }
    }
    return true;
  }
}

registerProcessor('pcm-capture', PcmCapture);
