// AudioWorklet: capture mic audio on the audio thread, batch it into ~100 ms
// frames, convert Float32 → 16-bit PCM, and post each frame to the main thread
// (which forwards it over the WebSocket). It also posts a cheap per-frame RMS
// level used for the 4-second silence auto-stop and the live waveform — both
// computed here so the UI thread stays smooth ("blazing fast") while speaking.
class PCMProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
        this._buf = [];
        this._frame = 1600;   // 100 ms @ 16 kHz
    }

    process(inputs) {
        const input = inputs[0];
        if (!input || !input[0]) return true;   // keep the processor alive
        const ch = input[0];
        for (let i = 0; i < ch.length; i++) this._buf.push(ch[i]);

        while (this._buf.length >= this._frame) {
            const slice = this._buf.splice(0, this._frame);
            const pcm = new Int16Array(slice.length);
            let sumSq = 0;
            for (let i = 0; i < slice.length; i++) {
                const s = Math.max(-1, Math.min(1, slice[i]));
                sumSq += s * s;
                pcm[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
            }
            const rms = Math.sqrt(sumSq / slice.length);
            // PCM bytes go out as a transferable; level rides alongside (tiny).
            this.port.postMessage({ pcm: pcm.buffer, level: rms }, [pcm.buffer]);
        }
        return true;
    }
}

registerProcessor('pcm-processor', PCMProcessor);
