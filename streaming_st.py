"""Live speech-to-text translation: streaming English ASR (faster-whisper + VAD)
piped into the wait-k simultaneous MT model (English -> Telugu).

Speak into the mic; English is transcribed incrementally and each new English
word drives the wait-k policy, so Telugu appears while you are still talking.

    python streaming_st.py --k 3 --adapter checkpoints/final
    python streaming_st.py --k 5 --whisper-model small.en --tgt-lang Telugu

Pipeline per utterance:
    mic (sounddevice, 16 kHz int16)
      -> webrtcvad frames  -> utterance segmentation (speech / trailing silence)
      -> faster-whisper     -> growing English transcript
      -> StreamingWaitKDecoder -> committed Telugu tokens (printed live)

Extra deps (on top of requirements.txt):
    pip install faster-whisper sounddevice webrtcvad numpy
"""
from __future__ import annotations

import argparse
import queue
import sys
import time
from collections import deque
from typing import List

import numpy as np
import torch

from src.load import load_model
from src.waitk import build_prompt, stop_token_ids

# ----------------------------------------------------------------------------
# Audio constants
# ----------------------------------------------------------------------------
SAMPLE_RATE = 16_000        # whisper + webrtcvad both expect 16 kHz mono
FRAME_MS = 30               # webrtcvad accepts 10 / 20 / 30 ms frames
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000          # 480 samples
FRAME_BYTES = FRAME_SAMPLES * 2                         # int16 -> 2 bytes/sample


# ----------------------------------------------------------------------------
# Streaming wait-k decoder
# ----------------------------------------------------------------------------
class StreamingWaitKDecoder:
    """Incremental wait-k decoding driven by a *growing* source word list.

    Mirrors src.waitk.wait_k_decode but never assumes the full source is known:
    target token ``t`` (1-indexed) is only committed once ``k + t - 1`` source
    words are available (or the utterance is final). Committed tokens are never
    revised, matching real simultaneous-translation commit-once semantics.
    """

    def __init__(self, model, tokenizer, k: int, target_language: str,
                 max_target_tokens: int = 256):
        self.model = model
        self.tokenizer = tokenizer
        self.k = k
        self.target_language = target_language
        self.max_target_tokens = max_target_tokens
        self.eos_ids = stop_token_ids(tokenizer)
        self.reset()

    def reset(self):
        self.committed: List[int] = []
        self.finished = False

    @property
    def text(self) -> str:
        return self.tokenizer.decode(self.committed, skip_special_tokens=True).strip()

    @torch.inference_mode()
    def feed(self, src_words: List[str], is_final: bool) -> str:
        """Commit as many target tokens as the wait-k policy allows given the
        currently available source words. Returns the newly committed text."""
        if self.finished:
            return ""
        S = len(src_words)
        before = len(self.committed)

        while len(self.committed) < self.max_target_tokens:
            t = len(self.committed) + 1
            needed = self.k + t - 1
            if not is_final and S < needed:
                break                       # wait for more source words (READ)
            num_src = min(needed, S)
            if num_src == 0:
                break

            prompt = build_prompt(self.tokenizer,
                                  " ".join(src_words[:num_src]),
                                  self.target_language)
            prompt_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.model.device)
            if self.committed:
                tail = torch.tensor([self.committed], device=self.model.device,
                                    dtype=prompt_ids.dtype)
                input_ids = torch.cat([prompt_ids, tail], dim=1)
            else:
                input_ids = prompt_ids

            logits = self.model(input_ids=input_ids).logits[0, -1]
            # Forbid EOS until the utterance is final AND all source is read.
            if not (is_final and num_src >= S):
                for eid in self.eos_ids:
                    logits[eid] = float("-inf")

            next_id = int(logits.argmax())
            if next_id in self.eos_ids:
                self.finished = True
                break
            self.committed.append(next_id)

        return self.tokenizer.decode(self.committed[before:],
                                     skip_special_tokens=True)


# ----------------------------------------------------------------------------
# Microphone capture + VAD utterance segmentation
# ----------------------------------------------------------------------------
def mic_frames(device=None):
    """Yield raw int16 PCM frames (FRAME_SAMPLES samples) from the default mic."""
    import sounddevice as sd

    q: "queue.Queue[bytes]" = queue.Queue()

    def callback(indata, frames, time_info, status):
        if status:
            print(status, file=sys.stderr)
        q.put(bytes(indata))

    with sd.RawInputStream(samplerate=SAMPLE_RATE, blocksize=FRAME_SAMPLES,
                           dtype="int16", channels=1, callback=callback,
                           device=device):
        while True:
            yield q.get()


def pcm16_to_float32(pcm: bytes) -> np.ndarray:
    return np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0


# ----------------------------------------------------------------------------
# Main loop
# ----------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    # MT model
    p.add_argument("--base", default="sarvamai/sarvam-translate")
    p.add_argument("--adapter", default=None, help="LoRA adapter dir (optional)")
    p.add_argument("--k", type=int, default=3, help="wait-k lag in source words")
    p.add_argument("--tgt-lang", default="Telugu")
    p.add_argument("--stability-trim", type=int, default=1,
                   help="while still speaking, drop the last N whisper words "
                        "(unstable, may be revised) before feeding wait-k")
    # ASR model
    p.add_argument("--whisper-model", default="base.en",
                   help="faster-whisper model: tiny.en/base.en/small.en/medium.en/large-v3")
    p.add_argument("--whisper-device", default=None, help="cuda / cpu (auto if unset)")
    p.add_argument("--whisper-compute", default=None,
                   help="float16 / int8_float16 / int8 (auto if unset)")
    # VAD / segmentation
    p.add_argument("--vad-aggressiveness", type=int, default=2, choices=[0, 1, 2, 3])
    p.add_argument("--silence-ms", type=int, default=700,
                   help="trailing silence that ends an utterance")
    p.add_argument("--retranscribe-ms", type=int, default=350,
                   help="re-run ASR every N ms of accumulated speech")
    p.add_argument("--mic-device", default=None, help="sounddevice input device id/name")
    return p.parse_args()


def make_whisper(args):
    from faster_whisper import WhisperModel

    device = args.whisper_device or ("cuda" if torch.cuda.is_available() else "cpu")
    compute = args.whisper_compute or ("float16" if device == "cuda" else "int8")
    print(f"[asr] faster-whisper {args.whisper_model} on {device} ({compute})", file=sys.stderr)
    return WhisperModel(args.whisper_model, device=device, compute_type=compute)


def transcribe(whisper, audio_f32: np.ndarray) -> str:
    segments, _ = whisper.transcribe(audio_f32, language="en", beam_size=1,
                                     condition_on_previous_text=False, vad_filter=False)
    return " ".join(s.text.strip() for s in segments).strip()


def render(english: str, telugu: str, final: bool):
    """Print the live EN/TE pair on a single, rewritten terminal line."""
    line = f"EN: {english}\n  -> TE: {telugu}"
    # \033[2K clears the line; \r returns to col 0. Two lines -> move up after.
    sys.stdout.write("\033[2K\r" + line.replace("\n", "\n\033[2K\r"))
    if final:
        sys.stdout.write("\n")
    else:
        # move cursor back up to overwrite next time (line has one '\n')
        sys.stdout.write("\033[1A")
    sys.stdout.flush()


def main():
    args = parse_args()

    print("[mt] loading wait-k MT model ...", file=sys.stderr)
    model, tokenizer = load_model(args.base, args.adapter)
    whisper = make_whisper(args)
    decoder = StreamingWaitKDecoder(model, tokenizer, k=args.k,
                                    target_language=args.tgt_lang)

    try:
        import webrtcvad
    except ImportError:
        sys.exit("webrtcvad not installed -> pip install webrtcvad")
    vad = webrtcvad.Vad(args.vad_aggressiveness)

    silence_frames = max(1, args.silence_ms // FRAME_MS)
    retr_frames = max(1, args.retranscribe_ms // FRAME_MS)

    print(f"\nWait-{args.k} live EN->{args.tgt_lang}. Speak now (Ctrl-C to quit).\n",
          file=sys.stderr)

    in_speech = False
    utt_pcm = bytearray()           # int16 PCM for the current utterance
    trailing_silence = 0
    frames_since_asr = 0
    last_english = ""

    def run_update(is_final: bool):
        nonlocal last_english
        audio = pcm16_to_float32(bytes(utt_pcm))
        if audio.size < FRAME_SAMPLES:
            return
        english = transcribe(whisper, audio)
        if not english:
            return
        last_english = english
        words = english.split()
        if not is_final and args.stability_trim > 0:
            words = words[:-args.stability_trim] or []   # drop unstable tail
        decoder.feed(words, is_final=is_final)
        render(english, decoder.text, final=is_final)

    try:
        for frame in mic_frames(args.mic_device):
            is_speech = vad.is_speech(frame, SAMPLE_RATE)

            if not in_speech:
                if is_speech:                       # utterance starts
                    in_speech = True
                    utt_pcm = bytearray(frame)
                    trailing_silence = 0
                    frames_since_asr = 0
                    decoder.reset()
                    last_english = ""
                continue

            # --- inside an utterance ---
            utt_pcm.extend(frame)
            frames_since_asr += 1
            trailing_silence = trailing_silence + 1 if not is_speech else 0

            if trailing_silence >= silence_frames:   # utterance ends
                run_update(is_final=True)
                in_speech = False
                utt_pcm = bytearray()
            elif frames_since_asr >= retr_frames:    # periodic partial update
                frames_since_asr = 0
                run_update(is_final=False)

    except KeyboardInterrupt:
        print("\n[done]", file=sys.stderr)


if __name__ == "__main__":
    main()
