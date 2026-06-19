#!/usr/bin/env python3
"""Probe NeMo cache-aware streaming for the Nemotron streaming STT model.

Goal: validate the exact streaming API + chunk sizing on the live GPU box
BEFORE folding it into voice/stt.py, so we don't iterate through Docker
rebuilds. Run this inside the container (it reuses the already-cached model):

    python /opt/app/scripts/stt_stream_probe.py /tmp/tts_out.wav

If you have no wav handy, synth one first via the TTS endpoint, or pass any
mono speech wav. The script:
  1. loads the model and dumps its streaming config (chunk sizes, methods),
  2. runs a true incremental streaming decode feeding small chunks (simulating
     the websocket), printing the running transcript per step,
  3. runs a one-shot batch transcribe for comparison,
  4. prints timing so we can confirm the per-chunk step is cheap.

Report the full output back and I'll wire the proven path into stt.py.
"""

from __future__ import annotations

import sys
import time

import numpy as np
import soundfile as sf
import torch


MODEL_ID = "nvidia/nemotron-speech-streaming-en-0.6b"
TARGET_SR = 16000
# Simulated websocket chunk: 16 kHz * 0.1 s = 1600 samples. The session will
# accumulate these and flush whenever it has enough for one encoder step.
WS_CHUNK_SAMPLES = 1600


def load_audio(path: str) -> np.ndarray:
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if sr != TARGET_SR:
        try:
            import soxr

            audio = soxr.resample(audio, sr, TARGET_SR, quality="HQ")
        except Exception:
            dur = len(audio) / float(sr)
            n = int(round(dur * TARGET_SR))
            audio = np.interp(
                np.linspace(0, dur, n, endpoint=False),
                np.linspace(0, dur, len(audio), endpoint=False),
                audio,
            ).astype(np.float32)
    return np.ascontiguousarray(audio, dtype=np.float32)


def dump_streaming_cfg(model) -> None:
    print("\n=== encoder streaming introspection ===")
    enc = model.encoder
    for name in ("setup_streaming_params", "get_initial_cache_state", "streaming_cfg"):
        print(f"  encoder.{name}: {'present' if hasattr(enc, name) else 'MISSING'}")
    print(
        "  model.conformer_stream_step:",
        "present" if hasattr(model, "conformer_stream_step") else "MISSING",
    )
    if hasattr(enc, "setup_streaming_params"):
        try:
            enc.setup_streaming_params()
        except Exception as exc:  # noqa: BLE001
            print("  setup_streaming_params() raised:", exc)
    cfg = getattr(enc, "streaming_cfg", None)
    if cfg is not None:
        for attr in (
            "chunk_size",
            "shift_size",
            "pre_encode_cache_size",
            "drop_extra_pre_encoded",
            "last_channel_cache_size",
        ):
            print(f"  streaming_cfg.{attr} = {getattr(cfg, attr, '<none>')}")


def stream_decode(model, audio: np.ndarray, device: str) -> str:
    """Incremental cache-aware decode, feeding fixed WS-sized chunks."""
    enc = model.encoder
    enc.setup_streaming_params()
    cfg = enc.streaming_cfg

    # samples per streaming step: chunk_size is in encoder frames; convert via
    # the model's subsampling + the preprocessor hop. NeMo exposes this via the
    # preprocessor window stride; chunk_size may be a list [non_causal, causal].
    chunk_frames = cfg.chunk_size
    if isinstance(chunk_frames, (list, tuple)):
        chunk_frames = chunk_frames[-1]
    sub = getattr(model.encoder, "subsampling_factor", 8)
    hop = int(TARGET_SR * model.cfg.preprocessor.window_stride)  # samples per frame
    step_samples = int(chunk_frames) * int(sub) * hop
    print(f"\n=== streaming decode (step ≈ {step_samples} samples / "
          f"{step_samples / TARGET_SR:.3f}s) ===")

    cache = model.encoder.get_initial_cache_state(batch_size=1)
    cache_last_channel, cache_last_time, cache_last_channel_len = cache
    previous_hypotheses = None
    pred_out_stream = None
    final_text = ""

    pending = np.zeros(0, dtype=np.float32)
    # split audio into WS-sized chunks, accumulate to step_samples, then step
    ws_chunks = [audio[i : i + WS_CHUNK_SAMPLES] for i in range(0, len(audio), WS_CHUNK_SAMPLES)]

    def run_step(samples: np.ndarray, is_last: bool):
        nonlocal cache_last_channel, cache_last_time, cache_last_channel_len
        nonlocal previous_hypotheses, pred_out_stream, final_text
        sig = torch.tensor(samples, dtype=torch.float32, device=device).unsqueeze(0)
        sig_len = torch.tensor([samples.shape[0]], dtype=torch.int64, device=device)
        processed, processed_len = model.preprocessor(input_signal=sig, length=sig_len)
        t0 = time.time()
        with torch.inference_mode():
            (
                pred_out_stream,
                transcribed,
                cache_last_channel,
                cache_last_time,
                cache_last_channel_len,
                previous_hypotheses,
            ) = model.conformer_stream_step(
                processed_signal=processed,
                processed_signal_length=processed_len,
                cache_last_channel=cache_last_channel,
                cache_last_time=cache_last_time,
                cache_last_channel_len=cache_last_channel_len,
                keep_all_outputs=is_last,
                previous_hypotheses=previous_hypotheses,
                previous_pred_out=pred_out_stream,
                drop_extra_pre_encoded=None,
                return_transcription=True,
            )
        dt = (time.time() - t0) * 1000
        txt = transcribed[0]
        txt = getattr(txt, "text", txt)
        final_text = txt
        print(f"  step ({samples.shape[0]:6d} smp, last={is_last}) "
              f"{dt:6.1f}ms -> {txt!r}")

    for ws in ws_chunks:
        pending = np.concatenate([pending, ws])
        while pending.shape[0] >= step_samples:
            run_step(pending[:step_samples], is_last=False)
            pending = pending[step_samples:]
    # flush remainder as the final step
    run_step(pending if pending.shape[0] else np.zeros(1, dtype=np.float32), is_last=True)
    return final_text


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: stt_stream_probe.py <speech.wav>")
        sys.exit(2)
    path = sys.argv[1]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}  loading {MODEL_ID} ...")

    import nemo.collections.asr as nemo_asr

    model = nemo_asr.models.ASRModel.from_pretrained(MODEL_ID)
    model.eval()
    model = model.to(device)

    dump_streaming_cfg(model)

    audio = load_audio(path)
    print(f"\naudio: {len(audio)} samples / {len(audio)/TARGET_SR:.2f}s")

    try:
        streamed = stream_decode(model, audio, device)
    except Exception as exc:  # noqa: BLE001
        import traceback

        print("\n!!! streaming decode FAILED:")
        traceback.print_exc()
        streamed = f"<failed: {exc}>"

    t0 = time.time()
    with torch.inference_mode():
        batch = model.transcribe([path], batch_size=1)
    item = batch[0] if isinstance(batch, (list, tuple)) else batch
    batch_text = getattr(item, "text", item)
    print(f"\n=== batch transcribe ({(time.time()-t0)*1000:.0f}ms) ===\n  {batch_text!r}")
    print(f"\n=== streamed final ===\n  {streamed!r}")
    print("\nMatch:", str(streamed).strip().lower() == str(batch_text).strip().lower())


if __name__ == "__main__":
    main()
