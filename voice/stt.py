"""NVIDIA NeMo STT wrapper for Parakeet Realtime EOU 120M (cache-aware streaming
FastConformer-RNNT with inline <EOU>/<EOB> endpoint markers)."""

from __future__ import annotations

import asyncio
import os
import re
import tempfile
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional

import numpy as np
import soundfile as sf

from gateway.config import Settings
import time

from voice.audio import decode_audio_bytes, to_mono_f32


class STTEngine:
    """Lazy, serialized ASR engine.

    The model is loaded on first use so the API can come up while HF downloads
    or gated model setup is still being fixed.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.model_id = settings.stt_model_id
        self.model_name = settings.stt_model_name
        self.device = settings.stt_device
        self.sample_rate = settings.stt_sample_rate
        self._model: Any = None
        self._load_error: Optional[str] = None
        self._lock = asyncio.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1)

    @property
    def status(self) -> str:
        if self._model is not None:
            return "ok"
        if self._load_error:
            return f"error: {self._load_error}"
        return "not_loaded"

    async def load(self) -> None:
        if self._model is not None:
            return
        async with self._lock:
            if self._model is not None:
                return
            loop = asyncio.get_running_loop()
            try:
                self._model = await loop.run_in_executor(self._executor, self._load_sync)
                self._load_error = None
            except Exception as exc:
                self._load_error = str(exc)
                raise

    def _load_sync(self) -> Any:
        import torch
        import nemo.collections.asr as nemo_asr

        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("STT_DEVICE=cuda requested, but torch.cuda.is_available() is false")

        model = nemo_asr.models.ASRModel.from_pretrained(self.model_id)
        model.eval()
        if self.device:
            model = model.to(self.device)
        self._configure_streaming_decoding(model)
        return model

    def _configure_streaming_decoding(self, model: Any) -> None:
        """Make the RNNT greedy decoder safe for cache-aware streaming.

        Cache-aware streaming (`conformer_stream_step`) only works with the
        label-looping greedy decoder (`greedy.loop_labels=True`); the non-looping
        path raises NotImplementedError on `partial_hypotheses`. But this NeMo
        build's label-looping decoder, when it computes timestamps, sets
        `Hypothesis.timestamp` to a *dict*, and the streaming chunk-merge
        (`Hypothesis.merge_`) does `timestamp.extend(...)` → crash on a dict.

        So: keep loop_labels=True, turn OFF timestamp/alignment computation
        (unused here), and install a dict-tolerant `merge_` as a belt-and-braces
        guard. Done once at load; recorded in `decoding_config` for introspection.
        """
        self.decoding_config: dict[str, Any] = {}
        self._patch_hypothesis_merge()
        try:
            from omegaconf import open_dict

            dec = model.cfg.decoding
            with open_dict(dec):
                dec.compute_timestamps = False
                dec.preserve_alignments = False
                greedy = dec.get("greedy", None)
                if greedy is not None:
                    greedy.loop_labels = True  # required for streaming continuation
                    greedy.preserve_alignments = False
                    greedy.compute_timestamps = False
            model.change_decoding_strategy(dec)
            after = model.cfg.decoding
            self.decoding_config = {
                "strategy": str(after.get("strategy", "?")),
                "loop_labels": (after.get("greedy", {}) or {}).get("loop_labels", "?"),
                "compute_timestamps": after.get("compute_timestamps", "?"),
                "merge_patched": True,
            }
        except Exception as exc:  # noqa: BLE001
            self.decoding_config = {"error": f"{type(exc).__name__}: {exc}", "merge_patched": True}

    @staticmethod
    def _patch_hypothesis_merge() -> None:
        """Make `Hypothesis.merge_` tolerate dict/tensor timestamps (idempotent).

        Normalizes both hypotheses' `timestamp` to compatible list form *before*
        the original merge runs, so the internal `timestamp.extend(...)` can never
        hit a dict or a tensor/list mismatch. Timestamps are unused downstream of
        the streaming text, so flattening them is safe."""
        try:
            from nemo.collections.asr.parts.utils import rnnt_utils
        except Exception:  # noqa: BLE001
            return
        Hyp = rnnt_utils.Hypothesis
        if getattr(Hyp.merge_, "_eou_patched", False):
            return
        orig = Hyp.merge_

        def _to_list(ts: Any) -> Any:
            if isinstance(ts, dict):
                ts = ts.get("timestep", [])
            try:
                import torch

                if isinstance(ts, torch.Tensor):
                    ts = ts.tolist()
            except Exception:  # noqa: BLE001
                pass
            return ts

        def merge_(self, other):  # type: ignore[no-untyped-def]
            self.timestamp = _to_list(getattr(self, "timestamp", None))
            if other is not None:
                other.timestamp = _to_list(getattr(other, "timestamp", None))
            return orig(self, other)

        merge_._eou_patched = True  # type: ignore[attr-defined]
        Hyp.merge_ = merge_

    async def create_stream_session(self) -> "STTStreamSession":
        """Per-connection cache-aware streaming decoder. Falls back silently
        (session.ok == False) if the model lacks the streaming API; callers
        should then use batch transcribe_audio on the buffered utterance."""
        await self.load()
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, STTStreamSession, self)

    async def stream_probe(self, data: bytes, ws_chunk_ms: int = 100) -> dict[str, Any]:
        """Diagnostic: run the real streaming path over `data` and compare with
        batch. Used by the debug endpoint to validate the NeMo streaming API
        remotely (no SSH). Everything runs in the serialized GPU executor."""
        await self.load()
        audio, sr = decode_audio_bytes(data, target_sr=self.sample_rate)
        loop = asyncio.get_running_loop()
        async with self._lock:
            return await loop.run_in_executor(
                self._executor, self._stream_probe_sync, audio, sr, ws_chunk_ms
            )

    def _stream_probe_sync(self, audio: np.ndarray, sr: int, ws_chunk_ms: int) -> dict[str, Any]:
        import traceback

        out: dict[str, Any] = {"audio_seconds": round(len(audio) / float(sr), 3)}
        out["decoding_config"] = getattr(self, "decoding_config", {})
        session = STTStreamSession(self)
        out["introspection"] = session.introspection
        out["ok"] = session.ok
        if not session.ok:
            out["error"] = session.error
        else:
            ws = max(1, int(sr * ws_chunk_ms / 1000))
            steps: list[dict[str, Any]] = []
            finals: list[str] = []
            partials: list[str] = []
            t_stream = time.time()
            for i in range(0, len(audio), ws):
                t0 = time.time()
                res = session.feed_sync(audio[i : i + ws])
                if not session.ok:
                    out["error"] = session.error
                    out["traceback"] = session.introspection.get("step_traceback")
                    break
                finals.extend(res["finals"])
                if res["partial"]:
                    partials.append(res["partial"])
                if res["finals"] or res["eou"] or res["eob"]:
                    steps.append({"ms": round((time.time() - t0) * 1000, 1),
                                  "partial": res["partial"], "finals": res["finals"],
                                  "eou": res["eou"], "eob": res["eob"]})
            if session.ok:
                t0 = time.time()
                res = session.finalize_sync()
                finals.extend(res["finals"])
                out["final_ms"] = round((time.time() - t0) * 1000, 1)
                if not session.ok:
                    out["error"] = session.error
                    out["traceback"] = session.introspection.get("final_traceback")
            out["stream_ms"] = round((time.time() - t_stream) * 1000, 1)
            out["raw_text"] = session._text  # cumulative hypothesis incl. tokens
            out["final_segments"] = finals
            out["streamed_text"] = " ".join(finals).strip()
            out["n_partials"] = len(partials)
            out["last_partials"] = partials[-5:]
            out["marker_steps"] = steps

        t0 = time.time()
        try:
            out["batch_text"] = self._transcribe_audio_sync(audio, sr)
            out["batch_ms"] = round((time.time() - t0) * 1000, 1)
            out["match"] = (
                str(out.get("streamed_text", "")).strip().lower()
                == str(out["batch_text"]).strip().lower()
            )
        except Exception as exc:  # noqa: BLE001
            out["batch_error"] = f"{type(exc).__name__}: {exc}"
        return out

    async def transcribe_bytes(self, data: bytes) -> str:
        audio, sr = decode_audio_bytes(data, target_sr=self.sample_rate)
        return await self.transcribe_audio(audio, sr)

    async def transcribe_audio(self, audio: np.ndarray, sample_rate: int) -> str:
        await self.load()
        audio = to_mono_f32(audio)
        if audio.size == 0:
            return ""

        loop = asyncio.get_running_loop()
        async with self._lock:
            return await loop.run_in_executor(
                self._executor,
                self._transcribe_audio_sync,
                audio,
                sample_rate,
            )

    def _transcribe_audio_sync(self, audio: np.ndarray, sample_rate: int) -> str:
        import torch

        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            sf.write(path, audio, sample_rate, subtype="PCM_16")
            with torch.inference_mode():
                result = self._model.transcribe([path], batch_size=1)
            return self._extract_text(result)
        finally:
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass

    def _extract_text(self, result: Any) -> str:
        item = result
        if isinstance(item, (list, tuple)) and item:
            item = item[0]
        if isinstance(item, str):
            return item
        if hasattr(item, "text"):
            return str(item.text)
        if isinstance(item, dict):
            for key in ("text", "transcript", "prediction"):
                if key in item:
                    return str(item[key])
        return "" if item is None else str(item)


class STTStreamSession:
    """Cache-aware streaming decode state for one connection / one turn.

    Audio is fed incrementally as it arrives (concurrent with VAD + turn), each
    block decoded via `conformer_stream_step` while keeping the encoder cache,
    so the transcript is already done when the turn ends — `finalize()` only
    flushes the tail. GPU work is serialized through the engine's single-worker
    executor + lock to coexist with any batch transcribe.

    If the loaded model does not expose the streaming API, `ok` stays False and
    callers fall back to batch transcription.
    """

    def __init__(self, engine: "STTEngine") -> None:
        self._engine = engine
        self._model = engine._model
        self._device = engine.device
        self._sr = engine.sample_rate
        self.ok = False
        self.error: Optional[str] = None
        self.step_samples = 0
        self.introspection: dict[str, Any] = {}
        self._pending = np.zeros(0, dtype=np.float32)
        self._text = ""  # full raw running hypothesis (incl. <EOU>/<EOB> tokens)
        # Endpoint markers emitted inline by Parakeet-EOU. Read from settings so a
        # token-string correction (validated via /debug/stt-stream) is config-only.
        self._eou_tokens = tuple(getattr(engine.settings, "stt_eou_tokens", ("<EOU>", "</s>")))
        self._eob_token = str(getattr(engine.settings, "stt_eob_token", "<EOB>"))
        self._emitted_segments = 0  # how many closed segments already reported
        self._setup()

    def _setup(self) -> None:
        try:
            model = self._model
            enc = model.encoder
            self.introspection = {
                "has_setup_streaming_params": hasattr(enc, "setup_streaming_params"),
                "has_get_initial_cache_state": hasattr(enc, "get_initial_cache_state"),
                "has_conformer_stream_step": hasattr(model, "conformer_stream_step"),
                "has_streaming_cfg": hasattr(enc, "streaming_cfg"),
            }
            if not all(
                (
                    self.introspection["has_setup_streaming_params"],
                    self.introspection["has_get_initial_cache_state"],
                    self.introspection["has_conformer_stream_step"],
                )
            ):
                self.error = "model lacks cache-aware streaming API"
                return

            enc.setup_streaming_params()
            # deterministic features (no dithering / extra padding) so re-running
            # the preprocessor over the growing buffer is stable frame-for-frame.
            try:
                model.preprocessor.featurizer.dither = 0.0
                model.preprocessor.featurizer.pad_to = 0
            except Exception:  # noqa: BLE001
                pass

            # Cache-aware streaming chunking, all in INPUT feature frames:
            #   * chunk_size / shift_size may be [first, subsequent] lists
            #   * each step feeds `pre_encode_cache_size` history frames + chunk
            #   * drop_extra_pre_encoded is 0 on step 0, else the configured value
            cfg = getattr(enc, "streaming_cfg", None)
            self._chunk_size = getattr(cfg, "chunk_size", 16) if cfg is not None else 16
            self._shift_size = getattr(cfg, "shift_size", self._chunk_size) if cfg is not None else self._chunk_size
            self._pre_encode = self._as_scalar(getattr(cfg, "pre_encode_cache_size", 0))
            self._drop = self._as_scalar(getattr(cfg, "drop_extra_pre_encoded", 0))
            sub = int(getattr(enc, "subsampling_factor", 8) or 8)
            window_stride = float(getattr(model.cfg.preprocessor, "window_stride", 0.01))
            self._hop = max(1, int(self._sr * window_stride))
            # accumulate raw audio until at least one fresh chunk is decodable
            self._min_append = self._pick(self._chunk_size, False) * self._hop
            self.introspection.update(
                {"chunk_size": self._chunk_size, "shift_size": self._shift_size,
                 "pre_encode_cache_size": self._pre_encode, "drop_extra_pre_encoded": self._drop,
                 "subsampling": sub, "hop": self._hop,
                 "last_channel_cache_size": getattr(cfg, "last_channel_cache_size", None)}
            )

            self._cache_lc, self._cache_lt, self._cache_lc_len = enc.get_initial_cache_state(
                batch_size=1
            )
            self._prev_hyp = None
            self._pred_out = None
            self._raw = np.zeros(0, dtype=np.float32)  # full turn audio (seam-free re-mel)
            self._feat = None  # [1, feat_dim, T] features of self._raw
            self._buf_idx = 0  # feature frames already consumed by the encoder
            self._step = 0
            self.ok = True
        except Exception as exc:  # noqa: BLE001
            import traceback

            self.error = f"{type(exc).__name__}: {exc}"
            self.introspection["setup_traceback"] = traceback.format_exc()
            self.ok = False

    def _parse(self, raw: str) -> list[dict[str, Any]]:
        """Split the cumulative hypothesis into segments delimited by inline
        endpoint tokens. Returns ordered segments, each:
            {"text": <stripped>, "kind": "eou" | "eob" | "open"}
        The trailing "open" segment (if any) is the live partial. Empty closed
        segments are dropped."""
        markers = list(self._eou_tokens)
        if self._eob_token:
            markers.append(self._eob_token)
        if not markers:
            return [{"text": raw.strip(), "kind": "open"}]
        pattern = re.compile("|".join(re.escape(m) for m in markers))
        eob = self._eob_token
        segments: list[dict[str, Any]] = []
        pos = 0
        for m in pattern.finditer(raw):
            text = raw[pos : m.start()].strip()
            kind = "eob" if m.group(0) == eob else "eou"
            if text or kind == "eou":
                segments.append({"text": text, "kind": kind})
            pos = m.end()
        tail = raw[pos:].strip()
        segments.append({"text": tail, "kind": "open"})
        return segments

    @staticmethod
    def _as_scalar(v: Any) -> int:
        if isinstance(v, (list, tuple)):
            return int(v[-1]) if v else 0
        return int(v or 0)

    @staticmethod
    def _pick(v: Any, first: bool) -> int:
        """chunk_size/shift_size may be [first_chunk, subsequent_chunk]."""
        if isinstance(v, (list, tuple)):
            return int(v[0] if first else v[-1])
        return int(v)

    def _preprocess_all(self) -> None:
        """(Re)compute mel features over the whole turn buffer. Cheap, and
        seam-free (unlike per-chunk preprocessing), so the streamed transcript
        matches batch quality."""
        import torch

        sig = torch.tensor(self._raw, dtype=torch.float32, device=self._device).unsqueeze(0)
        sig_len = torch.tensor([self._raw.shape[0]], dtype=torch.int64, device=self._device)
        with torch.inference_mode():
            self._feat, _ = self._model.preprocessor(input_signal=sig, length=sig_len)

    def _make_chunk(self, start: int, end: int, first: bool):
        """Slice feature frames [start:end] and prepend pre_encode_cache history
        (zero-padded if not enough), so the pre-encode conv has left context."""
        import torch
        import torch.nn.functional as F

        pre = 0 if first else self._pre_encode
        cache_start = max(0, start - pre)
        cache = self._feat[:, :, cache_start:start]
        if pre and cache.size(-1) < pre:
            cache = F.pad(cache, (pre - cache.size(-1), 0))
        new = self._feat[:, :, start:end]
        return torch.cat([cache, new], dim=-1) if pre else new

    def _run_chunk(self, chunk, drop: int, is_last: bool) -> dict[str, Any]:
        """One cache-aware encoder+decoder step over a prepared feature chunk."""
        import torch

        length = torch.tensor([chunk.size(-1)], dtype=torch.int64, device=self._device)
        with torch.inference_mode():
            (
                self._pred_out,
                transcribed,
                self._cache_lc,
                self._cache_lt,
                self._cache_lc_len,
                self._prev_hyp,
            ) = self._model.conformer_stream_step(
                processed_signal=chunk,
                processed_signal_length=length,
                cache_last_channel=self._cache_lc,
                cache_last_time=self._cache_lt,
                cache_last_channel_len=self._cache_lc_len,
                keep_all_outputs=is_last,
                previous_hypotheses=self._prev_hyp,
                previous_pred_out=self._pred_out,
                drop_extra_pre_encoded=drop,
                return_transcription=True,
            )
        item = transcribed[0] if isinstance(transcribed, (list, tuple)) else transcribed
        self._text = str(getattr(item, "text", item))
        return self._collect(is_last)

    def _collect(self, is_last: bool) -> dict[str, Any]:
        """Diff the parsed segments against what we've already reported."""
        segments = self._parse(self._text)
        closed = segments[:-1] if segments and segments[-1]["kind"] == "open" else segments
        partial = segments[-1]["text"] if segments and segments[-1]["kind"] == "open" else ""
        if is_last and partial:
            # flush the tail as a final segment even without an explicit token
            closed = closed + [{"text": partial, "kind": "eou"}]
            partial = ""

        new = closed[self._emitted_segments :]
        self._emitted_segments = len(closed)
        finals = [s["text"] for s in new if s["kind"] == "eou" and s["text"]]
        return {
            "partial": partial,
            "finals": finals,
            "eou": any(s["kind"] == "eou" for s in new),
            "eob": any(s["kind"] == "eob" for s in new),
            "raw": self._text,
        }

    @staticmethod
    def _empty_result() -> dict[str, Any]:
        return {"partial": "", "finals": [], "eou": False, "eob": False, "raw": ""}

    @staticmethod
    def _merge(acc: dict[str, Any], step: dict[str, Any]) -> dict[str, Any]:
        acc["partial"] = step["partial"]
        acc["finals"].extend(step["finals"])
        acc["eou"] = acc["eou"] or step["eou"]
        acc["eob"] = acc["eob"] or step["eob"]
        acc["raw"] = step["raw"]
        return acc

    def feed_sync(self, audio: np.ndarray) -> dict[str, Any]:
        """Accumulate audio, re-mel the turn, and decode every full chunk that
        is now available (cache-aware). Runs inside the engine executor+lock."""
        acc = self._empty_result()
        if not self.ok:
            return acc
        self._raw = np.concatenate([self._raw, to_mono_f32(audio)])
        # need at least one fresh chunk's worth of new audio before re-mel
        avail = self._raw.shape[0] - self._buf_idx * self._hop
        if avail < self._min_append:
            acc["partial"] = self._parse(self._text)[-1]["text"]
            return acc
        try:
            self._preprocess_all()
            total = self._feat.size(-1)
            # Each non-last chunk drops its trailing (lookahead) outputs, which
            # the *next* chunk re-decodes. Keep one chunk in reserve so the most
            # recent frames are never the "last" non-last chunk — finalize runs
            # them with keep_all_outputs=True, so a short final word isn't lost.
            reserve = self._pick(self._chunk_size, False)
            while True:
                first = self._step == 0
                cs = self._pick(self._chunk_size, first)
                if self._buf_idx + cs + reserve > total:
                    break
                chunk = self._make_chunk(self._buf_idx, self._buf_idx + cs, first)
                step = self._run_chunk(chunk, 0 if first else self._drop, is_last=False)
                self._merge(acc, step)
                self._buf_idx += cs
                self._step += 1
        except Exception as exc:  # noqa: BLE001
            import traceback
            self.error = f"step: {type(exc).__name__}: {exc}"
            self.introspection["step_traceback"] = traceback.format_exc()
            self.ok = False
        if not acc["raw"]:
            acc["partial"] = self._parse(self._text)[-1]["text"]
        return acc

    def finalize_sync(self) -> dict[str, Any]:
        """Decode any remaining tail frames as the final (keep_all_outputs) step.

        Pads with trailing silence first: this both lets the model emit a closing
        <EOU> and gives the RNNT decoder fresh frames to flush its last token even
        when `feed` already consumed every real frame as full (non-last) chunks —
        otherwise a short, fully-consumed utterance (one-word reply) decodes empty."""
        acc = self._empty_result()
        if not self.ok:
            acc["partial"] = self._parse(self._text)[-1]["text"]
            return acc
        try:
            if self._raw.shape[0] == 0:
                self._merge(acc, self._collect(is_last=True))
                return acc
            pad = np.zeros(int(0.2 * self._sr), dtype=np.float32)
            self._raw = np.concatenate([self._raw, pad])
            self._preprocess_all()
            total = self._feat.size(-1)
            while self._buf_idx < total:
                first = self._step == 0
                cs = self._pick(self._chunk_size, first)
                end = min(self._buf_idx + cs, total)
                chunk = self._make_chunk(self._buf_idx, end, first)
                step = self._run_chunk(chunk, 0 if first else self._drop, is_last=(end >= total))
                self._merge(acc, step)
                self._buf_idx = end
                self._step += 1
        except Exception as exc:  # noqa: BLE001
            import traceback
            self.error = f"final: {type(exc).__name__}: {exc}"
            self.introspection["final_traceback"] = traceback.format_exc()
            self.ok = False
        return acc

    async def feed(self, audio: np.ndarray) -> dict[str, Any]:
        if not self.ok:
            r = self._empty_result()
            r["partial"] = self._parse(self._text)[-1]["text"]
            return r
        loop = asyncio.get_running_loop()
        async with self._engine._lock:
            return await loop.run_in_executor(self._engine._executor, self.feed_sync, audio)

    async def finalize(self) -> dict[str, Any]:
        loop = asyncio.get_running_loop()
        async with self._engine._lock:
            return await loop.run_in_executor(self._engine._executor, self.finalize_sync)

    @property
    def text(self) -> str:
        """Full transcript with endpoint tokens stripped (all segments joined)."""
        segs = [s["text"] for s in self._parse(self._text) if s["kind"] != "eob" and s["text"]]
        return " ".join(segs).strip()

    def tail_audio(self, seconds: float) -> np.ndarray:
        """Last `seconds` of the turn audio — for an on-demand Smart Turn check."""
        n = int(seconds * self._sr)
        return self._raw[-n:] if n and self._raw.shape[0] > n else self._raw
