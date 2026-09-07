"""
transcriber.py
==============
Local, zero-cost speech-to-text with **word-level timestamps** using
faster-whisper (CTranslate2 backend - 4x faster and lighter than openai-whisper).

Output shape (also what gets cached to cache/<id>.transcript.json):

{
  "language": "en",
  "duration": 1832.4,
  "segments": [{"start":.., "end":.., "text":..}],
  "words":    [{"start":.., "end":.., "word":"hello", "prob":0.98}]
}

The flat `words` list is the important one - it drives both the LLM's
boundary snapping and the karaoke caption renderer.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Any, Iterable

from config import settings
from utils import read_json, write_json

log = logging.getLogger("transcriber")

_MODEL_CACHE: dict[tuple, Any] = {}


# ---------------------------------------------------------------------------
# CUDA library discovery
# ---------------------------------------------------------------------------
# ctranslate2 dlopen()s libcublas / libcudnn lazily, on the FIRST encode - not
# at model construction. When those libs come from pip wheels
# (nvidia-cublas-cu12, nvidia-cudnn-cu12) they live inside site-packages, which
# the dynamic linker does not search, so you get:
#
#     RuntimeError: Library libcublas.so.12 is not found or cannot be loaded
#
# Setting LD_LIBRARY_PATH from inside Python does not help: the loader caches it
# at process start. Loading each .so explicitly with RTLD_GLOBAL does, because
# subsequent dlopen() calls resolve against already-loaded globals.
_CUDA_PRELOADED = False

# Order matters - dependents last.
_CUDA_LIB_GLOBS = (
    "nvidia/cuda_runtime/lib/libcudart.so*",
    "nvidia/cublas/lib/libcublasLt.so*",
    "nvidia/cublas/lib/libcublas.so*",
    "nvidia/cudnn/lib/libcudnn_graph.so*",
    "nvidia/cudnn/lib/libcudnn_engines_precompiled.so*",
    "nvidia/cudnn/lib/libcudnn_engines_runtime_compiled.so*",
    "nvidia/cudnn/lib/libcudnn_heuristic.so*",
    "nvidia/cudnn/lib/libcudnn_ops.so*",
    "nvidia/cudnn/lib/libcudnn_cnn.so*",
    "nvidia/cudnn/lib/libcudnn_adv.so*",
    "nvidia/cudnn/lib/libcudnn.so*",
)


def preload_cuda_libs() -> int:
    """Best-effort: make pip-installed CUDA libs visible to ctranslate2.
    Returns how many shared objects were loaded. Never raises."""
    global _CUDA_PRELOADED
    if _CUDA_PRELOADED or sys.platform == "win32":
        return 0

    import ctypes
    import site

    roots: list[Path] = []
    for getter in ("getsitepackages", "getusersitepackages"):
        try:
            got = getattr(site, getter)()
            roots.extend(Path(p) for p in ([got] if isinstance(got, str) else got))
        except Exception:  # noqa: BLE001
            continue
    roots.append(Path(sys.prefix) / "lib")

    loaded = 0
    for pattern in _CUDA_LIB_GLOBS:
        for root in roots:
            for so in sorted(root.glob(pattern), reverse=True):  # prefer versioned
                try:
                    ctypes.CDLL(str(so), mode=ctypes.RTLD_GLOBAL)
                    loaded += 1
                    break
                except OSError:
                    continue
            else:
                continue
            break

    _CUDA_PRELOADED = True
    if loaded:
        log.debug("Preloaded %d CUDA shared libraries from site-packages", loaded)
    return loaded


def _is_cuda_runtime_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return any(t in msg for t in (
        "libcublas", "libcudnn", "libcudart", "cuda", "cudnn", "cublas",
        "out of memory", "no kernel image", "gpu",
    ))


# ---------------------------------------------------------------------------
# Device / precision auto-detection
# ---------------------------------------------------------------------------
def _resolve_device(device: str, compute_type: str) -> tuple[str, str]:
    if device == "auto":
        try:
            import ctranslate2
            device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
        except Exception:  # noqa: BLE001 - any import/driver problem => CPU
            device = "cpu"

    if compute_type == "auto":
        compute_type = "float16" if device == "cuda" else "int8"

    return device, compute_type


def load_model(size: str | None = None, device: str | None = None,
               compute_type: str | None = None):
    """Load (and memoise) a WhisperModel."""
    from faster_whisper import WhisperModel

    cfg = settings.whisper
    size = size or cfg.model_size
    device, compute_type = _resolve_device(device or cfg.device,
                                           compute_type or cfg.compute_type)
    key = (size, device, compute_type)

    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    if device == "cuda":
        preload_cuda_libs()

    log.info("Loading Whisper '%s' on %s (%s) - first run downloads weights",
             size, device, compute_type)
    model = WhisperModel(size, device=device, compute_type=compute_type)
    _MODEL_CACHE[key] = model
    return model


# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------
def transcribe(
    audio_path: str | Path,
    *,
    cache_key: str | None = None,
    language: str | None = None,
    model_size: str | None = None,
    force: bool = False,
) -> dict:
    """Transcribe with word timestamps; results cached by `cache_key`."""
    audio_path = Path(audio_path)
    cache_key = cache_key or audio_path.stem
    cache_file = settings.paths.cache / f"{cache_key}.transcript.json"

    if cache_file.exists() and not force:
        cached = read_json(cache_file)
        if cached and cached.get("words"):
            log.info("Cache hit - %d words from %s", len(cached["words"]), cache_file.name)
            return cached

    cfg = settings.whisper
    device, compute_type = _resolve_device(cfg.device, cfg.compute_type)

    log.info("Transcribing %s ...", audio_path.name)
    t0 = time.time()

    try:
        segments, words, info = _run_transcription(
            audio_path, device, compute_type, language, model_size)
    except Exception as exc:  # noqa: BLE001
        # CUDA problems surface on the FIRST encode, not at construction time,
        # so this retry has to wrap the whole streamed iteration.
        if device != "cuda" or not _is_cuda_runtime_error(exc):
            raise
        log.warning("GPU transcription failed: %s", exc)
        log.warning("Falling back to CPU/int8. This is slower but works. To fix "
                    "the GPU path: pip install nvidia-cublas-cu12 nvidia-cudnn-cu12")
        _MODEL_CACHE.pop((model_size or cfg.model_size, device, compute_type), None)
        segments, words, info = _run_transcription(
            audio_path, "cpu", "int8", language, model_size)

    result = {
        "language": info.language,
        "language_probability": round(float(info.language_probability or 0), 3),
        "duration": round(float(info.duration or 0), 3),
        "model": model_size or cfg.model_size,
        "segments": segments,
        "words": words,
    }

    log.info("Done: %d segments / %d words in %.1fs (lang=%s)",
             len(segments), len(words), time.time() - t0, info.language)
    write_json(cache_file, result)
    return result


def _run_transcription(audio_path: Path, device: str, compute_type: str,
                       language: str | None, model_size: str | None):
    """Run one full transcription pass. Raises on any backend failure."""
    cfg = settings.whisper
    model = load_model(model_size, device=device, compute_type=compute_type)

    segments_iter, info = model.transcribe(
        str(audio_path),
        language=language or cfg.language,
        beam_size=cfg.beam_size,
        word_timestamps=True,
        vad_filter=cfg.vad_filter,
        vad_parameters={"min_silence_duration_ms": 500} if cfg.vad_filter else None,
        condition_on_previous_text=False,   # reduces runaway repetition loops
    )

    segments: list[dict] = []
    words: list[dict] = []

    # faster-whisper streams lazily; iterating is what actually does the work,
    # which is exactly why backend errors land here rather than above.
    for seg in segments_iter:
        text = (seg.text or "").strip()
        if not text:
            continue
        segments.append({"start": round(seg.start, 3), "end": round(seg.end, 3), "text": text})
        for w in (seg.words or []):
            token = (w.word or "").strip()
            if not token:
                continue
            words.append({
                "start": round(w.start, 3),
                "end": round(max(w.end, w.start + 0.04), 3),
                "word": token,
                "prob": round(float(getattr(w, "probability", 1.0) or 1.0), 3),
            })
        if len(segments) % 25 == 0:
            log.info("  ... %s transcribed", _mmss(seg.end))

    return segments, words, info


def _mmss(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


# ---------------------------------------------------------------------------
# Helpers used by the analyzer and the caption renderer
# ---------------------------------------------------------------------------
def words_in_range(words: Iterable[dict], start: float, end: float) -> list[dict]:
    """Words whose midpoint falls inside [start, end]."""
    out = []
    for w in words:
        mid = (w["start"] + w["end"]) / 2
        if start <= mid <= end:
            out.append(w)
    return out


def snap_to_words(words: list[dict], start: float, end: float,
                  lead_in: float = 0.25, lead_out: float = 0.45) -> tuple[float, float]:
    """
    Pull clip boundaries onto real word edges so clips never open or close
    mid-syllable. Falls back to the raw values if nothing is nearby.
    """
    if not words:
        return max(0.0, start - lead_in), end + lead_out

    starts = [w["start"] for w in words]
    ends = [w["end"] for w in words]

    # nearest word start at or before the requested start
    candidates = [s for s in starts if s <= start + 0.6]
    new_start = max(candidates) if candidates else starts[0]

    # nearest word end at or after the requested end
    candidates = [e for e in ends if e >= end - 0.6]
    new_end = min(candidates) if candidates else ends[-1]

    new_start = max(0.0, new_start - lead_in)
    new_end = new_end + lead_out
    if new_end <= new_start:
        new_end = new_start + 1.0
    return round(new_start, 3), round(new_end, 3)


def transcript_lines(segments: list[dict]) -> str:
    """Compact `[start-end] text` form fed to the LLM (token-cheap)."""
    return "\n".join(f"[{s['start']:.1f}-{s['end']:.1f}] {s['text']}" for s in segments)


def plain_text(words: list[dict], start: float, end: float) -> str:
    return " ".join(w["word"] for w in words_in_range(words, start, end)).strip()