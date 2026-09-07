"""
utils.py
========
Small shared helpers: logging, subprocess wrappers around FFmpeg/ffprobe,
JSON caching, filename sanitising and time formatting.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

from config import settings

LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)-16s | %(message)s"


def setup_logging(level: str | None = None) -> None:
    logging.basicConfig(
        level=getattr(logging, (level or settings.log_level).upper(), logging.INFO),
        format=LOG_FORMAT,
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    # yt-dlp and httpx are extremely chatty at DEBUG
    for noisy in ("httpx", "httpcore", "urllib3", "google_genai", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


log = logging.getLogger("utils")


# ---------------------------------------------------------------------------
# Subprocess
# ---------------------------------------------------------------------------
class FFmpegError(RuntimeError):
    pass


def run(cmd: list[str], *, desc: str = "", check: bool = True) -> subprocess.CompletedProcess:
    """Run a command, capturing output. Raises FFmpegError with the tail of
    stderr on failure, which is what you actually need for debugging."""
    if desc:
        log.debug("%s", desc)
    log.debug("$ %s", " ".join(shlex.quote(c) for c in cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if check and proc.returncode != 0:
        tail = "\n".join((proc.stderr or "").strip().splitlines()[-25:])
        raise FFmpegError(f"Command failed ({proc.returncode}): {cmd[0]}\n{tail}")
    return proc


def ffmpeg(args: list[str], *, desc: str = "") -> None:
    base = [settings.ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y"]
    if settings.render.threads:
        base += ["-threads", str(settings.render.threads)]
    run(base + args, desc=desc)


def ffprobe_json(path: str | Path) -> dict[str, Any]:
    proc = run(
        [
            settings.ffprobe, "-v", "error",
            "-print_format", "json",
            "-show_format", "-show_streams",
            str(path),
        ],
        desc=f"probing {path}",
    )
    return json.loads(proc.stdout or "{}")


def video_info(path: str | Path) -> dict[str, Any]:
    """Return {width, height, duration, fps, has_audio} for a media file."""
    data = ffprobe_json(path)
    streams = data.get("streams", [])
    v = next((s for s in streams if s.get("codec_type") == "video"), None)
    a = next((s for s in streams if s.get("codec_type") == "audio"), None)
    if v is None:
        raise FFmpegError(f"No video stream found in {path}")

    fps = 30.0
    raw_fps = v.get("avg_frame_rate") or v.get("r_frame_rate") or "30/1"
    try:
        num, den = raw_fps.split("/")
        if float(den) != 0:
            fps = float(num) / float(den)
    except (ValueError, ZeroDivisionError):
        pass

    duration = 0.0
    for candidate in (v.get("duration"), data.get("format", {}).get("duration")):
        try:
            duration = float(candidate)
            break
        except (TypeError, ValueError):
            continue

    return {
        "width": int(v.get("width") or 0),
        "height": int(v.get("height") or 0),
        "duration": duration,
        "fps": round(fps, 3) or 30.0,
        "has_audio": a is not None,
    }


# ---------------------------------------------------------------------------
# JSON / cache
# ---------------------------------------------------------------------------
def read_json(path: str | Path) -> Any | None:
    p = Path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("Could not read JSON %s: %s", p, exc)
        return None


def write_json(path: str | Path, data: Any) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return p


def file_fingerprint(path: str | Path, extra: str = "") -> str:
    """Cheap, stable id for a media file: name + size + mtime (+ extra)."""
    p = Path(path)
    stat = p.stat()
    raw = f"{p.name}|{stat.st_size}|{int(stat.st_mtime)}|{extra}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Strings & time
# ---------------------------------------------------------------------------
_SLUG_RE = re.compile(r"[^\w\-]+", re.UNICODE)


def slugify(text: str, max_len: int = 60) -> str:
    text = (text or "clip").strip().lower().replace("&", " and ")
    text = _SLUG_RE.sub("-", text).strip("-")
    text = re.sub(r"-{2,}", "-", text)
    return (text[:max_len].rstrip("-")) or "clip"


def hhmmss(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def ass_timestamp(seconds: float) -> str:
    """ASS uses H:MM:SS.cc (centiseconds, single-digit hour)."""
    seconds = max(0.0, float(seconds))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    cs = int(round((seconds - int(seconds)) * 100))
    if cs == 100:  # rounding overflow
        cs, s = 0, s + 1
        if s == 60:
            s, m = 0, m + 1
            if m == 60:
                m, h = 0, h + 1
    return f"{h:d}:{m:02d}:{s:02d}.{cs:02d}"


def escape_filter_path(path: str | Path) -> str:
    """
    Escape a path so it survives FFmpeg's filtergraph parser.

    Windows is the painful case: `C:\\x\\y.ass` must become `C\\:/x/y.ass`
    before being wrapped in single quotes inside `subtitles='...'`.
    """
    p = str(Path(path).resolve()).replace("\\", "/")
    p = p.replace("'", r"\'")
    p = p.replace(":", r"\:")
    return p


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def even(n: float) -> int:
    """FFmpeg's H.264 encoder requires even dimensions."""
    i = int(round(n))
    return i - (i % 2)
