"""
downloader.py
=============
Video ingestion layer.

  * `fetch(source)` accepts a YouTube/any-yt-dlp-supported URL **or** a local
    file path and always returns a `SourceVideo`.
  * `extract_audio()` produces the 16 kHz mono WAV that faster-whisper wants
    (Whisper resamples internally anyway - doing it up front is faster and
    avoids surprises with exotic audio codecs).

Downloads are cached: re-running with the same URL will not re-download.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from pathlib import Path

from config import settings
from utils import ffmpeg, slugify, video_info

log = logging.getLogger("downloader")


@dataclass
class SourceVideo:
    path: Path            # local mp4 on disk
    title: str
    video_id: str         # stable id used for cache + output folder names
    duration: float
    width: int
    height: int
    fps: float
    url: str | None = None
    uploader: str | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["path"] = str(self.path)
        return d


def is_url(source: str) -> bool:
    return source.lower().startswith(("http://", "https://", "www."))


# ---------------------------------------------------------------------------
# YouTube / remote
# ---------------------------------------------------------------------------
def download(url: str, max_height: int = 1080) -> SourceVideo:
    """Download best <=max_height MP4 (video+audio merged) via yt-dlp."""
    import yt_dlp  # imported lazily so the module loads without yt-dlp present

    outdir = settings.paths.downloads
    outdir.mkdir(parents=True, exist_ok=True)

    ydl_opts = {
        # Ordered preference. The key detail: never fall through to a
        # progressive `best` stream without trying every adaptive combination
        # first - YouTube's muxed streams top out at 360p, which is how you end
        # up silently upscaling 640x360 into a 1080x1920 short.
        "format": (
            f"bestvideo[height<={max_height}][vcodec^=avc1]+bestaudio[ext=m4a]/"
            f"bestvideo[height<={max_height}][ext=mp4]+bestaudio/"
            f"bestvideo[height<={max_height}]+bestaudio/"
            f"bestvideo+bestaudio/"
            f"best[height>=720]/best"
        ),
        "format_sort": ["res", "fps", "vcodec:h264", "acodec:aac"],
        "merge_output_format": "mp4",
        "outtmpl": str(outdir / "%(id)s.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "retries": 5,
        "fragment_retries": 5,
        "concurrent_fragment_downloads": 4,
        "postprocessors": [
            {"key": "FFmpegVideoRemuxer", "preferedformat": "mp4"},
        ],
        # NOTE: do not pin `player_client` here. Pinning to a client that
        # yt-dlp has since deprecated (android, in particular) makes the
        # extractor return only low-resolution muxed formats, and the format
        # selector then quietly "succeeds" at 360p. Let yt-dlp pick.
    }

    log.info("Resolving %s", url)
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        if info.get("_type") == "playlist":
            info = info["entries"][0]

        vid = info["id"]
        existing = _find_existing(outdir, vid)
        if existing:
            log.info("Cache hit - reusing %s", existing.name)
            path = existing
        else:
            log.info("Downloading '%s' (%.0fs)", info.get("title"), info.get("duration") or 0)
            ydl.download([url])
            path = _find_existing(outdir, vid)
            if not path:
                raise FileNotFoundError(f"yt-dlp finished but no file matched id {vid}")

    meta = video_info(path)
    warn_low_resolution(meta["width"], meta["height"])
    return SourceVideo(
        path=path,
        title=info.get("title") or vid,
        video_id=vid,
        duration=meta["duration"] or float(info.get("duration") or 0),
        width=meta["width"],
        height=meta["height"],
        fps=meta["fps"],
        url=info.get("webpage_url") or url,
        uploader=info.get("uploader"),
    )


def warn_low_resolution(width: int, height: int, floor: int = 720) -> None:
    """A 9:16 short is rendered at 1080x1920. Anything under ~720p source gets
    upscaled and looks soft - worth saying out loud rather than shipping mush."""
    if height and height < floor:
        log.warning("!" * 68)
        log.warning("Source is only %dx%d. Output is upscaled to 1080x1920 and", width, height)
        log.warning("will look soft. If this is YouTube, you probably hit a")
        log.warning("360p muxed fallback - try:  pip install -U yt-dlp")
        log.warning("Then delete downloads/ and re-run to force a fresh pull.")
        log.warning("!" * 68)


def _find_existing(outdir: Path, vid: str) -> Path | None:
    for ext in (".mp4", ".mkv", ".webm", ".mov"):
        p = outdir / f"{vid}{ext}"
        if p.exists() and p.stat().st_size > 0:
            return p
    return None


# ---------------------------------------------------------------------------
# Local file
# ---------------------------------------------------------------------------
def load_local(file_path: str | Path) -> SourceVideo:
    path = Path(file_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"No such file: {path}")
    meta = video_info(path)
    warn_low_resolution(meta["width"], meta["height"])
    return SourceVideo(
        path=path,
        title=path.stem,
        video_id=slugify(path.stem, 40) or "local",
        duration=meta["duration"],
        width=meta["width"],
        height=meta["height"],
        fps=meta["fps"],
    )


def fetch(source: str, max_height: int = 1080) -> SourceVideo:
    """URL or path -> SourceVideo."""
    return download(source, max_height) if is_url(source) else load_local(source)


# ---------------------------------------------------------------------------
# Audio for the transcriber
# ---------------------------------------------------------------------------
def extract_audio(video: SourceVideo, force: bool = False) -> Path:
    """16 kHz mono WAV in the cache dir."""
    wav = settings.paths.cache / f"{video.video_id}.wav"
    if wav.exists() and wav.stat().st_size > 1024 and not force:
        log.info("Cache hit - reusing audio %s", wav.name)
        return wav

    log.info("Extracting audio -> %s", wav.name)
    ffmpeg(
        ["-i", str(video.path), "-vn", "-ac", "1", "-ar", "16000",
         "-acodec", "pcm_s16le", str(wav)],
        desc="extract audio",
    )
    return wav
