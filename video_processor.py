"""
video_processor.py
==================
Everything between "here is a clip range" and "here is a publish-ready MP4".

Pipeline per clip (two encodes, deliberately - it is far easier to debug and
the intermediate is disposable):

  STAGE 1  cut          source.mp4  --[-ss/-to, re-encode]-->  work/cut.mp4
  STAGE 2  ass          words       --[karaoke builder]------>  work/clip.ass
  STAGE 3  reframe+burn cut.mp4     --[filter_complex]------->  output/clip.mp4

Reframe modes
-------------
  blur  Full-frame blurred, zoom-filled background with the original 16:9
        video centred on top. Nothing is ever cropped away. Safest default.
  crop  Hard centre crop to 9:16. Maximum screen real-estate, loses the sides.
  face  Same as crop, but the crop window is centred on where faces actually
        are (median of OpenCV Haar detections sampled across the clip),
        clamped to the frame. Good for interviews and talking heads.

Captions
--------
Word-by-word karaoke: for each caption card we emit one ASS Dialogue event per
word, where the currently-spoken word is recoloured and scaled up with a short
\\t() transform. That is more events than \\k karaoke tags, but it renders
identically on every libass build and gives full control over the "pop".
"""

from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass
from pathlib import Path

from config import settings
from utils import (
    FFmpegError, ass_timestamp, clamp, escape_filter_path,
    even, ffmpeg, video_info,
)

log = logging.getLogger("video_processor")


@dataclass
class RenderResult:
    video_path: Path
    thumbnail_path: Path | None
    ass_path: Path | None
    width: int
    height: int
    duration: float
    tightening: dict | None = None
    style_key: str | None = None


# ===========================================================================
# STAGE 1 - cut
# ===========================================================================
def cut_clip(source: Path, start: float, end: float, dest: Path) -> Path:
    """
    Accurate cut with re-encode. `-ss` before `-i` is fast *and* frame-accurate
    in modern FFmpeg because it decodes from the preceding keyframe.
    Re-encoding (rather than `-c copy`) guarantees the clip starts at PTS 0,
    which keeps subtitle timing honest in stage 3.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    r = settings.render
    duration = max(0.2, end - start)

    args = [
        "-ss", f"{start:.3f}", "-i", str(source), "-t", f"{duration:.3f}",
        "-map", "0:v:0", "-map", "0:a:0?",
        "-c:v", "libx264", "-preset", r.preset, "-crf", "18",
        "-pix_fmt", "yuv420p", "-r", str(r.fps),
        "-c:a", "aac", "-b:a", r.audio_bitrate, "-ar", "48000", "-ac", "2",
        "-avoid_negative_ts", "make_zero", "-fflags", "+genpts",
        str(dest),
    ]
    ffmpeg(args, desc=f"cut {start:.1f}-{end:.1f}")
    return dest


# ===========================================================================
# STAGE 2 - ASS karaoke captions
# ===========================================================================
ASS_HEADER = """[Script Info]
ScriptType: v4.00+
WrapStyle: 2
ScaledBorderAndShadow: yes
YCbCr Matrix: TV.709
PlayResX: {play_x}
PlayResY: {play_y}

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Caption,{font},{size},{c_idle},{c_active},{c_outline},{c_shadow},-1,0,0,0,100,100,0,0,1,{outline},{shadow},2,{ml},{mr},{mv},1

[Events]
Format: Layer, Start, End, Style, MarginL, MarginR, MarginV, Effect, Text
"""


def _escape_ass_text(text: str) -> str:
    return (text.replace("\\", "\\\\")
                .replace("{", "\\{")
                .replace("}", "\\}")
                .replace("\n", " ")
                .strip())


def group_words_into_cards(words: list[dict], max_words: int, max_seconds: float) -> list[list[dict]]:
    """
    Chunk words into caption cards. Breaks on: word count, elapsed time,
    sentence-ending punctuation, and any pause longer than 0.6 s.
    """
    cards: list[list[dict]] = []
    current: list[dict] = []

    for w in words:
        if current:
            gap = w["start"] - current[-1]["end"]
            span = w["end"] - current[0]["start"]
            if (len(current) >= max_words or span >= max_seconds or gap > 0.6):
                cards.append(current)
                current = []
        current.append(w)
        if current[-1]["word"].rstrip().endswith((".", "!", "?")) and len(current) >= 2:
            cards.append(current)
            current = []

    if current:
        cards.append(current)
    return cards


def build_ass(
    words: list[dict],
    clip_start: float,
    clip_duration: float,
    dest: Path,
    *,
    play_x: int | None = None,
    play_y: int | None = None,
) -> Path:
    """
    Write an .ass file whose timings are RELATIVE to the cut clip
    (i.e. clip_start has already been subtracted).
    """
    cap = settings.captions
    r = settings.render
    play_x = play_x or r.width
    play_y = play_y or r.height

    header = ASS_HEADER.format(
        play_x=play_x, play_y=play_y,
        font=cap.font_name, size=cap.font_size,
        c_idle=cap.color_idle, c_active=cap.color_active,
        c_outline=cap.outline_color, c_shadow=cap.shadow_color,
        outline=cap.outline, shadow=cap.shadow,
        ml=cap.margin_h, mr=cap.margin_h, mv=cap.margin_v,
    )

    # Rebase and clamp to the clip window.
    local: list[dict] = []
    for w in words:
        s = w["start"] - clip_start
        e = w["end"] - clip_start
        if e <= 0 or s >= clip_duration:
            continue
        local.append({
            "start": clamp(s, 0.0, clip_duration),
            "end": clamp(max(e, s + 0.06), 0.0, clip_duration),
            "word": w["word"].strip(),
        })

    lines: list[str] = []
    cards = group_words_into_cards(local, cap.max_words_per_card, cap.max_card_seconds)

    for card in cards:
        if not card:
            continue
        card_start = card[0]["start"]
        card_end = card[-1]["end"]

        for i, word in enumerate(card):
            ev_start = word["start"] if i > 0 else card_start
            # Hold the highlight until the next word actually begins, so the
            # card never blinks out during a natural micro-pause.
            ev_end = card[i + 1]["start"] if i + 1 < len(card) else card_end
            ev_end = max(ev_end, ev_start + 0.08)

            pieces = []
            for j, other in enumerate(card):
                token = _escape_ass_text(other["word"])
                if cap.uppercase:
                    token = token.upper()
                if j == i:
                    pieces.append(
                        f"{{\\c{cap.color_active}\\t(0,{cap.pop_ms},"
                        f"\\fscx{cap.pop_scale}\\fscy{cap.pop_scale})}}{token}"
                        f"{{\\r\\c{cap.color_idle}}}"
                    )
                else:
                    pieces.append(token)
            text = " ".join(pieces)

            fade = "{\\fad(60,0)}" if i == 0 else ""
            lines.append(
                f"Dialogue: 0,{ass_timestamp(ev_start)},{ass_timestamp(ev_end)},"
                f"Caption,0,0,0,,{fade}{text}"
            )

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(header + "\n".join(lines) + "\n", encoding="utf-8")
    log.debug("ASS written: %s (%d cards, %d events)", dest.name, len(cards), len(lines))
    return dest


# ===========================================================================
# STAGE 3 - reframe + burn
# ===========================================================================
def detect_face_center(clip_path: Path, samples: int = 40) -> float:
    """
    Return the horizontal centre of interest as a 0..1 fraction of frame width.
    Median of Haar-cascade face detections; 0.5 when nothing is found.
    """
    try:
        import cv2
    except ImportError:
        log.warning("opencv not installed - 'face' mode falls back to centre crop.")
        return 0.5

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(cascade_path)
    if cascade.empty():
        log.warning("Haar cascade unavailable - using centre crop.")
        return 0.5

    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        return 0.5

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = float(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0) or 1.0
    if total <= 0:
        cap.release()
        return 0.5

    step = max(1, total // samples)
    centers: list[float] = []
    idx = 0
    while idx < total:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            break
        small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=5,
                                         minSize=(40, 40))
        if len(faces):
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])   # biggest face
            centers.append(((x + w / 2) * 2) / width)
        idx += step
    cap.release()

    if not centers:
        log.info("No faces detected - centring crop.")
        return 0.5
    center = float(statistics.median(centers))
    log.info("Face-aware crop centre: %.2f (from %d detections)", center, len(centers))
    return clamp(center, 0.0, 1.0)


def _build_filtergraph(src_w: int, src_h: int, mode: str, ass_file: Path | None,
                       face_center: float = 0.5, trim_video: str = "",
                       grade: str = "") -> str:
    r = settings.render
    W, H = r.width, r.height
    subs = ""
    if ass_file is not None:
        fonts = ""
        if settings.captions.fonts_dir:
            fonts = f":fontsdir='{escape_filter_path(settings.captions.fonts_dir)}'"
        subs = f",subtitles='{escape_filter_path(ass_file)}'{fonts}"

    # Silence removal has to happen FIRST, before anything timestamp-sensitive,
    # and the caption timings must already have been remapped to match.
    pre = f"{trim_video}," if trim_video else ""
    # Grade goes AFTER scaling/overlay but BEFORE subtitles, so caption colours
    # stay exactly as the style specified them.
    post = f",{grade}" if grade else ""
    src_is_vertical = src_h >= src_w

    if mode in ("crop", "face") and not src_is_vertical:
        crop_w = even(min(src_w, src_h * W / H))
        crop_h = even(min(src_h, crop_w * H / W))
        max_x = max(0, src_w - crop_w)
        x = int(clamp(face_center * src_w - crop_w / 2, 0, max_x))
        return (
            f"[0:v]{pre}crop={crop_w}:{crop_h}:{x}:{(src_h - crop_h) // 2},"
            f"scale={W}:{H}:flags=lanczos,setsar=1,fps={r.fps}{post}{subs}[v]"
        )

    # --- blur mode (and the fallback for already-vertical sources) ----------
    fg_h = even(H * r.fg_height_ratio)
    fg_w = even(min(W, src_w * fg_h / max(1, src_h)))
    if fg_w >= W:                       # standard 16:9 -> full-width strip
        fg_w = W
        fg_h = even(src_h * W / max(1, src_w))

    return (
        f"[0:v]{pre}split=2[bg][fg];"
        f"[bg]scale={W}:{H}:force_original_aspect_ratio=increase,"
        f"crop={W}:{H},gblur=sigma={r.blur_sigma},"
        f"eq=brightness={r.blur_darken}:saturation=1.15[bgb];"
        f"[fg]scale={fg_w}:{fg_h}:flags=lanczos,setsar=1[fgs];"
        f"[bgb][fgs]overlay=(W-w)/2:(H-h)/2:format=auto,"
        f"fps={r.fps},format=yuv420p{post}{subs}[v]"
    )


def render_vertical(
    cut_path: Path,
    dest: Path,
    ass_file: Path | None = None,
    mode: str | None = None,
    trim_filters: tuple[str, str] | None = None,
    grade: str = "",
    loudness_target: float | None = None,
) -> Path:
    """Reframe to 9:16 and burn in the captions in a single encode.
    `trim_filters` is an optional (video_select, audio_select) pair from
    optimizer.keeps_to_filters() for silence removal."""
    r = settings.render
    mode = (mode or r.reframe_mode).lower()
    info = video_info(cut_path)

    face_center = 0.5
    if mode == "face":
        face_center = detect_face_center(cut_path)

    trim_v, trim_a = trim_filters or ("", "")
    graph = _build_filtergraph(info["width"], info["height"], mode, ass_file,
                               face_center, trim_video=trim_v, grade=grade)

    args = ["-i", str(cut_path), "-filter_complex", graph, "-map", "[v]"]
    if info["has_audio"]:
        args += ["-map", "0:a:0"]
        afilters = [f for f in (trim_a,) if f]
        if r.loudnorm:
            target = loudness_target if loudness_target is not None else -14.0
            afilters.append(f"loudnorm=I={target:.1f}:TP=-1.5:LRA=11")
        if afilters:
            args += ["-af", ",".join(afilters)]
        args += ["-c:a", "aac", "-b:a", r.audio_bitrate, "-ar", "48000", "-ac", "2"]
    else:
        args += ["-an"]

    args += [
        "-c:v", "libx264", "-preset", r.preset, "-crf", str(r.crf),
        "-profile:v", "high", "-level", "4.1", "-pix_fmt", "yuv420p",
        "-g", str(r.fps * 2), "-movflags", "+faststart",
        "-metadata", "comment=Generated by AI Video Clipper",
        str(dest),
    ]

    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        ffmpeg(args, desc=f"render {dest.name} [{mode}]")
    except FFmpegError as exc:
        # Most common cause: a font that libass cannot resolve. Retry bare so
        # the user still gets a usable clip instead of a hard failure.
        if ass_file is not None:
            log.warning("Render with captions failed (%s). Retrying without captions.", exc)
            return render_vertical(cut_path, dest, ass_file=None, mode=mode,
                                   trim_filters=trim_filters, grade=grade,
                                   loudness_target=loudness_target)
        raise
    return dest


def make_thumbnail(video_path: Path, dest: Path, at_seconds: float = 1.0) -> Path | None:
    try:
        ffmpeg(
            ["-ss", f"{at_seconds:.2f}", "-i", str(video_path),
             "-frames:v", "1", "-q:v", "3", str(dest)],
            desc="thumbnail",
        )
        return dest
    except FFmpegError as exc:
        log.warning("Thumbnail failed: %s", exc)
        return None


# ===========================================================================
# Orchestration for a single clip
# ===========================================================================
def process_clip(
    source: Path,
    clip,                       # ai_analyzer.Clip
    words: list[dict],
    out_video: Path,
    work_dir: Path,
    *,
    mode: str | None = None,
    burn_captions: bool = True,
    thumbnail: bool = True,
    tighten: bool = False,
    style=None,
) -> RenderResult:
    work_dir.mkdir(parents=True, exist_ok=True)
    stem = out_video.stem

    cut_path = work_dir / f"{stem}.cut.mp4"
    cut_clip(source, clip.start_time, clip.end_time, cut_path)

    # --- optional silence removal --------------------------------------
    # Order is critical: plan the cuts, remap the word timings onto the new
    # timeline, THEN build the .ass. Building captions first would desync
    # every word the moment a gap is removed.
    trim_filters = None
    tighten_info = None
    local_words = words
    clip_start_for_ass = clip.start_time

    if tighten or (style is not None and style.tighten):
        import optimizer
        cut_info = video_info(cut_path)
        noise = style.silence_noise_db if style else -32
        min_sil = style.silence_min if style else 0.35
        pad = style.keep_pad if style else 0.12
        silences = optimizer.detect_silences(cut_path, noise_db=noise,
                                             min_silence=min_sil)
        keeps = optimizer.plan_tightening(cut_info["duration"], silences,
                                          keep_pad=pad)
        tighten_info = optimizer.tightening_summary(keeps, cut_info["duration"])
        if tighten_info["removed_seconds"] >= 0.4:
            log.info("  tightened: -%.1fs (%.0f%%) across %d cut(s)",
                     tighten_info["removed_seconds"], tighten_info["removed_pct"],
                     tighten_info["cut_count"])
            trim_filters = optimizer.keeps_to_filters(keeps)
            local_words = optimizer.remap_words(words, keeps, clip.start_time)
            clip_start_for_ass = 0.0     # remapped words are already clip-local
        else:
            log.info("  tightened: nothing worth removing")
            tighten_info = None

    ass_path = None
    if burn_captions:
        duration = (tighten_info["tightened_duration"] if tighten_info
                    else clip.duration)
        ass_path = build_ass(
            words=local_words,
            clip_start=clip_start_for_ass,
            clip_duration=duration,
            dest=work_dir / f"{stem}.ass",
        )

    import styles as _styles
    render_vertical(cut_path, out_video, ass_file=ass_path, mode=mode,
                    trim_filters=trim_filters,
                    grade=_styles.grade_filter(style) if style else "",
                    loudness_target=style.loudness_target if style else None)

    thumb = None
    if thumbnail:
        thumb = make_thumbnail(out_video, out_video.with_suffix(".jpg"),
                               at_seconds=min(1.5, clip.duration / 3))

    final = video_info(out_video)
    try:
        cut_path.unlink(missing_ok=True)     # intermediates are disposable
    except OSError:
        pass

    return RenderResult(
        video_path=out_video,
        thumbnail_path=thumb,
        ass_path=ass_path,
        width=final["width"],
        height=final["height"],
        duration=final["duration"],
        tightening=tighten_info,
        style_key=getattr(style, "key", None),
    )