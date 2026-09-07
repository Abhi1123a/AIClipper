"""
optimizer.py
============
Everything here targets *mechanical* handicaps - the things that measurably
stop a clip performing regardless of whether the content is good.

Read this before using it
-------------------------
Nobody outside YouTube, TikTok and Meta knows how their ranking systems work.
The systems change, they are not documented, and most "algorithm hack" advice
online is folklore repeated until it sounds official. Even the better sources
label their retention thresholds as synthesized from creator testing rather
than platform-confirmed.

So this module does NOT promise reach. What it does is remove specific,
verifiable defects:

  1. Captions rendered underneath platform UI (invisible to a sound-off viewer)
  2. Dead air and long pauses that push completion rate down
  3. Clips that open on filler words instead of the hook
  4. Clips outside the length band where completion is achievable
  5. Source material too low-resolution to look sharp after upscaling

Those are real problems with real fixes. Content quality, niche consistency and
posting frequency dominate everything here, and no tool can supply them.

The one genuinely reliable signal is your own analytics. `audit_report()`
writes machine-readable per-clip features so you can correlate them against
your real performance later, instead of trusting anyone's generic advice.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field, asdict

from utils import run
from config import settings

log = logging.getLogger("optimizer")


# ===========================================================================
# Platform profiles
# ===========================================================================
@dataclass
class PlatformProfile:
    """
    Safe-zone margins are in pixels on a 1080x1920 canvas, measured from the
    named edge. They come from published 2026 UI measurements and WILL drift as
    the apps redesign - treat them as a starting point and eyeball a real post.
    """
    key: str
    label: str
    blocked_bottom: int      # UI overlay height from the bottom edge
    blocked_top: int
    blocked_left: int
    blocked_right: int
    caption_margin_v: int    # where we actually place captions (> blocked_bottom)
    caption_margin_h: int
    duration_sweet_min: float
    duration_sweet_max: float
    duration_hard_max: float
    hashtag_count: int
    notes: str = ""

    @property
    def safe_height(self) -> int:
        return 1920 - self.blocked_top - self.blocked_bottom


PROFILES: dict[str, PlatformProfile] = {
    # TikTok has the heaviest UI, especially bottom and right.
    "tiktok": PlatformProfile(
        key="tiktok", label="TikTok",
        blocked_bottom=484, blocked_top=130, blocked_left=44, blocked_right=140,
        caption_margin_v=520, caption_margin_h=170,
        duration_sweet_min=15, duration_sweet_max=34, duration_hard_max=600,
        hashtag_count=5,
        notes="Heaviest UI of the three. Right third is risky for faces and text.",
    ),
    # Reels' audio attribution bar grew ~50px in late 2025.
    "reels": PlatformProfile(
        key="reels", label="Instagram Reels",
        blocked_bottom=440, blocked_top=140, blocked_left=60, blocked_right=120,
        caption_margin_v=480, caption_margin_h=130,
        duration_sweet_min=15, duration_sweet_max=45, duration_hard_max=900,
        hashtag_count=5,
        notes="Sends/DM shares are weighted heavily - make it worth forwarding.",
    ),
    # Shorts' subscribe button grew ~30% in late 2025.
    "shorts": PlatformProfile(
        key="shorts", label="YouTube Shorts",
        blocked_bottom=400, blocked_top=380, blocked_left=60, blocked_right=120,
        caption_margin_v=440, caption_margin_h=130,
        duration_sweet_min=20, duration_sweet_max=45, duration_hard_max=180,
        hashtag_count=3,
        notes="Shorts allows 3 minutes since Oct 2024. Also surfaces in search, "
              "so title keywords matter more here than hashtags.",
    ),
    # One render that survives all three. Costs vertical space; buys simplicity.
    "universal": PlatformProfile(
        key="universal", label="All platforms (safe everywhere)",
        blocked_bottom=484, blocked_top=380, blocked_left=60, blocked_right=140,
        caption_margin_v=540, caption_margin_h=170,
        duration_sweet_min=20, duration_sweet_max=40, duration_hard_max=180,
        hashtag_count=4,
        notes="Strictest margin from each platform. Post one file everywhere.",
    ),
}

DEFAULT_PROFILE = "universal"


def get_profile(key: str | None) -> PlatformProfile:
    return PROFILES.get((key or DEFAULT_PROFILE).lower(), PROFILES[DEFAULT_PROFILE])


def apply_profile_to_captions(profile: PlatformProfile) -> None:
    """Move captions above the platform's UI. This is the single most concrete
    fix in the module: a caption behind the UI is invisible, and ~70% of
    short-form viewing happens with the sound off."""
    cap = settings.captions
    if cap.margin_v < profile.blocked_bottom:
        log.info("Captions raised %dpx -> %dpx to clear %s UI (blocks bottom %dpx)",
                 cap.margin_v, profile.caption_margin_v, profile.label,
                 profile.blocked_bottom)
    cap.margin_v = profile.caption_margin_v
    cap.margin_h = max(cap.margin_h, profile.caption_margin_h)


# ===========================================================================
# Retention edit 1: remove dead air
# ===========================================================================
_SILENCE_START = re.compile(r"silence_start:\s*(-?[\d.]+)")
_SILENCE_END = re.compile(r"silence_end:\s*(-?[\d.]+)")


def detect_silences(media_path, noise_db: int = -32, min_silence: float = 0.35
                    ) -> list[tuple[float, float]]:
    """Return [(start, end)] of silent stretches, via FFmpeg's silencedetect."""
    proc = run(
        [settings.ffmpeg, "-hide_banner", "-nostdin", "-i", str(media_path),
         "-af", f"silencedetect=noise={noise_db}dB:d={min_silence}",
         "-f", "null", "-"],
        desc="silencedetect", check=False,
    )
    text = (proc.stderr or "") + (proc.stdout or "")
    starts = [float(m) for m in _SILENCE_START.findall(text)]
    ends = [float(m) for m in _SILENCE_END.findall(text)]
    return [(s, e) for s, e in zip(starts, ends) if e > s]


def plan_tightening(duration: float, silences: list[tuple[float, float]],
                    keep_pad: float = 0.12, min_removal: float = 0.30,
                    max_intervals: int = 60) -> list[tuple[float, float]]:
    """
    Convert silence ranges into the list of intervals to KEEP.

    `keep_pad` leaves a sliver of silence at each edge so speech does not sound
    clipped - removing every millisecond makes people sound breathless and is a
    common way tightened edits end up worse than the original.
    """
    cuts = []
    for s, e in silences:
        s2, e2 = s + keep_pad, e - keep_pad
        if e2 - s2 >= min_removal:
            cuts.append((s2, e2))
    if not cuts:
        return [(0.0, duration)]

    cuts.sort()
    merged = [list(cuts[0])]
    for s, e in cuts[1:]:
        if s <= merged[-1][1] + 0.05:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])

    # Longest removals first if we have to cap the filter expression size.
    if len(merged) > max_intervals:
        merged = sorted(sorted(merged, key=lambda c: c[1] - c[0],
                               reverse=True)[:max_intervals])

    keeps, cursor = [], 0.0
    for s, e in merged:
        if s > cursor + 0.05:
            keeps.append((round(cursor, 3), round(s, 3)))
        cursor = e
    if duration > cursor + 0.05:
        keeps.append((round(cursor, 3), round(duration, 3)))
    return keeps or [(0.0, duration)]


def remap_words(words: list[dict], keeps: list[tuple[float, float]],
                clip_start: float = 0.0) -> list[dict]:
    """
    Shift word timestamps onto the tightened timeline.

    Without this, trimming silence silently desyncs every caption - the whole
    reason word-level timing exists. Words falling inside a removed gap are
    snapped to the nearest surviving edge rather than dropped.
    """
    if not keeps:
        return [{**w, "start": w["start"] - clip_start, "end": w["end"] - clip_start}
                for w in words]

    # (source_start, source_end, offset_on_new_timeline)
    spans, acc = [], 0.0
    for s, e in keeps:
        spans.append((s, e, acc))
        acc += e - s
    total = acc

    def map_t(t: float) -> float:
        for s, e, off in spans:
            if s <= t <= e:
                return off + (t - s)
        for s, e, off in spans:          # inside a removed gap -> snap forward
            if t < s:
                return off
        return total

    out = []
    for w in words:
        s = map_t(w["start"] - clip_start)
        e = map_t(w["end"] - clip_start)
        if e <= s:
            e = s + 0.08
        out.append({**w, "start": round(s, 3), "end": round(e, 3)})
    return out


def keeps_to_filters(keeps: list[tuple[float, float]]) -> tuple[str, str]:
    """FFmpeg select/aselect expressions for the kept intervals."""
    expr = "+".join(f"between(t,{s:.3f},{e:.3f})" for s, e in keeps)
    return (f"select='{expr}',setpts=N/FRAME_RATE/TB",
            f"aselect='{expr}',asetpts=N/SR/TB")


def tightening_summary(keeps: list[tuple[float, float]], duration: float) -> dict:
    kept = sum(e - s for s, e in keeps)
    return {
        "original_duration": round(duration, 2),
        "tightened_duration": round(kept, 2),
        "removed_seconds": round(duration - kept, 2),
        "removed_pct": round(100 * (duration - kept) / max(0.01, duration), 1),
        "cut_count": max(0, len(keeps) - 1),
    }


# ===========================================================================
# Retention edit 2: don't open on filler
# ===========================================================================
FILLER_OPENERS = {
    "so", "and", "but", "um", "uh", "er", "like", "well", "yeah", "yep", "okay",
    "ok", "right", "anyway", "anyways", "basically", "actually", "just", "i",
    "you", "know", "mean", "then", "now", "look", "listen", "alright",
}


def trim_leading_filler(words: list[dict], max_trim: float = 2.0,
                        max_words: int = 4) -> tuple[float, list[str]]:
    """
    Return how many seconds of filler to cut off the front, plus what was cut.

    The first ~3 seconds decide whether a viewer stays. Opening on "So, um,
    yeah, anyway..." wastes the part that matters most. Conservative by design:
    never trims more than `max_trim` seconds or `max_words` words, and stops at
    the first real word.
    """
    if not words:
        return 0.0, []
    removed: list[str] = []
    origin = words[0]["start"]
    for w in words[:max_words]:
        token = re.sub(r"[^\w']", "", w["word"]).lower()
        if token and token in FILLER_OPENERS:
            if w["end"] - origin <= max_trim:
                removed.append(w["word"].strip())
                continue
        break
    if not removed:
        return 0.0, []
    trim_to = words[len(removed)]["start"] if len(removed) < len(words) else origin
    return round(max(0.0, trim_to - origin), 3), removed


# ===========================================================================
# Auditing
# ===========================================================================
@dataclass
class Finding:
    level: str        # "error" | "warn" | "info"
    code: str
    message: str


@dataclass
class ClipAudit:
    clip_index: int
    platform: str
    duration: float
    findings: list[Finding] = field(default_factory=list)
    features: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not any(f.level == "error" for f in self.findings)

    def to_dict(self) -> dict:
        return {
            "clip_index": self.clip_index,
            "platform": self.platform,
            "duration": self.duration,
            "passed": self.ok,
            "findings": [asdict(f) for f in self.findings],
            "features": self.features,
        }


def audit_clip(clip, profile: PlatformProfile, *, source_height: int = 0,
               render=None, words: list[dict] | None = None) -> ClipAudit:
    """Check one clip against the mechanical failure modes, and record the
    features worth correlating against your own analytics later."""
    cap = settings.captions
    audit = ClipAudit(clip_index=getattr(clip, "index", 0), platform=profile.key,
                      duration=clip.duration)
    f = audit.findings

    # --- captions behind UI: the one true error state ----------------------
    if cap.margin_v < profile.blocked_bottom:
        f.append(Finding("error", "captions_under_ui",
                         f"Captions sit {cap.margin_v}px from the bottom but "
                         f"{profile.label} blocks {profile.blocked_bottom}px. "
                         f"They will be hidden. Use --platform {profile.key}."))
    elif cap.margin_v < profile.blocked_bottom + 25:
        f.append(Finding("warn", "captions_tight",
                         "Captions clear the UI by under 25px - tight on some devices."))

    # --- duration ----------------------------------------------------------
    if clip.duration > profile.duration_hard_max:
        f.append(Finding("error", "too_long",
                         f"{clip.duration:.0f}s exceeds {profile.label}'s "
                         f"{profile.duration_hard_max:.0f}s limit."))
    elif clip.duration > profile.duration_sweet_max:
        f.append(Finding("warn", "above_sweet_spot",
                         f"{clip.duration:.0f}s is above the {profile.duration_sweet_min:.0f}-"
                         f"{profile.duration_sweet_max:.0f}s band where completion is "
                         f"easiest. Fine if the content earns it - completion rate "
                         f"matters more than length."))
    elif clip.duration < profile.duration_sweet_min:
        f.append(Finding("warn", "very_short",
                         f"{clip.duration:.0f}s is short; a weak hook has nowhere to hide."))

    # --- opener ------------------------------------------------------------
    if words:
        trim, removed = trim_leading_filler(words)
        if trim > 0:
            f.append(Finding("warn", "filler_opener",
                             f"Opens on filler ({' '.join(removed)}). "
                             f"--tighten trims {trim:.1f}s off the front."))
        opener = " ".join(w["word"] for w in words[:12])
        audit.features["opener_text"] = opener.strip()
        audit.features["opens_on_question"] = "?" in opener

    # --- source quality ----------------------------------------------------
    if source_height and source_height < 720:
        f.append(Finding("warn", "low_source_res",
                         f"Source is {source_height}p; upscaling to 1920 looks soft."))

    if render is not None:
        if (render.width, render.height) != (1080, 1920):
            f.append(Finding("warn", "unexpected_dimensions",
                             f"Rendered {render.width}x{render.height}, expected 1080x1920."))
        if getattr(render, "ass_path", None) is None:
            f.append(Finding("warn", "no_captions",
                             "No burned-in captions. Most short-form viewing is muted."))

    # --- features for your own analysis ------------------------------------
    audit.features.update({
        "viral_score": getattr(clip, "viral_score", None),
        "word_count": len(words or []),
        "words_per_second": round(len(words or []) / max(1.0, clip.duration), 2),
        "is_series_part": getattr(clip, "is_series", False),
        "part": getattr(clip, "part", 0),
        "hashtag_count": len(getattr(clip, "hashtags", []) or []),
        "title_length": len(getattr(clip, "hook_title", "") or ""),
        "caption_margin_v": cap.margin_v,
    })
    return audit


def print_audit(audits: list[ClipAudit]) -> None:
    icons = {"error": "FAIL", "warn": "WARN", "info": "INFO"}
    errors = sum(1 for a in audits for f in a.findings if f.level == "error")
    warns = sum(1 for a in audits for f in a.findings if f.level == "warn")

    print("\n" + "-" * 78)
    print(f"  PRE-FLIGHT AUDIT  ({audits[0].platform if audits else '-'})"
          f"   {errors} error(s), {warns} warning(s)")
    print("-" * 78)
    for a in audits:
        if not a.findings:
            print(f"  clip {a.clip_index}: clean ({a.duration:.0f}s)")
            continue
        print(f"  clip {a.clip_index} ({a.duration:.0f}s):")
        for f in a.findings:
            print(f"     [{icons.get(f.level, '?')}] {f.message}")
    print("-" * 78)
    print("  These are mechanical checks only. They cannot tell you whether the")
    print("  content is good, and passing them does not predict reach.")
    print("-" * 78 + "\n")