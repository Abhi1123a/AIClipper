"""
styles.py
=========
Editing style presets. A motivational clip and a brainrot clip want visually
opposite treatments, and using one look for everything makes every clip feel
like the same template.

Each style controls:
  * caption typography, colour, casing, words-per-card, pop animation
  * caption vertical placement (still clamped to the platform safe zone)
  * pacing: whether to trim silence, and how aggressively
  * grade: saturation, contrast, brightness of the final image
  * blur-mode framing (how much of the frame the sharp strip occupies)
  * the LLM's tone instructions for titles, captions and hashtags

Styles: humor | motivational | story | brainrot | clean

A note on `brainrot`
--------------------
The style exists because you asked for it and it is a real format. But it is
worth knowing what it interacts with: YouTube's Generic or Repetitive Content
policy (renamed from "inauthentic content" in July 2026) makes templated,
mass-produced uploads ineligible for monetization, and enforcement happens at
the CHANNEL level, not per video. In January 2026 YouTube deleted a set of
large faceless channels outright.

The policy does not target any visual style - it targets output that "looks
like it's made with a template" across a channel. So this module deliberately
VARIES its output per clip (see `variation_seed`) rather than stamping an
identical look on everything. That is not a loophole; it is the difference
between a style and a template. Original framing, commentary or editorial
judgement is what keeps a channel on the right side of it, and no code can
supply that.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field, replace

log = logging.getLogger("styles")


# ---------------------------------------------------------------------------
# ASS colours are &HAABBGGRR - alpha, BLUE, GREEN, RED. Not RGB.
# ---------------------------------------------------------------------------
WHITE = "&H00FFFFFF"
BLACK = "&H00000000"
AMBER = "&H0000E5FF"
GOLD = "&H0000C8FF"
CYAN = "&H00FFFF00"
LIME = "&H0000FF7F"
MAGENTA = "&H00FF00FF"
HOT_PINK = "&H00B469FF"
SOFT_BLUE = "&H00E0C080"
RED = "&H002020FF"


@dataclass
class EditStyle:
    key: str
    label: str
    description: str

    # --- captions ---------------------------------------------------------
    font_name: str = "Arial Black"
    font_size: int = 86
    color_idle: str = WHITE
    color_active: str = AMBER
    outline: int = 6
    shadow: int = 3
    uppercase: bool = True
    max_words_per_card: int = 4
    max_card_seconds: float = 1.8
    pop_scale: int = 115
    pop_ms: int = 110
    # Fraction of frame height for captions, measured from the bottom.
    # Clamped to the platform safe zone at apply time - never overrides it.
    caption_position: float = 0.28

    # --- pacing -----------------------------------------------------------
    tighten: bool = False
    silence_noise_db: int = -32
    silence_min: float = 0.35
    keep_pad: float = 0.12

    # --- image ------------------------------------------------------------
    saturation: float = 1.0
    contrast: float = 1.0
    brightness: float = 0.0
    fg_height_ratio: float = 0.62
    blur_sigma: int = 24

    # --- audio ------------------------------------------------------------
    loudness_target: float = -14.0

    # --- how the LLM should write titles/captions --------------------------
    tone_prompt: str = ""

    # --- per-clip variation ------------------------------------------------
    # Alternate accent colours cycled across clips so a batch does not look
    # like one template stamped repeatedly.
    accent_cycle: list[str] = field(default_factory=list)
    size_jitter: int = 0          # +/- px applied to font_size per clip
    position_jitter: float = 0.0  # +/- fraction applied to caption_position


# ===========================================================================
# Presets
# ===========================================================================
STYLES: dict[str, EditStyle] = {

    # -- HUMOR: punchy, loud, fast. Captions land like a punchline. --------
    "humor": EditStyle(
        key="humor", label="Humor",
        description="Fast, loud, high-contrast. Short caption bursts that land "
                    "like punchlines. Dead air removed hard.",
        font_name="Impact",
        font_size=96, color_idle=WHITE, color_active=LIME,
        outline=7, shadow=4, uppercase=True,
        max_words_per_card=3, max_card_seconds=1.2,
        pop_scale=132, pop_ms=80,
        caption_position=0.30,
        tighten=True, silence_min=0.28, keep_pad=0.08,
        saturation=1.18, contrast=1.08, brightness=0.02,
        fg_height_ratio=0.66, blur_sigma=20,
        loudness_target=-13.0,
        accent_cycle=[LIME, CYAN, AMBER, MAGENTA],
        size_jitter=6, position_jitter=0.02,
        tone_prompt=(
            "Write titles and captions with comic timing. Favour the setup-then-"
            "punchline shape, understatement, and specific absurd detail over "
            "generic 'this is hilarious' phrasing. Never explain the joke. "
            "Captions can be a single deadpan line."
        ),
    ),

    # -- MOTIVATIONAL: slower, weightier, cinematic. -----------------------
    "motivational": EditStyle(
        key="motivational", label="Motivational / Deep",
        description="Slower, cinematic and weighted. Fewer words on screen, "
                    "held longer. Cooler grade, gentle pacing.",
        font_name="Georgia",
        font_size=78, color_idle=WHITE, color_active=GOLD,
        outline=5, shadow=5, uppercase=False,
        max_words_per_card=3, max_card_seconds=2.4,
        pop_scale=106, pop_ms=220,
        caption_position=0.34,
        # Deliberately NOT tightened: pauses carry the weight in this format.
        tighten=False,
        saturation=0.88, contrast=1.12, brightness=-0.04,
        fg_height_ratio=0.58, blur_sigma=34,
        loudness_target=-15.0,
        accent_cycle=[GOLD, SOFT_BLUE, WHITE],
        size_jitter=4, position_jitter=0.02,
        tone_prompt=(
            "Write with restraint and weight. Plain, declarative sentences. No "
            "hype words, no exclamation marks, no emoji. Let the idea carry the "
            "line rather than the punctuation. Titles should sound like something "
            "a person would actually say out loud, not a slogan."
        ),
    ),

    # -- STORY: readable, neutral, built for multi-part series. ------------
    "story": EditStyle(
        key="story", label="Story / Narrative",
        description="Clean and highly readable. Longer caption cards for "
                    "dialogue, neutral grade, light pause trimming.",
        font_name="Arial Black",
        font_size=80, color_idle=WHITE, color_active=CYAN,
        outline=6, shadow=3, uppercase=False,
        max_words_per_card=5, max_card_seconds=2.0,
        pop_scale=112, pop_ms=130,
        caption_position=0.30,
        # Light trim only - conversation rhythm matters, so long pauses go but
        # natural beats stay.
        tighten=True, silence_min=0.55, keep_pad=0.18,
        saturation=1.0, contrast=1.02, brightness=0.0,
        fg_height_ratio=0.62, blur_sigma=26,
        loudness_target=-14.0,
        accent_cycle=[CYAN, AMBER, LIME],
        size_jitter=4, position_jitter=0.015,
        tone_prompt=(
            "Write like someone recounting what happened to a friend. Keep the "
            "chronology clear and never spoil the ending in the title. For a "
            "series, each part's title should describe that part's beat only."
        ),
    ),

    # -- BRAINROT: maximum stimulation, tiny cards, hard cuts. -------------
    "brainrot": EditStyle(
        key="brainrot", label="Brainrot / Max stimulation",
        description="Very fast, very loud, 1-2 words per card, saturated grade, "
                    "aggressive silence removal. See the module docstring on "
                    "platform policy before mass-producing this.",
        font_name="Impact",
        font_size=110, color_idle=WHITE, color_active=HOT_PINK,
        outline=9, shadow=5, uppercase=True,
        max_words_per_card=2, max_card_seconds=0.8,
        pop_scale=150, pop_ms=60,
        caption_position=0.32,
        tighten=True, silence_noise_db=-30, silence_min=0.20, keep_pad=0.05,
        saturation=1.42, contrast=1.16, brightness=0.05,
        fg_height_ratio=0.72, blur_sigma=16,
        loudness_target=-12.0,
        accent_cycle=[HOT_PINK, LIME, CYAN, AMBER, MAGENTA, RED],
        size_jitter=10, position_jitter=0.03,
        tone_prompt=(
            "Write in the blunt, high-energy register of viral short-form: short "
            "fragments, direct address, present tense. Still say something true "
            "and specific about the clip - vague hype ('this is insane') reads as "
            "filler and gets scrolled past."
        ),
    ),

    # -- CLEAN: the original neutral look, unchanged. ----------------------
    "clean": EditStyle(
        key="clean", label="Clean / Neutral",
        description="The default neutral look. No grade, no trimming, "
                    "balanced captions.",
        tone_prompt="",
    ),
}

DEFAULT_STYLE = "clean"

# Which style suits which clip strategy, when the user does not say.
STRATEGY_DEFAULT_STYLE = {
    "story": "story",
    "sequential": "story",
    "highlights": "clean",
}


def get_style(key: str | None) -> EditStyle:
    return STYLES.get((key or DEFAULT_STYLE).lower(), STYLES[DEFAULT_STYLE])


def list_styles() -> str:
    return "\n".join(f"  {s.key:<14} {s.description}" for s in STYLES.values())


# ===========================================================================
# Per-clip variation
# ===========================================================================
def variation_seed(*parts) -> int:
    """Stable seed from clip identity, so a given clip always renders the same
    way (re-runs are reproducible) while differing from its siblings."""
    raw = "|".join(str(p) for p in parts)
    return int(hashlib.sha1(raw.encode("utf-8")).hexdigest()[:8], 16)


def vary(style: EditStyle, seed: int) -> EditStyle:
    """
    Return a copy with small, deterministic per-clip differences.

    This is the difference between a *style* and a *template*. Ten clips that
    are pixel-identical in treatment are exactly what YouTube's Generic or
    Repetitive Content policy describes; ten clips that share a look but differ
    in accent, size and placement read as a consistent brand instead.
    """
    out = replace(style)

    if style.accent_cycle:
        out.color_active = style.accent_cycle[seed % len(style.accent_cycle)]
    if style.size_jitter:
        span = style.size_jitter * 2 + 1
        out.font_size = style.font_size + (seed % span) - style.size_jitter
    if style.position_jitter:
        steps = 5
        offset = ((seed >> 8) % steps) / (steps - 1)      # 0.0 .. 1.0
        out.caption_position = style.caption_position + \
            (offset * 2 - 1) * style.position_jitter

    return out


# ===========================================================================
# Application
# ===========================================================================
def apply_style(style: EditStyle, settings, profile=None) -> None:
    """
    Push a style into the live settings object.

    Safe-zone rule: the platform profile always wins. A style may ask for
    captions lower than the platform's UI allows; it gets clamped, because a
    caption behind the UI is invisible and that beats any aesthetic choice.
    """
    cap = settings.captions
    r = settings.render

    cap.font_name = style.font_name
    cap.font_size = style.font_size
    cap.color_idle = style.color_idle
    cap.color_active = style.color_active
    cap.outline = style.outline
    cap.shadow = style.shadow
    cap.uppercase = style.uppercase
    cap.max_words_per_card = style.max_words_per_card
    cap.max_card_seconds = style.max_card_seconds
    cap.pop_scale = style.pop_scale
    cap.pop_ms = style.pop_ms

    wanted = int(round(style.caption_position * r.height))
    if profile is not None and wanted < profile.caption_margin_v:
        log.debug("Style %s wanted captions at %dpx; clamped to %dpx for %s.",
                  style.key, wanted, profile.caption_margin_v, profile.label)
        wanted = profile.caption_margin_v
    cap.margin_v = wanted

    r.fg_height_ratio = style.fg_height_ratio
    r.blur_sigma = style.blur_sigma


def grade_filter(style: EditStyle) -> str:
    """FFmpeg `eq` fragment for this style's grade, or '' if neutral."""
    parts = []
    if abs(style.saturation - 1.0) > 0.01:
        parts.append(f"saturation={style.saturation:.2f}")
    if abs(style.contrast - 1.0) > 0.01:
        parts.append(f"contrast={style.contrast:.2f}")
    if abs(style.brightness) > 0.005:
        parts.append(f"brightness={style.brightness:.3f}")
    return f"eq={':'.join(parts)}" if parts else ""


def describe(style: EditStyle) -> str:
    bits = [
        f"{style.font_name} {style.font_size}px",
        f"{style.max_words_per_card}w/card",
        "trimmed" if style.tighten else "untrimmed",
    ]
    if abs(style.saturation - 1.0) > 0.01:
        bits.append(f"sat {style.saturation:.2f}")
    return " | ".join(bits)