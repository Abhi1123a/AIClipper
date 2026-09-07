"""
config.py
=========
Single source of truth for paths, API keys, model choices and render settings.

Every value can be overridden with an environment variable (or a .env file),
so you never hard-code a secret. Copy `.env.example` -> `.env` and fill it in.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Load .env sitting next to this file (does not override real env vars).
ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")


def _env(key: str, default: str = "") -> str:
    return os.getenv(key, default).strip()


def _env_int(key: str, default: int) -> int:
    try:
        return int(_env(key) or default)
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    try:
        return float(_env(key) or default)
    except ValueError:
        return default


def _env_bool(key: str, default: bool = False) -> bool:
    raw = _env(key).lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "y", "on"}


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
@dataclass
class Paths:
    root: Path = ROOT
    downloads: Path = ROOT / "downloads"      # raw source videos
    cache: Path = ROOT / "cache"              # transcripts + LLM answers
    work: Path = ROOT / "work"                # intermediate cuts, .ass files
    output: Path = ROOT / "output"            # final, publish-ready clips
    browser_profile: Path = ROOT / ".browser_profile"   # Playwright session

    def ensure(self) -> None:
        for p in (self.downloads, self.cache, self.work, self.output):
            p.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------
@dataclass
class WhisperConfig:
    # tiny | base | small | medium | large-v3 | distil-large-v3
    model_size: str = _env("WHISPER_MODEL", "small")
    # auto | cpu | cuda
    device: str = _env("WHISPER_DEVICE", "auto")
    # int8 | int8_float16 | float16 | float32  ("auto" picks a sane default)
    compute_type: str = _env("WHISPER_COMPUTE_TYPE", "auto")
    language: str | None = _env("WHISPER_LANGUAGE", "") or None   # None = detect
    beam_size: int = _env_int("WHISPER_BEAM_SIZE", 5)
    vad_filter: bool = _env_bool("WHISPER_VAD", True)


# ---------------------------------------------------------------------------
# LLM highlight detection
# ---------------------------------------------------------------------------
# Any OpenAI-compatible endpoint works. base_url, default model, env var holding
# the key. Add your own row here and it becomes a --provider choice for free.
OPENAI_PRESETS: dict[str, tuple[str, str, str]] = {
    "nim":        ("https://integrate.api.nvidia.com/v1", "meta/llama-3.3-70b-instruct", "NVIDIA_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1",        "deepseek/deepseek-chat-v3-0324:free", "OPENROUTER_API_KEY"),
    "cerebras":   ("https://api.cerebras.ai/v1",          "llama-3.3-70b",              "CEREBRAS_API_KEY"),
    "together":   ("https://api.together.xyz/v1",         "meta-llama/Llama-3.3-70B-Instruct-Turbo", "TOGETHER_API_KEY"),
    "ollama":     ("http://localhost:11434/v1",           "qwen2.5:14b-instruct",       ""),   # local, no key
    "lmstudio":   ("http://localhost:1234/v1",            "local-model",                ""),   # local, no key
    "openai":     ("", "", "OPENAI_API_KEY"),   # fully custom: set OPENAI_BASE_URL/MODEL
}

LOCAL_PROVIDERS = {"ollama", "lmstudio"}
KEYLESS_PROVIDERS = {"heuristic", "manual"} | LOCAL_PROVIDERS


@dataclass
class LLMConfig:
    # gemini | groq | manual | heuristic | nim | openrouter | cerebras |
    # together | ollama | lmstudio | openai
    provider: str = _env("LLM_PROVIDER", "gemini")
    gemini_api_key: str = _env("GEMINI_API_KEY")
    gemini_model: str = _env("GEMINI_MODEL", "gemini-2.5-flash")
    groq_api_key: str = _env("GROQ_API_KEY")
    groq_model: str = _env("GROQ_MODEL", "llama-3.3-70b-versatile")

    # --- generic OpenAI-compatible (NIM / OpenRouter / Cerebras / Ollama...) --
    openai_base_url: str = _env("OPENAI_BASE_URL")
    openai_api_key: str = _env("OPENAI_API_KEY")
    openai_model: str = _env("OPENAI_MODEL")

    # --- manual relay (paste into claude.ai / any chat UI by hand) -----------
    # When true, `manual` writes the prompt files and exits instead of blocking
    # on input. Re-run the same command later to pick up your saved answers.
    manual_export_only: bool = False

    def openai_settings(self, provider: str) -> tuple[str, str, str]:
        """Resolve (base_url, model, api_key) for an OpenAI-compatible provider.
        Explicit OPENAI_* env vars always win over the preset defaults."""
        base, model, key_env = OPENAI_PRESETS.get(provider, ("", "", "OPENAI_API_KEY"))
        base = self.openai_base_url or base
        model = self.openai_model or model
        key = self.openai_api_key or (_env(key_env) if key_env else "")
        if provider in LOCAL_PROVIDERS:
            key = key or "not-needed"       # SDK insists on a non-empty string
        return base, model, key

    temperature: float = _env_float("LLM_TEMPERATURE", 0.7)
    # Transcript characters per LLM request. Long videos are windowed.
    chunk_chars: int = _env_int("LLM_CHUNK_CHARS", 14000)
    # Manual mode pastes into a chat UI with a far larger context than a free
    # API tier, so it gets one big window instead. This matters most for story
    # mode: a model that only sees half the story cannot place the break points
    # of the half it cannot see.
    manual_chunk_chars: int = _env_int("LLM_MANUAL_CHUNK_CHARS", 120000)
    chunk_overlap_chars: int = _env_int("LLM_CHUNK_OVERLAP", 1200)
    max_retries: int = _env_int("LLM_MAX_RETRIES", 3)

    def resolved_provider(self) -> str:
        """Honour the chosen provider; fall back only when it cannot possibly work."""
        p = self.provider.lower()

        if p in KEYLESS_PROVIDERS:
            return p
        if p == "gemini" and self.gemini_api_key:
            return "gemini"
        if p == "groq" and self.groq_api_key:
            return "groq"
        if p in OPENAI_PRESETS:
            base, _model, key = self.openai_settings(p)
            if base and key:
                return p

        # Chosen provider is unusable - try anything else that is configured.
        if self.gemini_api_key:
            return "gemini"
        if self.groq_api_key:
            return "groq"
        for name in OPENAI_PRESETS:
            base, _model, key = self.openai_settings(name)
            if base and key and name not in LOCAL_PROVIDERS:
                return name
        return "heuristic"


# ---------------------------------------------------------------------------
# Clip selection rules
# ---------------------------------------------------------------------------
@dataclass
class ClipRules:
    # highlights = best isolated moments (default)
    # story      = chronological, contiguous parts covering the whole narrative
    # sequential = mechanical chop at sentence boundaries, no LLM at all
    strategy: str = _env("CLIP_STRATEGY", "highlights")
    min_duration: float = _env_float("CLIP_MIN_SECONDS", 15.0)
    max_duration: float = _env_float("CLIP_MAX_SECONDS", 59.0)
    target_count: int = _env_int("CLIP_COUNT", 5)
    # Discard a candidate that overlaps an accepted clip by more than this ratio
    max_overlap_ratio: float = _env_float("CLIP_MAX_OVERLAP", 0.35)
    # Padding added around the LLM's boundaries, then snapped to word edges
    lead_in: float = _env_float("CLIP_LEAD_IN", 0.25)
    lead_out: float = _env_float("CLIP_LEAD_OUT", 0.45)

    # --- story / sequential only -------------------------------------------
    # Gaps between consecutive parts smaller than this get closed, so the
    # series loses no dialogue. Larger gaps are left as deliberate cuts.
    part_gap_tolerance: float = _env_float("CLIP_PART_GAP_TOLERANCE", 4.0)
    # Preferred part length. Parts break at the first sentence end past this.
    part_target: float = _env_float("CLIP_PART_TARGET", 50.0)
    # In story/sequential mode, a part shorter than this fraction of
    # min_duration after boundary adjustment is merged into its neighbour.
    part_min_ratio: float = _env_float("CLIP_PART_MIN_RATIO", 0.7)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
@dataclass
class RenderConfig:
    width: int = _env_int("OUT_WIDTH", 1080)
    height: int = _env_int("OUT_HEIGHT", 1920)
    fps: int = _env_int("OUT_FPS", 30)
    crf: int = _env_int("OUT_CRF", 20)
    preset: str = _env("OUT_PRESET", "veryfast")
    audio_bitrate: str = _env("OUT_AUDIO_BITRATE", "192k")
    # blur | crop | face
    reframe_mode: str = _env("REFRAME_MODE", "blur")
    blur_sigma: int = _env_int("BLUR_SIGMA", 24)
    blur_darken: float = _env_float("BLUR_DARKEN", -0.10)
    # Height of the foreground strip in "blur" mode, as a fraction of 1920
    fg_height_ratio: float = _env_float("FG_HEIGHT_RATIO", 0.62)
    loudnorm: bool = _env_bool("LOUDNORM", True)
    threads: int = _env_int("FFMPEG_THREADS", 0)   # 0 = let FFmpeg decide


# ---------------------------------------------------------------------------
# Caption styling (burned-in ASS)
# ---------------------------------------------------------------------------
@dataclass
class CaptionConfig:
    font_name: str = _env("CAPTION_FONT", "Arial Black")
    font_size: int = _env_int("CAPTION_FONT_SIZE", 86)
    # ASS colours are &HAABBGGRR (alpha, blue, green, red) - NOT RGB.
    color_idle: str = _env("CAPTION_COLOR_IDLE", "&H00FFFFFF")     # white
    color_active: str = _env("CAPTION_COLOR_ACTIVE", "&H0000E5FF") # amber
    outline_color: str = _env("CAPTION_OUTLINE_COLOR", "&H00000000")
    shadow_color: str = _env("CAPTION_SHADOW_COLOR", "&H80000000")
    outline: int = _env_int("CAPTION_OUTLINE", 6)
    shadow: int = _env_int("CAPTION_SHADOW", 3)
    # Distance from the bottom edge, in pixels of a 1920-tall frame.
    margin_v: int = _env_int("CAPTION_MARGIN_V", 540)
    margin_h: int = _env_int("CAPTION_MARGIN_H", 90)
    uppercase: bool = _env_bool("CAPTION_UPPERCASE", True)
    max_words_per_card: int = _env_int("CAPTION_MAX_WORDS", 4)
    max_card_seconds: float = _env_float("CAPTION_MAX_CARD_SECONDS", 1.8)
    # Scale-up "pop" applied to the active word (percent)
    pop_scale: int = _env_int("CAPTION_POP_SCALE", 115)
    pop_ms: int = _env_int("CAPTION_POP_MS", 110)
    # Optional folder holding .ttf files, passed to FFmpeg as fontsdir
    fonts_dir: str = _env("CAPTION_FONTS_DIR", "")


# ---------------------------------------------------------------------------
# Publishing
# ---------------------------------------------------------------------------
@dataclass
class PublishConfig:
    # Never fully auto-publish unless you have explicitly opted in.
    semi_auto: bool = _env_bool("PUBLISH_SEMI_AUTO", True)
    headless: bool = _env_bool("PUBLISH_HEADLESS", False)
    youtube_client_secrets: str = _env("YOUTUBE_CLIENT_SECRETS", "client_secrets.json")
    youtube_token_file: str = _env("YOUTUBE_TOKEN_FILE", "youtube_token.json")
    youtube_privacy: str = _env("YOUTUBE_PRIVACY", "private")  # private|unlisted|public
    youtube_category_id: str = _env("YOUTUBE_CATEGORY_ID", "22")  # People & Blogs
    default_hashtags: list[str] = field(
        default_factory=lambda: [t for t in _env("DEFAULT_HASHTAGS", "").split(",") if t]
    )


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------
@dataclass
class Settings:
    paths: Paths = field(default_factory=Paths)
    whisper: WhisperConfig = field(default_factory=WhisperConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    clips: ClipRules = field(default_factory=ClipRules)
    render: RenderConfig = field(default_factory=RenderConfig)
    captions: CaptionConfig = field(default_factory=CaptionConfig)
    publish: PublishConfig = field(default_factory=PublishConfig)

    ffmpeg: str = _env("FFMPEG_BIN", "ffmpeg")
    ffprobe: str = _env("FFPROBE_BIN", "ffprobe")
    log_level: str = _env("LOG_LEVEL", "INFO")

    def validate(self) -> list[str]:
        """Return a list of human-readable problems (empty == good to go)."""
        problems: list[str] = []
        if not shutil.which(self.ffmpeg):
            problems.append(
                f"FFmpeg not found on PATH (looked for '{self.ffmpeg}'). "
                "See README > Installing FFmpeg."
            )
        if not shutil.which(self.ffprobe):
            problems.append(f"ffprobe not found on PATH (looked for '{self.ffprobe}').")
        if self.llm.resolved_provider() == "heuristic" and self.llm.provider != "heuristic":
            problems.append(
                "No usable LLM credentials found - falling back to the offline heuristic "
                "picker. Use '--provider manual' to relay the prompt through claude.ai "
                "by hand instead, or set a key (see .env.example)."
            )
        if self.render.reframe_mode not in {"blur", "crop", "face"}:
            problems.append(f"Unknown REFRAME_MODE '{self.render.reframe_mode}'.")
        return problems


settings = Settings()
settings.paths.ensure()
