"""
ai_analyzer.py
==============
Turns a timestamped transcript into 3-5 ranked, ready-to-cut short-form clips.

Providers (all free):
  * gemini    - Google Gemini free tier via the `google-genai` SDK
  * groq      - Groq free tier (Llama 3.3 70B) via the `groq` SDK
  * heuristic - no API key at all; local scoring on hook words, question
                density, numbers, and speech pacing. Deliberately included so
                the pipeline is *never* blocked on a key or a rate limit.

Long transcripts are windowed into overlapping chunks, each analysed
separately, then merged, de-overlapped and ranked by viral_score.

Every clip returned is guaranteed to:
  - respect ClipRules.min_duration / max_duration
  - start and end on a real word boundary (snapped against the word list)
  - not overlap an already-accepted clip beyond ClipRules.max_overlap_ratio
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field, asdict

from config import OPENAI_PRESETS, settings
from transcriber import plain_text, snap_to_words, transcript_lines
from utils import read_json, write_json

log = logging.getLogger("ai_analyzer")


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass
class Clip:
    start_time: float
    end_time: float
    viral_score: int
    hook_title: str
    reason: str = ""
    caption: str = ""
    hashtags: list[str] = field(default_factory=list)
    transcript: str = ""
    index: int = 0

    # --- series metadata (story / sequential strategies) -------------------
    part: int = 0            # 1-based position in the series; 0 = not a series
    total_parts: int = 0
    series_title: str = ""   # shared title for the whole story
    cliffhanger: str = ""    # what this part leaves unresolved

    @property
    def duration(self) -> float:
        return round(self.end_time - self.start_time, 3)

    @property
    def is_series(self) -> bool:
        return self.part > 0 and self.total_parts > 1

    def to_dict(self) -> dict:
        d = asdict(self)
        d["duration"] = self.duration
        return d


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a senior short-form video editor who has cut clips \
that collectively passed a billion views on TikTok, Reels and YouTube Shorts.

You are given a timestamped transcript of a long video. Select the segments \
that would perform best as standalone vertical shorts.

Selection rules - follow strictly:
1. Each clip must be between {min_s:.0f} and {max_s:.0f} seconds long.
2. A clip MUST be self-contained: it makes complete sense to someone who has \
never seen the source video and has no other context.
3. The FIRST 3 seconds must contain the hook - a bold claim, a surprising \
number, a question, a confession, a contradiction, or the start of a story. \
Never open on filler, throat-clearing, or "so anyway...".
4. Never cut mid-sentence at either boundary.
5. Prefer: contrarian opinions, specific numbers, personal stories with a \
turn, step-by-step advice, emotional peaks, myth-busting, before/after.
6. Avoid: rambling setup, inside jokes, sponsor reads, greetings, sign-offs, \
segments that reference on-screen visuals the viewer cannot see.
7. Clips must not overlap each other.

For each clip also write:
- hook_title: <=60 chars, punchy, no clickbait lying, no leading emoji spam.
- reason: one sentence on why this specific segment will retain attention.
- caption: a 1-2 sentence social caption ending in a question or a CTA.
- hashtags: 5-8 lowercase tags WITHOUT the '#' character, mixing one broad \
tag, several niche/topical tags, and one format tag (e.g. shorts, reels).
- viral_score: integer 1-100. Be harsh and use the full range; a score above \
85 should be rare.

Return ONLY valid JSON, no markdown fences, no commentary, in exactly this shape:
{{"clips": [{{"start_time": 123.4, "end_time": 168.9, "viral_score": 82, \
"hook_title": "...", "reason": "...", "caption": "...", \
"hashtags": ["...", "..."]}}]}}"""

USER_PROMPT = """Video title: {title}

Timestamped transcript (seconds):
---
{transcript}
---

Return the {n} strongest clips as JSON. Use timestamps from the transcript above."""


# ---------------------------------------------------------------------------
# Story mode: a sequential series, not a greatest-hits reel
# ---------------------------------------------------------------------------
STORY_SYSTEM_PROMPT = """You are a short-form editor who cuts single continuous \
stories into multi-part vertical series - the format where a radio segment, \
confession or interview is posted as Part 1, Part 2, Part 3 and viewers follow \
to get the ending.

You are given a timestamped transcript. Split it into consecutive parts.

This is NOT a highlights reel. You are not looking for the best moments. You \
are dividing one continuous story so that someone who watches every part in \
order hears the whole thing, with nothing important missing.

Rules - follow strictly:
1. Parts must be in chronological order and must not overlap.
2. Parts must be CONTIGUOUS: each part begins where the previous part ended. \
Do not skip material, even if a stretch is slow - context the audience needs \
is more valuable here than pace.
3. Each part must be between {min_s:.0f} and {max_s:.0f} seconds. Aim for \
around {target_s:.0f} seconds.
4. Break at natural beats: the end of an exchange, a revelation, a question \
asked but not yet answered. Never break mid-sentence.
5. Part 1 must establish the premise fast - who these people are and what the \
situation is - because most viewers meet the story there.
6. Every part except the final one must END on something unresolved. That is \
what makes someone look for the next part.
7. Cover the story from its beginning to its resolution. Skip only true \
non-story material at the edges (ads, station idents, unrelated chatter).

For each part provide:
- part: the 1-based part number, in order.
- hook_title: <=60 chars describing THIS part's beat. Do not write "Part 2" - \
numbering is added automatically.
- cliffhanger: one short line naming what this part leaves unresolved. Empty \
string for the final part.
- reason: one sentence on why the break lands where it does.
- caption: 1-2 sentences of social caption for this part.
- hashtags: 5-8 lowercase tags WITHOUT '#'.
- viral_score: integer 1-100 for this part's individual strength. Parts are \
kept regardless of score - this is only for your own ordering notes.

Also provide a `series_title`: one short title for the whole story, reused \
across every part.

Return ONLY valid JSON, no markdown fences, no commentary:
{{"series_title": "...", "clips": [{{"part": 1, "start_time": 12.0, \
"end_time": 64.5, "viral_score": 80, "hook_title": "...", "cliffhanger": "...", \
"reason": "...", "caption": "...", "hashtags": ["...", "..."]}}]}}"""

STORY_USER_PROMPT = """Video title: {title}

Timestamped transcript (seconds):
---
{transcript}
---

Split this into {n} consecutive parts as JSON, covering the story from start to \
finish. Use timestamps from the transcript above."""


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def analyze(
    transcript: dict,
    *,
    title: str = "",
    cache_key: str | None = None,
    count: int | None = None,
    provider: str | None = None,
    strategy: str | None = None,
    style: str | None = None,
    force: bool = False,
) -> list[Clip]:
    """
    strategy:
      "highlights" - best isolated moments, ranked, de-overlapped (default)
      "story"      - chronological contiguous parts covering the whole narrative
      "sequential" - mechanical chop at sentence boundaries, no LLM required
    `count=None` means auto: 5 for highlights, "however many the story needs"
    for story/sequential.
    """
    rules = settings.clips
    strategy = (strategy or rules.strategy).lower()
    provider = (provider or settings.llm.resolved_provider()).lower()

    tone = ""
    if style:
        import styles as _styles
        tone = _styles.get_style(style).tone_prompt

    segments = transcript.get("segments", [])
    words = transcript.get("words", [])
    if not segments:
        log.error("Empty transcript - nothing to analyse.")
        return []

    media_duration = transcript.get("duration", 0.0) or (segments[-1]["end"] if segments else 0)
    auto_count = _auto_count(strategy, media_duration, rules)
    count = count if count and count > 0 else auto_count

    cache_file = None
    if cache_key:
        style_tag = f".{style}" if style and style != "clean" else ""
        cache_file = settings.paths.cache / f"{cache_key}.{strategy}{style_tag}.clips.json"
        if cache_file.exists() and not force:
            cached = read_json(cache_file)
            if cached:
                log.info("Cache hit - %d clips from %s", len(cached), cache_file.name)
                return [Clip(**c) for c in _strip_derived(cached)]

    log.info("Analysing %d segments | strategy=%s | provider=%s | target=%d parts",
             len(segments), strategy, provider, count)

    # --- sequential: purely mechanical, never calls an LLM -----------------
    if strategy == "sequential":
        clips = _sequential_clips(segments, words, media_duration, count)

    # --- story: LLM chooses the break points, then we enforce continuity ---
    elif strategy == "story":
        if provider == "heuristic":
            log.warning("Story mode with no LLM - falling back to sequential chopping. "
                        "Break points will be mechanical rather than dramatic.")
            clips = _sequential_clips(segments, words, media_duration, count)
        else:
            raw, series_title = _llm_candidates(
                segments, title, count, provider, tag=cache_key or "clip",
                story=True, tone=tone)
            if not raw:
                log.warning("LLM returned nothing usable - falling back to sequential.")
                clips = _sequential_clips(segments, words, media_duration, count)
            else:
                clips = _story_postprocess(raw, words, media_duration, count, series_title
                                           or title)

    # --- highlights: original behaviour, untouched -------------------------
    else:
        if provider == "heuristic":
            raw = _heuristic_candidates(segments, words, count * 2)
        else:
            raw, _ = _llm_candidates(segments, title, count, provider,
                                     tag=cache_key or "clip", tone=tone)
            if not raw:
                log.warning("LLM returned nothing usable - falling back to heuristic.")
                raw = _heuristic_candidates(segments, words, count * 2)
        clips = _postprocess(raw, words, media_duration, count)

    if cache_file:
        write_json(cache_file, [c.to_dict() for c in clips])

    if clips and clips[0].is_series:
        total = clips[0].total_parts
        covered = sum(c.duration for c in clips)
        log.info("Series '%s': %d parts, %.0fs of %.0fs source covered (%.0f%%)",
                 clips[0].series_title or title, total, covered, media_duration,
                 100 * covered / max(1.0, media_duration))
    else:
        log.info("Selected %d clips: %s", len(clips),
                 ", ".join(f"{c.viral_score}@{c.start_time:.0f}s" for c in clips))
    return clips


def _auto_count(strategy: str, media_duration: float, rules) -> int:
    """How many parts/clips to aim for when the user did not say."""
    if strategy in ("story", "sequential"):
        # Enough parts to cover the source at the preferred part length.
        est = int(round(media_duration / max(10.0, rules.part_target)))
        return max(2, min(est, 40))     # 40 parts is already an absurd series
    return rules.target_count


def _strip_derived(dicts: list[dict]) -> list[dict]:
    """Remove computed keys so dicts can be re-inflated into Clip()."""
    allowed = set(Clip.__dataclass_fields__.keys())
    return [{k: v for k, v in d.items() if k in allowed} for d in dicts]


# ---------------------------------------------------------------------------
# LLM path
# ---------------------------------------------------------------------------
def _llm_candidates(segments: list[dict], title: str, count: int, provider: str,
                    tag: str = "clip", story: bool = False,
                    tone: str = "") -> tuple[list[dict], str]:
    cfg = settings.llm
    rules = settings.clips
    max_chars = cfg.manual_chunk_chars if provider == "manual" else cfg.chunk_chars
    chunks = _chunk_transcript(segments, max_chars, cfg.chunk_overlap_chars)
    log.info("Transcript split into %d LLM window(s)", len(chunks))

    if story and len(chunks) > 1:
        log.warning(
            "Story mode across %d windows: the model sees each window in "
            "isolation, so break points near the seams may be weaker. Raise "
            "%s to fit the transcript in one window if you can.",
            len(chunks),
            "LLM_MANUAL_CHUNK_CHARS" if provider == "manual" else "LLM_CHUNK_CHARS",
        )

    if story:
        system = STORY_SYSTEM_PROMPT.format(
            min_s=rules.min_duration, max_s=rules.max_duration,
            target_s=min(rules.part_target, rules.max_duration))
        user_tpl = STORY_USER_PROMPT
        # Parts are local to their window, so split the target across windows.
        per_chunk = max(2, round(count / len(chunks)))
    else:
        system = SYSTEM_PROMPT.format(
            min_s=rules.min_duration, max_s=rules.max_duration)
        user_tpl = USER_PROMPT
        per_chunk = max(3, min(count, 5)) if len(chunks) == 1 else max(2, count // 2 + 1)

    if tone:
        system += f"\n\nVOICE AND TONE for every title, caption and hashtag:\n{tone}"

    results: list[dict] = []
    series_title = ""
    pending_manual = 0

    for i, chunk in enumerate(chunks, 1):
        prompt = user_tpl.format(title=title or "Untitled", transcript=chunk, n=per_chunk)
        log.info("  window %d/%d (%d chars)", i, len(chunks), len(chunk))
        try:
            text = _call_llm(provider, system, prompt, tag=f"{tag}_w{i:02d}")
            if text is _EXPORTED:            # manual mode, nothing to parse yet
                pending_manual += 1
                continue
            parsed, found_title = _parse_json(text, want_series_title=True)
            results.extend(parsed)
            series_title = series_title or found_title
        except Exception as exc:  # noqa: BLE001 - never let one window kill the run
            log.warning("  window %d failed: %s", i, exc)
        if i < len(chunks) and provider != "manual":
            time.sleep(1.2)  # stay comfortably inside free-tier RPM limits

    if pending_manual:
        raise ManualPromptsExported(pending_manual)
    return results, series_title


class ManualPromptsExported(RuntimeError):
    """Raised when --export-prompts wrote prompt files and there is nothing
    more to do until the user comes back with answers."""

    def __init__(self, count: int):
        super().__init__(f"{count} prompt file(s) exported - awaiting your answers")
        self.count = count


_EXPORTED = object()   # sentinel


def _call_llm(provider: str, system: str, prompt: str, tag: str = "clip"):
    cfg = settings.llm
    last_exc: Exception | None = None

    # Manual relay is human-paced: no retries, no backoff, no timeout.
    if provider == "manual":
        return _call_manual(system, prompt, tag)

    for attempt in range(1, cfg.max_retries + 1):
        try:
            if provider == "gemini":
                return _call_gemini(system, prompt)
            if provider == "groq":
                return _call_groq(system, prompt)
            if provider in OPENAI_PRESETS:
                return _call_openai_compatible(provider, system, prompt)
            raise ValueError(f"Unknown provider '{provider}'")
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            msg = str(exc).lower()
            # Back off harder on rate limits, which free tiers hand out freely.
            wait = 20 * attempt if ("429" in msg or "rate" in msg or "quota" in msg) else 3 * attempt
            if attempt < cfg.max_retries:
                log.warning("  LLM attempt %d failed (%s) - retrying in %ds", attempt, exc, wait)
                time.sleep(wait)
    raise RuntimeError(f"LLM failed after {cfg.max_retries} attempts: {last_exc}")


def _call_gemini(system: str, prompt: str) -> str:
    from google import genai
    from google.genai import types

    cfg = settings.llm
    client = genai.Client(api_key=cfg.gemini_api_key)
    resp = client.models.generate_content(
        model=cfg.gemini_model,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system,
            temperature=cfg.temperature,
            response_mime_type="application/json",
            max_output_tokens=8192,
        ),
    )
    return resp.text or ""


def _call_groq(system: str, prompt: str) -> str:
    from groq import Groq

    cfg = settings.llm
    client = Groq(api_key=cfg.groq_api_key)
    resp = client.chat.completions.create(
        model=cfg.groq_model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        temperature=cfg.temperature,
        response_format={"type": "json_object"},
        max_tokens=8000,
    )
    return resp.choices[0].message.content or ""


def _call_openai_compatible(provider: str, system: str, prompt: str) -> str:
    """
    One code path for every OpenAI-compatible endpoint: NVIDIA NIM, OpenRouter,
    Cerebras, Together, and local servers (Ollama / LM Studio).

    Not every backend honours `response_format=json_object`, so we ask for it,
    and silently retry without it if the server rejects the parameter.
    """
    from openai import OpenAI

    cfg = settings.llm
    base_url, model, api_key = cfg.openai_settings(provider)
    if not base_url:
        raise RuntimeError(
            f"No base_url for provider '{provider}'. Set OPENAI_BASE_URL in .env."
        )

    client = OpenAI(base_url=base_url, api_key=api_key, timeout=180.0)
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]
    kwargs = dict(model=model, messages=messages,
                  temperature=cfg.temperature, max_tokens=8000)

    try:
        resp = client.chat.completions.create(
            response_format={"type": "json_object"}, **kwargs)
    except Exception as exc:  # noqa: BLE001
        if "response_format" not in str(exc).lower():
            raise
        log.debug("%s rejected response_format - retrying without it", provider)
        resp = client.chat.completions.create(**kwargs)

    return resp.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# Manual relay: use claude.ai (or any chat UI you already pay for) as the model
# ---------------------------------------------------------------------------
_MANUAL_BANNER = """
{bar}
  MANUAL STEP {tag}
{bar}
  1. Open the prompt file below and copy ALL of it:

       {prompt_path}

  2. Paste it into claude.ai (or ChatGPT, Gemini, whatever you have).
  3. Bring the answer back, either way:

       [A]  Save the reply to:  {resp_path}
            ...then press ENTER here.

       [B]  Type  paste  and press ENTER, then paste the JSON directly and
            finish with a line containing only  END

{bar}"""


def _call_manual(system: str, prompt: str, tag: str) -> str | object:
    """
    Write the full prompt to a file, wait for the human to bring back the
    model's JSON. Costs nothing, uses a subscription you already have, and
    gets you a frontier model on the one step where model quality matters.
    """
    cache = settings.paths.cache
    prompt_path = cache / f"prompt_{tag}.md"
    resp_path = cache / f"response_{tag}.json"

    prompt_path.write_text(
        f"<!-- Paste this whole file into claude.ai. Reply with JSON only. -->\n\n"
        f"## Instructions\n\n{system}\n\n---\n\n## Task\n\n{prompt}\n",
        encoding="utf-8",
    )

    # Already answered on a previous run? Just use it.
    if resp_path.exists() and resp_path.stat().st_size > 2:
        log.info("  using saved answer %s", resp_path.name)
        return resp_path.read_text(encoding="utf-8")

    if settings.llm.manual_export_only:
        log.info("  prompt exported -> %s", prompt_path)
        return _EXPORTED

    bar = "=" * 74
    print(_MANUAL_BANNER.format(bar=bar, tag=tag, prompt_path=prompt_path,
                                resp_path=resp_path))

    while True:
        choice = input(">>> ENTER when saved, or type 'paste' / 'skip': ").strip().lower()

        if choice == "skip":
            log.warning("  window %s skipped by user", tag)
            return "{}"

        if choice == "paste":
            print("--- paste the JSON, then a line containing only END ---")
            lines: list[str] = []
            while True:
                try:
                    line = input()
                except EOFError:
                    break
                if line.strip() == "END":
                    break
                lines.append(line)
            text = "\n".join(lines).strip()
            if text:
                resp_path.write_text(text, encoding="utf-8")   # cached for reruns
                return text
            print("Nothing captured - try again.")
            continue

        if resp_path.exists() and resp_path.stat().st_size > 2:
            return resp_path.read_text(encoding="utf-8")
        print(f"Could not find {resp_path.name} (or it is empty). Save it and press ENTER.")


_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.MULTILINE)


def _parse_json(text: str, want_series_title: bool = False):
    """Tolerant JSON extraction - models occasionally add fences or prose."""
    empty = ([], "") if want_series_title else []
    if not text:
        return empty
    cleaned = _FENCE_RE.sub("", text).strip()

    data = None
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        # Grab the outermost {...} or [...] and retry.
        for opener, closer in (("{", "}"), ("[", "]")):
            i, j = cleaned.find(opener), cleaned.rfind(closer)
            if i != -1 and j > i:
                try:
                    data = json.loads(cleaned[i:j + 1])
                    break
                except json.JSONDecodeError:
                    continue
    if data is None:
        log.warning("Could not parse LLM output as JSON (first 200 chars): %s", cleaned[:200])
        return empty

    series_title = ""
    clips: list[dict] = []

    if isinstance(data, dict):
        series_title = str(data.get("series_title") or data.get("title") or "").strip()
        for key in ("clips", "parts", "results", "highlights", "segments", "data"):
            if isinstance(data.get(key), list):
                clips = [c for c in data[key] if isinstance(c, dict)]
                break
        else:
            clips = [data] if "start_time" in data else []
    elif isinstance(data, list):
        clips = [c for c in data if isinstance(c, dict)]

    return (clips, series_title) if want_series_title else clips


def _chunk_transcript(segments: list[dict], max_chars: int, overlap_chars: int) -> list[str]:
    full = transcript_lines(segments)
    if len(full) <= max_chars:
        return [full]

    lines = full.split("\n")
    chunks, buf, size = [], [], 0
    for line in lines:
        if size + len(line) > max_chars and buf:
            chunks.append("\n".join(buf))
            # keep a tail of the previous window so clips near the seam survive
            tail, tail_size = [], 0
            for prev in reversed(buf):
                if tail_size + len(prev) > overlap_chars:
                    break
                tail.insert(0, prev)
                tail_size += len(prev)
            buf, size = tail, tail_size
        buf.append(line)
        size += len(line) + 1
    if buf:
        chunks.append("\n".join(buf))
    return chunks


# ---------------------------------------------------------------------------
# Heuristic fallback (no API key required)
# ---------------------------------------------------------------------------
HOOK_WORDS = {
    "secret", "mistake", "never", "always", "nobody", "everyone", "truth",
    "worst", "best", "biggest", "actually", "realize", "realized", "shocked",
    "crazy", "insane", "problem", "why", "how", "reason", "wrong", "failed",
    "million", "billion", "thousand", "percent", "money", "free", "stop",
    "warning", "proof", "learned", "changed", "hate", "love", "first", "last",
}


def _heuristic_candidates(segments: list[dict], words: list[dict], want: int) -> list[dict]:
    """Sliding-window scoring: hook vocabulary + questions + numbers + pacing."""
    rules = settings.clips
    target = (rules.min_duration + rules.max_duration) / 2
    if not segments:
        return []

    candidates = []
    for i, seg in enumerate(segments):
        start = seg["start"]
        texts, j = [], i
        # Grow the window one segment at a time and score EVERY valid length,
        # so short punchy clips can beat long rambling ones.
        while j < len(segments) and segments[j]["end"] - start <= rules.max_duration:
            texts.append(segments[j]["text"])
            end = segments[j]["end"]
            j += 1
            if end - start < rules.min_duration:
                continue

            text = " ".join(texts)
            low = text.lower()
            tokens = re.findall(r"[a-z']+", low)
            if not tokens:
                continue

            hook_hits = sum(1 for t in tokens if t in HOOK_WORDS)
            questions = low.count("?")
            numbers = len(re.findall(r"\b\d+([.,]\d+)?\b", text))
            wps = len(tokens) / max(1.0, end - start)      # speech density
            opener_bonus = 12 if any(t in HOOK_WORDS for t in tokens[:12]) else 0
            # Reward density rather than raw totals, else longer always wins.
            density = (hook_hits * 6 + questions * 8 + numbers * 3) / max(1.0, (end - start) / 30.0)
            length_penalty = abs((end - start) - target) * 0.9
            closure_bonus = 8 if text.rstrip().endswith((".", "!", "?")) else 0

            score = density + min(wps, 4.0) * 6 + opener_bonus + closure_bonus - length_penalty
            candidates.append({
                "start_time": start,
                "end_time": end,
                "viral_score": int(max(1, min(95, 38 + score * 0.7))),
                "hook_title": _title_from_text(text),
                "reason": "Heuristic pick: high hook-word density and pacing.",
                "caption": text[:180].strip(),
                "hashtags": [],
                "_raw_text": text,
            })

    # Keep a deep pool: _postprocess drops anything overlapping an accepted
    # clip, and a shallow pool would leave it with nothing to fall back on.
    candidates.sort(key=lambda c: c["viral_score"], reverse=True)
    return candidates[: max(want * 40, 400)]


def _title_from_text(text: str, max_len: int = 60) -> str:
    sentence = re.split(r"(?<=[.!?])\s+", text.strip())[0]
    sentence = re.sub(r"\s+", " ", sentence).strip(" -,")
    if len(sentence) > max_len:
        sentence = sentence[:max_len].rsplit(" ", 1)[0] + "..."
    return sentence or "Untitled clip"


# ---------------------------------------------------------------------------
# Story mode: enforce chronology, continuity and part numbering
# ---------------------------------------------------------------------------
def _story_postprocess(raw: list[dict], words: list[dict], media_duration: float,
                       count: int, series_title: str) -> list[Clip]:
    """
    The LLM chooses where the story breaks. This function makes those breaks
    physically valid: chronological, non-overlapping, gap-free within tolerance,
    and inside the duration bounds. Unlike highlights mode, nothing is dropped
    for being unexciting - dropping a part would put a hole in the story.
    """
    rules = settings.clips
    prepared: list[Clip] = []

    for item in raw:
        clip = _clip_from_item(item, words, media_duration, trim_long=True)
        if clip is None:
            continue
        clip.cliffhanger = str(item.get("cliffhanger") or "").strip()
        try:
            clip.part = int(item.get("part") or 0)
        except (TypeError, ValueError):
            clip.part = 0
        prepared.append(clip)

    if not prepared:
        return []

    # Chronological order is the whole point - trust timestamps over the
    # model's own `part` numbers, which drift when a transcript is windowed.
    prepared.sort(key=lambda c: c.start_time)

    # Drop near-duplicate starts (happens at chunk seams, where two windows
    # both propose a part beginning at the same beat).
    deduped: list[Clip] = []
    for clip in prepared:
        if deduped and abs(clip.start_time - deduped[-1].start_time) < 2.0:
            # keep whichever is longer, it usually has the better break point
            if clip.duration > deduped[-1].duration:
                deduped[-1] = clip
            continue
        deduped.append(clip)

    # Enforce continuity: close small gaps, resolve overlaps.
    stitched: list[Clip] = []
    for clip in deduped:
        if stitched:
            prev = stitched[-1]
            if clip.start_time < prev.end_time:          # overlap -> cut prev short
                prev.end_time = round(clip.start_time, 3)
            else:
                gap = clip.start_time - prev.end_time
                if 0 < gap <= rules.part_gap_tolerance:  # small gap -> close it
                    prev.end_time = round(clip.start_time, 3)
            if prev.duration < rules.min_duration * rules.part_min_ratio:
                # Adjustment left a stub: absorb it into this part instead.
                clip.start_time = prev.start_time
                clip.transcript = plain_text(words, clip.start_time, clip.end_time)
                stitched.pop()
        if clip.duration > rules.max_duration:
            clip.end_time = round(clip.start_time + rules.max_duration, 3)
        stitched.append(clip)

    # Final tidy: enforce bounds, renumber, refresh transcripts.
    parts = [c for c in stitched
             if c.duration >= rules.min_duration * rules.part_min_ratio]
    if count and len(parts) > count * 2:
        log.warning("Model proposed %d parts for a target of %d - keeping all of "
                    "them, the story matters more than the count.", len(parts), count)

    total = len(parts)
    for i, clip in enumerate(parts, 1):
        clip.index = i
        clip.part = i
        clip.total_parts = total
        clip.series_title = series_title or clip.series_title
        clip.transcript = plain_text(words, clip.start_time, clip.end_time) or clip.transcript
    return parts


# ---------------------------------------------------------------------------
# Sequential mode: no LLM, guaranteed full coverage
# ---------------------------------------------------------------------------
def _sequential_clips(segments: list[dict], words: list[dict],
                      media_duration: float, count: int) -> list[Clip]:
    """
    Walk the transcript start to finish and break at the first sentence end
    past the target length. Covers 100% of the spoken content, costs nothing,
    and needs no model. The break points are mechanical rather than dramatic -
    that is the trade-off against story mode.
    """
    rules = settings.clips
    target = min(rules.part_target, rules.max_duration)
    if not segments:
        return []

    parts: list[Clip] = []
    start = segments[0]["start"]
    texts: list[str] = []

    def emit(end: float, body: list[str]) -> None:
        if end - start < rules.min_duration * rules.part_min_ratio:
            return
        text = " ".join(body).strip()
        parts.append(Clip(
            start_time=round(start, 3),
            end_time=round(end, 3),
            viral_score=50,
            hook_title=_title_from_text(text),
            reason="Sequential split at a sentence boundary.",
            caption=text[:180].strip(),
            hashtags=[],
            transcript=plain_text(words, start, end) or text,
        ))

    for seg in segments:
        texts.append(seg["text"])
        elapsed = seg["end"] - start
        ends_sentence = seg["text"].rstrip().endswith((".", "!", "?"))
        if (elapsed >= target and ends_sentence) or elapsed >= rules.max_duration:
            emit(seg["end"], texts)
            start, texts = seg["end"], []

    if texts:
        last_end = min(segments[-1]["end"], media_duration or segments[-1]["end"])
        if last_end - start >= rules.min_duration * rules.part_min_ratio:
            emit(last_end, texts)
        elif parts:
            # Tail too short to stand alone - glue it onto the previous part.
            parts[-1].end_time = round(min(last_end, parts[-1].start_time
                                           + rules.max_duration), 3)
            parts[-1].transcript = plain_text(words, parts[-1].start_time,
                                              parts[-1].end_time)

    total = len(parts)
    for i, clip in enumerate(parts, 1):
        clip.index = i
        clip.part = i
        clip.total_parts = total
        if i < total:
            clip.cliffhanger = "Continues in the next part."
    return parts


def _clip_from_item(item: dict, words: list[dict], media_duration: float,
                    trim_long: bool = True) -> Clip | None:
    """Shared validation/snapping used by both highlights and story modes."""
    rules = settings.clips
    try:
        start = float(item.get("start_time", item.get("start", -1)))
        end = float(item.get("end_time", item.get("end", -1)))
    except (TypeError, ValueError):
        return None
    if start < 0 or end <= start:
        return None

    if trim_long and end - start > rules.max_duration:
        end = start + rules.max_duration
    if end - start < rules.min_duration * rules.part_min_ratio:
        return None

    start, end = snap_to_words(words, start, end, rules.lead_in, rules.lead_out)
    if media_duration:
        end = min(end, media_duration - 0.05)
    if end <= start:
        return None
    if end - start > rules.max_duration:
        end = start + rules.max_duration

    tags = item.get("hashtags") or []
    if isinstance(tags, str):
        tags = [t for t in re.split(r"[,\s]+", tags) if t]
    tags = [str(t).lstrip("#").strip().lower() for t in tags if str(t).strip()]

    score = item.get("viral_score", 50)
    try:
        score = int(round(float(score)))
    except (TypeError, ValueError):
        score = 50

    return Clip(
        start_time=round(start, 3),
        end_time=round(end, 3),
        viral_score=max(1, min(100, score)),
        hook_title=str(item.get("hook_title") or "Untitled clip").strip()[:100],
        reason=str(item.get("reason") or "").strip(),
        caption=str(item.get("caption") or "").strip(),
        hashtags=tags[:12],
        transcript=plain_text(words, start, end) or str(item.get("_raw_text", "")).strip(),
    )


# ---------------------------------------------------------------------------
# Post-processing: validate, snap, de-overlap, rank
# ---------------------------------------------------------------------------
def _postprocess(raw: list[dict], words: list[dict], media_duration: float, count: int) -> list[Clip]:
    rules = settings.clips
    prepared: list[Clip] = []

    for item in raw:
        clip = _clip_from_item(item, words, media_duration, trim_long=True)
        if clip is None or clip.duration < rules.min_duration * 0.85:
            continue
        prepared.append(clip)

    prepared.sort(key=lambda c: c.viral_score, reverse=True)

    accepted: list[Clip] = []
    for clip in prepared:
        if any(_overlap_ratio(clip, other) > rules.max_overlap_ratio for other in accepted):
            continue
        accepted.append(clip)
        if len(accepted) >= count:
            break

    accepted.sort(key=lambda c: c.start_time)   # chronological, nicer to review
    for i, clip in enumerate(accepted, 1):
        clip.index = i
    return accepted


def _overlap_ratio(a: Clip, b: Clip) -> float:
    overlap = min(a.end_time, b.end_time) - max(a.start_time, b.start_time)
    if overlap <= 0:
        return 0.0
    return overlap / min(a.duration, b.duration)