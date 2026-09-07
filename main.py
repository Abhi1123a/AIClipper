#!/usr/bin/env python3
"""
main.py
=======
One-command pipeline:

    python main.py --url https://www.youtube.com/watch?v=XXXX
    python main.py --file ./podcast.mp4 --clips 5 --mode face
    python main.py --url <URL> --publish youtube --dry-run

Stages: download -> audio -> transcribe -> analyse -> cut -> reframe ->
        caption -> package -> (optional) publish

Every expensive stage is cached, so re-running after a crash or tweaking
caption styling does not re-download or re-transcribe anything.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from config import settings
from utils import setup_logging, hhmmss

log = logging.getLogger("main")

BANNER = r"""
   _   ___   __   ___ _ _
  /_\ |_ _| / _| / __| (_)_ __ _ __  ___ _ _
 / _ \ | | | (_ | (__| | | '_ \ '_ \/ -_) '_|
/_/ \_\___| \___|\___|_|_| .__/ .__/\___|_|
                         |_|  |_|   long video -> vertical shorts
"""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="main.py",
        description="Turn long videos into captioned 9:16 short clips.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python main.py --url https://youtu.be/XXXX\n"
            "  python main.py --file talk.mp4 --clips 3 --mode blur --whisper-model medium\n"
            "  python main.py --url https://youtu.be/XXXX --publish youtube tiktok\n"
            "  python main.py --login tiktok       # one-time browser session setup\n"
        ),
    )
    src = p.add_mutually_exclusive_group()
    src.add_argument("--url", help="YouTube (or any yt-dlp supported) URL")
    src.add_argument("--file", help="Path to a local video file")

    p.add_argument("--strategy", choices=["highlights", "story", "sequential"],
                   default=settings.clips.strategy,
                   help="highlights = best isolated moments (default); "
                        "story = chronological parts covering the whole narrative; "
                        "sequential = mechanical chop, no LLM needed")
    p.add_argument("--clips", default=str(settings.clips.target_count),
                   help="How many clips/parts, or 'auto' to let story mode cover "
                        f"the whole source (default {settings.clips.target_count})")
    p.add_argument("--min-duration", type=float, default=settings.clips.min_duration)
    p.add_argument("--max-duration", type=float, default=settings.clips.max_duration)
    p.add_argument("--mode", choices=["blur", "crop", "face"],
                   default=settings.render.reframe_mode,
                   help="9:16 reframing strategy (default %(default)s)")
    p.add_argument("--provider", default=None,
                   choices=["gemini", "groq", "manual", "heuristic", "nim",
                            "openrouter", "cerebras", "together", "ollama",
                            "lmstudio", "openai"],
                   help="Highlight-detection backend. 'manual' relays the prompt "
                        "through claude.ai by hand; 'ollama'/'lmstudio' run locally")
    p.add_argument("--export-prompts", action="store_true",
                   help="With --provider manual: write the prompt files and exit. "
                        "Re-run the same command once you have saved the answers.")
    p.add_argument("--whisper-model", default=settings.whisper.model_size,
                   help="tiny|base|small|medium|large-v3|distil-large-v3")
    p.add_argument("--whisper-device", choices=["auto", "cpu", "cuda"],
                   default=settings.whisper.device,
                   help="Force transcription device. Use 'cpu' if CUDA libs are missing")
    p.add_argument("--language", default=None,
                   help="Force transcription language (e.g. en). Default: auto-detect")
    p.add_argument("--max-height", type=int, default=1080,
                   help="Max source download height (default 1080)")

    p.add_argument("--style", default="auto",
                   choices=["auto", "humor", "motivational", "story",
                            "brainrot", "clean"],
                   help="Editing style: typography, pacing, grade and caption "
                        "rhythm. 'auto' picks from the clip strategy (default)")
    p.add_argument("--no-style-variation", action="store_true",
                   help="Render every clip with identical styling. Off by "
                        "default: varied output avoids looking mass-produced")
    p.add_argument("--platform", choices=["tiktok", "reels", "shorts", "universal"],
                   default="universal",
                   help="Place captions clear of that platform's UI safe zone. "
                        "'universal' uses the strictest margin from all three "
                        "(default) so one file is safe everywhere")
    p.add_argument("--tighten", action="store_true",
                   help="Remove dead air and long pauses; caption timings are "
                        "remapped to match. Typically cuts 10-30%% of runtime")
    p.add_argument("--no-audit", action="store_true",
                   help="Skip the pre-flight check for captions-under-UI, "
                        "duration and filler openers")
    p.add_argument("--no-captions", action="store_true", help="Skip burned-in captions")
    p.add_argument("--no-thumbnail", action="store_true", help="Skip thumbnail extraction")
    p.add_argument("--force", action="store_true",
                   help="Ignore transcript/analysis caches and redo everything")

    p.add_argument("--publish", nargs="*", choices=["youtube", "tiktok", "instagram"],
                   default=[], help="Publish after rendering")
    p.add_argument("--login", choices=["tiktok", "instagram", "youtube_studio"],
                   help="Open a browser to save a login session, then exit")

    p.add_argument("--dry-run", action="store_true",
                   help="Analyse and print the clip plan; render nothing")
    p.add_argument("--log-level", default=settings.log_level,
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args(argv)


def apply_overrides(args: argparse.Namespace) -> None:
    args.clips = 0 if str(args.clips).lower() == "auto" else int(args.clips)
    if args.clips:
        settings.clips.target_count = args.clips
    settings.clips.strategy = args.strategy
    settings.clips.min_duration = args.min_duration
    settings.clips.max_duration = args.max_duration
    settings.render.reframe_mode = args.mode
    settings.whisper.model_size = args.whisper_model
    settings.whisper.device = args.whisper_device
    if args.language:
        settings.whisper.language = args.language
    if args.provider:
        settings.llm.provider = args.provider
    if args.export_prompts:
        settings.llm.provider = "manual"
        settings.llm.manual_export_only = True


def preflight() -> bool:
    problems = settings.validate()
    fatal = [p for p in problems if "FFmpeg" in p or "ffprobe" in p]
    for msg in problems:
        (log.error if msg in fatal else log.warning)(msg)
    return not fatal


def print_plan(clips) -> None:
    if not clips:
        return
    series = clips[0].is_series
    print("\n" + "-" * 78)
    if series:
        title = clips[0].series_title or "Series"
        print(f"  SERIES: {title}  ({clips[0].total_parts} parts, post in order)")
        print("-" * 78)
        print(f"{'PART':<6}{'START':<10}{'DUR':<8}{'GAP':<7}BEAT")
    else:
        print(f"{'#':<3}{'SCORE':<7}{'START':<10}{'DUR':<7}HOOK")
    print("-" * 78)

    prev_end = None
    for c in clips:
        if series:
            gap = "-" if prev_end is None else f"{c.start_time - prev_end:+.1f}s"
            print(f"{str(c.part) + '/' + str(c.total_parts):<6}{hhmmss(c.start_time):<10}"
                  f"{c.duration:>5.1f}s  {gap:<7}{c.hook_title[:38]}")
            prev_end = c.end_time
        else:
            print(f"{c.index:<3}{c.viral_score:<7}{hhmmss(c.start_time):<10}"
                  f"{c.duration:>5.1f}s  {c.hook_title[:44]}")

    if series:
        covered = sum(c.duration for c in clips)
        span = clips[-1].end_time - clips[0].start_time
        print("-" * 78)
        print(f"  Covers {covered:.0f}s across a {span:.0f}s span "
              f"({100 * covered / max(1.0, span):.0f}% - GAP column shows any skips)")
    print("-" * 78 + "\n")


def run(args: argparse.Namespace) -> int:
    # Imports are deferred so `--help` works without heavy deps installed.
    import downloader
    import transcriber
    from ai_analyzer import ManualPromptsExported, analyze
    from transcriber import words_in_range
    from video_processor import process_clip
    import publisher
    import optimizer
    import styles

    t_start = time.time()

    # --- 1. Ingest ---------------------------------------------------------
    source_arg = args.url or args.file
    log.info("[1/6] Ingesting %s", source_arg)
    source = downloader.fetch(source_arg, max_height=args.max_height)
    log.info("      '%s' | %s | %dx%d @ %.2ffps",
             source.title, hhmmss(source.duration), source.width, source.height, source.fps)

    # --- 2. Audio ----------------------------------------------------------
    log.info("[2/6] Extracting audio")
    audio = downloader.extract_audio(source, force=args.force)

    # --- 3. Transcribe -----------------------------------------------------
    log.info("[3/6] Transcribing (word-level timestamps)")
    transcript = transcriber.transcribe(
        audio,
        cache_key=source.video_id,
        language=args.language,
        model_size=args.whisper_model,
        force=args.force,
    )
    words = transcript["words"]
    if not words:
        log.error("No speech detected - nothing to clip.")
        return 2

    # --- 4. Analyse --------------------------------------------------------
    import styles as _styles
    resolved_style = (_styles.STRATEGY_DEFAULT_STYLE.get(args.strategy, "clean")
                      if args.style == "auto" else args.style)

    log.info("[4/6] Finding highlights")
    try:
        clips = analyze(
            transcript,
            title=source.title,
            cache_key=f"{source.video_id}-{args.strategy}-{args.clips}",
            count=args.clips,
            provider=settings.llm.provider if args.provider or args.export_prompts else None,
            strategy=args.strategy,
            style=resolved_style,
            force=args.force,
        )
    except ManualPromptsExported as exc:
        print(f"\n  {exc.count} prompt file(s) written to {settings.paths.cache}")
        print("  Paste each into claude.ai, save the JSON replies next to them as")
        print("  response_<same-tag>.json, then re-run this exact command without")
        print("  --export-prompts.\n")
        return 0
    if not clips:
        log.error("No clips selected. Try --clips 3, a longer video, or --provider heuristic.")
        return 3

    print_plan(clips)

    # --- 5. Render ---------------------------------------------------------
    profile = optimizer.get_profile(args.platform)
    optimizer.apply_profile_to_captions(profile)

    if args.style == "auto":
        log.info("Style auto-selected: %s (from --strategy %s)",
                 resolved_style, args.strategy)
    base_style = styles.get_style(resolved_style)
    log.info("Style: %s | %s", base_style.label, styles.describe(base_style))

    if not args.no_audit:
        pre = [optimizer.audit_clip(c, profile, source_height=source.height,
                                    words=words_in_range(words, c.start_time, c.end_time))
               for c in clips]
        optimizer.print_audit(pre)
        blocking = [a for a in pre if not a.ok]
        if blocking and not args.dry_run:
            log.warning("%d clip(s) have blocking issues - rendering anyway, but "
                        "read the audit above.", len(blocking))

    if args.dry_run:
        log.info("Dry run - stopping before render.")
        return 0

    log.info("[5/6] Rendering %d clip(s) | mode=%s | platform=%s | style=%s",
             len(clips), args.mode, profile.label, base_style.label)
    work_dir = settings.paths.work / source.video_id
    bundles: list[publisher.ClipBundle] = []
    rendered_clips = []
    renders = []

    for clip in clips:
        log.info("  clip %d/%d  score=%d  %s -> %s  '%s'",
                 clip.index, len(clips), clip.viral_score,
                 hhmmss(clip.start_time), hhmmss(clip.end_time), clip.hook_title)
        clip_style = base_style
        if not args.no_style_variation:
            seed = styles.variation_seed(source.video_id, clip.index,
                                         clip.hook_title)
            clip_style = styles.vary(base_style, seed)
        styles.apply_style(clip_style, settings, profile)

        try:
            result = process_clip(
                source=source.path,
                clip=clip,
                words=words_in_range(words, clip.start_time - 0.5, clip.end_time + 0.5),
                out_video=work_dir / f"clip_{clip.index:02d}.mp4",
                work_dir=work_dir,
                mode=args.mode,
                burn_captions=not args.no_captions,
                thumbnail=not args.no_thumbnail,
                tighten=args.tighten,
                style=clip_style,
            )
            bundles.append(publisher.organize(clip, source, result, ass_path=result.ass_path))
            rendered_clips.append(clip)
            renders.append(result)
        except Exception as exc:  # noqa: BLE001 - one bad clip must not kill the run
            log.exception("  clip %d failed: %s", clip.index, exc)

    if not bundles:
        log.error("Every clip failed to render. Run with --log-level DEBUG for FFmpeg output.")
        return 4

    summary = publisher.write_run_summary(bundles, source, rendered_clips)

    if not args.no_audit:
        post = [optimizer.audit_clip(c, profile, source_height=source.height,
                                     render=r,
                                     words=words_in_range(words, c.start_time, c.end_time))
                for c, r in zip(rendered_clips, renders)]
        from utils import write_json
        write_json(bundles[0].directory.parent / "audit.json",
                   {"platform": profile.key,
                    "profile": {"blocked_bottom": profile.blocked_bottom,
                                "caption_margin_v": profile.caption_margin_v,
                                "sweet_spot": [profile.duration_sweet_min,
                                               profile.duration_sweet_max],
                                "notes": profile.notes},
                    "clips": [a.to_dict() for a in post]})

    # --- 6. Publish --------------------------------------------------------
    if args.publish:
        log.info("[6/6] Publishing to: %s", ", ".join(args.publish))
        publisher.publish(bundles, args.publish, dry_run=args.dry_run)
    else:
        log.info("[6/6] Publishing skipped (use --publish youtube tiktok instagram)")

    elapsed = time.time() - t_start
    print("\n" + "=" * 78)
    print(f"  DONE - {len(bundles)} clip(s) in {elapsed / 60:.1f} min")
    print(f"  Output : {bundles[0].directory.parent}")
    print(f"  Summary: {summary}")
    print("=" * 78 + "\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging(args.log_level)
    print(BANNER)

    if args.login:
        import publisher
        publisher.browser_login(args.login)
        return 0

    if not (args.url or args.file):
        log.error("Provide --url <YOUTUBE_URL> or --file <path.mp4>  (see --help)")
        return 1

    apply_overrides(args)
    if not preflight():
        return 1

    try:
        return run(args)
    except KeyboardInterrupt:
        log.warning("Interrupted. Caches are intact - re-run to resume.")
        return 130
    except Exception as exc:  # noqa: BLE001
        log.exception("Pipeline failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())