"""
publisher.py
============
Two jobs:

  1. ORGANISE  - lay out /output/<video_id>/clip_01_<slug>/ with the MP4,
     thumbnail, .ass, transcript, ready-to-paste caption .txt files, and a
     `metadata.json` carrying per-platform titles/descriptions/hashtags.
     This half has no dependencies beyond the stdlib and always runs.

  2. PUBLISH   - optional.
       * YouTube Shorts via the official Data API v3 (free quota: 10,000
         units/day; an upload costs 1,600 -> ~6 uploads/day). Fully automated
         and fully within Google's terms.
       * TikTok / Instagram have no free public upload API for personal
         accounts, so those use Playwright in SEMI-AUTOMATIC mode: the browser
         opens with your existing logged-in session, the file is attached and
         the caption is filled in, then it STOPS and waits for you to review
         and press Post yourself.

  ⚠️  Read `PLATFORM_NOTES` below before enabling browser automation. Fully
      unattended posting to TikTok/Instagram may violate their Terms of
      Service and can get an account rate-limited or banned. The default
      (`PUBLISH_SEMI_AUTO=true`) keeps a human in the loop on purpose.
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from config import settings
from utils import slugify, write_json

log = logging.getLogger("publisher")

PLATFORM_NOTES = """\
YouTube  : official Data API v3, free quota. Shorts = vertical AND <=180s.
           A '#Shorts' tag in the title/description helps classification.
Instagram: Graph API publishing requires a Business/Creator account linked to
           a Facebook Page. Personal accounts have no free upload API ->
           semi-automated browser flow only.
TikTok   : the Content Posting API requires an approved developer app; direct
           post is gated. Semi-automated browser flow only for personal use.
"""

# Platform hard limits (chars)
LIMITS = {
    "youtube": {"title": 100, "description": 5000, "tags": 15},
    "instagram": {"caption": 2200, "hashtags": 30},
    "tiktok": {"caption": 2200, "hashtags": 20},
}


@dataclass
class ClipBundle:
    directory: Path
    video: Path
    metadata: Path
    thumbnail: Path | None = None


# ===========================================================================
# 1. Organise + metadata
# ===========================================================================
def _tags(clip, extra: list[str]) -> list[str]:
    seen, out = set(), []
    for t in list(clip.hashtags) + list(extra) + list(settings.publish.default_hashtags):
        t = str(t).lstrip("#").strip().lower().replace(" ", "")
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return out


def _hash_str(tags: list[str], limit: int) -> str:
    return " ".join(f"#{t}" for t in tags[:limit])


def build_metadata(clip, source, render, video_rel: str) -> dict:
    """Per-platform, length-checked, copy-paste-ready metadata."""
    title = (clip.hook_title or "Untitled").strip().rstrip(".")
    caption = (clip.caption or title).strip()

    # --- series handling -----------------------------------------------------
    is_series = getattr(clip, "is_series", False)
    part, total = getattr(clip, "part", 0), getattr(clip, "total_parts", 0)
    series_title = (getattr(clip, "series_title", "") or source.title).strip()
    is_final = is_series and part >= total

    if is_series:
        # Part number goes first: it is the single most useful thing a scroller
        # can see, and it is what makes them look for the rest.
        base = f"{series_title} (Part {part}/{total})"
        title = f"Part {part}: {title}"
        if is_final:
            tail = "Thanks for watching the whole story."
        else:
            tail = f"Part {part + 1} is up next - follow so you don't miss it."
        cliff = (getattr(clip, "cliffhanger", "") or "").strip()
        caption = " ".join(x for x in (f"Part {part}/{total}.", caption, cliff, tail) if x)
    else:
        base = title

    yt_tags = _tags(clip, ["shorts", "viral"] + (["series", "part" + str(part)] if is_series else []))
    ig_tags = _tags(clip, ["reels", "reelsinstagram", "explore"])
    tt_tags = _tags(clip, ["fyp", "foryou", "tiktok"])

    yt_title = base if len(base) <= 90 else base[:87].rsplit(" ", 1)[0] + "..."
    yt_title = f"{yt_title} #Shorts"[: LIMITS["youtube"]["title"]]

    attribution = f"\n\nFull video: {source.url}" if getattr(source, "url", None) else ""
    yt_description = (
        f"{caption}\n\n{_hash_str(yt_tags, 8)}{attribution}"
    )[: LIMITS["youtube"]["description"]]

    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "clip_index": clip.index,
        "source": {
            "title": source.title,
            "video_id": source.video_id,
            "url": getattr(source, "url", None),
            "uploader": getattr(source, "uploader", None),
        },
        "timing": {
            "start_time": clip.start_time,
            "end_time": clip.end_time,
            "duration": clip.duration,
        },
        "analysis": {
            "viral_score": clip.viral_score,
            "hook_title": clip.hook_title,
            "reason": clip.reason,
        },
        "render": {
            "file": video_rel,
            "width": render.width,
            "height": render.height,
            "aspect_ratio": "9:16",
            "duration": round(render.duration, 2),
            "captions_burned": render.ass_path is not None,
        },
        "transcript": clip.transcript,
        "platforms": {
            "youtube_shorts": {
                "title": yt_title,
                "description": yt_description,
                "tags": yt_tags[: LIMITS["youtube"]["tags"]],
                "category_id": settings.publish.youtube_category_id,
                "privacy_status": settings.publish.youtube_privacy,
                "made_for_kids": False,
            },
            "instagram_reels": {
                "caption": f"{caption}\n\n{_hash_str(ig_tags, LIMITS['instagram']['hashtags'])}"[
                    : LIMITS["instagram"]["caption"]
                ],
                "hashtags": ig_tags[: LIMITS["instagram"]["hashtags"]],
                "share_to_feed": True,
            },
            "tiktok": {
                "caption": f"{caption} {_hash_str(tt_tags, LIMITS['tiktok']['hashtags'])}"[
                    : LIMITS["tiktok"]["caption"]
                ],
                "hashtags": tt_tags[: LIMITS["tiktok"]["hashtags"]],
                "allow_comments": True,
            },
        },
        "status": {"youtube": "pending", "instagram": "pending", "tiktok": "pending"},
    }

    if is_series:
        meta["series"] = {
            "series_title": series_title,
            "part": part,
            "total_parts": total,
            "is_first": part == 1,
            "is_final": is_final,
            "cliffhanger": getattr(clip, "cliffhanger", ""),
            # Post in this order. Posting a series out of order kills it.
            "post_order": part,
            "previous_part": f"clip_{part - 1:02d}" if part > 1 else None,
            "next_part": f"clip_{part + 1:02d}" if not is_final else None,
        }
    return meta


def organize(clip, source, render, ass_path: Path | None = None) -> ClipBundle:
    """Move a rendered clip and its siblings into its own publish folder."""
    root = settings.paths.output / slugify(f"{source.video_id}-{source.title}", 50)
    folder = root / f"clip_{clip.index:02d}_{slugify(clip.hook_title, 40)}"
    folder.mkdir(parents=True, exist_ok=True)

    video_dest = folder / f"clip_{clip.index:02d}.mp4"
    if Path(render.video_path).resolve() != video_dest.resolve():
        shutil.move(str(render.video_path), video_dest)
    render.video_path = video_dest

    thumb_dest = None
    if render.thumbnail_path and Path(render.thumbnail_path).exists():
        thumb_dest = folder / "thumbnail.jpg"
        shutil.move(str(render.thumbnail_path), thumb_dest)
        render.thumbnail_path = thumb_dest

    if ass_path and Path(ass_path).exists():
        shutil.copy2(ass_path, folder / "captions.ass")

    meta = build_metadata(clip, source, render, video_dest.name)
    meta_path = write_json(folder / "metadata.json", meta)

    # Human-friendly copy/paste files
    p = meta["platforms"]
    (folder / "caption_youtube.txt").write_text(
        f"{p['youtube_shorts']['title']}\n\n{p['youtube_shorts']['description']}\n",
        encoding="utf-8")
    (folder / "caption_instagram.txt").write_text(
        p["instagram_reels"]["caption"] + "\n", encoding="utf-8")
    (folder / "caption_tiktok.txt").write_text(
        p["tiktok"]["caption"] + "\n", encoding="utf-8")
    (folder / "transcript.txt").write_text(clip.transcript + "\n", encoding="utf-8")

    log.info("Packaged -> %s", folder)
    return ClipBundle(directory=folder, video=video_dest, metadata=meta_path,
                      thumbnail=thumb_dest)


def write_run_summary(bundles: list[ClipBundle], source, clips) -> Path:
    root = bundles[0].directory.parent if bundles else settings.paths.output
    summary = {
        "source": {"title": source.title, "video_id": source.video_id,
                   "url": getattr(source, "url", None),
                   "duration": round(source.duration, 2)},
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "clip_count": len(bundles),
        "clips": [
            {
                "index": c.index,
                "hook_title": c.hook_title,
                "viral_score": c.viral_score,
                "start_time": c.start_time,
                "end_time": c.end_time,
                "duration": c.duration,
                "folder": str(b.directory.relative_to(settings.paths.output)),
                "video": str(b.video.relative_to(settings.paths.output)),
            }
            for c, b in zip(clips, bundles)
        ],
    }
    return write_json(root / "summary.json", summary)


# ===========================================================================
# 2a. YouTube - official API (free quota, fully automated, ToS-compliant)
# ===========================================================================
YOUTUBE_SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]


def _youtube_service():
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build

    cfg = settings.publish
    token_path = Path(cfg.youtube_token_file)
    creds = None

    if token_path.exists():
        creds = Credentials.from_authorized_user_file(str(token_path), YOUTUBE_SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            secrets = Path(cfg.youtube_client_secrets)
            if not secrets.exists():
                raise FileNotFoundError(
                    f"{secrets} missing. Create an OAuth 'Desktop app' client in "
                    "Google Cloud Console (YouTube Data API v3 enabled) and save "
                    "the JSON there."
                )
            flow = InstalledAppFlow.from_client_secrets_file(str(secrets), YOUTUBE_SCOPES)
            creds = flow.run_local_server(port=0)
        token_path.write_text(creds.to_json(), encoding="utf-8")

    return build("youtube", "v3", credentials=creds, cache_discovery=False)


def upload_to_youtube(bundle: ClipBundle, *, dry_run: bool = False) -> str | None:
    """Resumable upload of one clip. Returns the video id."""
    import json
    from googleapiclient.http import MediaFileUpload

    meta = json.loads(bundle.metadata.read_text(encoding="utf-8"))
    yt = meta["platforms"]["youtube_shorts"]

    if dry_run:
        log.info("[dry-run] would upload '%s' (%s)", yt["title"], bundle.video.name)
        return None

    service = _youtube_service()
    body = {
        "snippet": {
            "title": yt["title"],
            "description": yt["description"],
            "tags": yt["tags"],
            "categoryId": yt["category_id"],
        },
        "status": {
            "privacyStatus": yt["privacy_status"],
            "selfDeclaredMadeForKids": yt["made_for_kids"],
        },
    }
    media = MediaFileUpload(str(bundle.video), chunksize=4 * 1024 * 1024,
                            resumable=True, mimetype="video/mp4")
    request = service.videos().insert(part="snippet,status", body=body, media_body=media)

    response, last_pct = None, -1
    while response is None:
        status, response = request.next_chunk()
        if status:
            pct = int(status.progress() * 100)
            if pct >= last_pct + 20:
                last_pct = pct
                log.info("  uploading %s ... %d%%", bundle.video.name, pct)

    video_id = response.get("id")
    log.info("Uploaded: https://youtube.com/shorts/%s", video_id)

    meta["status"]["youtube"] = "uploaded"
    meta.setdefault("published", {})["youtube_video_id"] = video_id
    write_json(bundle.metadata, meta)
    return video_id


# ===========================================================================
# 2b. TikTok / Instagram - Playwright, semi-automated
# ===========================================================================
_UPLOAD_URLS = {
    "tiktok": "https://www.tiktok.com/tiktokstudio/upload",
    "instagram": "https://www.instagram.com/",
    "youtube_studio": "https://studio.youtube.com/",
}


def _persistent_browser(playwright, headless: bool):
    """
    One persistent Chromium profile so you log in ONCE by hand and the session
    is reused. Nothing is scraped or stored beyond the browser profile itself.
    """
    profile = settings.paths.browser_profile
    profile.mkdir(parents=True, exist_ok=True)
    return playwright.chromium.launch_persistent_context(
        user_data_dir=str(profile),
        headless=headless,
        viewport={"width": 1440, "height": 950},
        args=["--disable-blink-features=AutomationControlled"],
    )


def browser_login(platform: str = "tiktok") -> None:
    """Open a browser so you can log in once; the session persists afterwards."""
    from playwright.sync_api import sync_playwright

    url = _UPLOAD_URLS.get(platform, _UPLOAD_URLS["tiktok"])
    with sync_playwright() as pw:
        ctx = _persistent_browser(pw, headless=False)
        page = ctx.pages[0] if ctx.pages else ctx.new_page()
        page.goto(url)
        input(f"\n>>> Log in to {platform} in the browser, then press ENTER here to save the session... ")
        ctx.close()
    log.info("Session saved to %s", settings.paths.browser_profile)


def post_semi_auto(bundle: ClipBundle, platform: str, *, timeout_ms: int = 120_000) -> bool:
    """
    Open the upload page, attach the MP4, prefill the caption - then STOP.
    You review and click Post yourself.

    Selectors on these sites change frequently. If a step fails the browser is
    left open at the upload page so you can finish manually; treat the
    selectors below as a starting point to maintain, not a stable contract.
    """
    import json
    from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

    meta = json.loads(bundle.metadata.read_text(encoding="utf-8"))
    caption = (meta["platforms"]["tiktok"]["caption"] if platform == "tiktok"
               else meta["platforms"]["instagram_reels"]["caption"])

    log.info("Opening %s uploader for %s", platform, bundle.video.name)
    with sync_playwright() as pw:
        ctx = _persistent_browser(pw, headless=False)   # never headless here
        page = ctx.pages[0] if ctx.pages else ctx.new_page()
        try:
            page.goto(_UPLOAD_URLS[platform], timeout=timeout_ms)
            page.wait_for_timeout(4000)

            if platform == "tiktok":
                # TikTok renders the uploader inside an iframe on some locales.
                target = page
                for frame in page.frames:
                    if "upload" in (frame.url or ""):
                        target = frame
                        break
                target.locator("input[type='file']").first.set_input_files(str(bundle.video))
                page.wait_for_timeout(12_000)           # wait for processing
                editor = target.locator("div[contenteditable='true']").first
                editor.click()
                page.keyboard.press("Control+A")
                page.keyboard.press("Backspace")
                editor.type(caption, delay=12)

            else:  # instagram
                page.get_by_role("link", name="New post").first.click(timeout=20_000)
                page.locator("input[type='file']").first.set_input_files(str(bundle.video))
                page.wait_for_timeout(6000)
                for label in ("Next", "Weiter", "Siguiente"):
                    try:
                        page.get_by_role("button", name=label).first.click(timeout=8000)
                        page.wait_for_timeout(2500)
                        page.get_by_role("button", name=label).first.click(timeout=8000)
                        break
                    except PWTimeout:
                        continue
                page.wait_for_timeout(2500)
                page.locator("div[contenteditable='true']").first.type(caption, delay=12)

            print("\n" + "=" * 68)
            print(f"  {platform.upper()} draft ready: {bundle.video.name}")
            print("  Review the preview, cover frame and caption in the browser,")
            print("  then click Post/Publish YOURSELF.")
            print("=" * 68)
            input(">>> Press ENTER here once you are done (or to close)... ")

            meta["status"][platform] = "posted_semi_auto"
            write_json(bundle.metadata, meta)
            return True

        except Exception as exc:  # noqa: BLE001
            log.error("%s automation failed (%s). Finish manually in the open window.",
                      platform, exc)
            input(">>> Press ENTER to close the browser... ")
            return False
        finally:
            ctx.close()


def publish(bundles: list[ClipBundle], platforms: list[str], *, dry_run: bool = False) -> None:
    for bundle in bundles:
        for platform in platforms:
            try:
                if platform == "youtube":
                    upload_to_youtube(bundle, dry_run=dry_run)
                elif platform in ("tiktok", "instagram"):
                    if dry_run:
                        log.info("[dry-run] would open %s uploader for %s",
                                 platform, bundle.video.name)
                    else:
                        post_semi_auto(bundle, platform)
                else:
                    log.warning("Unknown platform '%s' - skipped.", platform)
            except Exception as exc:  # noqa: BLE001
                log.error("Publishing %s to %s failed: %s", bundle.video.name, platform, exc)
                