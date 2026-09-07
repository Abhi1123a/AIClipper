# AI Video Clipper

Turn a long YouTube video or local MP4 into **3-5 vertical (9:16) short clips** with
burned-in word-by-word animated captions, smart reframing, and publish-ready metadata
for YouTube Shorts, Instagram Reels and TikTok.

Everything runs locally except an optional free-tier LLM call. **No paid API is required
at any point** — with no API key at all it falls back to an offline heuristic clip picker.

```
long_video.mp4
   └─ yt-dlp ────────► source video
       └─ faster-whisper ────────► word-level transcript (local, free)
           └─ Gemini / Groq free tier ────────► ranked clip list + captions + hashtags
               └─ FFmpeg ────────► 1080x1920 MP4 + karaoke captions + thumbnail
                   └─ output/<video>/clip_01_.../  (mp4, metadata.json, captions, transcript)
```

---

## 1. Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.10 – 3.12 | 3.13/3.14 mostly work, but CUDA and audio wheels lag behind — if you hit odd import or backend errors, 3.12 is the safe choice |
| FFmpeg + ffprobe | Must be on `PATH` |
| ~2 GB disk | Whisper weights are downloaded on first run |
| GPU (optional) | NVIDIA + CUDA 12 makes transcription 5-15x faster |

### Installing FFmpeg

**Windows**
```powershell
winget install Gyan.FFmpeg
# or: choco install ffmpeg-full
# then close and reopen the terminal so PATH refreshes
ffmpeg -version
```

**macOS**
```bash
brew install ffmpeg
```

**Linux (Debian/Ubuntu)**
```bash
sudo apt update && sudo apt install -y ffmpeg
```

---

## 2. Install

```bash
git clone <your-repo> ai-video-clipper && cd ai-video-clipper

# virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt

# only if you want browser-based posting
python -m playwright install chromium
```

**NVIDIA GPU users:** `faster-whisper` needs cuBLAS and cuDNN 9 for CUDA:
```bash
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12
```
If CUDA initialisation fails for any reason, the transcriber logs a warning and
automatically falls back to CPU/int8 — the run does not break.

---

## 3. Configure

```bash
cp .env.example .env        # Windows: copy .env.example .env
```

Then add **one** free key (or neither):

- **Gemini** — https://aistudio.google.com/apikey → `GEMINI_API_KEY=...`
- **Groq** — https://console.groq.com/keys → `GROQ_API_KEY=...` and `LLM_PROVIDER=groq`
- **Neither** — the pipeline runs with `--provider heuristic`. It still works; the clip
  choices are just noticeably less intelligent.

---

## 3b. Choosing a brain (the `--provider` flag)

The pipeline makes **1–3 LLM calls per video**, total. Requests-per-minute is
therefore almost irrelevant here — what matters is the daily/monthly ceiling and
whether the tier still exists in six months.

| Provider | Cost | Good for | Watch out for |
|---|---|---|---|
| `manual` | £0 (uses your existing Claude Pro) | Best clip quality, no key, no quota | ~30 s of copy-paste per video |
| `gemini` | Free tier | Fully unattended runs | Google cut free quotas in late 2025; verify current limits |
| `groq` | Free tier | Fast, easy | Low tokens-per-minute; long transcripts get windowed |
| `nim` | Free (dev use) | 100+ models, one key | See the NIM note below |
| `openrouter` | Free `:free` models | Widest model choice | Free models rotate/disappear |
| `ollama` | £0 forever | True long-term independence | Needs ~8 GB RAM; a 14B model is weaker than Gemini |
| `heuristic` | £0 | Emergencies, offline | Noticeably dumber picks |

### `--provider manual` — relay through claude.ai by hand

This is the "give Claude web the work, then hand it back to the code" mode you
probably want. The pipeline stops after transcription, hands you a prompt file,
and waits.

```bash
# 1. transcribe + write the prompt, then exit
python main.py --url "https://youtu.be/XXXX" --provider manual --export-prompts

#    -> cache/prompt_<id>_w01.md
#    Open it, copy everything, paste into claude.ai.
#    Save Claude's JSON reply as cache/response_<id>_w01.json

# 2. re-run the identical command WITHOUT --export-prompts
python main.py --url "https://youtu.be/XXXX" --provider manual
```

Or stay in one terminal session: drop `--export-prompts` and it blocks with an
interactive prompt where you can either point it at a saved file or type
`paste`, dump the JSON inline, and end with a line containing only `END`.

Answers are cached, so re-rendering with different caption styling never asks
you to paste anything twice.

### Story mode + manual: the best combination in the tool

`--strategy story --provider manual` is worth calling out, because it pairs the
task that most needs a strong model with the strongest model you have access to.
Placing story break points requires holding the whole narrative in mind at once
— exactly what a windowed free-tier call is bad at.

```bash
python main.py --url "https://youtu.be/XXXX" \
  --strategy story --clips auto --provider manual --export-prompts

# paste cache/prompt_<id>-story-0_w01.md into claude.ai
# save the JSON reply as cache/response_<id>-story-0_w01.json

python main.py --url "https://youtu.be/XXXX" \
  --strategy story --clips auto --provider manual
```

Manual mode uses a much larger window than the API providers
(`LLM_MANUAL_CHUNK_CHARS`, default 120,000 chars ≈ 3-4 hours of speech), so a
typical segment arrives as **one prompt file containing the entire story**. If a
transcript still splits across windows, you'll get a warning — break points near
the seams are weaker, because the model cannot place breaks in a half it never
saw.

Prompt and response filenames include the strategy, so highlights and story runs
on the same video never overwrite each other's answers.

**When the reply comes back, sanity-check the plan table before rendering.** The
`GAP` column is the thing to read: `+0.0s` everywhere means the series is
airtight; a large positive number means the model skipped material. If it did,
say so in the chat ("part 3 jumps 40 seconds, give me contiguous parts") and
save the corrected reply — no re-transcription, no re-download.

**Worth being clear about the trade-off:** this is a human relay, not
automation. It's the right choice when you're clipping a handful of videos a
week and want the best possible clip selection. It is the wrong choice if you
want a cron job. Also check your plan's terms before building a workflow around
manually relaying a subscription chat UI into a commercial pipeline.

### A note on NVIDIA NIM

NIM is an OpenAI-compatible catalogue at `integrate.api.nvidia.com/v1`, and it
does have a genuinely useful free developer tier — NVIDIA staff have publicly referenced roughly 40 requests per minute as the practical baseline, though reporting on whether access is credit-metered or purely rate-limited is contradictory. Either way, 40 RPM is
~40× more than this tool needs.

The real catch is the terms, not the throughput: NVIDIA defines production use as anything beyond development, testing, research or evaluation — including activity serving real end-users — and says production requires NVIDIA AI Enterprise. A pipeline that posts to
your live social accounts is arguably on the wrong side of that line. For
prototyping, it's excellent. For "something I can use for a long time," I'd
reach for `ollama` (nothing to revoke) or `manual` (nothing to meter) instead.

```bash
# NIM
NVIDIA_API_KEY=nvapi-xxxx python main.py --url "..." --provider nim

# Local, unlimited, no account:
#   ollama pull qwen2.5:14b-instruct && ollama serve
python main.py --url "..." --provider ollama
```

Free tiers across the board change often — one provider's model list dropped to two entries overnight in May 2026 — so the code is
built to degrade rather than crash: an unusable provider falls back to another
configured one, and finally to the offline heuristic.

---

## 3c. Picking a strategy (`--strategy`)

Not every video wants "best moments". A story needs *continuity*, and a
highlights reel of a story is just confusing.

| Strategy | What it does | Use it for |
|---|---|---|
| `highlights` (default) | Finds the 3-5 strongest isolated moments, ranked by hook potential, de-overlapped | Podcasts, interviews, talks, tutorials — anything where clips stand alone |
| `story` | Splits the video into **consecutive, contiguous parts** in chronological order, each ending on a cliffhanger | Second Date Update, confessions, true crime, any single narrative with a beginning and an end |
| `sequential` | Mechanically chops at sentence boundaries. No LLM, no cost, 100% coverage | Guaranteed full coverage, or when you have no API key at all |

```bash
# One story, split into as many parts as it needs
python main.py --url "https://youtu.be/XXXX" --strategy story --clips auto

# Exactly 6 parts
python main.py --url "https://youtu.be/XXXX" --strategy story --clips 6

# No LLM at all, full coverage
python main.py --file segment.mp4 --strategy sequential --clips auto
```

`--clips auto` sizes the series from the source length and `CLIP_PART_TARGET`
(default 50 s), so a 10-minute segment becomes ~12 parts.

### What story mode guarantees

The model picks *where the story breaks*; the code then makes those breaks
physically valid, because models are unreliable about boundary arithmetic:

- **Chronological order**, sorted by timestamp — the model's own `part` numbers
  are ignored, since they drift when a long transcript is windowed.
- **No overlaps.** If part 3 starts before part 2 ends, part 2 is cut short.
- **No accidental gaps.** Gaps under `CLIP_PART_GAP_TOLERANCE` (default 4 s) are
  closed so no dialogue is lost. Larger gaps are treated as deliberate skips and
  shown in the `GAP` column of the plan table, so you can see them and decide.
- **Nothing dropped for being boring.** Unlike highlights mode, a low-scoring
  part is still kept — removing it would put a hole in the story.
- **Part numbering** flows into `metadata.json` as a `series` block with
  `part`, `total_parts`, `is_final`, `cliffhanger`, `next_part` and `post_order`.
  Titles become `Series Title (Part 2/8)` and captions get an automatic
  "Part 3 is up next" CTA on every part except the last.

The plan table shows continuity at a glance before you render:

```
  SERIES: He Called The Radio Station About His Date  (4 parts, post in order)
PART  START     DUR     GAP    BEAT
1/4   00:00:00   50.1s  -      The date was going perfectly
2/4   00:00:50   44.7s  +0.0s  She left and never explained
3/4   00:01:35   55.2s  +0.0s  The brother in the black truck
4/4   00:03:10   49.8s  +40.1s Marcus finally hears the truth
```

That `+40.1s` is the tool telling you it skipped 40 seconds between parts 3 and
4. If that skip matters, raise `--clips` or lower `CLIP_PART_TARGET` and re-run.

**One thing to get right when posting a series:** post in `post_order`, and
don't let part 2 go out before part 1 has had time to find an audience. A series
where part 1 flops and part 4 goes viral just annoys people. Many creators pin a
comment linking the next part.

---

## 4. Run

```bash
# YouTube URL, everything default
python main.py --url "https://www.youtube.com/watch?v=XXXXXXXXXXX"

# Local file, 3 clips, face-aware crop, better transcription
python main.py --file ./podcast.mp4 --clips 3 --mode face --whisper-model medium

# See the clip plan without spending any render time
python main.py --url "https://youtu.be/XXXX" --dry-run

# Render and then upload to YouTube (official API) + open TikTok drafts
python main.py --url "https://youtu.be/XXXX" --publish youtube tiktok
```

### Key flags

| Flag | Purpose |
|---|---|
| `--url` / `--file` | Source (mutually exclusive) |
| `--clips N` \| `auto` | Number of clips/parts. `auto` sizes a series to the source |
| `--strategy` | `highlights` \| `story` \| `sequential` |
| `--mode blur\|crop\|face` | 9:16 reframing strategy |
| `--whisper-model` | `tiny`→`large-v3`. `small` is the sweet spot on CPU |
| `--provider` | `gemini` \| `groq` \| `heuristic` |
| `--min-duration` / `--max-duration` | Clip length bounds in seconds |
| `--no-captions` | Skip burned-in subtitles |
| `--dry-run` | Analyse and print the plan, render nothing |
| `--force` | Ignore transcript/analysis caches |
| `--login tiktok` | One-time browser login, session is saved |

### Reframe modes

- **`blur`** (default) — blurred zoom-filled background with the full 16:9 frame centred
  on top. Nothing is ever cropped away. Safest for screen-shares, gameplay, slides.
- **`crop`** — hard centre crop to 9:16. Biggest subject on screen, loses the sides.
- **`face`** — same as crop but the crop window is centred on the median face position
  detected across the clip (OpenCV Haar cascade). Best for interviews and talking heads.

---

## 4b. Reach: what this tool can and cannot do

Short version: **it can remove mechanical handicaps, and that is all.**

Nobody outside YouTube, TikTok and Meta knows how their ranking systems work.
They are undocumented, they change, and most "algorithm hack" advice is folklore
repeated until it sounds official — even the better sources label their
retention thresholds as synthesized from creator testing rather than confirmed.
Anyone selling you a guaranteed reach method is guessing or lying.

What genuinely holds across platforms is unglamorous: **completion rate beats
length**, and the first ~3 seconds decide whether someone stays. A 45s clip
watched to the end outperforms a 15s clip abandoned halfway. So the levers worth
pulling are the ones that raise completion — not tricks.

### `--platform` — stop hiding your captions behind the UI

This one is not a theory, it is a measurable defect. Each app covers part of the
frame with its own interface, and ~70% of short-form viewing is muted, so a
caption behind the UI is a caption nobody reads.

| Platform | Blocks bottom | Captions placed at |
|---|---|---|
| TikTok | ~484 px | 520 px |
| Instagram Reels | ~440 px | 480 px |
| YouTube Shorts | ~400 px | 440 px |
| `universal` (default) | — | 540 px |

```bash
python main.py --url "..." --platform tiktok     # per-platform render
python main.py --url "..."                       # universal: safe everywhere
```

**If you rendered clips before this was added, your captions were at 400 px and
were partly hidden on TikTok and Reels.** The default is now 540. These pixel
numbers drift as the apps redesign — check a real post occasionally.

### `--tighten` — remove dead air

Cuts silences over ~0.35 s and **remaps every word timestamp onto the new
timeline** so captions stay in sync. Typically removes 10–30% of runtime, which
raises completion rate for the same content. A 0.12 s pad is left at each edge,
because trimming every millisecond makes people sound breathless.

```bash
python main.py --url "..." --platform tiktok --tighten
```

### The pre-flight audit

Runs by default before rendering and flags: captions under UI (the only hard
error), clips outside the platform's completion-friendly band, openers made of
filler ("So, um, yeah, anyway..."), sub-720p sources, and missing captions.

```
  clip 1 (72s):
     [FAIL] Captions sit 400px from the bottom but TikTok blocks 484px...
     [WARN] Opens on filler (So um yeah anyway). --tighten trims 2.0s off the front.
```

### The part that actually matters: your own data

`audit.json` records per-clip features — duration, words-per-second, opener
text, whether it opens on a question, viral_score, hashtag count, series part.
Log what each clip did in your real analytics and correlate. **Twenty of your
own posts tell you more than any generic best-practice list**, because they
control for your niche, your audience and your face.

YouTube Analytics API is free and works with the OAuth token `publisher.py`
already creates, so this is the natural next thing to build.

## 4c. Editing styles (`--style`)

One look for every clip makes everything feel like the same template. Styles
change typography, caption rhythm, pacing, colour grade and the tone the LLM
writes titles in.

| Style | Look | Pacing |
|---|---|---|
| `humor` | Impact 96px, lime accent, 3-word bursts, punchy grade | Aggressive trim |
| `motivational` | Georgia 78px lowercase, gold accent, held cards, cooler desaturated grade | **No trim** — pauses carry the weight |
| `story` | Arial Black 80px, 5-word cards for dialogue, neutral grade | Light trim only |
| `brainrot` | Impact 110px, hot pink, 2-word cards, saturated +42% | Very aggressive trim |
| `clean` | The original neutral look | None |

```bash
python main.py --url "..." --style motivational
python main.py --url "..." --strategy story --style story --platform tiktok
python main.py --url "..." --style auto     # picks from --strategy (default)
```

The style also shapes what the LLM writes. `motivational` is instructed toward
plain declarative sentences with no hype words or exclamation marks; `humor`
toward setup-then-punchline with no joke-explaining. Cache keys include the
style, so switching styles re-asks the model rather than reusing old captions.

**The safe zone always wins.** A style can ask for captions lower than the
platform UI allows and it gets clamped — an invisible caption beats any
aesthetic choice.

### Per-clip variation (and why it matters)

By default each clip gets slightly different accent colour, font size and
caption placement, derived deterministically from the clip's identity — so a
given clip always renders identically on re-runs, but its siblings differ.

This is not cosmetic. YouTube's **Generic or Repetitive Content** policy
(renamed from "inauthentic content" in July 2026) makes templated,
mass-produced uploads ineligible for monetization, and **enforcement is at the
channel level, not per video** — in January 2026 YouTube deleted a set of large
faceless channels outright. The policy targets output that "looks like it's made
with a template," not any particular visual style or the use of AI.

Variation helps a batch avoid reading as one stamp repeated. It is not a
loophole, and it does not substitute for the thing the policy actually asks for:
original framing, commentary or editorial judgement. Use `--no-style-variation`
if you specifically want a locked, uniform brand look.

**On `brainrot` specifically:** it's a real format and the style works. But it's
the style most likely to be produced at volume with the least original input,
which is exactly the profile those policies describe. Worth going in with your
eyes open — especially since you're clipping other people's radio content, where
"added original value" is already the harder question.

### Things this tool deliberately does not do

- **Engagement pods, follow/unfollow, bought views, comment bots.** These violate
  every platform's terms and are the fastest route to suppressed reach.
- **"Best time to post" tables.** Generic timing charts are weak evidence. Your
  own analytics show when *your* audience is online.
- **Hashtag stuffing.** Three relevant tags is the working range on Shorts,
  where title keywords matter more because Shorts appear in search.

The honest summary: consistency, niche focus and content quality dominate
everything above. This removes the technical reasons a good clip underperforms.
It cannot make a boring clip interesting.

---

## 5. Output layout

```
output/
└── dQw4w9WgXcQ-my-video-title/
    ├── summary.json
    ├── clip_01_the-biggest-mistake-beginners-make/
    │   ├── clip_01.mp4              1080x1920, captions burned in
    │   ├── thumbnail.jpg
    │   ├── captions.ass             editable subtitle source
    │   ├── metadata.json            per-platform titles/descriptions/hashtags
    │   ├── caption_youtube.txt      copy-paste ready
    │   ├── caption_instagram.txt
    │   ├── caption_tiktok.txt
    │   └── transcript.txt
    └── clip_02_.../
```

`metadata.json` carries the clip timing, the viral score and reasoning, and a
length-validated block per platform:

```json
{
  "analysis": { "viral_score": 87, "hook_title": "...", "reason": "..." },
  "platforms": {
    "youtube_shorts": { "title": "... #Shorts", "description": "...", "tags": [...] },
    "instagram_reels": { "caption": "...", "hashtags": [...] },
    "tiktok": { "caption": "...", "hashtags": [...] }
  },
  "status": { "youtube": "pending", "instagram": "pending", "tiktok": "pending" }
}
```

---

## 6. Publishing

### YouTube — fully automated, official, free

Uses the YouTube Data API v3. The free quota is 10,000 units/day and an upload costs
1,600 units, so about **6 uploads per day**.

1. Google Cloud Console → new project → enable **YouTube Data API v3**
2. Credentials → OAuth client ID → **Desktop app** → download JSON
3. Save it as `client_secrets.json` in the project root
4. `python main.py --file clip.mp4 --publish youtube` — a browser opens once for consent

Uploads default to `privacy_status: private` so you can review before going public.
Change `YOUTUBE_PRIVACY` in `.env` when you trust the output.

### TikTok / Instagram — semi-automated

Neither platform offers free unrestricted upload APIs for personal accounts, so these
use Playwright to open the uploader, attach the file, and prefill the caption — then
**stop and wait for you to click Post**.

```bash
python main.py --login tiktok        # log in once; session persists
python main.py --url "..." --publish tiktok
```

> **Read this before enabling browser automation.** Fully unattended posting to TikTok
> or Instagram may violate their Terms of Service and can get an account rate-limited or
> banned. `publisher.py` is deliberately built to keep a human in the loop, and the UI
> selectors in it *will* break when those sites redesign — treat them as a starting point
> you maintain, not a stable contract. Also make sure you actually have the right to
> repost the source video; clipping someone else's content is a copyright question, not a
> technical one.

---

## 7. Tuning the look

All caption styling lives in `.env` / `config.py::CaptionConfig`:

| Setting | Effect |
|---|---|
| `CAPTION_FONT_SIZE` | 86 is good at 1080x1920; go 96-110 for a bolder look |
| `CAPTION_MARGIN_V` | Distance from the bottom. 400 ≈ lower third; raise to ~700 to clear TikTok's UI |
| `CAPTION_COLOR_ACTIVE` | ASS colours are `&HAABBGGRR` — **blue-green-red**, not RGB |
| `CAPTION_MAX_WORDS` | 3-4 reads best on mobile |
| `CAPTION_UPPERCASE` | `false` for a softer, more editorial feel |
| `FG_HEIGHT_RATIO` | In blur mode, how tall the sharp video strip is |

Because transcripts and analyses are cached, re-running after a styling tweak only
re-renders — no re-download, no re-transcription.

---

## 8. Troubleshooting

| Symptom | Fix |
|---|---|
| `FFmpeg not found on PATH` | Install FFmpeg (section 1) and reopen the terminal |
| Captions render in the wrong font | The font name must exist on the system. Linux: `sudo apt install fonts-dejavu ttf-mscorefonts-installer`, or set `CAPTION_FONTS_DIR=./fonts` and drop a `.ttf` in there |
| Captions missing entirely | The render auto-retries without subtitles if libass fails — run with `--log-level DEBUG` to see the real FFmpeg error |
| `Library libcublas.so.12 is not found` / `Unable to load cudnn` | `pip install nvidia-cublas-cu12 nvidia-cudnn-cu12`. The transcriber preloads these from site-packages automatically, so no `LD_LIBRARY_PATH` needed. To skip the GPU entirely: `--whisper-device cpu` |
| yt-dlp 403 / "video unavailable" | `pip install -U yt-dlp` — extractors break often and are fixed fast |
| Source downloads at 640x360 | You hit YouTube's muxed fallback. `pip install -U yt-dlp`, delete `downloads/`, re-run. The downloader now warns when source height < 720 |
| LLM returns 429 | Free-tier rate limit. The code backs off and retries; or switch `--provider groq` |
| No clips selected | Video may be too short or mostly music. Try `--clips 3 --min-duration 10` |
| Transcription is very slow | Use `--whisper-model base`, or `distil-large-v3` on GPU |
| Windows path errors in filters | Handled by `utils.escape_filter_path`; if you still hit it, avoid non-ASCII characters in folder names |

---

## 9. Module map

| File | Responsibility |
|---|---|
| `config.py` | All settings, env loading, preflight validation |
| `utils.py` | Logging, FFmpeg/ffprobe wrappers, JSON cache, path escaping |
| `downloader.py` | yt-dlp ingestion, local file loading, audio extraction |
| `transcriber.py` | faster-whisper word-level transcription + boundary snapping |
| `ai_analyzer.py` | Gemini/Groq/heuristic highlight selection, JSON repair, de-overlapping |
| `video_processor.py` | Cutting, 9:16 reframing, ASS karaoke captions, final encode |
| `styles.py` | Editing style presets, per-clip variation |
| `optimizer.py` | Platform safe zones, silence trimming, pre-flight audit |
| `publisher.py` | Output packaging, metadata.json, YouTube API, Playwright posting |
| `main.py` | CLI + pipeline orchestration |

---

## 10. Sensible next steps

- **Dynamic face tracking** — currently `face` mode computes one static crop centre per
  clip. Sample per-second positions, smooth them, and drive `crop=x` via FFmpeg's
  `sendcmd` filter for true subject tracking.
- **B-roll and zoom punches** — add a scale keyframe every ~4 s to break up talking heads.
- **Silence trimming** — cut pauses over ~350 ms with a `select` filter for tighter pacing.
- **A/B hook testing** — render two openers per clip and compare 3-second retention.
- **Scheduling** — wrap `main.py` in cron / Task Scheduler to watch a channel RSS feed.