# AI Video Generator (Free & Local)

A fully automated YouTube video generation pipeline that runs **entirely on Apple Silicon** with zero API costs. From script to publish-ready video in one command.

## What It Does

```
Script + Scene Plan → Narration → AI Images → Video Assembly → Thumbnails → Captions → Package
```

## Features

### Zero-Cost Local Generation
- Runs entirely on Apple Silicon — no cloud GPU, no API keys, no subscriptions
- MFLUX (FLUX schnell 4-bit) generates images locally using MLX + Metal
- edge-tts provides free, high-quality narration (Microsoft's neural voices)
- FFmpeg handles all encoding locally with hardware acceleration

### AI-Powered Narration
- Natural text-to-speech via edge-tts (en-US-AndrewNeural voice)
- Adjustable speech rate (default -5% for deliberate delivery)
- Strategic pauses auto-inserted after statistics, percentages, and time shifts
- Sentence boundary data captured for precise caption/overlay sync
- Fallback to local Qwen3-TTS (MLX) if cloud isn't available

### AI Image Generation
- FLUX schnell 4-bit model via MFLUX — optimized for Apple Silicon
- Multi-image per scene: 1 image every ~8 seconds of narration (60-70 per video)
- 1536×864 native generation → LANCZOS upscale to 1920×1080 (saves memory)
- Automatic retry (3 attempts) with 50KB quality gate
- Style prefix injection to maintain visual consistency across all images

### Professional Video Assembly
- **Ken Burns effects** — 8-direction camera movement (zoom in/out, pan, diagonal)
- **Lower-third text overlays** — Semi-transparent bar + white text, sentence-boundary synced
- **Multi-overlay per scene** — 2-4 overlays distributed across sub-images (every 12-15s)
- **Background music** — Per-section mood switching (tension/momentum/neutral/resolve)
- **Music mixing** — -22dB bed volume, 2s crossfades between moods, 3s fade in/out
- **Logo watermark** — Persistent branding at configurable opacity and position
- **1-second scene gaps** — Prevents audio overlap, gives breathing room
- **LUFS mastering** — Final audio normalized to -14 LUFS / -1.0 dBTP (YouTube standard)
- **CRF 18 encoding** — Proper HD bitrate for YouTube upload

### YouTube Shorts Pipeline
- Independent vertical pipeline (1080×1920) — not a crop of horizontal content
- All-Pillow rendering via single `VideoClip(make_frame)` for stability
- Text overlays with word-wrap, fade-in, and pre-emptive timing (0.3s before narration)
- Single overlay rule — only latest active cue renders (no stacking)
- Pattern interrupt — 8% punch-in zoom at midpoint of each image
- Mid-sentence loop strategy for seamless replay
- Subscribe CTA with varied phrasing per short
- 24fps output, background music at -22dB

### Thumbnail Generation
- A/B variant system: Variant A (threat/red) + Variant B (opportunity/gold)
- Safe zone enforcement: text constrained to center 70% (avoids YouTube duration badge)
- Feed pop: +15% contrast, +10% saturation for dark mode visibility
- MAX 3 words per thumbnail — visual proof, not description
- MFLUX generates the base image, Pillow composites text

### Auto-Generated Captions (SRT)
- Sentence-level SRT from edge-tts boundary data
- Proper inter-scene gap offsets
- Ready for YouTube Studio upload (Subtitles → English → Upload with timing)
- Improves SEO for technical terms that auto-captions botch

### Multi-Platform Distribution
- **X Thread Generator** — Script → 6-tweet thread with stats + CTA
- **Substack Newsletter** — Script → article with TL;DR, sections, subscribe CTA
- **LinkedIn Post** — 150-word business angle + hashtags (link in first comment)
- One command generates all three: `vidgen distribute`

### Data Visualization Pipeline
- `visual_type: "data"` scenes render charts on atmospheric backgrounds
- Templates: big_stat, bar_comparison, line_trend, comparison, bullet_list
- Gaussian blur background + dark gradient + grain texture for cohesion
- Matplotlib/Pillow renders charts as transparent PNGs composited onto MFLUX backgrounds

### Community Tab Content
- Auto-generates 2 square-cropped scene images (1080×1080) per video
- Auto-generates 280-char engagement posts (poll + stat-hook)
- Integrated into the `vidgen produce` output

### Batch Processing & Resume
- Queue system: add multiple videos, process overnight
- `caffeinate` integration prevents Mac sleep during multi-hour runs
- Full resume support — intermediate artifacts persist to disk
- Per-stage and per-short resume on failure
- `generate_overnight.sh` script for hands-off batch generation

### News Scanner (Reactive Content)
- Monitors HackerNews, TechCrunch, ArXiv, Reddit, ProductHunt
- Scores stories by brand name, conflict terms, numbers, recency
- Dedup via similarity threshold
- Outputs ranked candidates for script writing

### Production-Ready CLI
- Single entry point: `vidgen` with subcommands for every operation
- Input validation before generation starts
- Configurable via centralized YAML (voice, visual, pipeline, shorts, music)
- Pydantic models for type-safe data flow throughout the pipeline

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4) — 16GB+ RAM recommended (36GB ideal)
- Python 3.11+
- FFmpeg (`brew install ffmpeg`)

## Quick Start

```bash
# Clone
git clone https://github.com/mukulgadd/Ai-Video-Generator-Free.git
cd Ai-Video-Generator-Free

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
.venv/bin/pip install mflux
.venv/bin/pip install hatchling moviepy pydantic pydub scipy click pyyaml httpx feedparser
.venv/bin/pip install "pillow>=12.1.1"
.venv/bin/pip install -e . --no-deps

# Verify
.venv/bin/vidgen --help
```

## Usage

```bash
# Validate inputs before generation
vidgen validate script.md -p scene_plan.json

# Generate a full video
vidgen run script.md -p scene_plan.json

# Full content package (video + shorts + distribution)
vidgen produce script.md -p scene_plan.json

# Resume a failed/interrupted job
vidgen produce script.md -p scene_plan.json --resume

# Generate Shorts only
vidgen short shorts/my_short.md

# Generate distribution content
vidgen distribute script.md --url https://youtu.be/your_video

# Generate captions from existing narration
vidgen captions jobs/your_job/narration/

# Batch processing (overnight)
vidgen queue-add script.md -p plan.json
vidgen queue-run
vidgen queue-status
```

## Tech Stack

| Component | Tool | Notes |
|-----------|------|-------|
| Narration | edge-tts (Andrew voice) | Free cloud TTS, rate=-5%, strategic pauses |
| Image Gen | MFLUX (FLUX schnell 4-bit) | Local on Apple Silicon, ~2 min/image |
| Video Assembly | MoviePy 2.x + FFmpeg | Ken Burns, overlays, music, CRF 18 |
| Thumbnails | MFLUX + Pillow | A/B variants, safe zones |
| CLI | Click | All operations via `vidgen` command |
| Config | PyYAML + Pydantic | Centralized YAML config |
| Testing | pytest | 296 tests |

## Pipeline Features

- **Multi-image per scene** — 1 image per ~8s of narration (60-70 images per 10-min video)
- **Ken Burns effects** — 8-direction camera movement (zoom, pan, diagonal)
- **Background music** — Per-section mood switching (tension/momentum/neutral/resolve) at -22dB
- **Lower-third overlays** — Sentence-boundary synced, pre-emptive timing
- **LUFS mastering** — Final audio normalized to -14 LUFS / -1.0 dBTP
- **Image retry** — 3 attempts + 50KB quality gate
- **Resume support** — Intermediate artifacts persist; pick up where you left off
- **Data visualization** — Chart overlays on atmospheric backgrounds (bar, line, stat, comparison)

## Configuration

All settings are in `config.yaml`:

- Voice engine and settings
- Image generation dimensions and model path
- Video encoding parameters
- Background music volume and crossfades
- Shorts dimensions and overlay styling
- Pipeline timeouts and memory limits

## Project Structure

```
├── src/vidgen/          # Pipeline source (25 modules)
│   ├── cli.py           # Click CLI entry point
│   ├── pipeline.py      # Long-form video orchestrator
│   ├── producer.py      # Unified produce command
│   ├── narration.py     # TTS (edge-tts + Qwen3 fallback)
│   ├── imaging.py       # MFLUX image generation
│   ├── assembly.py      # Video composition (MoviePy)
│   ├── shorts_assembly.py  # Vertical shorts assembler
│   ├── distribution.py  # X/Substack/LinkedIn generators
│   ├── captions.py      # SRT generator
│   └── ...
├── tests/               # 296 tests
├── templates/           # Script + scene plan templates
├── config.yaml          # Pipeline configuration
├── pyproject.toml       # Package definition
├── SETUP.md             # Detailed setup instructions
└── generate_overnight.sh  # Batch generation with caffeinate
```

## Generation Timing (Apple Silicon M3 Pro, 36GB)

| Stage | Time |
|-------|------|
| TTS Narration | ~2 min |
| Image Generation | ~80-90 min |
| Video Assembly | ~85 min |
| Thumbnails | ~5 min |
| **Total per video** | **~3 hours** |

Overnight batch: 2 videos per night with `caffeinate`.

## Running Tests

```bash
VIDGEN_PLACEHOLDER=1 .venv/bin/pytest tests/ -v
```

The `VIDGEN_PLACEHOLDER` env var enables placeholder mode so tests don't require GPU/model access.

## License

MIT

## Author

Mukul Gaddhyan
