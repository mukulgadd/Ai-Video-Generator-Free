# AI Video Generator (Free & Local)

A fully automated YouTube video generation pipeline that runs **entirely on Apple Silicon** with zero API costs. From script to publish-ready video in one command.

## What It Does

```
Script + Scene Plan → Narration → AI Images → Video Assembly → Thumbnails → Captions → Package
```

- **Text-to-Speech** — Natural narration via edge-tts (cloud, free, instant)
- **AI Image Generation** — MFLUX (FLUX schnell 4-bit) running locally on Apple Silicon
- **Video Assembly** — Ken Burns effects, lower-third overlays, background music, logo watermark
- **Shorts Pipeline** — Vertical short-form content with text overlays and pattern interrupts
- **Thumbnails** — A/B variants with safe zones and feed-pop adjustments
- **Captions** — Auto-generated SRT from sentence boundary data
- **Distribution** — X threads, Substack newsletters, LinkedIn posts from any script

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
