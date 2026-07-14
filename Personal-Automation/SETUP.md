# Setup Guide — vidgen (YouTube Video Generator)

## Prerequisites

- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.11+ (Homebrew: `brew install python@3.14`)
- FFmpeg (`brew install ffmpeg`)

## Environment Setup

The project uses a Python virtual environment. The system pip is managed by Homebrew and won't install packages globally — you must use the venv.

### Create and activate venv

```bash
cd /Users/mukul.gaddhyan/Git/personal-automation

# Create venv (one-time)
python3 -m venv .venv

# Activate (every new terminal session)
source .venv/bin/activate
```

### Important: conda interference

If you see `(base)` in your prompt alongside `(.venv)`, conda is overriding pip. The fix:

```bash
# Always use the explicit venv pip path
.venv/bin/pip install <package>

# Or deactivate conda completely before activating venv
conda deactivate
conda deactivate  # twice if needed (base + any nested env)
source .venv/bin/activate
```

### Install dependencies

```bash
# Install mflux (image generation) first — it's the heaviest dep
.venv/bin/pip install mflux

# Install remaining deps
.venv/bin/pip install hatchling
.venv/bin/pip install moviepy pydantic pydub scipy pytest hypothesis pytest-cov

# Install vidgen in editable mode (no deps, they're already installed)
.venv/bin/pip install -e . --no-deps

# Force pillow 12+ (mflux needs it, moviepy warns but works fine)
.venv/bin/pip install "pillow>=12.1.1"
```

### Verify installation

```bash
# All imports work
.venv/bin/python -c "from vidgen.cli import main; from vidgen.imaging import ImageGenerator; print('OK')"

# mflux CLI available
.venv/bin/mflux-generate --help

# vidgen CLI available
.venv/bin/vidgen --help

# Tests pass
.venv/bin/pytest tests/ -v
```

## Installed Packages (Key)

| Package | Version | Purpose |
|---------|---------|---------|
| mflux | 0.18.0 | FLUX image generation on Apple Silicon (MLX) |
| mlx | 0.31.2 | Apple ML framework |
| mlx-metal | 0.31.2 | Metal GPU backend for MLX |
| moviepy | 2.2.1 | Programmatic video composition |
| pydantic | 2.13.4 | Data validation models |
| click | 8.4.2 | CLI framework |
| pillow | 12.2.0 | Image processing (text overlays) |
| pytest | 9.1.1 | Test framework |
| hypothesis | 6.155.7 | Property-based testing |
| torch | 2.12.1 | ML framework (mflux dep) |

## Known Issues

### Pillow version conflict (cosmetic)

```
moviepy 2.2.1 requires pillow<12.0
mflux 0.18.0 requires pillow>=12.1.1
```

We install pillow 12.2.0. MoviePy works fine with it — the constraint is overly strict. No runtime issues.

### pip alias points to system Python

If `pip install` gives "externally-managed-environment" error even inside the venv, always use:

```bash
.venv/bin/pip install <package>
```

This is caused by conda/Homebrew shell aliases overriding the venv's pip.

### mflux API (v0.18 uses CLI, not Python API)

The `imaging.py` module currently falls back to placeholder images. For real image generation, mflux is invoked via its CLI:

```bash
.venv/bin/mflux-generate \
  --model madroid/flux.1-schnell-mflux-4bit \
  --prompt "your prompt" \
  --steps 4 \
  --width 1920 --height 1080 \
  -o output.png
```

First run downloads model weights (~4GB from HuggingFace).

## Running the Pipeline

```bash
# Activate venv
source .venv/bin/activate

# Validate inputs
.venv/bin/vidgen validate templates/script_template.md -p templates/scene_plan_template.json

# Run single video (overnight)
.venv/bin/vidgen run script.md -p scene_plan.json

# Batch mode
.venv/bin/vidgen queue-add script1.md -p plan1.json --priority 1
.venv/bin/vidgen queue-run

# Check queue status
.venv/bin/vidgen queue-status
```

## Project Structure

```
personal-automation/
├── .venv/                  # Python virtual environment (not in git)
├── .kiro/                  # Kiro specs and steering files
├── src/vidgen/             # Pipeline source code (16 modules)
├── tests/                  # 184 tests
├── templates/              # Script and scene plan templates
├── config.yaml             # Pipeline configuration
├── pyproject.toml          # Package definition
├── jobs/                   # Working directories (generated)
└── output/                 # Final video packages (generated)
```

## TTS Setup (Pending)

TTS model not yet installed. Options:
- **Qwen3-TTS**: `pip install qwen3-tts-apple-silicon` (via MLX)
- **Chatterbox-TTS**: MPS-optimized for Apple Silicon

Will be decided after quality comparison testing.
