"""Configuration loading, validation, and serialization for the vidgen pipeline.

Uses Pydantic BaseModel for automatic validation, consistent with models.py.
Configuration is loaded from YAML with support for CLI overrides.
"""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


# --- Configuration Models ---


class VoiceConfig(BaseModel):
    """Voice/TTS configuration settings."""

    engine: str = "edge-tts"  # "edge-tts" or "qwen3"
    model: str = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
    speaker: str = "aiden"
    voice: str = "en-US-AndrewNeural"  # edge-tts voice name
    rate: str = "-5%"  # edge-tts rate adjustment
    instruct: str = ""
    pace_wpm: int = 150
    sample_rate: int = 24000
    sentence_pause_ms: int = 400
    section_pause_ms: int = 1000


class VisualConfig(BaseModel):
    """Visual style configuration for image generation."""

    style_prefix: str = "professional tech illustration, clean minimalist style, blue and purple color scheme"
    color_palette: list[str] = Field(
        default_factory=lambda: ["#1a1a2e", "#16213e", "#0f3460", "#e94560"]
    )
    font_family: str = "Inter"
    image_model: str = "./models/schnell-4bit"
    image_steps: int = 4
    image_gen_width: int = 1536
    image_gen_height: int = 864
    image_output_width: int = 1920
    image_output_height: int = 1080
    seed_strategy: str = "fixed"


class BrandingConfig(BaseModel):
    """Channel branding configuration."""

    intro_template: Path | None = None
    outro_template: Path | None = None
    watermark: Path | None = None
    thumbnail_font: str = "Helvetica"
    thumbnail_colors: list[str] = Field(
        default_factory=lambda: ["#ffffff", "#e94560"]
    )


class DataOverlayConfig(BaseModel):
    """Configuration for data visualization overlays on MFLUX backgrounds."""

    enabled: bool = True
    blur_radius: int = 8  # Gaussian blur on background (0 = disabled)
    dark_overlay_base: int = 120  # Base alpha for dark overlay (0-255)
    grain_opacity: float = 0.05  # Grain texture opacity (0 = disabled)


class ShortsConfig(BaseModel):
    """Configuration for YouTube Shorts generation pipeline."""

    image_gen_width: int = 768
    image_gen_height: int = 1344
    image_output_width: int = 1080
    image_output_height: int = 1920
    ken_burns_zoom: float = 0.10
    image_pace_seconds: int = 6
    text_font_size_impact: int = 96
    text_font_size_normal: int = 72
    text_font_size_danger: int = 80
    text_color_impact: str = "#ffffff"
    text_color_normal: str = "#ffffff"
    text_color_danger: str = "#e94560"
    logo_size: int = 60
    logo_opacity: float = 0.30
    logo_position: str = "bottom-left"
    music_volume_db: int = -22
    music_fade_in: float = 1.0
    music_fade_out: float = 2.0
    music_dir: Path = Path("channel/assets/music")
    narration_rate: str = "+3%"  # Slightly faster for Shorts (long-form uses -5%)
    output_dir: Path = Path("output/shorts")


class PipelineConfig(BaseModel):
    """Top-level pipeline configuration combining all sub-configs."""

    voice: VoiceConfig = Field(default_factory=VoiceConfig)
    visual: VisualConfig = Field(default_factory=VisualConfig)
    branding: BrandingConfig = Field(default_factory=BrandingConfig)
    shorts: "ShortsConfig" = Field(default_factory=lambda: ShortsConfig())
    data_overlay: DataOverlayConfig = Field(default_factory=DataOverlayConfig)
    generation_ratio_minutes: float = 1.0
    timeout_buffer: float = 1.5
    memory_limit_gb: int = 32
    output_dir: Path = Path("./output")
    jobs_dir: Path = Path("./jobs")
    crossfade_duration_ms: int = 500
    background_music_db: int = -18
    max_retries: int = 3
    image_timeout_seconds: int = 120


# --- Functions ---


def load_config(path: Path, overrides: dict[str, Any] | None = None) -> PipelineConfig:
    """Load configuration from a YAML file and apply optional CLI overrides.

    Args:
        path: Path to the YAML configuration file.
        overrides: Optional dict of dot-notation keys to override values.
            Example: {"voice.pace_wpm": 155, "memory_limit_gb": 24}

    Returns:
        A validated PipelineConfig instance.

    Raises:
        FileNotFoundError: If the config file doesn't exist.
        yaml.YAMLError: If the YAML is malformed.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path) as f:
        raw = yaml.safe_load(f) or {}

    # Map YAML top-level "pipeline" key into flat PipelineConfig fields
    data: dict[str, Any] = {}
    if "voice" in raw:
        data["voice"] = raw["voice"]
    if "visual" in raw:
        data["visual"] = raw["visual"]
    if "branding" in raw:
        data["branding"] = raw["branding"]
    if "shorts" in raw:
        data["shorts"] = raw["shorts"]
    if "pipeline" in raw:
        pipeline_data = raw["pipeline"]
        for key, value in pipeline_data.items():
            data[key] = value

    # Apply CLI overrides using dot-notation
    if overrides:
        for key, value in overrides.items():
            _apply_override(data, key, value)

    return PipelineConfig(**data)


def _apply_override(data: dict[str, Any], key: str, value: Any) -> None:
    """Apply a dot-notation override to the config data dict.

    Example: "voice.pace_wpm" -> data["voice"]["pace_wpm"] = value
    """
    parts = key.split(".")
    if len(parts) == 1:
        data[parts[0]] = value
    elif len(parts) == 2:
        section, field = parts
        if section not in data:
            data[section] = {}
        if isinstance(data[section], dict):
            data[section][field] = value
        else:
            # If already parsed into a model, convert back to dict
            data[section] = dict(data[section]) if hasattr(data[section], "__iter__") else {}
            data[section][field] = value


def validate_config(config: PipelineConfig) -> list[str]:
    """Validate configuration values are within acceptable ranges.

    Returns a list of error strings. Empty list means config is valid.
    """
    errors: list[str] = []

    # Voice validation
    if not (140 <= config.voice.pace_wpm <= 160):
        errors.append(
            f"voice.pace_wpm must be between 140 and 160, got {config.voice.pace_wpm}"
        )
    if config.voice.sample_rate not in (22050, 24000, 44100, 48000):
        errors.append(
            f"voice.sample_rate must be 22050, 24000, 44100, or 48000, got {config.voice.sample_rate}"
        )
    if not (200 <= config.voice.sentence_pause_ms <= 800):
        errors.append(
            f"voice.sentence_pause_ms must be between 200 and 800, got {config.voice.sentence_pause_ms}"
        )
    if not (800 <= config.voice.section_pause_ms <= 1500):
        errors.append(
            f"voice.section_pause_ms must be between 800 and 1500, got {config.voice.section_pause_ms}"
        )

    # Visual validation
    if config.visual.seed_strategy not in ("fixed", "random"):
        errors.append(
            f"visual.seed_strategy must be 'fixed' or 'random', got '{config.visual.seed_strategy}'"
        )

    # Pipeline validation
    if config.memory_limit_gb <= 0:
        errors.append(
            f"memory_limit_gb must be greater than 0, got {config.memory_limit_gb}"
        )
    if config.timeout_buffer <= 1.0:
        errors.append(
            f"timeout_buffer must be greater than 1.0, got {config.timeout_buffer}"
        )
    if config.generation_ratio_minutes <= 0:
        errors.append(
            f"generation_ratio_minutes must be greater than 0, got {config.generation_ratio_minutes}"
        )
    if config.crossfade_duration_ms <= 0:
        errors.append(
            f"crossfade_duration_ms must be greater than 0, got {config.crossfade_duration_ms}"
        )
    if config.max_retries < 0:
        errors.append(
            f"max_retries must be >= 0, got {config.max_retries}"
        )
    if config.image_timeout_seconds <= 0:
        errors.append(
            f"image_timeout_seconds must be greater than 0, got {config.image_timeout_seconds}"
        )

    return errors


def serialize_config(config: PipelineConfig) -> str:
    """Serialize a PipelineConfig to YAML string for round-trip testing.

    The output format matches the expected config.yaml structure with
    voice, visual, branding, and pipeline sections.
    """
    data: dict[str, Any] = {
        "voice": config.voice.model_dump(),
        "visual": config.visual.model_dump(),
        "branding": _serialize_branding(config.branding),
        "pipeline": {
            "generation_ratio_minutes": config.generation_ratio_minutes,
            "timeout_buffer": config.timeout_buffer,
            "memory_limit_gb": config.memory_limit_gb,
            "output_dir": str(config.output_dir),
            "jobs_dir": str(config.jobs_dir),
            "crossfade_duration_ms": config.crossfade_duration_ms,
            "background_music_db": config.background_music_db,
            "max_retries": config.max_retries,
            "image_timeout_seconds": config.image_timeout_seconds,
        },
    }
    return yaml.dump(data, default_flow_style=False, sort_keys=False)


def _serialize_branding(branding: BrandingConfig) -> dict[str, Any]:
    """Serialize BrandingConfig, converting Path objects to strings or None."""
    return {
        "intro_template": str(branding.intro_template) if branding.intro_template else None,
        "outro_template": str(branding.outro_template) if branding.outro_template else None,
        "watermark": str(branding.watermark) if branding.watermark else None,
        "thumbnail_font": branding.thumbnail_font,
        "thumbnail_colors": branding.thumbnail_colors,
    }
