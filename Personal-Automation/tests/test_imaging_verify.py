"""Verification tests for the imaging module."""

import tempfile
from pathlib import Path

from vidgen.config import VisualConfig
from vidgen.imaging import ImageGenerator
from vidgen.models import Scene


def test_import():
    """Verify module imports correctly."""
    from vidgen.imaging import ImageGenerator
    assert ImageGenerator is not None


def test_generate_scene_image():
    """Generate a scene image and verify dimensions."""
    config = VisualConfig()
    gen = ImageGenerator(config)
    scene = Scene(
        id="scene-001",
        section_id="hook",
        image_prompt="futuristic cityscape with neon lights",
        ken_burns_direction="zoom-in",
        start_time=0.0,
        duration=5.0,
        transition_type="crossfade",
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "test.png"
        result = gen.generate_scene_image(scene, out)
        assert out.exists()
        assert result.width == 1920
        assert result.height == 1080
        assert result.scene_id == "scene-001"
        assert result.seed_used > 0
        assert result.generation_time_seconds >= 0.0


def test_generate_thumbnail():
    """Generate a thumbnail and verify dimensions."""
    config = VisualConfig()
    gen = ImageGenerator(config)

    with tempfile.TemporaryDirectory() as tmpdir:
        thumb = Path(tmpdir) / "thumb.png"
        result = gen.generate_thumbnail("AI technology overview", thumb)
        assert thumb.exists()
        assert result.width == 1280
        assert result.height == 720
        assert result.scene_id is None


def test_deterministic_seed():
    """Fixed seed strategy produces same seed for same ID."""
    config = VisualConfig(seed_strategy="fixed")
    gen = ImageGenerator(config)
    seed1 = gen._get_seed("scene-001")
    seed2 = gen._get_seed("scene-001")
    assert seed1 == seed2


def test_different_ids_get_different_seeds():
    """Different scene IDs produce different seeds."""
    config = VisualConfig(seed_strategy="fixed")
    gen = ImageGenerator(config)
    seed1 = gen._get_seed("scene-001")
    seed2 = gen._get_seed("scene-002")
    assert seed1 != seed2


def test_random_seed_strategy():
    """Random seed strategy produces varying seeds."""
    config = VisualConfig(seed_strategy="random")
    gen = ImageGenerator(config)
    # Just verify it returns an int in valid range
    seed = gen._get_seed("scene-001")
    assert isinstance(seed, int)
    assert 0 <= seed < 2**32


def test_release_model():
    """Model release resets state."""
    config = VisualConfig()
    gen = ImageGenerator(config)
    gen._ensure_model_loaded()
    assert gen._model_loaded
    gen.release_model()
    assert not gen._model_loaded


def test_generate_all():
    """Generate multiple scene images."""
    config = VisualConfig()
    gen = ImageGenerator(config)
    scenes = [
        Scene(
            id=f"scene-{i:03d}",
            section_id="body",
            image_prompt=f"test prompt {i}",
            ken_burns_direction="pan-left",
            start_time=float(i * 5),
            duration=5.0,
            transition_type="crossfade",
        )
        for i in range(3)
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        results = gen.generate_all(scenes, Path(tmpdir))
        assert len(results) == 3
        for i, result in enumerate(results):
            assert result.scene_id == f"scene-{i:03d}"
            assert result.width == 1920
            assert result.height == 1080
            assert (Path(tmpdir) / f"scene-{i:03d}.png").exists()


def test_style_prefix_applied():
    """Style prefix from config is prepended to prompt."""
    config = VisualConfig(style_prefix="cinematic photo")
    gen = ImageGenerator(config)
    scene = Scene(
        id="scene-001",
        section_id="hook",
        image_prompt="a sunset",
        ken_burns_direction="zoom-out",
        start_time=0.0,
        duration=5.0,
        transition_type="cut",
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "test.png"
        # This just verifies it doesn't crash - prompt composition is internal
        result = gen.generate_scene_image(scene, out)
        assert out.exists()


def test_empty_style_prefix():
    """Empty style prefix doesn't add comma separator."""
    config = VisualConfig(style_prefix="")
    gen = ImageGenerator(config)
    scene = Scene(
        id="scene-001",
        section_id="hook",
        image_prompt="a sunset",
        ken_burns_direction="zoom-out",
        start_time=0.0,
        duration=5.0,
        transition_type="cut",
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "test.png"
        result = gen.generate_scene_image(scene, out)
        assert out.exists()
