"""Tests for the upgraded ThumbnailGenerator (Token Economy style)."""

from pathlib import Path

import pytest
from PIL import Image

from vidgen.config import BrandingConfig
from vidgen.thumbnails import ThumbnailGenerator


@pytest.fixture
def config() -> BrandingConfig:
    return BrandingConfig()


@pytest.fixture
def generator(config: BrandingConfig) -> ThumbnailGenerator:
    return ThumbnailGenerator(config)


@pytest.fixture
def sample_images(tmp_path: Path) -> list[Path]:
    """Create 5 sample scene images for testing."""
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    paths = []
    for i in range(5):
        img = Image.new("RGB", (1920, 1080), color=(50 + i * 30, 40, 80))
        path = images_dir / f"scene_{i + 1:03d}_01.png"
        img.save(str(path))
        paths.append(path)
    return paths


class TestThumbnailGenerator:
    def test_generates_three_variants_by_default(self, generator, tmp_path, sample_images):
        output_dir = tmp_path / "thumbnails"
        paths = generator.generate_variants(
            title="Why 90% of AI Startups Fail",
            style_prefix="tech illustration",
            output_dir=output_dir,
            scene_images=sample_images,
        )
        assert len(paths) == 3
        for p in paths:
            assert p.exists()
            assert p.suffix == ".png"

    def test_output_dimensions_are_1280x720(self, generator, tmp_path, sample_images):
        output_dir = tmp_path / "thumbnails"
        paths = generator.generate_variants(
            title="Test Title",
            style_prefix="",
            output_dir=output_dir,
            scene_images=sample_images,
        )
        for p in paths:
            img = Image.open(str(p))
            assert img.size == (1280, 720)

    def test_custom_thumbnail_text(self, generator, tmp_path, sample_images):
        output_dir = tmp_path / "thumbnails"
        paths = generator.generate_variants(
            title="Full Long Title Here",
            style_prefix="",
            output_dir=output_dir,
            scene_images=sample_images,
            thumbnail_text="90% FAIL",
        )
        assert len(paths) == 3
        # All variants should exist and be valid images
        for p in paths:
            img = Image.open(str(p))
            assert img.mode == "RGB"

    def test_accent_word_and_color(self, generator, tmp_path, sample_images):
        output_dir = tmp_path / "thumbnails"
        paths = generator.generate_variants(
            title="The $50K Automation Secret",
            style_prefix="",
            output_dir=output_dir,
            scene_images=sample_images,
            thumbnail_text="$50K SAVED",
            accent_word="$50K",
            accent_color="#f59e0b",
        )
        assert len(paths) == 3

    def test_works_without_scene_images(self, generator, tmp_path):
        """Falls back to gradient background when no scene images provided."""
        output_dir = tmp_path / "thumbnails"
        paths = generator.generate_variants(
            title="AI Startups Fail",
            style_prefix="",
            output_dir=output_dir,
            scene_images=None,
        )
        assert len(paths) == 3
        for p in paths:
            img = Image.open(str(p))
            assert img.size == (1280, 720)

    def test_works_with_empty_scene_images_list(self, generator, tmp_path):
        output_dir = tmp_path / "thumbnails"
        paths = generator.generate_variants(
            title="Something Cool",
            style_prefix="",
            output_dir=output_dir,
            scene_images=[],
        )
        assert len(paths) == 3

    def test_variant_count_parameter(self, generator, tmp_path, sample_images):
        output_dir = tmp_path / "thumbnails"
        paths = generator.generate_variants(
            title="Test",
            style_prefix="",
            output_dir=output_dir,
            count=5,
            scene_images=sample_images,
        )
        assert len(paths) == 5

    def test_creates_output_directory(self, generator, tmp_path, sample_images):
        output_dir = tmp_path / "nested" / "thumbnails"
        assert not output_dir.exists()
        generator.generate_variants(
            title="Test",
            style_prefix="",
            output_dir=output_dir,
            scene_images=sample_images,
        )
        assert output_dir.exists()

    def test_apply_text_overlay_legacy(self, generator, tmp_path):
        """Test the legacy interface still works."""
        # Create a source image
        source = tmp_path / "source.png"
        Image.new("RGB", (1280, 720), color=(30, 30, 60)).save(str(source))

        output = tmp_path / "output.png"
        result = generator.apply_text_overlay(source, "90% FAIL", output)
        assert result == output
        assert output.exists()
        img = Image.open(str(output))
        assert img.size == (1280, 720)


class TestTextExtraction:
    def test_extracts_parenthetical_hook(self, generator):
        text = generator._extract_thumbnail_text(
            "Why 90% of AI Startups Fail (5 Fatal Patterns Nobody Talks About)"
        )
        # Should extract from parentheses: "5 Fatal Patterns Nobody Talks About"
        # After filtering: "5 Fatal Patterns Talks" or similar
        assert len(text.split()) <= 5
        assert text == text.upper()

    def test_filters_common_words(self, generator):
        text = generator._extract_thumbnail_text("How to Build the Best AI Model in the World")
        words = text.split()
        # Should filter "how", "to", "the", "in"
        assert len(words) <= 4
        for w in words:
            assert w.lower() not in {"how", "to", "the", "in"}

    def test_short_title_unchanged(self, generator):
        text = generator._extract_thumbnail_text("AI Wins Big")
        assert text == "AI WINS BIG"

    def test_always_uppercase(self, generator):
        text = generator._extract_thumbnail_text("Some Random Title About Things")
        assert text == text.upper()


class TestAccentWordPicker:
    def test_picks_numbers(self, generator):
        words = ["90%", "WILL", "FAIL"]
        assert generator._pick_accent_word(words) == "90%"

    def test_picks_power_words(self, generator):
        words = ["STARTUPS", "WILL", "FAIL"]
        assert generator._pick_accent_word(words) == "FAIL"

    def test_picks_longest_when_no_power_word(self, generator):
        words = ["STARTUPS", "TODAY"]
        assert generator._pick_accent_word(words) == "STARTUPS"

    def test_numbers_take_priority_over_power_words(self, generator):
        words = ["$50K", "FAIL", "SAVED"]
        assert generator._pick_accent_word(words) == "$50K"
