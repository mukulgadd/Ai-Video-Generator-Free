"""Tests for the shorts pipeline orchestrator."""

import subprocess
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from vidgen.config import PipelineConfig, ShortsConfig
from vidgen.models import ShortGenerationResult, ShortScript
from vidgen.shorts_parser import ShortParseError, parse_short_script
from vidgen.shorts_pipeline import ShortsPipeline


VALID_SCRIPT_CONTENT = """\
---
title: "Pipeline Test Short"
duration_target: 35
source_video: "test"
music_mood: "neutral"
style_prefix: "dark cinematic, vertical composition"
hook_text: "Hook text"
---

## Narration

This is the narration text for the pipeline test. It needs to be at least thirty words to pass validation. Here we add some more context about AI and technology to make it sound realistic for a tech channel.

## Text Overlays

| time | text | style |
|------|------|-------|
| 0 | Test overlay | impact |
| 5 | Second | normal |

## Images

- A dark cinematic scene one, vertical composition, no text, no numbers, no readable labels
- A dark cinematic scene two, vertical composition, no text, no numbers, no readable labels
- A dark cinematic scene three, vertical composition, no text, no numbers, no readable labels
"""


@pytest.fixture
def script_path(tmp_path: Path) -> Path:
    path = tmp_path / "test_short.md"
    path.write_text(VALID_SCRIPT_CONTENT)
    return path


@pytest.fixture
def config_path(tmp_path: Path) -> Path:
    """Create a minimal config.yaml for testing."""
    config = tmp_path / "config.yaml"
    config.write_text("""\
voice:
  model: "test-model"
  speaker: "aiden"
visual:
  image_model: "test"
shorts:
  output_dir: "{output_dir}"
  music_dir: "{music_dir}"
""".format(
        output_dir=str(tmp_path / "output" / "shorts"),
        music_dir=str(tmp_path / "music"),
    ))
    return config


class TestShortsPipelineInit:
    def test_init_with_default_config(self):
        pipeline = ShortsPipeline(config_path=Path("/nonexistent/config.yaml"))
        assert pipeline.shorts_config is not None
        assert pipeline.shorts_config.image_output_width == 1080

    def test_init_loads_config_file(self, config_path: Path):
        pipeline = ShortsPipeline(config_path=config_path)
        assert pipeline.config is not None


class TestShortsPipelineValidate:
    def test_validate_valid_script(self, script_path: Path):
        pipeline = ShortsPipeline(config_path=Path("/nonexistent"))
        script = pipeline._validate(script_path)
        assert script.title == "Pipeline Test Short"
        assert script.duration_target == 35

    def test_validate_invalid_script_raises(self, tmp_path: Path):
        bad = tmp_path / "bad.md"
        bad.write_text("not a valid script")
        pipeline = ShortsPipeline(config_path=Path("/nonexistent"))
        with pytest.raises(ShortParseError):
            pipeline._validate(bad)


class TestShortsPipelineFindMusic:
    def test_find_specific_track(self, tmp_path: Path):
        music_dir = tmp_path / "music"
        music_dir.mkdir()
        track = music_dir / "custom.mp3"
        track.write_bytes(b"fake mp3")

        pipeline = ShortsPipeline(config_path=Path("/nonexistent"))
        pipeline.shorts_config = ShortsConfig(music_dir=music_dir)

        result = pipeline._find_music("neutral", "custom.mp3")
        assert result == track

    def test_find_mood_track(self, tmp_path: Path):
        music_dir = tmp_path / "music"
        music_dir.mkdir()
        track = music_dir / "suspense.mp3"
        track.write_bytes(b"fake mp3")

        pipeline = ShortsPipeline(config_path=Path("/nonexistent"))
        pipeline.shorts_config = ShortsConfig(music_dir=music_dir)

        result = pipeline._find_music("suspense", None)
        assert result == track

    def test_find_fallback_any_track(self, tmp_path: Path):
        music_dir = tmp_path / "music"
        music_dir.mkdir()
        track = music_dir / "random.mp3"
        track.write_bytes(b"fake mp3")

        pipeline = ShortsPipeline(config_path=Path("/nonexistent"))
        pipeline.shorts_config = ShortsConfig(music_dir=music_dir)

        result = pipeline._find_music("nonexistent_mood", None)
        assert result == track

    def test_find_no_music_returns_none(self, tmp_path: Path):
        music_dir = tmp_path / "empty_music"

        pipeline = ShortsPipeline(config_path=Path("/nonexistent"))
        pipeline.shorts_config = ShortsConfig(music_dir=music_dir)

        result = pipeline._find_music("neutral", None)
        assert result is None


class TestShortsPipelineRun:
    """Test the full run() with mocked heavy stages."""

    @patch("vidgen.shorts_pipeline.ShortsPipeline._narrate")
    @patch("vidgen.shorts_pipeline.ShortsPipeline._generate_images")
    @patch("vidgen.shorts_pipeline.ShortsPipeline._assemble")
    def test_run_success_with_mocked_stages(
        self, mock_assemble, mock_images, mock_narrate, script_path, tmp_path
    ):
        # Set up mocks
        narration_wav = tmp_path / "narration.wav"
        subprocess.run(
            ["ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono",
             "-t", "5", str(narration_wav)],
            capture_output=True, check=True,
        )
        mock_narrate.return_value = narration_wav

        img = tmp_path / "img.png"
        from PIL import Image
        Image.new("RGB", (1080, 1920), (50, 50, 50)).save(img)
        mock_images.return_value = [img, img, img]

        assembly_mp4 = tmp_path / "assembly" / "short.mp4"
        assembly_mp4.parent.mkdir(parents=True)
        assembly_mp4.write_bytes(b"fake mp4 content " * 100)
        mock_assemble.return_value = assembly_mp4

        pipeline = ShortsPipeline(config_path=Path("/nonexistent"))
        pipeline.shorts_config = ShortsConfig(output_dir=tmp_path / "output" / "shorts")
        pipeline.config.jobs_dir = tmp_path / "jobs"

        result = pipeline.run(script_path)
        assert result.success is True
        assert result.output_path is not None
        assert result.output_path.exists()

    def test_run_failure_returns_error(self, tmp_path):
        bad_script = tmp_path / "bad.md"
        bad_script.write_text("not valid")

        pipeline = ShortsPipeline(config_path=Path("/nonexistent"))
        result = pipeline.run(bad_script)
        assert result.success is False
        assert result.error is not None
        assert "failed" in result.error.lower() or "parse" in result.error.lower()


class TestShortsPipelineRunBatch:
    @patch("vidgen.shorts_pipeline.ShortsPipeline.run")
    def test_batch_processes_all_scripts(self, mock_run, tmp_path):
        mock_run.return_value = ShortGenerationResult(
            success=True,
            output_path=tmp_path / "out.mp4",
            duration_seconds=60.0,
        )

        scripts = [tmp_path / f"script_{i}.md" for i in range(3)]
        for s in scripts:
            s.write_text(VALID_SCRIPT_CONTENT)

        pipeline = ShortsPipeline(config_path=Path("/nonexistent"))
        results = pipeline.run_batch(scripts)

        assert len(results) == 3
        assert all(r.success for r in results)
        assert mock_run.call_count == 3

    @patch("vidgen.shorts_pipeline.ShortsPipeline.run")
    def test_batch_continues_on_failure(self, mock_run, tmp_path):
        """A failed Short should not stop the batch."""
        mock_run.side_effect = [
            ShortGenerationResult(success=True, output_path=tmp_path / "a.mp4", duration_seconds=30),
            ShortGenerationResult(success=False, error="stage failed", duration_seconds=5),
            ShortGenerationResult(success=True, output_path=tmp_path / "c.mp4", duration_seconds=30),
        ]

        scripts = [tmp_path / f"s{i}.md" for i in range(3)]
        for s in scripts:
            s.write_text("x")

        pipeline = ShortsPipeline(config_path=Path("/nonexistent"))
        results = pipeline.run_batch(scripts)

        assert len(results) == 3
        assert results[0].success is True
        assert results[1].success is False
        assert results[2].success is True
