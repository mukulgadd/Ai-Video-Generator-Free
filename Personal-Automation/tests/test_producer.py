"""Tests for the unified content producer."""

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from vidgen.producer import ContentProducer, ProduceResult


VALID_SCRIPT = """\
---
title: "Why 90% of AI Startups Will Fail by 2028"
topic_slug: why-ai-startups-fail
target_duration_minutes: 11
niche: ai-tech-business
---

## Hook

Jasper AI raised 125 million dollars at a 1.5 billion valuation in 2022. By 2025 they laid off most staff.

## Introduction

In this video we identify five failure patterns.

## Section 1: The Funding Mirage

The first pattern is the funding mirage. In 1999, raising a hundred million felt like winning. CB Insights reports 400 of 1200 funded AI startups shut down.

## Section 2: The Thin Wrapper

A thin wrapper takes a model and adds a prompt. No moat. Jasper built on GPT-3. ChatGPT launched for free. Revenue dropped forty percent.

## Conclusion

The ninety percent failure rate is a prediction of concentration. Subscribe for weekly analysis.
"""

VALID_SCENE_PLAN = """\
{
  "video_title": "Why 90% of AI Startups Will Fail by 2028",
  "topic_slug": "why-ai-startups-fail",
  "style_prefix": "dark cinematic tech illustration",
  "scenes": [
    {"id": "scene_001", "section_id": "hook", "image_prompt": "startup graveyard", "ken_burns": "zoom-in", "start_time": 0, "duration": 30, "transition": "crossfade"},
    {"id": "scene_002", "section_id": "section_1", "image_prompt": "champagne money", "ken_burns": "pan-left", "start_time": 30, "duration": 60, "transition": "crossfade"},
    {"id": "scene_003", "section_id": "section_2", "image_prompt": "thin wrapper", "ken_burns": "zoom-out", "start_time": 90, "duration": 60, "transition": "crossfade"},
    {"id": "scene_004", "section_id": "conclusion", "image_prompt": "concentration", "ken_burns": "pan-right", "start_time": 150, "duration": 30, "transition": "cut"}
  ],
  "total_duration_seconds": 180
}
"""


@pytest.fixture
def script_path(tmp_path: Path) -> Path:
    path = tmp_path / "003_why_ai_startups_fail.md"
    path.write_text(VALID_SCRIPT)
    return path


@pytest.fixture
def scene_plan_path(tmp_path: Path) -> Path:
    path = tmp_path / "003_why_ai_startups_fail_plan.json"
    path.write_text(VALID_SCENE_PLAN)
    return path


@pytest.fixture
def shorts_dir(tmp_path: Path) -> Path:
    """Create mock short scripts matching number 003."""
    sdir = tmp_path / "shorts"
    sdir.mkdir()
    for i, name in enumerate(["thin_wrapper", "capability_treadmill", "survival_signs"], 1):
        (sdir / f"003_short_{i}_{name}.md").write_text(f"""\
---
title: "Short {i}"
duration_target: 35
source_video: "003"
music_mood: "suspense"
style_prefix: "dark cinematic"
hook_text: "Hook {i}"
---

## Narration

This is narration for short number {i}. It needs at least thirty words to pass validation so we add more words here about AI startups and failure patterns and data.

## Text Overlays

| time | text | style |
|------|------|-------|
| 0 | Test | impact |
| 5 | More | normal |

## Images

- A dark scene one, no text, no numbers, no readable labels
- A dark scene two, no text, no numbers, no readable labels
- A dark scene three, no text, no numbers, no readable labels
""")
    return sdir


class TestContentProducerInit:
    def test_init_default_config(self):
        producer = ContentProducer(config_path=Path("/nonexistent"))
        assert producer.config is not None

    def test_init_with_real_config(self):
        config = Path("config.yaml")
        if config.exists():
            producer = ContentProducer(config_path=config)
            assert producer.config.shorts is not None


class TestExtractNumber:
    def test_standard_format(self):
        producer = ContentProducer(config_path=Path("/nonexistent"))
        assert producer._extract_number(Path("scripts/003_why_ai_startups_fail.md")) == "003"

    def test_single_digit(self):
        producer = ContentProducer(config_path=Path("/nonexistent"))
        assert producer._extract_number(Path("1_test.md")) == "1"

    def test_no_number(self):
        producer = ContentProducer(config_path=Path("/nonexistent"))
        assert producer._extract_number(Path("no_number_here.md")) == "000"


class TestSlugifyTitle:
    def test_long_title_truncated(self):
        producer = ContentProducer(config_path=Path("/nonexistent"))
        result = producer._slugify_title(
            "Why 90% of AI Startups Will Fail by 2028 — And the Pattern That Predicts"
        )
        # Should be short and readable
        assert len(result.split()) <= 6
        assert "—" not in result

    def test_removes_percentage(self):
        producer = ContentProducer(config_path=Path("/nonexistent"))
        result = producer._slugify_title("Why 90% of AI Startups Fail")
        assert "90%" not in result
        assert "AI Startups" in result

    def test_simple_title(self):
        producer = ContentProducer(config_path=Path("/nonexistent"))
        result = producer._slugify_title("Open Source vs Closed AI")
        assert result == "Open Source vs Closed AI"


class TestProduceDistribution:
    def test_generates_all_files(self, script_path: Path, tmp_path: Path):
        from vidgen.parsers import parse_script

        producer = ContentProducer(config_path=Path("/nonexistent"))
        script = parse_script(script_path)
        output_dir = tmp_path / "output" / "test"
        output_dir.mkdir(parents=True)

        ok = producer._produce_distribution(script, output_dir, "https://youtu.be/test")
        assert ok is True

        dist_dir = output_dir / "distribution"
        assert (dist_dir / "thread.txt").exists()
        assert (dist_dir / "newsletter.md").exists()
        assert (dist_dir / "linkedin_post.txt").exists()

    def test_thread_contains_cta(self, script_path: Path, tmp_path: Path):
        from vidgen.parsers import parse_script

        producer = ContentProducer(config_path=Path("/nonexistent"))
        script = parse_script(script_path)
        output_dir = tmp_path / "output" / "test"
        output_dir.mkdir(parents=True)

        producer._produce_distribution(script, output_dir, "https://youtu.be/xyz")
        thread = (output_dir / "distribution" / "thread.txt").read_text()
        assert "@TokenEconomyAI" in thread
        assert "https://youtu.be/xyz" in thread

    def test_newsletter_has_structure(self, script_path: Path, tmp_path: Path):
        from vidgen.parsers import parse_script

        producer = ContentProducer(config_path=Path("/nonexistent"))
        script = parse_script(script_path)
        output_dir = tmp_path / "output" / "test"
        output_dir.mkdir(parents=True)

        producer._produce_distribution(script, output_dir, "")
        newsletter = (output_dir / "distribution" / "newsletter.md").read_text()
        assert "## TL;DR" in newsletter
        assert "## The Bottom Line" in newsletter
        assert "tokeneconomyai.substack.com" in newsletter


class TestProduceShorts:
    @patch("vidgen.shorts_pipeline.ShortsPipeline")
    def test_finds_matching_shorts(self, mock_pipeline_cls, script_path, shorts_dir, tmp_path):
        mock_pipeline = MagicMock()
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.output_path = tmp_path / "fake_short.mp4"
        mock_result.output_path.write_bytes(b"x" * 100_000)
        mock_pipeline.run.return_value = mock_result
        mock_pipeline_cls.return_value = mock_pipeline

        producer = ContentProducer(config_path=Path("/nonexistent"))
        output_dir = tmp_path / "output" / "test"
        output_dir.mkdir(parents=True)

        completed, total = producer._produce_shorts(
            script_path, output_dir, shorts_dir, resume=False
        )
        assert total == 3
        assert completed == 3
        assert mock_pipeline.run.call_count == 3

    def test_no_shorts_returns_zero(self, script_path, tmp_path):
        producer = ContentProducer(config_path=Path("/nonexistent"))
        output_dir = tmp_path / "output" / "test"
        output_dir.mkdir(parents=True)

        completed, total = producer._produce_shorts(
            script_path, output_dir, tmp_path / "empty_shorts", resume=False
        )
        assert total == 0
        assert completed == 0

    @patch("vidgen.shorts_pipeline.ShortsPipeline")
    def test_resume_skips_existing(self, mock_pipeline_cls, script_path, shorts_dir, tmp_path):
        producer = ContentProducer(config_path=Path("/nonexistent"))
        output_dir = tmp_path / "output" / "test"
        shorts_output = output_dir / "shorts"
        shorts_output.mkdir(parents=True)

        # Pre-create 3 valid short MP4s
        for i in range(1, 4):
            (shorts_output / f"003_short_{i}_test.mp4").write_bytes(b"x" * 100_000)

        completed, total = producer._produce_shorts(
            script_path, output_dir, shorts_dir, resume=True
        )
        # Should skip all — pipeline never called
        assert completed == 3
        assert total == 3
        mock_pipeline_cls.assert_not_called()


class TestProduceVideo:
    @patch("vidgen.pipeline.PipelineOrchestrator")
    def test_copies_video_on_success(self, mock_orch_cls, script_path, scene_plan_path, tmp_path):
        from vidgen.parsers import parse_scene_plan, parse_script

        script = parse_script(script_path)
        scene_plan = parse_scene_plan(scene_plan_path)

        # Mock orchestrator
        mock_orch = MagicMock()
        mock_result = MagicMock()
        mock_result.success = True
        mock_orch.run.return_value = mock_result
        mock_orch_cls.return_value = mock_orch

        producer = ContentProducer(config_path=Path("/nonexistent"))
        producer.config.jobs_dir = tmp_path / "jobs"

        # Create fake video artifact where pipeline would put it
        job_dir = tmp_path / "jobs" / "produce_why-ai-startups-fail"
        (job_dir / "assembly").mkdir(parents=True)
        (job_dir / "assembly" / "video_raw.mp4").write_bytes(b"video" * 100_000)
        (job_dir / "thumbnails").mkdir()
        (job_dir / "thumbnails" / "variant_1.png").write_bytes(b"thumb" * 1000)

        output_dir = tmp_path / "output" / "test"
        output_dir.mkdir(parents=True)

        ok = producer._produce_video(script, scene_plan, output_dir, resume=False)
        assert ok is True
        assert (output_dir / "video" / "final.mp4").exists()
        assert (output_dir / "video" / "thumbnail.png").exists()

    def test_resume_skips_existing_video(self, script_path, scene_plan_path, tmp_path):
        from vidgen.parsers import parse_scene_plan, parse_script

        script = parse_script(script_path)
        scene_plan = parse_scene_plan(scene_plan_path)

        producer = ContentProducer(config_path=Path("/nonexistent"))
        output_dir = tmp_path / "output" / "test"
        video_dir = output_dir / "video"
        video_dir.mkdir(parents=True)
        (video_dir / "final.mp4").write_bytes(b"v" * 200_000)

        ok = producer._produce_video(script, scene_plan, output_dir, resume=True)
        assert ok is True  # Skipped successfully


class TestProduceMetadata:
    def test_writes_metadata_json(self, script_path, scene_plan_path, tmp_path):
        from vidgen.parsers import parse_scene_plan, parse_script

        script = parse_script(script_path)
        scene_plan = parse_scene_plan(scene_plan_path)

        producer = ContentProducer(config_path=Path("/nonexistent"))
        output_dir = tmp_path / "output" / "test"
        output_dir.mkdir(parents=True)

        producer._write_metadata(script, scene_plan, output_dir, "https://youtu.be/abc")

        meta_path = output_dir / "metadata.json"
        assert meta_path.exists()
        data = json.loads(meta_path.read_text())
        assert data["video_url"] == "https://youtu.be/abc"
        assert "platforms" in data
        assert data["platforms"]["substack"] == "https://tokeneconomyai.substack.com"


class TestFullProduce:
    @patch("vidgen.producer.ContentProducer._produce_video")
    @patch("vidgen.producer.ContentProducer._produce_shorts")
    def test_full_produce_integrates_all_stages(
        self, mock_shorts, mock_video, script_path, scene_plan_path, tmp_path
    ):
        mock_video.return_value = True
        mock_shorts.return_value = (3, 3)

        producer = ContentProducer(config_path=Path("/nonexistent"))
        producer.config.output_dir = tmp_path / "output"

        result = producer.produce(
            script_path=script_path,
            scene_plan_path=scene_plan_path,
            video_url="https://youtu.be/test",
        )

        assert result.success is True
        assert result.video_ok is True
        assert result.shorts_completed == 3
        assert result.distribution_ok is True
        assert result.output_dir is not None
        assert (result.output_dir / "distribution" / "thread.txt").exists()
        assert (result.output_dir / "distribution" / "newsletter.md").exists()
        assert (result.output_dir / "metadata.json").exists()
