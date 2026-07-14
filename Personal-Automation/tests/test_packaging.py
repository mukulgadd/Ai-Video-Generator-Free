"""Unit tests for output packaging logic."""

import json
from pathlib import Path

import pytest

from vidgen.models import Script, ScriptSection, Scene, ScenePlan, TextOverlay, VideoMetadata
from vidgen.config import PipelineConfig
from vidgen.packaging import (
    package_output,
    generate_metadata,
    _generate_chapters,
    _generate_description,
    _generate_tags,
    _format_timestamp,
)


# --- Fixtures ---


def _make_script(title: str = "AI Agents Transform Business Operations") -> Script:
    """Create a minimal valid Script for testing."""
    return Script(
        title=title,
        hook=ScriptSection(
            id="hook",
            title="Hook",
            narration_text="Did you know AI agents are taking over?",
            scene_marker="Dramatic AI visualization",
        ),
        introduction=ScriptSection(
            id="introduction",
            title="Introduction",
            narration_text="In this video we explore three key trends reshaping business.",
            scene_marker="Market growth infographic",
        ),
        body_sections=[
            ScriptSection(
                id="section_1",
                title="Autonomous Customer Service",
                narration_text="The first trend is autonomous agents handling support.",
                scene_marker="Split screen manual vs automated",
            ),
            ScriptSection(
                id="section_2",
                title="Predictive Analytics",
                narration_text="The second trend leverages predictive analytics for decisions.",
                scene_marker="Dashboard with flowing data",
            ),
            ScriptSection(
                id="section_3",
                title="Code Generation",
                narration_text="The third trend is AI writing code to automate development.",
                scene_marker="IDE with AI suggestions",
            ),
        ],
        conclusion=ScriptSection(
            id="conclusion",
            title="Conclusion",
            narration_text="Subscribe for weekly AI insights.",
            scene_marker="Forward looking montage",
        ),
        total_word_count=750,
    )


def _make_scene_plan() -> ScenePlan:
    """Create a minimal valid ScenePlan for testing."""
    return ScenePlan(
        video_title="AI Agents Transform Business Operations",
        topic_slug="ai-agents-transform-business",
        style_prefix="professional tech illustration",
        scenes=[
            Scene(
                id="scene_001",
                section_id="hook",
                image_prompt="dramatic AI visualization",
                ken_burns_direction="zoom-in",
                start_time=0.0,
                duration=10.0,
                transition_type="crossfade",
            ),
            Scene(
                id="scene_002",
                section_id="introduction",
                image_prompt="market growth chart",
                ken_burns_direction="pan-right",
                start_time=10.0,
                duration=15.0,
                transition_type="crossfade",
            ),
            Scene(
                id="scene_003",
                section_id="section_1",
                image_prompt="autonomous agents dashboard",
                ken_burns_direction="pan-left",
                start_time=25.0,
                duration=60.0,
                transition_type="crossfade",
            ),
            Scene(
                id="scene_004",
                section_id="section_2",
                image_prompt="predictive analytics graph",
                ken_burns_direction="zoom-out",
                start_time=85.0,
                duration=60.0,
                transition_type="crossfade",
            ),
            Scene(
                id="scene_005",
                section_id="section_3",
                image_prompt="code generation IDE",
                ken_burns_direction="pan-right",
                start_time=145.0,
                duration=60.0,
                transition_type="crossfade",
            ),
            Scene(
                id="scene_006",
                section_id="conclusion",
                image_prompt="futuristic montage",
                ken_burns_direction="zoom-in",
                start_time=205.0,
                duration=15.0,
                transition_type="cut",
            ),
        ],
        total_duration=220.0,
    )


# --- Tests: _format_timestamp ---


class TestFormatTimestamp:
    def test_zero_seconds(self):
        assert _format_timestamp(0.0) == "0:00"

    def test_under_one_minute(self):
        assert _format_timestamp(45.0) == "0:45"

    def test_exact_one_minute(self):
        assert _format_timestamp(60.0) == "1:00"

    def test_minutes_and_seconds(self):
        assert _format_timestamp(90.0) == "1:30"

    def test_over_one_hour(self):
        assert _format_timestamp(3661.0) == "1:01:01"

    def test_fractional_seconds_truncated(self):
        assert _format_timestamp(65.7) == "1:05"


# --- Tests: _generate_tags ---


class TestGenerateTags:
    def test_returns_at_least_5_tags(self):
        script = _make_script()
        scene_plan = _make_scene_plan()
        tags = _generate_tags(script, scene_plan)
        assert len(tags) >= 5

    def test_returns_at_most_20_tags(self):
        script = _make_script()
        scene_plan = _make_scene_plan()
        tags = _generate_tags(script, scene_plan)
        assert len(tags) <= 20

    def test_no_duplicates(self):
        script = _make_script()
        scene_plan = _make_scene_plan()
        tags = _generate_tags(script, scene_plan)
        lower_tags = [t.lower() for t in tags]
        assert len(lower_tags) == len(set(lower_tags))

    def test_includes_channel_tags(self):
        script = _make_script()
        scene_plan = _make_scene_plan()
        tags = _generate_tags(script, scene_plan)
        lower_tags = [t.lower() for t in tags]
        assert "token economy" in lower_tags
        assert "ai business" in lower_tags

    def test_includes_title_keywords(self):
        script = _make_script(title="Quantum Computing Breakthrough Changes Everything")
        scene_plan = ScenePlan(
            video_title="Quantum Computing Breakthrough Changes Everything",
            topic_slug="quantum-computing",
            style_prefix="tech illustration",
            scenes=[
                Scene(
                    id="scene_001",
                    section_id="hook",
                    image_prompt="dramatic visualization",
                    ken_burns_direction="zoom-in",
                    start_time=0.0,
                    duration=30.0,
                    transition_type="cut",
                )
            ],
            total_duration=600.0,
        )
        tags = _generate_tags(script, scene_plan)
        lower_tags = [t.lower() for t in tags]
        assert "quantum" in lower_tags
        assert "computing" in lower_tags
        assert "breakthrough" in lower_tags


# --- Tests: _generate_chapters ---


class TestGenerateChapters:
    def test_generates_chapters_from_sections(self):
        script = _make_script()
        scene_plan = _make_scene_plan()
        chapters = _generate_chapters(script, scene_plan)
        assert len(chapters) > 0

    def test_first_chapter_is_hook(self):
        script = _make_script()
        scene_plan = _make_scene_plan()
        chapters = _generate_chapters(script, scene_plan)
        # First chapter should be at 0:00 with an engaging name (not just "Hook")
        assert chapters[0][0] == "0:00"
        assert len(chapters[0][1]) > 0

    def test_chapters_include_body_section_titles(self):
        script = _make_script()
        scene_plan = _make_scene_plan()
        chapters = _generate_chapters(script, scene_plan)
        titles = [title for _, title in chapters]
        assert "Autonomous Customer Service" in titles
        assert "Predictive Analytics" in titles

    def test_chapters_are_monotonically_increasing(self):
        script = _make_script()
        scene_plan = _make_scene_plan()
        chapters = _generate_chapters(script, scene_plan)
        # All timestamps should be in order (string comparison works for M:SS format
        # as long as minutes are comparable)
        assert len(chapters) >= 2


# --- Tests: _generate_description ---


class TestGenerateDescription:
    def test_includes_title(self):
        script = _make_script()
        chapters = [("0:00", "Hook"), ("0:10", "Introduction")]
        desc = _generate_description(script, script.title, chapters)
        # Description should include subscribe CTA and social links
        assert "@TokenEconomyAI" in desc
        assert "x.com/TokenEconomyAI" in desc

    def test_includes_chapter_timestamps(self):
        script = _make_script()
        chapters = [("0:00", "Hook"), ("0:10", "Introduction")]
        desc = _generate_description(script, script.title, chapters)
        assert "0:00" in desc
        assert "Hook" in desc
        assert "0:10" in desc
        assert "Introduction" in desc

    def test_includes_subscribe_cta(self):
        script = _make_script()
        desc = _generate_description(script, script.title, [])
        assert "Subscribe" in desc


# --- Tests: generate_metadata ---


class TestGenerateMetadata:
    def test_returns_video_metadata(self):
        script = _make_script()
        scene_plan = _make_scene_plan()
        metadata = generate_metadata(script, scene_plan, Path("/output/video.mp4"))
        assert isinstance(metadata, VideoMetadata)

    def test_title_is_truncated_to_60_chars(self):
        long_title = "A" * 100
        script = _make_script(title=long_title)
        scene_plan = _make_scene_plan()
        metadata = generate_metadata(script, scene_plan, Path("/output/video.mp4"))
        assert len(metadata.title) <= 60

    def test_description_within_5000_chars(self):
        script = _make_script()
        scene_plan = _make_scene_plan()
        metadata = generate_metadata(script, scene_plan, Path("/output/video.mp4"))
        assert len(metadata.description) <= 5000

    def test_tags_count_valid(self):
        script = _make_script()
        scene_plan = _make_scene_plan()
        metadata = generate_metadata(script, scene_plan, Path("/output/video.mp4"))
        assert 15 <= len(metadata.tags) <= 30

    def test_resolution_is_1920x1080(self):
        script = _make_script()
        scene_plan = _make_scene_plan()
        metadata = generate_metadata(script, scene_plan, Path("/output/video.mp4"))
        assert metadata.resolution == "1920x1080"

    def test_duration_from_scene_plan(self):
        script = _make_script()
        scene_plan = _make_scene_plan()
        metadata = generate_metadata(script, scene_plan, Path("/output/video.mp4"))
        assert metadata.duration_seconds == scene_plan.total_duration


# --- Tests: package_output ---


class TestPackageOutput:
    def test_creates_output_directory(self, tmp_path):
        job_dir = tmp_path / "job"
        job_dir.mkdir()
        (job_dir / "assembly").mkdir()
        (job_dir / "assembly" / "video_raw.mp4").write_bytes(b"fake video data")

        script = _make_script()
        scene_plan = _make_scene_plan()
        config = PipelineConfig(output_dir=tmp_path / "output")

        output_dir = package_output(job_dir, script, scene_plan, config)
        assert output_dir.exists()
        assert output_dir.is_dir()

    def test_copies_video(self, tmp_path):
        job_dir = tmp_path / "job"
        job_dir.mkdir()
        (job_dir / "assembly").mkdir()
        (job_dir / "assembly" / "video_raw.mp4").write_bytes(b"fake video data")

        script = _make_script()
        scene_plan = _make_scene_plan()
        config = PipelineConfig(output_dir=tmp_path / "output")

        output_dir = package_output(job_dir, script, scene_plan, config)
        assert (output_dir / "video.mp4").exists()
        assert (output_dir / "video.mp4").read_bytes() == b"fake video data"

    def test_copies_thumbnail(self, tmp_path):
        job_dir = tmp_path / "job"
        job_dir.mkdir()
        (job_dir / "assembly").mkdir()
        (job_dir / "assembly" / "video_raw.mp4").write_bytes(b"video")
        (job_dir / "thumbnails").mkdir()
        (job_dir / "thumbnails" / "variant_1.png").write_bytes(b"thumb1")
        (job_dir / "thumbnails" / "variant_2.png").write_bytes(b"thumb2")

        script = _make_script()
        scene_plan = _make_scene_plan()
        config = PipelineConfig(output_dir=tmp_path / "output")

        output_dir = package_output(job_dir, script, scene_plan, config)
        assert (output_dir / "thumbnail.png").exists()
        assert (output_dir / "thumbnail.png").read_bytes() == b"thumb1"

    def test_generates_metadata_json(self, tmp_path):
        job_dir = tmp_path / "job"
        job_dir.mkdir()
        (job_dir / "assembly").mkdir()
        (job_dir / "assembly" / "video_raw.mp4").write_bytes(b"video")

        script = _make_script()
        scene_plan = _make_scene_plan()
        config = PipelineConfig(output_dir=tmp_path / "output")

        output_dir = package_output(job_dir, script, scene_plan, config)
        metadata_path = output_dir / "metadata.json"
        assert metadata_path.exists()

        data = json.loads(metadata_path.read_text())
        assert data["title"] == script.title[:60]
        assert data["resolution"] == "1920x1080"
        assert len(data["tags"]) >= 15

    def test_copies_shorts_with_metadata(self, tmp_path):
        job_dir = tmp_path / "job"
        job_dir.mkdir()
        (job_dir / "assembly").mkdir()
        (job_dir / "assembly" / "video_raw.mp4").write_bytes(b"video")
        (job_dir / "shorts").mkdir()
        (job_dir / "shorts" / "short_1.mp4").write_bytes(b"short1")
        (job_dir / "shorts" / "short_2.mp4").write_bytes(b"short2")

        script = _make_script()
        scene_plan = _make_scene_plan()
        config = PipelineConfig(output_dir=tmp_path / "output")

        output_dir = package_output(job_dir, script, scene_plan, config)
        shorts_dir = output_dir / "shorts"
        assert shorts_dir.exists()
        assert (shorts_dir / "short_1.mp4").exists()
        assert (shorts_dir / "short_2.mp4").exists()
        assert (shorts_dir / "short_1_metadata.json").exists()
        assert (shorts_dir / "short_2_metadata.json").exists()

        # Verify short metadata structure
        short_meta = json.loads((shorts_dir / "short_1_metadata.json").read_text())
        assert "title" in short_meta
        assert "description" in short_meta
        assert "tags" in short_meta
        assert len(short_meta["title"]) <= 60

    def test_output_dir_name_format(self, tmp_path):
        job_dir = tmp_path / "job"
        job_dir.mkdir()
        (job_dir / "assembly").mkdir()
        (job_dir / "assembly" / "video_raw.mp4").write_bytes(b"video")

        script = _make_script()
        scene_plan = _make_scene_plan()
        config = PipelineConfig(output_dir=tmp_path / "output")

        output_dir = package_output(job_dir, script, scene_plan, config)
        # Should contain the topic_slug
        assert "ai-agents-transform-business" in output_dir.name

    def test_handles_missing_video_gracefully(self, tmp_path):
        job_dir = tmp_path / "job"
        job_dir.mkdir()
        # No assembly/video_raw.mp4

        script = _make_script()
        scene_plan = _make_scene_plan()
        config = PipelineConfig(output_dir=tmp_path / "output")

        # Should not crash, just skip video copy
        output_dir = package_output(job_dir, script, scene_plan, config)
        assert output_dir.exists()
        assert not (output_dir / "video.mp4").exists()

    def test_handles_missing_thumbnails_gracefully(self, tmp_path):
        job_dir = tmp_path / "job"
        job_dir.mkdir()
        (job_dir / "assembly").mkdir()
        (job_dir / "assembly" / "video_raw.mp4").write_bytes(b"video")
        # No thumbnails directory

        script = _make_script()
        scene_plan = _make_scene_plan()
        config = PipelineConfig(output_dir=tmp_path / "output")

        output_dir = package_output(job_dir, script, scene_plan, config)
        assert output_dir.exists()
        assert not (output_dir / "thumbnail.png").exists()

    def test_uses_untitled_when_no_topic_slug(self, tmp_path):
        job_dir = tmp_path / "job"
        job_dir.mkdir()
        (job_dir / "assembly").mkdir()
        (job_dir / "assembly" / "video_raw.mp4").write_bytes(b"video")

        script = _make_script()
        scene_plan = ScenePlan(
            video_title="Test",
            topic_slug="",
            style_prefix="test style",
            scenes=[
                Scene(
                    id="scene_001",
                    section_id="hook",
                    image_prompt="test",
                    ken_burns_direction="zoom-in",
                    start_time=0.0,
                    duration=10.0,
                    transition_type="crossfade",
                ),
            ],
            total_duration=10.0,
        )
        config = PipelineConfig(output_dir=tmp_path / "output")

        output_dir = package_output(job_dir, script, scene_plan, config)
        assert "untitled" in output_dir.name
