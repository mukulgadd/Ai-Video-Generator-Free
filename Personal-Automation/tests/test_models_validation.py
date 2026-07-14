"""Test that model validators reject invalid input correctly."""

import pytest
from pydantic import ValidationError

from vidgen.models import (
    JobState,
    QualityCheck,
    QualityReport,
    Scene,
    ScenePlan,
    Script,
    ScriptSection,
    ShortSegment,
    TextOverlay,
    VideoMetadata,
)


class TestSceneValidation:
    def test_invalid_ken_burns_direction(self):
        with pytest.raises(ValidationError, match="ken_burns_direction"):
            Scene(
                id="s", section_id="s", image_prompt="p",
                ken_burns_direction="invalid", start_time=0, duration=10, transition_type="cut",
            )

    def test_invalid_transition_type(self):
        with pytest.raises(ValidationError, match="transition_type"):
            Scene(
                id="s", section_id="s", image_prompt="p",
                ken_burns_direction="zoom-in", start_time=0, duration=10, transition_type="wipe",
            )

    def test_negative_duration(self):
        with pytest.raises(ValidationError):
            Scene(
                id="s", section_id="s", image_prompt="p",
                ken_burns_direction="zoom-in", start_time=0, duration=-5, transition_type="cut",
            )

    def test_negative_start_time(self):
        with pytest.raises(ValidationError):
            Scene(
                id="s", section_id="s", image_prompt="p",
                ken_burns_direction="zoom-in", start_time=-1, duration=10, transition_type="cut",
            )


class TestTextOverlayValidation:
    def test_invalid_position(self):
        with pytest.raises(ValidationError, match="Position must be one of"):
            TextOverlay(text="hi", position="left", appear_at=0, duration=5)

    def test_negative_duration(self):
        with pytest.raises(ValidationError):
            TextOverlay(text="hi", position="top", appear_at=0, duration=-1)


class TestJobStateValidation:
    def test_invalid_status(self):
        with pytest.raises(ValidationError, match="Job status"):
            JobState(job_id="x", status="running")

    def test_all_valid_statuses(self):
        for status in ["queued", "in-progress", "completed", "failed", "timed-out"]:
            state = JobState(job_id="x", status=status)
            assert state.status == status


class TestVideoMetadataValidation:
    def test_too_few_tags(self):
        with pytest.raises(ValidationError, match="Tags must have 5-30"):
            VideoMetadata(
                title="T", description="D", tags=["a", "b"],
                duration_seconds=100, resolution="1920x1080", file_path="f",
            )

    def test_too_many_tags(self):
        with pytest.raises(ValidationError, match="Tags must have 5-30"):
            VideoMetadata(
                title="T", description="D", tags=["t"] * 31,
                duration_seconds=100, resolution="1920x1080", file_path="f",
            )

    def test_title_too_long(self):
        with pytest.raises(ValidationError):
            VideoMetadata(
                title="x" * 101, description="D", tags=["t"] * 15,
                duration_seconds=100, resolution="r", file_path="f",
            )

    def test_description_too_long(self):
        with pytest.raises(ValidationError):
            VideoMetadata(
                title="T", description="x" * 5001, tags=["t"] * 15,
                duration_seconds=100, resolution="r", file_path="f",
            )


class TestShortSegmentValidation:
    def test_end_time_before_start_time(self):
        with pytest.raises(ValidationError, match="end_time.*must be greater than start_time"):
            ShortSegment(
                start_time=100, end_time=50, duration=50,
                source_scenes=["s1"], hook_caption="h", title="t", description="d", tags=[],
            )

    def test_equal_start_and_end_time(self):
        with pytest.raises(ValidationError):
            ShortSegment(
                start_time=100, end_time=100, duration=0,
                source_scenes=["s1"], hook_caption="h", title="t", description="d", tags=[],
            )


class TestQualityReportValidation:
    def test_inconsistent_passed_with_failed_checks(self):
        check = QualityCheck(name="test", passed=False, details="failed")
        with pytest.raises(ValidationError, match="cannot be marked passed"):
            QualityReport(passed=True, checks=[check], timestamp="2024-01-01")

    def test_consistent_all_passed(self):
        check = QualityCheck(name="test", passed=True, details="ok")
        report = QualityReport(passed=True, checks=[check], timestamp="2024-01-01")
        assert report.passed is True

    def test_consistent_some_failed(self):
        checks = [
            QualityCheck(name="a", passed=True, details="ok"),
            QualityCheck(name="b", passed=False, details="fail"),
        ]
        report = QualityReport(passed=False, checks=checks, timestamp="2024-01-01")
        assert report.passed is False


class TestScriptValidation:
    def test_empty_body_sections(self):
        with pytest.raises(ValidationError, match="at least 1 body section"):
            Script(
                title="T",
                hook=ScriptSection(id="h", title="H", narration_text="t", scene_marker="v"),
                introduction=ScriptSection(id="i", title="I", narration_text="t", scene_marker="v"),
                body_sections=[],
                conclusion=ScriptSection(id="c", title="C", narration_text="t", scene_marker="v"),
                total_word_count=100,
            )

    def test_negative_word_count(self):
        with pytest.raises(ValidationError):
            Script(
                title="T",
                hook=ScriptSection(id="h", title="H", narration_text="t", scene_marker="v"),
                introduction=ScriptSection(id="i", title="I", narration_text="t", scene_marker="v"),
                body_sections=[ScriptSection(id="s1", title="S", narration_text="t", scene_marker="v")],
                conclusion=ScriptSection(id="c", title="C", narration_text="t", scene_marker="v"),
                total_word_count=-1,
            )


class TestScriptSectionValidation:
    def test_invalid_emphasis_markers_negative(self):
        with pytest.raises(ValidationError, match="non-negative"):
            ScriptSection(id="x", title="X", narration_text="text", scene_marker="v", emphasis_markers=[(-1, 5)])

    def test_invalid_emphasis_markers_start_gte_end(self):
        with pytest.raises(ValidationError, match="start must be less than end"):
            ScriptSection(id="x", title="X", narration_text="text", scene_marker="v", emphasis_markers=[(5, 5)])


class TestScenePlanValidation:
    def test_empty_scenes(self):
        with pytest.raises(ValidationError, match="at least one scene"):
            ScenePlan(video_title="T", style_prefix="s", scenes=[], total_duration=100)
