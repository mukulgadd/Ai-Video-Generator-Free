"""Smoke tests for the QualityController module."""

import tempfile
from pathlib import Path

import pytest

from vidgen.config import PipelineConfig
from vidgen.models import (
    QualityCheck,
    QualityReport,
    Scene,
    ScenePlan,
    Script,
    ScriptSection,
    TextOverlay,
)
from vidgen.quality import QualityController


@pytest.fixture
def config():
    return PipelineConfig()


@pytest.fixture
def controller(config):
    return QualityController(config)


@pytest.fixture
def sample_script():
    return Script(
        title="Test Video",
        hook=ScriptSection(
            id="hook",
            title="Hook",
            narration_text="This is a hook to grab attention quickly.",
            scene_marker="[SCENE: dramatic opening]",
        ),
        introduction=ScriptSection(
            id="introduction",
            title="Introduction",
            narration_text="Welcome to this video about testing.",
            scene_marker="[SCENE: intro graphic]",
        ),
        body_sections=[
            ScriptSection(
                id="section_1",
                title="Section 1",
                narration_text="This is the body content of the video.",
                scene_marker="[SCENE: main content]",
            ),
        ],
        conclusion=ScriptSection(
            id="conclusion",
            title="Conclusion",
            narration_text="Thanks for watching.",
            scene_marker="[SCENE: outro]",
        ),
        total_word_count=200,
    )


@pytest.fixture
def sample_scene_plan():
    return ScenePlan(
        video_title="Test Video",
        style_prefix="test style",
        scenes=[
            Scene(
                id="scene_001",
                section_id="hook",
                image_prompt="dramatic opening",
                ken_burns_direction="zoom-in",
                start_time=0.0,
                duration=10.0,
                transition_type="crossfade",
            ),
        ],
        total_duration=60.0,
    )


class TestQualityControllerInstantiation:
    def test_creates_with_config(self, controller):
        assert controller.config is not None

    def test_has_all_check_methods(self, controller):
        assert callable(controller.run_all_checks)
        assert callable(controller.check_video_specs)
        assert callable(controller.check_duration)
        assert callable(controller.check_audio_gaps)
        assert callable(controller.check_image_integrity)
        assert callable(controller.check_subtitle_accuracy)
        assert callable(controller.check_short_specs)


class TestCheckImageIntegrity:
    def test_valid_images_pass(self, controller):
        with tempfile.TemporaryDirectory() as tmp:
            img_dir = Path(tmp)
            (img_dir / "scene_001.png").write_bytes(b"x" * 20000)
            (img_dir / "scene_002.png").write_bytes(b"x" * 15000)
            result = controller.check_image_integrity(img_dir)
            assert result.passed
            assert "2 images valid" in result.details

    def test_small_image_fails(self, controller):
        with tempfile.TemporaryDirectory() as tmp:
            img_dir = Path(tmp)
            (img_dir / "scene_001.png").write_bytes(b"x" * 20000)
            (img_dir / "scene_002.png").write_bytes(b"x" * 5000)
            result = controller.check_image_integrity(img_dir)
            assert not result.passed
            assert "scene_002.png" in result.details

    def test_empty_directory_fails(self, controller):
        with tempfile.TemporaryDirectory() as tmp:
            img_dir = Path(tmp)
            result = controller.check_image_integrity(img_dir)
            assert not result.passed
            assert "No image files found" in result.details

    def test_custom_min_size(self, controller):
        with tempfile.TemporaryDirectory() as tmp:
            img_dir = Path(tmp)
            (img_dir / "scene_001.png").write_bytes(b"x" * 500)
            result = controller.check_image_integrity(img_dir, min_size_bytes=100)
            assert result.passed

    def test_zero_byte_file_fails(self, controller):
        with tempfile.TemporaryDirectory() as tmp:
            img_dir = Path(tmp)
            (img_dir / "scene_001.png").write_bytes(b"")
            result = controller.check_image_integrity(img_dir)
            assert not result.passed


class TestCheckVideoSpecs:
    def test_returns_quality_check(self, controller):
        with tempfile.TemporaryDirectory() as tmp:
            fake_video = Path(tmp) / "video.mp4"
            fake_video.write_bytes(b"not a real video")
            result = controller.check_video_specs(fake_video)
            assert isinstance(result, QualityCheck)
            assert result.name == "video_specs"

    def test_graceful_when_ffprobe_unavailable(self, controller, monkeypatch):
        """When ffprobe is not available, should pass with warning."""
        import subprocess

        original_run = subprocess.run

        def fake_run(*args, **kwargs):
            raise FileNotFoundError("ffprobe not found")

        monkeypatch.setattr(subprocess, "run", fake_run)

        with tempfile.TemporaryDirectory() as tmp:
            fake_video = Path(tmp) / "video.mp4"
            fake_video.write_bytes(b"fake")
            result = controller.check_video_specs(fake_video)
            assert result.passed
            assert "skipped" in result.details.lower() or "not available" in result.details.lower()


class TestCheckDuration:
    def test_returns_quality_check(self, controller):
        with tempfile.TemporaryDirectory() as tmp:
            fake_video = Path(tmp) / "video.mp4"
            fake_video.write_bytes(b"not a real video")
            result = controller.check_duration(fake_video)
            assert isinstance(result, QualityCheck)
            assert result.name == "duration"

    def test_graceful_without_ffprobe(self, controller, monkeypatch):
        import subprocess

        def fake_run(*args, **kwargs):
            raise FileNotFoundError("ffprobe not found")

        monkeypatch.setattr(subprocess, "run", fake_run)

        with tempfile.TemporaryDirectory() as tmp:
            fake_video = Path(tmp) / "video.mp4"
            fake_video.write_bytes(b"fake")
            result = controller.check_duration(fake_video)
            assert result.passed
            assert "skipped" in result.details.lower()


class TestCheckAudioGaps:
    def test_returns_quality_check(self, controller):
        with tempfile.TemporaryDirectory() as tmp:
            fake_audio = Path(tmp) / "audio.wav"
            fake_audio.write_bytes(b"fake audio data")
            result = controller.check_audio_gaps(fake_audio)
            assert isinstance(result, QualityCheck)
            assert result.name == "audio_gaps"

    def test_graceful_without_ffmpeg(self, controller, monkeypatch):
        import subprocess

        def fake_run(*args, **kwargs):
            raise FileNotFoundError("ffmpeg not found")

        monkeypatch.setattr(subprocess, "run", fake_run)

        with tempfile.TemporaryDirectory() as tmp:
            fake_audio = Path(tmp) / "audio.wav"
            fake_audio.write_bytes(b"fake")
            result = controller.check_audio_gaps(fake_audio)
            assert result.passed
            assert "skipped" in result.details.lower() or "not available" in result.details.lower()


class TestCheckShortSpecs:
    def test_returns_quality_check(self, controller):
        with tempfile.TemporaryDirectory() as tmp:
            fake_short = Path(tmp) / "short.mp4"
            fake_short.write_bytes(b"fake short")
            result = controller.check_short_specs(fake_short)
            assert isinstance(result, QualityCheck)
            assert result.name == "short_specs"

    def test_graceful_without_ffprobe(self, controller, monkeypatch):
        import subprocess

        def fake_run(*args, **kwargs):
            raise FileNotFoundError("ffprobe not found")

        monkeypatch.setattr(subprocess, "run", fake_run)

        with tempfile.TemporaryDirectory() as tmp:
            fake_short = Path(tmp) / "short.mp4"
            fake_short.write_bytes(b"fake")
            result = controller.check_short_specs(fake_short)
            assert result.passed
            assert "skipped" in result.details.lower()


class TestCheckSubtitleAccuracy:
    def test_returns_quality_check(self, controller, sample_script):
        with tempfile.TemporaryDirectory() as tmp:
            fake_video = Path(tmp) / "video.mp4"
            fake_video.write_bytes(b"fake video")
            result = controller.check_subtitle_accuracy(fake_video, sample_script)
            assert isinstance(result, QualityCheck)
            assert result.name == "subtitle_accuracy"
            # Placeholder always passes
            assert result.passed


class TestRunAllChecks:
    def test_returns_quality_report(self, controller, sample_script, sample_scene_plan):
        with tempfile.TemporaryDirectory() as tmp:
            job_dir = Path(tmp)
            # Create minimal structure
            (job_dir / "images").mkdir()
            (job_dir / "images" / "scene_001.png").write_bytes(b"x" * 20000)
            (job_dir / "narration").mkdir()
            (job_dir / "narration" / "scene_001.wav").write_bytes(b"x" * 50000)

            report = controller.run_all_checks(job_dir, sample_script, sample_scene_plan)
            assert isinstance(report, QualityReport)
            assert len(report.checks) > 0
            assert report.timestamp  # Has a timestamp

    def test_missing_video_fails(self, controller, sample_script, sample_scene_plan):
        with tempfile.TemporaryDirectory() as tmp:
            job_dir = Path(tmp)
            (job_dir / "images").mkdir()
            (job_dir / "images" / "scene_001.png").write_bytes(b"x" * 20000)
            (job_dir / "narration").mkdir()
            (job_dir / "narration" / "scene_001.wav").write_bytes(b"x" * 50000)

            report = controller.run_all_checks(job_dir, sample_script, sample_scene_plan)
            # Missing video file should cause at least one check to fail
            video_checks = [c for c in report.checks if c.name == "video_specs"]
            assert len(video_checks) == 1
            assert not video_checks[0].passed
            assert "not found" in video_checks[0].details.lower()

    def test_report_passed_false_when_check_fails(
        self, controller, sample_script, sample_scene_plan
    ):
        with tempfile.TemporaryDirectory() as tmp:
            job_dir = Path(tmp)
            (job_dir / "images").mkdir()
            # Image below threshold
            (job_dir / "images" / "scene_001.png").write_bytes(b"x" * 100)
            (job_dir / "narration").mkdir()
            (job_dir / "narration" / "scene_001.wav").write_bytes(b"x" * 50000)

            report = controller.run_all_checks(job_dir, sample_script, sample_scene_plan)
            assert not report.passed
