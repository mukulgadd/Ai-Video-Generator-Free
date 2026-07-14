"""Verification tests for pipeline orchestrator (Task 6.2)."""

import tempfile
from datetime import timedelta
from pathlib import Path

import pytest

from vidgen.config import PipelineConfig
from vidgen.models import Script, ScriptSection, Scene, ScenePlan
from vidgen.pipeline import (
    STAGE_ORDER,
    PipelineOrchestrator,
    PipelineResult,
    StageResult,
    calculate_timeouts,
    release_gpu_memory,
    retry_with_backoff,
    BACKOFF_BASE_SECONDS,
    MAX_RETRIES,
)


# --- Fixtures ---


@pytest.fixture
def sample_script() -> Script:
    """A minimal script for testing (~6 min video at 150 wpm)."""
    return Script(
        title="Test Video",
        hook=ScriptSection(id="hook", title="Hook", narration_text="Test hook text with words"),
        introduction=ScriptSection(id="intro", title="Intro", narration_text="Intro text content"),
        body_sections=[
            ScriptSection(id="s1", title="Section 1", narration_text="Body content here"),
        ],
        conclusion=ScriptSection(id="conclusion", title="Conclusion", narration_text="Outro text"),
        total_word_count=900,
    )


@pytest.fixture
def sample_scene_plan() -> ScenePlan:
    """A minimal scene plan for testing."""
    return ScenePlan(
        video_title="Test Video",
        style_prefix="professional style",
        scenes=[
            Scene(
                id="scene_001",
                section_id="hook",
                image_prompt="Test prompt",
                ken_burns_direction="zoom-in",
                start_time=0.0,
                duration=10.0,
                transition_type="crossfade",
            )
        ],
        total_duration=360.0,
    )


@pytest.fixture
def config() -> PipelineConfig:
    return PipelineConfig()


@pytest.fixture
def job_dir(tmp_path: Path) -> Path:
    return tmp_path / "test_job"


# --- STAGE_ORDER Tests ---


class TestStageOrder:
    def test_stage_order_length(self):
        assert len(STAGE_ORDER) == 8

    def test_stage_order_starts_with_validate(self):
        assert STAGE_ORDER[0] == "validate"

    def test_stage_order_ends_with_packaging(self):
        assert STAGE_ORDER[-1] == "packaging"

    def test_stage_order_correct_sequence(self):
        expected = [
            "validate",
            "narration",
            "imaging",
            "assembly",
            "thumbnails",
            "shorts",
            "quality",
            "packaging",
        ]
        assert STAGE_ORDER == expected


# --- Timeout Calculation Tests ---


class TestCalculateTimeouts:
    def test_job_timeout_exceeds_estimated_time(self, sample_script, config):
        timeouts = calculate_timeouts(sample_script, config)
        estimated = (sample_script.estimated_duration_seconds / 60) * config.generation_ratio_minutes * 3600
        assert timeouts["job_total"] > estimated

    def test_job_timeout_equals_estimated_times_buffer(self, sample_script, config):
        timeouts = calculate_timeouts(sample_script, config)
        target_minutes = sample_script.estimated_duration_seconds / 60
        expected = target_minutes * config.generation_ratio_minutes * 3600 * config.timeout_buffer
        assert abs(timeouts["job_total"] - expected) < 0.01

    def test_all_stages_have_timeouts(self, sample_script, config):
        timeouts = calculate_timeouts(sample_script, config)
        for stage in STAGE_ORDER:
            assert stage in timeouts
        assert "job_total" in timeouts

    def test_imaging_gets_largest_share(self, sample_script, config):
        timeouts = calculate_timeouts(sample_script, config)
        # Imaging gets 45% of total, narration gets 25%
        assert timeouts["imaging"] > timeouts["narration"]

    def test_validate_and_packaging_have_fixed_timeouts(self, sample_script, config):
        timeouts = calculate_timeouts(sample_script, config)
        assert timeouts["validate"] == 60
        assert timeouts["packaging"] == 60

    def test_quality_has_fixed_timeout(self, sample_script, config):
        timeouts = calculate_timeouts(sample_script, config)
        assert timeouts["quality"] == 300


# --- Release GPU Memory Tests ---


class TestReleaseGpuMemory:
    def test_does_not_crash(self):
        """release_gpu_memory should not raise even without MLX installed."""
        release_gpu_memory()


# --- Retry With Backoff Tests ---


class TestRetryWithBackoff:
    def test_succeeds_on_first_try(self):
        result = retry_with_backoff(lambda: ["artifact.txt"], max_retries=3)
        assert result == ["artifact.txt"]

    def test_succeeds_on_second_try(self):
        attempts = {"count": 0}

        def flaky_fn():
            attempts["count"] += 1
            if attempts["count"] < 2:
                raise RuntimeError("fail")
            return ["output.wav"]

        result = retry_with_backoff(flaky_fn, max_retries=3)
        assert result == ["output.wav"]
        assert attempts["count"] == 2

    def test_raises_after_max_retries(self):
        def always_fail():
            raise RuntimeError("permanent failure")

        with pytest.raises(RuntimeError, match="permanent failure"):
            retry_with_backoff(always_fail, max_retries=1)

    def test_returns_empty_list_on_success(self):
        result = retry_with_backoff(lambda: [], max_retries=3)
        assert result == []


# --- Pipeline Orchestrator Tests ---


class TestPipelineOrchestrator:
    def test_creates_state_tracker(self, config, job_dir):
        orch = PipelineOrchestrator(config=config, job_dir=job_dir)
        assert orch.state_tracker is not None
        assert orch.config == config

    def test_run_completes_all_stages(self, config, job_dir, sample_script, sample_scene_plan):
        orch = PipelineOrchestrator(config=config, job_dir=job_dir)
        result = orch.run(sample_script, sample_scene_plan)

        assert result.success is True
        assert len(result.stage_results) == 8
        assert result.total_duration_seconds > 0
        assert result.output_dir == config.output_dir

    def test_run_stage_results_have_correct_names(self, config, job_dir, sample_script, sample_scene_plan):
        orch = PipelineOrchestrator(config=config, job_dir=job_dir)
        result = orch.run(sample_script, sample_scene_plan)

        stage_names = [sr.stage for sr in result.stage_results]
        assert stage_names == STAGE_ORDER

    def test_run_marks_job_completed(self, config, job_dir, sample_script, sample_scene_plan):
        orch = PipelineOrchestrator(config=config, job_dir=job_dir)
        orch.run(sample_script, sample_scene_plan)

        state = orch.get_state()
        assert state.status == "completed"

    def test_resume_skips_completed_stages(self, config, job_dir, sample_script, sample_scene_plan):
        orch = PipelineOrchestrator(config=config, job_dir=job_dir)

        # Pre-mark some stages as completed and create stub files
        orch.state_tracker.mark_stage_started("validate")
        orch.state_tracker.mark_stage_completed("validate", [])
        orch.state_tracker.mark_stage_started("narration")
        orch.state_tracker.mark_stage_completed("narration", ["narration/scene_001.wav"])

        # Create stub narration files so assembly can find them
        import struct
        import wave
        narration_dir = job_dir / "narration"
        narration_dir.mkdir(parents=True, exist_ok=True)
        for scene in sample_scene_plan.scenes:
            wav_path = narration_dir / f"{scene.id}.wav"
            with wave.open(str(wav_path), "w") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(24000)
                samples = int(scene.duration * 24000)
                wf.writeframes(struct.pack(f"<{samples}h", *([0] * samples)))

        result = orch.run(sample_script, sample_scene_plan, resume=True)

        assert result.success is True
        # Should only have results for non-skipped stages
        executed_stages = [sr.stage for sr in result.stage_results]
        assert "validate" not in executed_stages
        assert "narration" not in executed_stages
        assert "imaging" in executed_stages

    def test_estimate_duration(self, config, job_dir, sample_script):
        orch = PipelineOrchestrator(config=config, job_dir=job_dir)
        duration = orch.estimate_duration(sample_script)

        assert isinstance(duration, timedelta)
        assert duration.total_seconds() > 0

        # 900 words at 150wpm = 6 min, at 1hr/min ratio = 6 hours
        expected_hours = (sample_script.estimated_duration_seconds / 60) * config.generation_ratio_minutes
        assert abs(duration.total_seconds() - expected_hours * 3600) < 1.0

    def test_get_state_returns_job_state(self, config, job_dir):
        orch = PipelineOrchestrator(config=config, job_dir=job_dir)
        state = orch.get_state()
        assert state.job_id is not None
        assert state.status == "queued"

    def test_stage_failure_stops_pipeline(self, config, job_dir, sample_script, sample_scene_plan):
        orch = PipelineOrchestrator(config=config, job_dir=job_dir)

        # Override a handler to fail
        def failing_handler():
            raise RuntimeError("Image generation failed")

        orch._stage_handlers["imaging"] = failing_handler

        # Set max_retries to 1 to speed up test
        orch.config = config.model_copy(update={"max_retries": 1})

        result = orch.run(sample_script, sample_scene_plan)

        assert result.success is False
        assert "imaging" in result.error
        # validate and narration should have succeeded
        assert result.stage_results[0].stage == "validate"
        assert result.stage_results[0].success is True
        assert result.stage_results[1].stage == "narration"
        assert result.stage_results[1].success is True
        # imaging failed
        assert result.stage_results[2].stage == "imaging"
        assert result.stage_results[2].success is False


# --- StageResult and PipelineResult Model Tests ---


class TestResultModels:
    def test_stage_result_defaults(self):
        result = StageResult(stage="validate", success=True)
        assert result.duration_seconds == 0.0
        assert result.artifacts == []
        assert result.error is None

    def test_pipeline_result_defaults(self):
        result = PipelineResult(job_id="test-123", success=True)
        assert result.total_duration_seconds == 0.0
        assert result.stage_results == []
        assert result.output_dir is None
        assert result.error is None

    def test_stage_result_with_error(self):
        result = StageResult(stage="imaging", success=False, error="GPU OOM")
        assert result.success is False
        assert result.error == "GPU OOM"
