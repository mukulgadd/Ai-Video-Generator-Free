"""Pipeline orchestrator for the vidgen video generation pipeline.

Coordinates stage execution, manages state transitions, handles timeouts,
retries, and resume logic. Each stage runs sequentially with GPU memory
released between heavy stages.
"""

import gc
import logging
import signal
import time
from datetime import timedelta
from pathlib import Path
from typing import Callable

from pydantic import BaseModel, Field

from vidgen.config import PipelineConfig
from vidgen.models import JobState, ScenePlan, Script
from vidgen.state import JobStateTracker

logger = logging.getLogger(__name__)

# --- Constants ---

STAGE_ORDER = [
    "validate",
    "narration",
    "imaging",
    "assembly",
    "thumbnails",
    "shorts",
    "quality",
    "packaging",
]

MAX_RETRIES = 3
BACKOFF_BASE_SECONDS = 5


# --- Result Models ---


class StageResult(BaseModel):
    """Result of executing a single pipeline stage."""

    stage: str
    success: bool
    duration_seconds: float = 0.0
    artifacts: list[str] = Field(default_factory=list)
    error: str | None = None


class PipelineResult(BaseModel):
    """Result of a full pipeline execution."""

    job_id: str
    success: bool
    total_duration_seconds: float = 0.0
    stage_results: list[StageResult] = Field(default_factory=list)
    output_dir: Path | None = None
    error: str | None = None


# --- Timeout Calculation ---


def calculate_timeouts(script: Script, config: PipelineConfig) -> dict[str, float]:
    """Calculate per-stage and overall job timeouts based on generation ratio + 50% buffer.

    The generation ratio defines hours per minute of final video. The timeout buffer
    (default 1.5x) is applied to prevent premature termination on overnight runs.

    Args:
        script: The script being processed (provides estimated duration).
        config: Pipeline config with generation_ratio_minutes and timeout_buffer.

    Returns:
        Dict mapping stage names (and "job_total") to timeout values in seconds.
    """
    target_minutes = script.estimated_duration_seconds / 60
    ratio = config.generation_ratio_minutes  # hours per minute of video
    buffer = config.timeout_buffer  # 1.5x

    total_hours = target_minutes * ratio
    total_seconds = total_hours * 3600
    job_timeout = total_seconds * buffer

    return {
        "validate": 60,
        "narration": total_seconds * 0.25 * buffer,
        "imaging": total_seconds * 0.45 * buffer,
        "assembly": total_seconds * 0.15 * buffer,
        "thumbnails": total_seconds * 0.05 * buffer,
        "shorts": total_seconds * 0.05 * buffer,
        "quality": 300,
        "packaging": 60,
        "job_total": job_timeout,
    }


# --- Memory Management ---


def release_gpu_memory() -> None:
    """Release GPU memory between stages.

    Calls gc.collect() and clears MLX Metal cache if available.
    Gracefully handles missing MLX dependency.
    """
    gc.collect()
    try:
        import mlx.core as mx

        mx.metal.clear_cache()
        logger.debug("MLX Metal cache cleared")
    except ImportError:
        pass


# --- Retry Utility ---


def retry_with_backoff(
    fn: Callable[[], list[str]], max_retries: int = MAX_RETRIES
) -> list[str]:
    """Execute fn with exponential backoff retry (5s, 10s, 20s).

    Args:
        fn: A callable that returns a list of artifact paths on success.
        max_retries: Maximum number of attempts before raising.

    Returns:
        The return value of fn on success.

    Raises:
        The last exception if all retries are exhausted.
    """
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            wait = BACKOFF_BASE_SECONDS * (2**attempt)
            logger.warning(
                f"Attempt {attempt + 1} failed, retrying in {wait}s: {e}"
            )
            time.sleep(wait)
            release_gpu_memory()
    # Should not reach here, but satisfy type checker
    return []  # pragma: no cover


# --- Pipeline Orchestrator ---


class PipelineOrchestrator:
    """Coordinates stage execution with state tracking, timeouts, and retries.

    The orchestrator runs stages sequentially, persists state after each stage,
    and supports resuming from the last completed stage after a crash or timeout.
    """

    def __init__(self, config: PipelineConfig, job_dir: Path) -> None:
        self.config = config
        self.job_dir = job_dir
        self.state_tracker = JobStateTracker(job_dir / "state.json")
        self._terminated = False
        self._stage_handlers: dict[str, Callable[[], list[str]]] = {
            "validate": self._run_validate,
            "narration": self._run_narration,
            "imaging": self._run_imaging,
            "assembly": self._run_assembly,
            "thumbnails": self._run_thumbnails,
            "shorts": self._run_shorts,
            "quality": self._run_quality,
            "packaging": self._run_packaging,
        }

    def run(
        self, script: Script, scene_plan: ScenePlan, resume: bool = False
    ) -> PipelineResult:
        """Execute all stages sequentially. If resume=True, skip completed stages.

        Args:
            script: The parsed video script.
            scene_plan: The parsed scene plan.
            resume: If True, skip previously completed stages.

        Returns:
            PipelineResult with success status, timing, and stage results.
        """
        # Store inputs for stage handlers
        self._script = script
        self._scene_plan = scene_plan
        # Register signal handler for graceful termination
        original_handler = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGTERM, self._handle_sigterm)

        job_start = time.time()
        state = self.state_tracker.load()
        job_id = state.job_id
        timeouts = calculate_timeouts(script, self.config)
        stage_results: list[StageResult] = []

        try:
            for stage in STAGE_ORDER:
                # Check for termination signal
                if self._terminated:
                    self.state_tracker.mark_timed_out("Received SIGTERM signal")
                    break

                # Check overall job timeout
                elapsed = time.time() - job_start
                if elapsed > timeouts["job_total"]:
                    self.state_tracker.mark_timed_out(
                        f"Job timeout exceeded ({elapsed:.0f}s > {timeouts['job_total']:.0f}s)"
                    )
                    break

                # Skip completed stages on resume
                if resume and self.state_tracker.is_stage_completed(stage):
                    logger.info(f"Skipping completed stage: {stage}")
                    continue

                # Execute stage
                result = self.run_stage(stage, timeouts.get(stage, 3600))
                stage_results.append(result)

                if not result.success:
                    self.state_tracker.mark_failed(
                        result.error or f"Stage {stage} failed"
                    )
                    return PipelineResult(
                        job_id=job_id,
                        success=False,
                        total_duration_seconds=time.time() - job_start,
                        stage_results=stage_results,
                        error=result.error,
                    )

                # Release GPU memory between heavy stages
                if stage in ("narration", "imaging"):
                    release_gpu_memory()

            # All stages completed (or terminated)
            if not self._terminated:
                self.state_tracker.mark_completed()

            return PipelineResult(
                job_id=job_id,
                success=not self._terminated,
                total_duration_seconds=time.time() - job_start,
                stage_results=stage_results,
                output_dir=self.config.output_dir,
            )
        finally:
            signal.signal(signal.SIGTERM, original_handler)

    def run_stage(self, stage: str, timeout: float) -> StageResult:
        """Execute a single stage with timeout and retry logic.

        Args:
            stage: Name of the stage to execute.
            timeout: Timeout in seconds for this stage (warning only).

        Returns:
            StageResult with success status, duration, and artifacts.
        """
        self.state_tracker.mark_stage_started(stage)
        stage_start = time.time()

        try:
            handler = self._stage_handlers[stage]
            artifacts = retry_with_backoff(handler, self.config.max_retries)
            duration = time.time() - stage_start

            # Check stage timeout (warning only, continue unless job timeout exceeded)
            if duration > timeout:
                logger.warning(
                    f"Stage {stage} exceeded timeout "
                    f"({duration:.0f}s > {timeout:.0f}s)"
                )

            self.state_tracker.mark_stage_completed(stage, artifacts or [])
            return StageResult(
                stage=stage,
                success=True,
                duration_seconds=duration,
                artifacts=artifacts or [],
            )
        except Exception as e:
            duration = time.time() - stage_start
            error_msg = f"Stage {stage} failed after retries: {str(e)}"
            logger.error(error_msg)
            return StageResult(
                stage=stage,
                success=False,
                duration_seconds=duration,
                error=error_msg,
            )

    def get_state(self) -> JobState:
        """Return current job state from disk."""
        return self.state_tracker.load()

    def estimate_duration(self, script: Script) -> timedelta:
        """Estimate total generation time based on script length and generation ratio.

        Args:
            script: The script to estimate generation time for.

        Returns:
            Estimated duration as a timedelta.
        """
        target_minutes = script.estimated_duration_seconds / 60
        total_hours = target_minutes * self.config.generation_ratio_minutes
        return timedelta(hours=total_hours)

    def _handle_sigterm(self, signum: int, frame: object) -> None:
        """Handle SIGTERM for graceful shutdown."""
        logger.info("Received SIGTERM, shutting down gracefully...")
        self._terminated = True

    # --- Stage Handlers ---

    def _run_validate(self) -> list[str]:
        """Validate input files — script and scene plan alignment."""
        logger.info("Running validation stage...")
        from vidgen.parsers import validate_script_scene_alignment
        validate_script_scene_alignment(self._script, self._scene_plan)
        return []

    def _run_narration(self) -> list[str]:
        """Generate narration audio for all scenes using Qwen3-TTS."""
        logger.info("Running narration stage...")
        from vidgen.narration import NarrationEngine

        engine = NarrationEngine(self.config.voice)
        narration_dir = self.job_dir / "narration"

        try:
            results = engine.generate_all(
                self._script, self._scene_plan.scenes, narration_dir
            )
            self._narration_results = results
            return [str(r.audio_path) for r in results]
        finally:
            engine.release_model()

    def _run_imaging(self) -> list[str]:
        """Generate scene images using MFLUX. Generates multiple images per scene
        based on narration duration (~1 image per 12 seconds of audio)."""
        logger.info("Running imaging stage...")
        from vidgen.imaging import ImageGenerator

        generator = ImageGenerator(self.config.visual)
        images_dir = self.job_dir / "images"

        # Build narration duration map for multi-image generation
        narration_durations: dict[str, float] = {}
        narration_results = getattr(self, "_narration_results", None)
        if narration_results:
            for nr in narration_results:
                narration_durations[nr.scene_id] = nr.duration_seconds

        try:
            results = generator.generate_all(
                self._scene_plan.scenes, images_dir, narration_durations=narration_durations
            )
            self._image_results = results
            return [str(r.image_path) for r in results]
        finally:
            generator.release_model()

    def _run_assembly(self) -> list[str]:
        """Assemble final video from narration + images using MoviePy/FFmpeg."""
        logger.info("Running assembly stage...")
        from vidgen.assembly import VideoAssembler
        from vidgen.models import ImageResult, NarrationResult

        assembler = VideoAssembler(self.config)
        output_path = self.job_dir / "assembly" / "video_raw.mp4"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Use in-memory results or reconstruct from disk (for resume)
        narration_results = getattr(self, "_narration_results", None)
        image_results = getattr(self, "_image_results", None)

        if narration_results is None:
            narration_results = self._reconstruct_narration_results()
        if image_results is None:
            image_results = self._reconstruct_image_results()

        result = assembler.assemble(
            scenes=self._scene_plan.scenes,
            narration_results=narration_results,
            image_results=image_results,
            output_path=output_path,
            background_music=self._find_background_music(),
        )
        self._assembly_result = result
        return [str(result.video_path)]

    def _find_background_music(self) -> Path | None:
        """Find the default background music track for the long-form video."""
        music_dir = Path("channel/assets/music")
        # Prefer neutral as the default/fallback track
        for name in ["neutral.mp3", "suspense.mp3", "upbeat.mp3"]:
            path = music_dir / name
            if path.exists():
                return path
        return None

    def _reconstruct_narration_results(self) -> list:
        """Reconstruct NarrationResult objects from files on disk."""
        import wave
        from vidgen.models import NarrationResult

        narration_dir = self.job_dir / "narration"
        results = []
        if not narration_dir.exists():
            return results
        for scene in self._scene_plan.scenes:
            path = narration_dir / f"{scene.id}.wav"
            if path.exists():
                with wave.open(str(path), "r") as wf:
                    duration = wf.getnframes() / wf.getframerate()
                results.append(NarrationResult(
                    scene_id=scene.id,
                    audio_path=path,
                    duration_seconds=duration,
                    word_count=0,
                    actual_wpm=0.0,
                ))
        return results

    def _reconstruct_image_results(self) -> list:
        """Reconstruct ImageResult objects from files on disk.
        
        Handles both single-image (scene_001.png) and multi-image
        (scene_001_01.png, scene_001_02.png) naming patterns.
        """
        from vidgen.models import ImageResult

        images_dir = self.job_dir / "images"
        results = []
        if not images_dir.exists():
            return results
        for scene in self._scene_plan.scenes:
            # Try single image first
            path = images_dir / f"{scene.id}.png"
            if path.exists():
                results.append(ImageResult(
                    scene_id=scene.id,
                    image_path=path,
                    width=1920,
                    height=1080,
                    generation_time_seconds=0.0,
                    seed_used=0,
                ))
            else:
                # Try multi-image pattern (scene_001_01.png, scene_001_02.png, ...)
                multi_images = sorted(images_dir.glob(f"{scene.id}_*.png"))
                for img_path in multi_images:
                    results.append(ImageResult(
                        scene_id=scene.id,
                        image_path=img_path,
                        width=1920,
                        height=1080,
                        generation_time_seconds=0.0,
                        seed_used=0,
                    ))
        return results

    def _run_thumbnails(self) -> list[str]:
        """Generate thumbnail variants using scene images as backgrounds.
        
        Generates A/B variants:
        - Variants 1-2: Threat/problem angle (red accent, negative keyword)
        - Variant 3: Opportunity/solution angle (gold accent, positive keyword)
        YouTube's A/B test feature picks the winner automatically.
        """
        logger.info("Running thumbnails stage...")
        from vidgen.thumbnails import ThumbnailGenerator

        generator = ThumbnailGenerator(self.config.branding)
        thumbs_dir = self.job_dir / "thumbnails"

        # Collect existing scene images for use as thumbnail backgrounds
        images_dir = self.job_dir / "images"
        scene_images = sorted(images_dir.glob("*.png")) if images_dir.exists() else []

        # Check scene plan for explicit thumbnail hints
        thumbnail_text = None
        accent_word = None
        accent_color = None
        if hasattr(self._scene_plan, "thumbnail_text") and self._scene_plan.thumbnail_text:
            thumbnail_text = self._scene_plan.thumbnail_text
        if hasattr(self._scene_plan, "thumbnail_accent_word") and self._scene_plan.thumbnail_accent_word:
            accent_word = self._scene_plan.thumbnail_accent_word
        if hasattr(self._scene_plan, "thumbnail_accent_color") and self._scene_plan.thumbnail_accent_color:
            accent_color = self._scene_plan.thumbnail_accent_color

        # Generate Variant A (threat angle — typically red)
        paths = generator.generate_variants(
            title=self._script.title,
            style_prefix=self.config.visual.style_prefix,
            output_dir=thumbs_dir,
            count=2,
            scene_images=scene_images,
            thumbnail_text=thumbnail_text,
            accent_word=accent_word,
            accent_color=accent_color,
        )

        # Generate Variant B (opportunity angle — gold)
        alt_text = getattr(self._scene_plan, "thumbnail_text_alt", None)
        alt_word = getattr(self._scene_plan, "thumbnail_text_alt", None) and getattr(self._scene_plan, "thumbnail_accent_word_alt", None)
        alt_color = getattr(self._scene_plan, "thumbnail_accent_color_alt", None) or "#f59e0b"

        if alt_text:
            alt_paths = generator.generate_variants(
                title=self._script.title,
                style_prefix=self.config.visual.style_prefix,
                output_dir=thumbs_dir,
                count=1,
                scene_images=scene_images,
                thumbnail_text=alt_text,
                accent_word=alt_word,
                accent_color=alt_color,
            )
            # Rename to variant_3 (since generate_variants starts at 1)
            if alt_paths:
                import shutil
                final_alt = thumbs_dir / "variant_3.png"
                shutil.move(str(alt_paths[0]), str(final_alt))
                paths.append(final_alt)
        return [str(p) for p in paths]

    def _run_shorts(self) -> list[str]:
        """Extract vertical Shorts from the main video. DISABLED — needs redesign."""
        logger.info("Shorts extraction skipped (disabled — needs independent vertical generation)")
        return []

    def _run_quality(self) -> list[str]:
        """Run automated quality checks on generated assets."""
        logger.info("Running quality control stage...")
        from vidgen.quality import QualityController

        controller = QualityController(self.config)
        video_path = self.job_dir / "assembly" / "video_raw.mp4"

        if not video_path.exists():
            logger.warning("Main video not found, skipping QC")
            return []

        report = controller.run_all_checks(self.job_dir, self._script, self._scene_plan)
        passed = sum(1 for c in report.checks if c.passed)
        total = len(report.checks)
        logger.info(f"QC: {passed}/{total} checks passed")
        for check in report.checks:
            if not check.passed:
                logger.warning(f"QC FAILED: {check.name} — {check.details}")

        return [f"qc:{check.name}={'PASS' if check.passed else 'FAIL'}" for check in report.checks]

    def _run_packaging(self) -> list[str]:
        """Package final output — video, thumbnails, shorts, metadata."""
        logger.info("Running output packaging stage...")
        from vidgen.packaging import package_output

        output_dir = package_output(
            job_dir=self.job_dir,
            script=self._script,
            scene_plan=self._scene_plan,
            config=self.config,
        )
        return [str(output_dir)]
