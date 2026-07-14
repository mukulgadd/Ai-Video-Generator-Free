"""Shorts pipeline orchestrator — generates YouTube Shorts from script files.

Independent from the long-form pipeline. Stages: validate → narrate → image → assemble → package.
Each Short takes ~11 minutes to generate (TTS + 5 images + assembly).
"""

import logging
import random
import shutil
import time
from pathlib import Path

from vidgen.config import PipelineConfig, ShortsConfig, VisualConfig, load_config
from vidgen.models import (
    ImageCue,
    ImageResult,
    NarrationResult,
    OverlayCue,
    Scene,
    ShortGenerationResult,
    ShortScript,
)
from vidgen.shorts_assembly import ShortsAssembler
from vidgen.shorts_parser import ShortParseError, parse_short_script, validate_short_script

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path("config.yaml")


class ShortsPipeline:
    """Orchestrates YouTube Short generation from script to upload-ready MP4.

    Stages:
    1. Validate — parse script, check format, verify dependencies
    2. Narrate — generate voiceover WAV using NarrationEngine
    3. Image — generate vertical images using ImageGenerator
    4. Assemble — compose vertical video with overlays, music, branding
    5. Package — copy output to final location
    """

    def __init__(self, config_path: Path | None = None) -> None:
        path = config_path or DEFAULT_CONFIG_PATH
        if path.exists():
            self.config = load_config(path)
        else:
            self.config = PipelineConfig()
        self.shorts_config = self.config.shorts

    def run(self, script_path: Path) -> ShortGenerationResult:
        """Generate a single YouTube Short from a script file.

        Args:
            script_path: Path to the Short script markdown file.

        Returns:
            ShortGenerationResult with success status and output path.
        """
        start_time = time.time()
        script_name = script_path.stem

        logger.info(f"Starting Short generation: {script_name}")

        try:
            # Stage 1: Validate
            script = self._validate(script_path)
            logger.info(f"  Validated: '{script.title}' ({len(script.narration_text.split())} words)")

            # Stage 2: Narrate
            job_dir = self.config.jobs_dir / "shorts" / script_name
            job_dir.mkdir(parents=True, exist_ok=True)
            narration_path = self._narrate(script, job_dir)
            logger.info(f"  Narrated: {narration_path.name}")

            # Stage 3: Image
            image_paths = self._generate_images(script, job_dir)
            logger.info(f"  Generated {len(image_paths)} vertical images")

            # Stage 4: Assemble
            assembly_path = self._assemble(script, narration_path, image_paths, job_dir)
            logger.info(f"  Assembled: {assembly_path.name}")

            # Stage 5: Package
            output_path = self._package(assembly_path, script_name)
            logger.info(f"  Packaged: {output_path}")

            duration = time.time() - start_time
            logger.info(f"Short completed: {script_name} in {duration:.1f}s")

            return ShortGenerationResult(
                success=True,
                output_path=output_path,
                duration_seconds=duration,
            )

        except Exception as e:
            duration = time.time() - start_time
            error_msg = f"Short generation failed: {e}"
            logger.error(error_msg)
            return ShortGenerationResult(
                success=False,
                error=error_msg,
                duration_seconds=duration,
            )

    def run_batch(self, script_paths: list[Path]) -> list[ShortGenerationResult]:
        """Generate multiple Shorts sequentially.

        Memory is released between each Short to stay within 32GB cap.
        A failed Short does not stop the batch.

        Args:
            script_paths: List of Short script markdown files.

        Returns:
            List of results, one per script.
        """
        results: list[ShortGenerationResult] = []
        total = len(script_paths)

        logger.info(f"Starting Shorts batch: {total} scripts")
        batch_start = time.time()

        for i, path in enumerate(script_paths, 1):
            logger.info(f"Processing {i}/{total}: {path.name}")
            result = self.run(path)
            results.append(result)

            status = "✓" if result.success else "✗"
            logger.info(f"  {status} {path.name} ({result.duration_seconds:.0f}s)")

        batch_duration = time.time() - batch_start
        completed = sum(1 for r in results if r.success)
        failed = total - completed

        logger.info(
            f"Batch complete: {completed}/{total} succeeded, "
            f"{failed} failed, {batch_duration:.0f}s total"
        )

        return results

    def _validate(self, script_path: Path) -> ShortScript:
        """Stage 1: Parse and validate the Short script."""
        script = parse_short_script(script_path)

        errors = validate_short_script(script)
        if errors:
            logger.warning(f"Validation warnings: {errors}")
            # Warnings don't stop generation, just log them

        return script

    def _narrate(self, script: ShortScript, job_dir: Path) -> Path:
        """Stage 2: Generate narration audio using NarrationEngine."""
        from vidgen.narration import NarrationEngine

        narration_dir = job_dir / "narration"
        narration_dir.mkdir(parents=True, exist_ok=True)
        output_path = narration_dir / "narration.wav"

        # Skip if already generated (resume support)
        if output_path.exists() and output_path.stat().st_size > 1000:
            logger.info("  Narration exists, skipping")
            return output_path

        engine = NarrationEngine(self.config.voice)
        # Override rate for Shorts (faster pacing)
        if hasattr(self.shorts_config, 'narration_rate'):
            engine.config = engine.config.model_copy(update={"rate": self.shorts_config.narration_rate})
        try:
            # Create a dummy Scene for the narration engine interface
            dummy_scene = Scene(
                id="short_narration",
                section_id="short",
                image_prompt="",
                ken_burns_direction="zoom-in",
                start_time=0.0,
                duration=float(script.duration_target),
                transition_type="cut",
            )
            result = engine.generate_scene_audio(
                dummy_scene, script.narration_text, output_path
            )
            return result.audio_path
        finally:
            engine.release_model()

    def _generate_images(self, script: ShortScript, job_dir: Path) -> list[Path]:
        """Stage 3: Generate vertical images using ImageGenerator."""
        from vidgen.imaging import ImageGenerator

        images_dir = job_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        # Check for existing images (resume support)
        existing = sorted(images_dir.glob("*.png"))
        if len(existing) >= len(script.image_cues):
            valid = [p for p in existing if p.stat().st_size > 50_000]
            if len(valid) >= len(script.image_cues):
                logger.info(f"  {len(valid)} images exist, skipping")
                return valid[:len(script.image_cues)]

        # Create vertical visual config override
        vertical_config = VisualConfig(
            style_prefix=script.style_prefix or self.config.visual.style_prefix,
            color_palette=self.config.visual.color_palette,
            font_family=self.config.visual.font_family,
            image_model=self.config.visual.image_model,
            image_steps=self.config.visual.image_steps,
            image_gen_width=self.shorts_config.image_gen_width,
            image_gen_height=self.shorts_config.image_gen_height,
            image_output_width=self.shorts_config.image_output_width,
            image_output_height=self.shorts_config.image_output_height,
            seed_strategy=self.config.visual.seed_strategy,
        )

        generator = ImageGenerator(vertical_config)
        image_paths: list[Path] = []

        try:
            for i, cue in enumerate(script.image_cues):
                output_path = images_dir / f"image_{i + 1:02d}.png"

                # Build prompt with style prefix and no-text suffix
                prompt = cue.prompt
                if not prompt.endswith("no readable labels"):
                    prompt += ", no text, no numbers, no readable labels"

                # Create dummy scene for generator interface
                scene = Scene(
                    id=f"short_img_{i + 1:02d}",
                    section_id="short",
                    image_prompt=prompt,
                    style_prefix=vertical_config.style_prefix,
                    ken_burns_direction="zoom-in",
                    start_time=cue.start_time,
                    duration=cue.duration,
                    transition_type="cut",
                    visual_type=getattr(cue, 'visual_type', 'scene'),
                    data_overlay=getattr(cue, 'data_overlay', None),
                )

                result = generator.generate_scene_image(scene, output_path)
                image_paths.append(result.image_path)

            return image_paths
        finally:
            generator.release_model()

    def _assemble(self, script: ShortScript, narration_path: Path,
                  image_paths: list[Path], job_dir: Path) -> Path:
        """Stage 4: Assemble the Short video."""
        assembly_dir = job_dir / "assembly"
        assembly_dir.mkdir(parents=True, exist_ok=True)
        output_path = assembly_dir / "short.mp4"

        # Find music track
        music_path = self._find_music(script.music_mood, script.music_track)

        # Find logo
        logo_path = Path("channel/assets/TokenEconomy_logo.png")
        if not logo_path.exists():
            logo_path = None

        assembler = ShortsAssembler(self.shorts_config)
        assembler.assemble(
            narration_path=narration_path,
            image_paths=image_paths,
            image_cues=script.image_cues,
            overlay_cues=script.overlay_cues,
            hook_text=script.hook_text,
            output_path=output_path,
            music_path=music_path,
            logo_path=logo_path,
            duration_target=script.duration_target,
        )

        return output_path

    def _package(self, assembly_path: Path, script_name: str) -> Path:
        """Stage 5: Copy to final output location."""
        output_dir = self.shorts_config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        final_path = output_dir / f"{script_name}.mp4"
        shutil.copy2(assembly_path, final_path)
        return final_path

    def _find_music(self, mood: str, specific_track: str | None) -> Path | None:
        """Find the appropriate music track."""
        music_dir = self.shorts_config.music_dir

        if specific_track:
            path = music_dir / specific_track
            if path.exists():
                return path
            logger.warning(f"Specified music track not found: {path}")

        # Look for mood-named track
        mood_path = music_dir / f"{mood}.mp3"
        if mood_path.exists():
            return mood_path

        # Fall back to any available track
        if music_dir.exists():
            tracks = list(music_dir.glob("*.mp3"))
            if tracks:
                return random.choice(tracks)

        # No music available — continue without (per REQ-5 clarification)
        logger.warning(f"No music found in {music_dir}, generating without music")
        return None
