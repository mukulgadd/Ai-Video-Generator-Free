"""Image generator - wraps MFLUX for local image generation on Apple Silicon."""

import hashlib
import logging
import time
from pathlib import Path

from PIL import Image

from vidgen.config import VisualConfig
from vidgen.models import ImageResult, Scene

logger = logging.getLogger(__name__)


class ImageGenerator:
    """Generates scene images using MFLUX (FLUX models, 4-bit quantized).
    
    Runs locally on Apple Silicon via MLX. Images are generated at 1920x1080
    (16:9 landscape) for long-form video scenes.
    """

    def __init__(self, config: VisualConfig) -> None:
        self.config = config
        self._model = None
        self._model_loaded = False

    def _ensure_model_loaded(self) -> None:
        """Lazily load the MFLUX model."""
        if self._model_loaded:
            return
        logger.info(f"Loading image model: {self.config.image_model}")
        # Actual MFLUX model loading happens here when dependency is available
        # For now mark as loaded - real inference to be wired when mflux is installed
        self._model_loaded = True
        logger.info("Image model loaded successfully")

    def generate_scene_image(self, scene: Scene, output_path: Path) -> ImageResult:
        """Generate a single scene image at 1920x1080.
        
        Args:
            scene: Scene with image_prompt and style info.
            output_path: Where to write the PNG file.
            
        Returns:
            ImageResult with path, dimensions, timing, and seed.
        """
        self._ensure_model_loaded()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        start_time = time.time()
        
        # Build full prompt with style prefix
        full_prompt = f"{self.config.style_prefix}, {scene.image_prompt}" if self.config.style_prefix else scene.image_prompt
        
        # Determine seed
        seed = self._get_seed(scene.id)
        
        # Generate at config resolution (max for 36GB M3 Pro) and upscale to output resolution.
        # MFLUX needs dims divisible by 16. Full 1920x1080 exceeds GPU memory.
        gen_width, gen_height = self.config.image_gen_width, self.config.image_gen_height
        final_width, final_height = self.config.image_output_width, self.config.image_output_height
        self._generate(full_prompt, output_path, gen_width, gen_height, seed)
        
        # Upscale to final resolution using LANCZOS
        self._upscale(output_path, final_width, final_height)
        
        # Apply data overlay if this is a "data" scene
        if getattr(scene, 'visual_type', 'scene') == 'data' and getattr(scene, 'data_overlay', None):
            self._apply_data_overlay(output_path, scene.data_overlay)
        
        generation_time = time.time() - start_time
        logger.info(f"Generated image for {scene.id}: {final_width}x{final_height} in {generation_time:.1f}s (seed={seed})")
        
        return ImageResult(
            scene_id=scene.id,
            image_path=output_path,
            width=final_width,
            height=final_height,
            generation_time_seconds=generation_time,
            seed_used=seed,
        )

    def generate_all(self, scenes: list[Scene], output_dir: Path, narration_durations: dict[str, float] | None = None) -> list[ImageResult]:
        """Generate scene images sequentially. When narration_durations is provided,
        generates multiple images per scene (one every ~8 seconds) for visual variety.
        
        Args:
            scenes: List of scenes to generate images for.
            output_dir: Directory to write PNG files into.
            narration_durations: Optional dict of scene_id -> duration in seconds.
                When provided, generates ceil(duration/8) images per scene.
            
        Returns:
            List of ImageResult objects (multiple per scene when multi-image enabled).
        """
        self._ensure_model_loaded()
        output_dir.mkdir(parents=True, exist_ok=True)
        results: list[ImageResult] = []
        
        sub_image_interval = 8.0  # seconds per sub-image (was 12s, tightened for retention)
        total_images = 0
        
        for i, scene in enumerate(scenes):
            # Determine how many images this scene needs
            if narration_durations and scene.id in narration_durations:
                duration = narration_durations[scene.id]
                num_images = max(1, int(duration / sub_image_interval + 0.5))
            else:
                num_images = 1
            
            total_images += num_images
            base_seed = self._get_seed(scene.id)
            
            for img_idx in range(num_images):
                # Name: scene_001_01.png, scene_001_02.png, etc.
                if num_images == 1:
                    output_path = output_dir / f"{scene.id}.png"
                else:
                    output_path = output_dir / f"{scene.id}_{img_idx + 1:02d}.png"
                
                # Skip if a valid image already exists (resume support)
                if output_path.exists() and output_path.stat().st_size > 50_000:
                    logger.info(f"Skipping existing valid image: {output_path.name}")
                    results.append(ImageResult(
                        scene_id=scene.id,
                        image_path=output_path,
                        width=self.config.image_output_width,
                        height=self.config.image_output_height,
                        generation_time_seconds=0.0,
                        seed_used=base_seed + img_idx * 1000,
                    ))
                    continue

                # Use different seed for each sub-image
                seed = base_seed + img_idx * 1000
                
                # Select prompt: use image_prompts list if available (cycling), else single prompt
                prompt_override = None
                if hasattr(scene, 'image_prompts') and scene.image_prompts:
                    prompt_override = scene.image_prompts[img_idx % len(scene.image_prompts)]
                
                # Generate with quality gate: retry up to 3 times if blank/corrupt
                result = None
                for attempt in range(3):
                    attempt_seed = seed + attempt * 7
                    result = self._generate_single(
                        scene, output_path, attempt_seed,
                        prompt_override=prompt_override,
                        is_first_sub_image=(img_idx < 2),  # Data overlay on first 2 sub-images (~16s)
                    )
                    if self._passes_quality_gate(output_path):
                        break
                    logger.warning(f"Image quality gate failed for {output_path.name} (attempt {attempt + 1}/3)")
                    if attempt < 2:
                        output_path.unlink(missing_ok=True)
                
                if result:
                    results.append(result)
            
            logger.info(f"Completed scene {i + 1}/{len(scenes)} ({num_images} images)")
        
        logger.info(f"Total images generated: {total_images}")
        return results

    def _generate_single(self, scene: Scene, output_path: Path, seed: int, prompt_override: str | None = None, is_first_sub_image: bool = True) -> ImageResult:
        """Generate a single image for a scene with a specific seed.
        
        If prompt_override is given, uses that instead of scene.image_prompt.
        If the scene has visual_type == "data" AND this is the first sub-image,
        applies DataOverlayRenderer after generating the MFLUX background.
        Data overlay only appears on the first sub-image — remaining ones are atmospheric.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        start_time = time.time()
        
        base_prompt = prompt_override or scene.image_prompt
        full_prompt = f"{self.config.style_prefix}, {base_prompt}" if self.config.style_prefix else base_prompt
        
        gen_width, gen_height = self.config.image_gen_width, self.config.image_gen_height
        final_width, final_height = self.config.image_output_width, self.config.image_output_height
        self._generate(full_prompt, output_path, gen_width, gen_height, seed)
        self._upscale(output_path, final_width, final_height)
        
        # Apply data overlay ONLY on first sub-image of a data scene
        if is_first_sub_image and getattr(scene, 'visual_type', 'scene') == 'data' and getattr(scene, 'data_overlay', None):
            self._apply_data_overlay(output_path, scene.data_overlay)
        
        generation_time = time.time() - start_time
        logger.info(f"Generated {output_path.name}: {final_width}x{final_height} in {generation_time:.1f}s (seed={seed})")
        
        return ImageResult(
            scene_id=scene.id,
            image_path=output_path,
            width=final_width,
            height=final_height,
            generation_time_seconds=generation_time,
            seed_used=seed,
        )

    def generate_thumbnail(self, prompt: str, output_path: Path) -> ImageResult:
        """Generate a single thumbnail base image at 1280x720.
        
        Args:
            prompt: Image generation prompt for the thumbnail.
            output_path: Where to write the PNG file.
            
        Returns:
            ImageResult with path and dimensions.
        """
        self._ensure_model_loaded()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        start_time = time.time()
        full_prompt = f"{self.config.style_prefix}, {prompt}" if self.config.style_prefix else prompt
        seed = self._get_seed("thumbnail")
        
        width, height = 1280, 720
        self._generate(full_prompt, output_path, width, height, seed)
        
        generation_time = time.time() - start_time
        logger.info(f"Generated thumbnail: {width}x{height} in {generation_time:.1f}s")
        
        return ImageResult(
            scene_id=None,
            image_path=output_path,
            width=width,
            height=height,
            generation_time_seconds=generation_time,
            seed_used=seed,
        )

    def release_model(self) -> None:
        """Explicitly release GPU memory used by the image model."""
        if self._model is not None:
            del self._model
            self._model = None
        self._model_loaded = False
        logger.info("Image model released")

    def _passes_quality_gate(self, image_path: Path) -> bool:
        """Check if a generated image passes quality thresholds.

        Detects:
        - Missing or tiny files (< 50KB)
        - Blank/near-blank images (low pixel variance)
        - Mostly-black or mostly-white images

        Returns True if image is acceptable, False if it should be retried.
        """
        # Check 1: File exists and minimum size
        if not image_path.exists():
            return False
        if image_path.stat().st_size < 50_000:
            return False

        try:
            from PIL import Image
            import numpy as np

            img = Image.open(image_path).convert("RGB")
            arr = np.array(img, dtype=np.float32)

            # Check 2: Pixel variance — blank images have near-zero std deviation
            # A real MFLUX image typically has std > 30. Blank/corrupt is < 10.
            pixel_std = arr.std()
            if pixel_std < 15.0:
                logger.debug(f"Quality gate: {image_path.name} pixel_std={pixel_std:.1f} (too low)")
                return False

            # Check 3: Mean brightness — reject mostly black (< 10) or mostly white (> 245)
            pixel_mean = arr.mean()
            if pixel_mean < 10.0 or pixel_mean > 245.0:
                logger.debug(f"Quality gate: {image_path.name} mean={pixel_mean:.1f} (too extreme)")
                return False

            return True
        except Exception as e:
            logger.debug(f"Quality gate: {image_path.name} failed to analyze: {e}")
            return False

    def _apply_data_overlay(self, image_path: Path, overlay_data: dict) -> None:
        """Apply data visualization overlay to a generated image.
        
        Uses DataOverlayRenderer to composite charts/stats onto the MFLUX background.
        Only called when scene has visual_type == "data".
        """
        try:
            from vidgen.config import DataOverlayConfig, load_config
            from vidgen.data_overlay import DataOverlayRenderer

            # Load config (try from default config.yaml)
            config_path = Path("config.yaml")
            if config_path.exists():
                full_config = load_config(config_path)
                overlay_config = full_config.data_overlay
            else:
                overlay_config = DataOverlayConfig()

            if not overlay_config.enabled:
                return

            renderer = DataOverlayRenderer(overlay_config)
            renderer.render(image_path, overlay_data, image_path)  # Overwrite in place

        except Exception as e:
            logger.warning(f"Data overlay failed for {image_path.name}: {e}")

    def _upscale(self, image_path: Path, target_width: int, target_height: int) -> None:
        """Upscale an image to target dimensions using LANCZOS resampling."""
        img = Image.open(str(image_path))
        if img.size == (target_width, target_height):
            return
        upscaled = img.resize((target_width, target_height), Image.LANCZOS)
        upscaled.save(str(image_path), "PNG")
        logger.debug(f"Upscaled {img.size[0]}x{img.size[1]} → {target_width}x{target_height}")

    def _get_seed(self, identifier: str) -> int:
        """Get seed based on strategy: fixed (deterministic from ID) or random."""
        if self.config.seed_strategy == "fixed":
            # Deterministic seed from identifier for reproducibility
            return int(hashlib.md5(identifier.encode()).hexdigest()[:8], 16)
        else:
            import random
            return random.randint(0, 2**32 - 1)

    def _generate(self, prompt: str, output_path: Path, width: int, height: int, seed: int) -> None:
        """Generate image using mflux-generate CLI.
        
        Shells out to the mflux-generate command which runs FLUX models
        on Apple Silicon via MLX Metal. Falls back to placeholder if
        mflux-generate is not found in PATH or if VIDGEN_PLACEHOLDER=1 is set.

        Retries up to 3 times on failure/timeout before falling back to placeholder.
        Validates output file size to catch degenerate (near-empty) images.
        """
        import os
        import shutil
        import subprocess

        # Allow forcing placeholder mode (for tests or when model isn't downloaded)
        if os.environ.get("VIDGEN_PLACEHOLDER", "0") == "1":
            self._generate_placeholder(output_path, width, height, seed)
            return

        mflux_bin = shutil.which("mflux-generate")
        if mflux_bin is None:
            # Check venv bin directly
            venv_bin = Path(__file__).resolve().parents[2] / ".venv" / "bin" / "mflux-generate"
            if venv_bin.exists():
                mflux_bin = str(venv_bin)

        if mflux_bin is None:
            logger.debug("mflux-generate not found, generating placeholder image")
            self._generate_placeholder(output_path, width, height, seed)
            return

        cmd = [
            mflux_bin,
            "--model", self.config.image_model,
            "--prompt", prompt,
            "--steps", str(self.config.image_steps),
            "--width", str(width),
            "--height", str(height),
            "--seed", str(seed),
            "--output", str(output_path),
        ]

        # Minimum file size for a valid 1536x864 PNG (~100KB minimum for real content)
        min_valid_size = 50_000

        max_retries = 3
        for attempt in range(1, max_retries + 1):
            logger.info(f"Running mflux-generate (seed={seed}, steps={self.config.image_steps}, attempt {attempt}/{max_retries})...")
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=300,  # 5 min timeout per image (1536x864 takes ~2 min)
                )
                if result.returncode != 0:
                    logger.warning(f"mflux-generate failed (attempt {attempt}): {result.stderr[:300]}")
                elif not output_path.exists():
                    logger.warning(f"mflux-generate produced no output file (attempt {attempt})")
                elif output_path.stat().st_size < min_valid_size:
                    logger.warning(
                        f"mflux-generate output too small ({output_path.stat().st_size} bytes, "
                        f"min {min_valid_size}) — likely degenerate (attempt {attempt})"
                    )
                    output_path.unlink(missing_ok=True)
                else:
                    # Success — valid image generated
                    logger.info(f"mflux-generate succeeded: {output_path} ({output_path.stat().st_size} bytes)")
                    return

            except subprocess.TimeoutExpired:
                logger.warning(f"mflux-generate timed out (attempt {attempt})")
                output_path.unlink(missing_ok=True)
            except Exception as e:
                logger.warning(f"mflux-generate error (attempt {attempt}): {e}")

            # Wait before retry (increasing backoff)
            if attempt < max_retries:
                import time
                wait = attempt * 10  # 10s, 20s
                logger.info(f"Retrying in {wait}s...")
                time.sleep(wait)

        # All retries exhausted — fall back to placeholder
        logger.error(f"mflux-generate failed after {max_retries} attempts, using placeholder")
        self._generate_placeholder(output_path, width, height, seed)

    def _generate_placeholder(self, output_path: Path, width: int, height: int, seed: int) -> None:
        """Generate a placeholder gradient image for pipeline testing."""
        import random
        rng = random.Random(seed)
        
        # Create a gradient image with some color variation based on seed
        r1, g1, b1 = rng.randint(20, 60), rng.randint(20, 60), rng.randint(40, 80)
        r2, g2, b2 = rng.randint(60, 120), rng.randint(40, 100), rng.randint(80, 160)
        
        img = Image.new("RGB", (width, height))
        pixels = img.load()
        for y in range(height):
            t = y / height
            r = int(r1 + (r2 - r1) * t)
            g = int(g1 + (g2 - g1) * t)
            b = int(b1 + (b2 - b1) * t)
            for x in range(width):
                pixels[x, y] = (r, g, b)
        
        img.save(str(output_path), "PNG")
