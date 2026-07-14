"""Unified content producer — generates all deliverables from a single script.

One script in → complete content package out:
  output/{number} - {title}/
    ├── video/final.mp4, thumbnail.png
    ├── shorts/short_1.mp4, short_2.mp4, short_3.mp4
    ├── distribution/thread.txt, newsletter.md, linkedin_post.txt
    └── metadata.json

Orchestrates the video pipeline, shorts pipeline, and distribution generators
in sequence with resume support.
"""

import json
import logging
import shutil
import time
from pathlib import Path

from pydantic import BaseModel, Field

from vidgen.config import PipelineConfig, load_config
from vidgen.distribution import generate_linkedin_post, generate_newsletter, generate_thread
from vidgen.models import Script, ScenePlan
from vidgen.parsers import parse_scene_plan, parse_script

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = Path("config.yaml")


class ProduceResult(BaseModel):
    """Result of a full produce run."""

    success: bool
    output_dir: Path | None = None
    video_ok: bool = False
    shorts_completed: int = 0
    shorts_total: int = 0
    distribution_ok: bool = False
    total_duration_seconds: float = 0.0
    error: str | None = None


class ContentProducer:
    """Orchestrates all content generation from a single script + scene plan.

    Stages:
    1. Video — long-form generation via PipelineOrchestrator
    2. Shorts — all matching short scripts via ShortsPipeline
    3. Distribution — thread, newsletter, LinkedIn post
    4. Package — organize everything into the final folder structure
    """

    def __init__(self, config_path: Path | None = None) -> None:
        path = config_path or DEFAULT_CONFIG
        if path.exists():
            self.config = load_config(path)
        else:
            self.config = PipelineConfig()

    def produce(
        self,
        script_path: Path,
        scene_plan_path: Path,
        shorts_dir: Path | None = None,
        video_url: str = "",
        resume: bool = False,
    ) -> ProduceResult:
        """Generate all content from a single script.

        Args:
            script_path: Path to the video script .md file.
            scene_plan_path: Path to the scene plan .json file.
            shorts_dir: Directory containing short scripts for this video.
                        Defaults to shorts/ and auto-detects by source_video number.
            video_url: YouTube URL (for distribution CTA links).
            resume: If True, skip stages whose output already exists.

        Returns:
            ProduceResult with status of each stage.
        """
        start_time = time.time()

        # Parse inputs
        script = parse_script(script_path)
        scene_plan = parse_scene_plan(scene_plan_path)

        # Derive output folder name: "003 - Why AI Startups Fail"
        number = self._extract_number(script_path)
        title_slug = self._slugify_title(script.title)
        folder_name = f"{number} - {title_slug}"

        output_dir = Path(self.config.output_dir) / folder_name
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Producing: {folder_name}")
        logger.info(f"Output: {output_dir}")

        result = ProduceResult(success=False, output_dir=output_dir)

        # --- Stage 1: Long-form video ---
        video_ok = self._produce_video(
            script, scene_plan, output_dir, resume
        )
        result.video_ok = video_ok

        # --- Stage 2: Shorts ---
        shorts_completed, shorts_total = self._produce_shorts(
            script_path, output_dir, shorts_dir, resume
        )
        result.shorts_completed = shorts_completed
        result.shorts_total = shorts_total

        # --- Stage 3: Distribution ---
        dist_ok = self._produce_distribution(script, output_dir, video_url)
        result.distribution_ok = dist_ok

        # --- Stage 4: Community Tab posts ---
        self._produce_community_posts(script, output_dir, video_url)

        # --- Stage 5: Metadata ---
        self._write_metadata(script, scene_plan, output_dir, video_url)

        # --- Stage 6: Captions (SRT) ---
        self._produce_captions(scene_plan, output_dir)

        # Overall result
        result.total_duration_seconds = time.time() - start_time
        result.success = video_ok  # Video is the critical path
        logger.info(
            f"Produce complete: video={'✓' if video_ok else '✗'}, "
            f"shorts={shorts_completed}/{shorts_total}, "
            f"distribution={'✓' if dist_ok else '✗'}, "
            f"time={result.total_duration_seconds:.0f}s"
        )
        return result

    def _produce_video(
        self, script: Script, scene_plan: ScenePlan,
        output_dir: Path, resume: bool
    ) -> bool:
        """Stage 1: Generate long-form video + thumbnail."""
        from vidgen.pipeline import PipelineOrchestrator

        video_dir = output_dir / "video"
        video_dir.mkdir(parents=True, exist_ok=True)

        # Check if already done (resume support)
        final_video = video_dir / "final.mp4"
        if resume and final_video.exists() and final_video.stat().st_size > 100_000:
            logger.info("  Video exists, skipping")
            return True

        # Run pipeline into a temp job dir
        job_dir = Path(self.config.jobs_dir) / f"produce_{scene_plan.topic_slug}"
        job_dir.mkdir(parents=True, exist_ok=True)

        orchestrator = PipelineOrchestrator(self.config, job_dir)
        try:
            pipeline_result = orchestrator.run(script, scene_plan, resume=resume)

            if pipeline_result.success:
                # Copy video to output
                source_video = job_dir / "assembly" / "video_raw.mp4"
                if source_video.exists():
                    shutil.copy2(source_video, final_video)

                # Copy best thumbnail
                thumb_dir = job_dir / "thumbnails"
                if thumb_dir.exists():
                    thumbs = sorted(thumb_dir.glob("variant_*.png"))
                    if thumbs:
                        shutil.copy2(thumbs[0], video_dir / "thumbnail.png")

                logger.info(f"  Video: {final_video.name} ({final_video.stat().st_size // (1024*1024)}MB)")
                return True
            else:
                logger.error(f"  Video pipeline failed: {pipeline_result.error}")
                return False
        except Exception as e:
            logger.error(f"  Video pipeline error: {e}")
            return False

    def _produce_shorts(
        self, script_path: Path, output_dir: Path,
        shorts_dir: Path | None, resume: bool
    ) -> tuple[int, int]:
        """Stage 2: Generate all shorts for this video."""
        from vidgen.shorts_pipeline import ShortsPipeline

        shorts_output = output_dir / "shorts"
        shorts_output.mkdir(parents=True, exist_ok=True)

        # Find short scripts matching this video number
        number = self._extract_number(script_path)
        search_dir = shorts_dir or Path("shorts")

        if not search_dir.exists():
            logger.info("  No shorts directory found, skipping")
            return 0, 0

        short_scripts = sorted(search_dir.glob(f"{number}_short_*.md"))
        if not short_scripts:
            logger.info(f"  No short scripts found for {number}, skipping")
            return 0, 0

        total = len(short_scripts)
        completed = 0

        # Check resume — count existing valid shorts
        if resume:
            existing = [f for f in shorts_output.glob("*.mp4") if f.stat().st_size > 50_000]
            if len(existing) >= total:
                logger.info(f"  {len(existing)} shorts exist, skipping")
                return len(existing), total

        pipeline = ShortsPipeline(
            config_path=Path("config.yaml") if Path("config.yaml").exists() else None
        )

        for short_script in short_scripts:
            short_name = short_script.stem
            output_path = shorts_output / f"{short_name}.mp4"

            # Skip if exists and valid (per-short resume)
            if resume and output_path.exists() and output_path.stat().st_size > 50_000:
                logger.info(f"  Short {short_name} exists, skipping")
                completed += 1
                continue

            try:
                result = pipeline.run(short_script)
                if result.success and result.output_path and result.output_path.exists():
                    shutil.copy2(result.output_path, output_path)
                    completed += 1
                    logger.info(f"  Short: {short_name}.mp4 ✓")
                else:
                    logger.warning(f"  Short {short_name} failed: {result.error}")
            except Exception as e:
                logger.warning(f"  Short {short_name} error: {e}")

        logger.info(f"  Shorts: {completed}/{total} completed")
        return completed, total

    def _produce_distribution(
        self, script: Script, output_dir: Path, video_url: str
    ) -> bool:
        """Stage 3: Generate distribution content (thread, newsletter, LinkedIn)."""
        dist_dir = output_dir / "distribution"
        dist_dir.mkdir(parents=True, exist_ok=True)

        try:
            thread = generate_thread(script, video_url=video_url)
            (dist_dir / "thread.txt").write_text(thread)

            newsletter = generate_newsletter(script, video_url=video_url)
            (dist_dir / "newsletter.md").write_text(newsletter)

            linkedin = generate_linkedin_post(script, video_url=video_url)
            (dist_dir / "linkedin_post.txt").write_text(linkedin)

            logger.info("  Distribution: thread.txt + newsletter.md + linkedin_post.txt ✓")
            return True
        except Exception as e:
            logger.error(f"  Distribution failed: {e}")
            return False

    def _produce_community_posts(
        self, script: Script, output_dir: Path, video_url: str
    ) -> None:
        """Stage 4: Generate Community Tab posts (square crops + engagement text)."""
        from vidgen.community import generate_community_posts

        try:
            # Find scene images from the job directory
            job_dir = Path(self.config.jobs_dir)
            # Look for images in the produce job or the video dir
            images_dir = output_dir / "video"
            scene_images: list[Path] = []

            # Check multiple possible locations for scene images
            for candidate in [
                job_dir / f"produce_{script.title[:30].lower().replace(' ', '-')}",
                *job_dir.glob("produce_*"),
            ]:
                img_dir = candidate / "images"
                if img_dir.exists():
                    scene_images = sorted(img_dir.glob("*.png"))
                    if scene_images:
                        break

            if not scene_images:
                logger.info("  Community: no scene images found, skipping")
                return

            community_dir = output_dir / "community"
            results = generate_community_posts(
                script=script,
                scene_images=scene_images,
                output_dir=community_dir,
                video_url=video_url,
            )
            if results:
                logger.info(f"  Community: {len(results)} posts generated ✓")
        except Exception as e:
            logger.warning(f"  Community posts failed: {e}")

    def _write_metadata(
        self, script: Script, scene_plan: ScenePlan,
        output_dir: Path, video_url: str
    ) -> None:
        """Stage 5: Write per-platform metadata files + consolidated metadata.json."""
        from vidgen.packaging import generate_metadata
        from vidgen.upload_package import generate_upload_package

        # --- New format: per-platform files in metadata/ folder ---
        try:
            # Find matching short scripts
            number = self._extract_number(
                Path(f"{scene_plan.topic_slug or 'unknown'}.md")
            )
            # Try to find short scripts by searching the shorts/ dir
            shorts_dir = Path("shorts")
            short_scripts: list[Path] = []
            if shorts_dir.exists():
                # Match by number from the output folder name
                folder_number = output_dir.name.split(" - ")[0].strip()
                short_scripts = sorted(shorts_dir.glob(f"{folder_number}_short_*.md"))

            generate_upload_package(
                script=script,
                scene_plan=scene_plan,
                video_url=video_url,
                short_scripts=short_scripts if short_scripts else None,
                output_path=output_dir,
            )
            logger.info("  Metadata: metadata/ (youtube.md, twitter.md, linkedin.md, substack.md, community.md) ✓")
        except Exception as e:
            logger.warning(f"  Upload package failed: {e}")

        # --- Legacy format: metadata.json (for backward compat) ---
        try:
            video_path = output_dir / "video" / "final.mp4"
            if video_path.exists():
                meta = generate_metadata(script, scene_plan, video_path)
                meta_dict = meta.model_dump(mode="json")
            else:
                meta_dict = {
                    "title": script.title,
                    "description": "",
                    "tags": [],
                    "duration_seconds": script.estimated_duration_seconds,
                    "resolution": "1920x1080",
                }

            meta_dict["video_url"] = video_url
            meta_dict["platforms"] = {
                "youtube": video_url,
                "substack": "https://tokeneconomyai.substack.com",
                "x": "https://x.com/TokenEconomyAI",
                "linkedin": "Token Economy AI",
            }

            (output_dir / "metadata.json").write_text(
                json.dumps(meta_dict, indent=2, default=str)
            )
        except Exception as e:
            logger.warning(f"  Metadata JSON write failed: {e}")

    def _produce_captions(self, scene_plan: ScenePlan, output_dir: Path) -> None:
        """Stage 6: Generate SRT captions from sentence boundary data."""
        from vidgen.captions import generate_srt

        try:
            # Find narration directory in the produce job folder
            job_dir = Path(self.config.jobs_dir) / f"produce_{scene_plan.topic_slug}"
            narration_dir = job_dir / "narration"

            if not narration_dir.exists():
                logger.info("  Captions: no narration dir found, skipping")
                return

            video_dir = output_dir / "video"
            video_dir.mkdir(parents=True, exist_ok=True)
            srt_path = video_dir / "captions.srt"

            result = generate_srt(narration_dir=narration_dir, output_path=srt_path)
            if result:
                logger.info(f"  Captions: {srt_path.name} ✓")
            else:
                logger.info("  Captions: no boundary data available, skipping")
        except Exception as e:
            logger.warning(f"  Captions failed: {e}")

    def _extract_number(self, script_path: Path) -> str:
        """Extract the script number (e.g. '003') from filename."""
        # Pattern: 003_why_ai_startups_fail.md → "003"
        stem = script_path.stem
        parts = stem.split("_", 1)
        if parts and parts[0].isdigit():
            return parts[0]
        return "000"

    def _slugify_title(self, title: str) -> str:
        """Create a short readable folder name from a video title.

        'Why 90% of AI Startups Will Fail by 2028 — And the Pattern...'
        → 'Why AI Startups Fail'
        """
        import re
        # Remove everything after dash/colon/parenthesis
        short = re.split(r'[—\-:(\[]', title)[0].strip()
        # Remove percentage/numbers for cleaner folder name
        short = re.sub(r'\d+%?\s*of\s+', '', short)
        # Remove "by YYYY"
        short = re.sub(r'\s+by\s+\d{4}', '', short)
        # Remove "Will" for brevity
        short = short.replace(" Will ", " ")
        # Trim to reasonable length
        words = short.split()[:6]
        return " ".join(words).strip()
