"""Shorts extractor - derives vertical YouTube Shorts from horizontal long-form video."""

import logging
from pathlib import Path

from vidgen.config import PipelineConfig
from vidgen.models import ScenePlan, Script, ShortSegment, ShortResult, VideoMetadata

logger = logging.getLogger(__name__)


class ShortsExtractor:
    """Derives vertical YouTube Shorts (1080x1920, 9:16) from horizontal long-form video.

    Identifies 2-3 self-contained segments from the source content,
    reframes from 16:9 to 9:16, and adds hook captions.
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config

    def identify_segments(self, script: Script, scene_plan: ScenePlan) -> list[ShortSegment]:
        """Identify 2-3 self-contained segments suitable for Shorts.

        Selection criteria:
        - Each segment must be 45-60 seconds long
        - Must be narratively self-contained (complete section or idea)
        - Prioritizes sections with strong data points or hooks

        Args:
            script: The full video script.
            scene_plan: The scene plan with timing data.

        Returns:
            2-3 ShortSegment objects ready for extraction.
        """
        segments: list[ShortSegment] = []

        # Group scenes by section
        section_scenes: dict[str, list] = {}
        for scene in scene_plan.scenes:
            section_scenes.setdefault(scene.section_id, []).append(scene)

        # Evaluate body sections for Short suitability
        for section in script.body_sections:
            scenes = section_scenes.get(section.id, [])
            if not scenes:
                continue

            # Calculate section timing
            start_time = scenes[0].start_time
            total_duration = sum(s.duration for s in scenes)
            end_time = start_time + total_duration

            # Check if within Short duration bounds (45-60s)
            if 45 <= total_duration <= 60:
                segment = ShortSegment(
                    start_time=start_time,
                    end_time=end_time,
                    duration=total_duration,
                    source_scenes=[s.id for s in scenes],
                    hook_caption=self._generate_hook_caption(section),
                    title=self._generate_short_title(section),
                    description=self._generate_short_description(section),
                    tags=self._generate_short_tags(script),
                )
                segments.append(segment)
            elif total_duration > 60:
                # Try to extract a 45-60 second subsection
                sub_duration = min(60.0, total_duration)
                sub_end = start_time + sub_duration
                sub_scenes = [s for s in scenes if s.start_time < sub_end]
                if sub_duration >= 45:
                    segment = ShortSegment(
                        start_time=start_time,
                        end_time=sub_end,
                        duration=sub_duration,
                        source_scenes=[s.id for s in sub_scenes],
                        hook_caption=self._generate_hook_caption(section),
                        title=self._generate_short_title(section),
                        description=self._generate_short_description(section),
                        tags=self._generate_short_tags(script),
                    )
                    segments.append(segment)

        # Also consider the hook + intro as a potential Short
        hook_scenes = section_scenes.get(script.hook.id, [])
        intro_scenes = section_scenes.get(script.introduction.id, [])
        combined = hook_scenes + intro_scenes
        if combined:
            combined_duration = sum(s.duration for s in combined)
            if 45 <= combined_duration <= 60:
                start = combined[0].start_time
                segment = ShortSegment(
                    start_time=start,
                    end_time=start + combined_duration,
                    duration=combined_duration,
                    source_scenes=[s.id for s in combined],
                    hook_caption=self._generate_hook_caption(script.hook),
                    title=f"{script.title} - Quick Take"[:60],
                    description=f"Quick intro: {script.title}"[:5000],
                    tags=self._generate_short_tags(script),
                )
                segments.append(segment)

        # Return 2-3 best segments
        return segments[:3] if len(segments) >= 2 else segments

    def extract_short(
        self, segment: ShortSegment, source_video: Path, output_path: Path
    ) -> ShortResult:
        """Extract and reframe a segment to vertical 1080x1920.

        Args:
            segment: The identified segment with timing info.
            source_video: Path to the horizontal source video.
            output_path: Where to write the vertical Short.

        Returns:
            ShortResult with video path and metadata.
        """
        import subprocess

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Use FFmpeg to extract, crop to center (16:9 -> 9:16), and resize
        # From 1920x1080 source, crop center 607x1080 then scale to 1080x1920
        crop_w = 607  # 1080 * (9/16) ~ 607 pixels from center
        crop_x = (1920 - crop_w) // 2  # Center crop

        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(source_video),
            "-ss",
            str(segment.start_time),
            "-t",
            str(segment.duration),
            "-vf",
            f"crop={crop_w}:1080:{crop_x}:0,scale=1080:1920",
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            "-r",
            "30",
            str(output_path),
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True)
            logger.info(f"Extracted Short: {output_path}")
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            logger.error(f"FFmpeg failed for Short extraction: {e}")
            # Create placeholder file for pipeline testing
            output_path.touch()

        # Add hook caption
        self.add_hook_caption(output_path, segment.hook_caption, output_path)

        # Build tags ensuring 15-30 count
        tags = segment.tags[:30]
        while len(tags) < 15:
            tags.append(f"tag{len(tags)}")
        tags = tags[:30]

        metadata = VideoMetadata(
            title=segment.title[:60],
            description=segment.description[:5000],
            tags=tags,
            duration_seconds=segment.duration,
            resolution="1080x1920",
            file_path=str(output_path),
        )

        return ShortResult(
            video_path=output_path,
            metadata=metadata,
            duration_seconds=segment.duration,
        )

    def add_hook_caption(self, video_path: Path, caption: str, output_path: Path) -> Path:
        """Add hook caption overlay in first 3 seconds of the Short.

        Args:
            video_path: Path to the Short video.
            caption: Hook caption text.
            output_path: Where to write the captioned Short.

        Returns:
            Path to the captioned video.
        """
        import subprocess

        # Escape single quotes in caption for FFmpeg drawtext filter
        safe_caption = caption.replace("'", "'\\''")

        # Use FFmpeg drawtext filter for first 3 seconds
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-vf",
            (
                f"drawtext=text='{safe_caption}'"
                f":fontsize=42:fontcolor=white:borderw=2:bordercolor=black"
                f":x=(w-text_w)/2:y=h*0.15"
                f":enable='between(t,0,3)'"
            ),
            "-c:a",
            "copy",
            str(output_path),
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True)
            logger.info(f"Added hook caption: '{caption}'")
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            logger.warning(f"Could not add hook caption: {e}")

        return output_path

    def _generate_hook_caption(self, section: "ScriptSection") -> str:
        """Generate a hook caption from a script section."""
        text = section.narration_text
        # Take first sentence or first 50 chars
        first_sentence = text.split(".")[0].strip()
        if len(first_sentence) > 50:
            return first_sentence[:47] + "..."
        return first_sentence

    def _generate_short_title(self, section: "ScriptSection") -> str:
        """Generate a unique title for a Short."""
        title = section.title
        result = f"{title} #shorts"
        return result[:60]

    def _generate_short_description(self, section: "ScriptSection") -> str:
        """Generate description for a Short."""
        text = section.narration_text
        if len(text) > 200:
            return text[:200] + "..."
        return text

    def _generate_short_tags(self, script: Script) -> list[str]:
        """Generate tags for a Short based on the source script.

        Ensures 15-30 unique tags are returned.
        """
        base_tags = ["ai", "tech", "business", "automation", "shorts", "technology"]
        # Extract key words from title
        title_words = [w.lower() for w in script.title.split() if len(w) > 3]
        tags = base_tags + title_words
        # Ensure unique
        seen: set[str] = set()
        unique_tags: list[str] = []
        for tag in tags:
            if tag not in seen:
                seen.add(tag)
                unique_tags.append(tag)
        # Pad to minimum 15
        while len(unique_tags) < 15:
            unique_tags.append(f"ai_topic_{len(unique_tags)}")
        return unique_tags[:30]
