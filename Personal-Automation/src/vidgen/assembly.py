"""Video assembler - composes final video from generated assets using MoviePy + FFmpeg."""

import logging
import subprocess
from pathlib import Path

from vidgen.config import PipelineConfig
from vidgen.models import AssemblyResult, ImageResult, NarrationResult, Scene

logger = logging.getLogger(__name__)


class VideoAssembler:
    """Composes final horizontal video (1920x1080, 30fps) from scene images and narration.

    Uses MoviePy for programmatic video composition with Ken Burns effects,
    crossfade transitions, subtitle burn-in, and background music mixing.
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config

    def assemble(
        self,
        scenes: list[Scene],
        narration_results: list[NarrationResult],
        image_results: list[ImageResult],
        output_path: Path,
        background_music: Path | None = None,
    ) -> AssemblyResult:
        """Compose final video from all assets.

        Args:
            scenes: List of scenes with timing and effect specifications.
            narration_results: Generated audio segments per scene.
            image_results: Generated images per scene.
            output_path: Where to write the final MP4.
            background_music: Optional path to background music file.

        Returns:
            AssemblyResult with video path, duration, resolution, fps, and file size.
        """
        from moviepy import (
            AudioFileClip,
            CompositeVideoClip,
            ImageClip,
            concatenate_videoclips,
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Build lookup maps
        narration_map = {nr.scene_id: nr for nr in narration_results}
        # Group images by scene_id (multiple images per scene now)
        from collections import defaultdict
        images_by_scene: dict[str, list] = defaultdict(list)
        for ir in image_results:
            images_by_scene[ir.scene_id].append(ir)

        # Ken Burns directions — 8 effects tuned for horizontal 16:9 landscape
        kb_directions = [
            "zoom-in", "pan-right", "diagonal-tl", "zoom-out",
            "pan-left", "pan-up", "diagonal-br", "pan-down",
        ]
        clips = []

        for scene in scenes:
            narration = narration_map.get(scene.id)
            scene_images = images_by_scene.get(scene.id, [])

            if not narration or not scene_images:
                logger.warning(f"Missing assets for scene {scene.id}, skipping")
                continue

            narration_duration = narration.duration_seconds
            gap_seconds = 1.0
            num_sub_images = len(scene_images)

            # Calculate duration per sub-image (narration split evenly, gap at end)
            sub_duration = narration_duration / num_sub_images
            audio = AudioFileClip(str(narration.audio_path))

            for idx, img_result in enumerate(scene_images):
                # Each sub-image gets an equal portion of time
                start_t = idx * sub_duration
                end_t = (idx + 1) * sub_duration

                # Last sub-image of the scene gets the extra 1s gap
                is_last = (idx == num_sub_images - 1)
                clip_duration = sub_duration + (gap_seconds if is_last else 0)

                img_clip = ImageClip(str(img_result.image_path)).with_duration(clip_duration)

                # Alternate Ken Burns direction for visual variety
                kb_dir = kb_directions[idx % len(kb_directions)]
                img_clip = self._apply_ken_burns(img_clip, kb_dir, clip_duration)

                # Force fixed 1920x1080 canvas — prevents black frames from
                # variable-size Ken Burns clips confusing concatenation
                img_clip = CompositeVideoClip(
                    [img_clip], size=(1920, 1080)
                ).with_duration(clip_duration)

                # Slice audio for this sub-image's portion
                audio_start = start_t
                audio_end = min(end_t, narration_duration - 0.01)  # Epsilon to avoid float precision edge
                if audio_start < narration_duration - 0.01:
                    sub_audio = audio.subclipped(audio_start, audio_end)
                    img_clip = img_clip.with_audio(sub_audio)

                # Distribute text overlays across sub-images
                # Each overlay is assigned to the sub-image that contains its appear_at time
                overlays = scene.text_overlays  # Already normalized from model validator
                for overlay in overlays:
                    # Check if this overlay's appear_at falls within this sub-image's time window
                    if start_t <= overlay.appear_at < end_t:
                        # Adjust appear_at relative to this sub-image's start
                        relative_appear = overlay.appear_at - start_t
                        img_clip = self._add_text_overlay_at(
                            img_clip, overlay, relative_appear, narration.audio_path, start_t
                        )

                clips.append(img_clip)

        if not clips:
            raise RuntimeError("No valid clips to assemble")

        # Concatenate clips — all are fixed 1920x1080 so chain is safe
        final = concatenate_videoclips(clips, method="chain")

        # Mix background music if provided
        if background_music and background_music.exists():
            final = self._mix_background_music(final, background_music, scenes)

        # Add logo watermark (bottom-right, semi-transparent)
        final = self._add_logo_watermark(final)

        # Write final video
        final.write_videofile(
            str(output_path),
            fps=30,
            codec="libx264",
            audio_codec="aac",
            audio_bitrate="192k",
            logger=None,
            ffmpeg_params=["-crf", "18", "-preset", "veryfast"],
        )

        # Capture duration before closing
        final_duration = final.duration

        # Get file size (before LUFS mastering)
        file_size = output_path.stat().st_size

        # Clean up MoviePy resources
        final.close()
        for clip in clips:
            clip.close()

        # LUFS mastering: normalize to -14 LUFS (YouTube standard) with -1.0 dBTP ceiling
        # NOTE: Temporarily disabled — loudnorm filter can truncate audio when 
        # MoviePy composite audio has duration mismatch. YouTube normalizes on their end anyway.
        # self._normalize_loudness(output_path)

        return AssemblyResult(
            video_path=output_path,
            duration_seconds=final_duration,
            resolution=(1920, 1080),
            fps=30,
            file_size_bytes=file_size,
        )

    def _apply_ken_burns(self, clip, direction: str, duration: float):
        """Apply Ken Burns pan/zoom effect to a still image clip.

        Tuned for horizontal 1920×1080 output:
        - Zooms: 20% (softer than portrait, more cinematic)
        - Horizontal pans: 20% of width (384px travel)
        - Vertical pans: 10% of height (subtle, avoids jarring movement)
        - Diagonals: combined subtle pan + zoom for cinematic feel

        Args:
            clip: The ImageClip to animate.
            direction: Effect name.
            duration: Clip duration in seconds.

        Returns:
            Clip with Ken Burns motion applied.
        """
        w, h = clip.size

        if direction == "zoom-in":
            # 1.0 → 1.2 center zoom
            def resize_func(t):
                progress = t / duration if duration > 0 else 0
                return 1.0 + 0.2 * progress
            return clip.resized(resize_func)

        elif direction == "zoom-out":
            # 1.2 → 1.0 reverse zoom
            def resize_func(t):
                progress = t / duration if duration > 0 else 0
                return 1.2 - 0.2 * progress
            return clip.resized(resize_func)

        elif direction == "pan-left":
            # Fixed 1.15x, pan right→left (20% of width)
            enlarged = clip.resized(1.15)
            travel = int(w * 0.2)

            def position_func(t):
                progress = t / duration if duration > 0 else 0
                return (-int(travel * progress), 0)
            return enlarged.with_position(position_func)

        elif direction == "pan-right":
            # Fixed 1.15x, pan left→right (20% of width)
            enlarged = clip.resized(1.15)
            travel = int(w * 0.2)

            def position_func(t):
                progress = t / duration if duration > 0 else 0
                return (-travel + int(travel * progress), 0)
            return enlarged.with_position(position_func)

        elif direction == "pan-up":
            # Fixed 1.15x, pan bottom→top (10% of height — subtle for landscape)
            enlarged = clip.resized(1.15)
            travel = int(h * 0.1)

            def position_func(t):
                progress = t / duration if duration > 0 else 0
                return (0, -int(travel * progress))
            return enlarged.with_position(position_func)

        elif direction == "pan-down":
            # Fixed 1.15x, pan top→bottom (10% of height)
            enlarged = clip.resized(1.15)
            travel = int(h * 0.1)

            def position_func(t):
                progress = t / duration if duration > 0 else 0
                return (0, -travel + int(travel * progress))
            return enlarged.with_position(position_func)

        elif direction == "diagonal-tl":
            # Diagonal pan: top-left to bottom-right with slight zoom
            enlarged = clip.resized(1.2)
            travel_x = int(w * 0.12)
            travel_y = int(h * 0.08)

            def position_func(t):
                progress = t / duration if duration > 0 else 0
                return (-int(travel_x * progress), -int(travel_y * progress))
            return enlarged.with_position(position_func)

        elif direction == "diagonal-br":
            # Diagonal pan: bottom-right to top-left with slight zoom
            enlarged = clip.resized(1.2)
            travel_x = int(w * 0.12)
            travel_y = int(h * 0.08)

            def position_func(t):
                progress = t / duration if duration > 0 else 0
                return (-travel_x + int(travel_x * progress),
                        -travel_y + int(travel_y * progress))
            return enlarged.with_position(position_func)

        else:
            # Default fallback: gentle 1.0 → 1.15 center zoom
            def resize_func(t):
                progress = t / duration if duration > 0 else 0
                return 1.0 + 0.15 * progress
            return clip.resized(resize_func)

    def _add_text_overlay_at(self, clip, overlay, relative_appear: float, narration_path: Path | None = None, scene_offset: float = 0.0):
        """Add lower-third style text overlay at a specific time within the clip.
        
        Uses sentence boundary timestamps for precise timing when available.
        No text should be baked into AI-generated images (MFLUX renders text poorly).

        Args:
            clip: The video clip to add the overlay to.
            overlay: TextOverlay model with text, position, appear_at, duration.
            relative_appear: When to show the overlay relative to clip start.
            narration_path: Path to narration audio (for boundary lookup).
            scene_offset: The offset of this sub-image within the full scene narration.
        """
        from moviepy import CompositeVideoClip, TextClip, ColorClip

        if not overlay:
            return clip

        try:
            import json as _json
            
            clip_dur = clip.duration
            appear_at = relative_appear
            
            # Try to use sentence boundaries for precise timing
            boundaries_path = narration_path.with_suffix(".boundaries.json") if narration_path else None
            if boundaries_path and boundaries_path.exists():
                try:
                    boundaries = _json.loads(boundaries_path.read_text())
                    # Match overlay text to the sentence that contains those words
                    overlay_words = set(overlay.text.lower().split())
                    overlay_words = {w for w in overlay_words if len(w) > 2}
                    
                    best_score = 0
                    best_time = None
                    for b in boundaries:
                        sent_words = set(b["text"].lower().split())
                        overlap = len(overlay_words & sent_words)
                        if overlap > best_score:
                            best_score = overlap
                            best_time = b["start"]
                    
                    if best_time is not None and best_score >= 2:
                        # Use boundary time relative to this sub-image
                        boundary_relative = best_time - scene_offset
                        # Only accept if within ±3s of original timing (prevents wild jumps)
                        if 0 <= boundary_relative < clip_dur and abs(boundary_relative - appear_at) <= 3.0:
                            appear_at = boundary_relative
                            logger.debug(f"Overlay '{overlay.text}' synced to {appear_at:.1f}s via boundaries")
                except Exception:
                    pass  # Fall back to manual timing

            # Pre-emptive timing: appear 0.3s before
            appear_at = min(max(0, appear_at - 0.3), max(0, clip_dur - 3.0))
            duration = min(overlay.duration, clip_dur - appear_at)
            if duration < 1.5:
                duration = min(1.5, clip_dur - appear_at)
            if duration <= 0:
                return clip

            # Create text with stroke for legibility over any background
            txt_clip = TextClip(
                text=overlay.text,
                font_size=44,
                color="white",
                font="Helvetica",
                stroke_color="black",
                stroke_width=2,
            )

            # Create semi-transparent dark bar behind text
            bar_height = 70
            bar = ColorClip(
                size=(clip.w, bar_height),
                color=(10, 10, 20),
            ).with_opacity(0.75).with_duration(duration)

            # Position bar and text
            if overlay.position == "top":
                bar_y = 30
            elif overlay.position == "center":
                bar_y = (clip.h - bar_height) // 2
            else:  # bottom (default)
                bar_y = clip.h - bar_height - 40

            bar = bar.with_position((0, bar_y)).with_start(appear_at)
            txt_clip = txt_clip.with_position(("center", bar_y + 15)).with_start(appear_at)
            txt_clip = txt_clip.with_duration(duration)

            return CompositeVideoClip([clip, bar, txt_clip], size=(clip.w, clip.h)).with_duration(clip_dur)
        except Exception as e:
            logger.warning(f"Could not add text overlay: {e}")
            return clip

    def _add_logo_watermark(self, clip):
        """Add semi-transparent Token Economy logo watermark to bottom-right corner.
        
        Uses the channel logo PNG if available, otherwise skips gracefully.
        Logo is displayed at 180x180 pixels, 42% opacity, 20px from edges.
        """
        from moviepy import CompositeVideoClip, ImageClip as StaticImageClip

        # Look for logo in standard locations
        logo_paths = [
            Path("channel/assets/TokenEconomy_logo.png"),
            Path("assets/logo.png"),
        ]
        
        logo_path = None
        for p in logo_paths:
            if p.exists():
                logo_path = p
                break
        
        if logo_path is None:
            logger.debug("No logo file found, skipping watermark")
            return clip

        try:
            logo = StaticImageClip(str(logo_path))
            # Resize to 80x80
            logo = logo.resized((180, 180))
            # Set opacity and duration
            logo = logo.with_opacity(0.42).with_duration(clip.duration)
            # Position: bottom-right with 20px margin
            logo = logo.with_position((clip.w - 200, clip.h - 200))
            
            return CompositeVideoClip([clip, logo])
        except Exception as e:
            logger.warning(f"Could not add logo watermark: {e}")
            return clip

    def _mix_background_music(self, video_clip, music_path: Path, scenes: list | None = None):
        """Mix background music at -22dB with per-section mood switching.

        If scenes have music_mood tags, selects different tracks per section
        and crossfades between them. Falls back to single-track loop if no
        mood tags are present or if mood-specific tracks aren't found.

        Args:
            video_clip: The assembled video clip with narration audio.
            music_path: Default music file (used when no mood tags exist).
            scenes: Optional list of scenes with music_mood and timing.

        Returns:
            Video clip with mixed background music.
        """
        from moviepy import AudioFileClip, CompositeAudioClip, concatenate_audioclips

        music_dir = Path("channel/assets/music")
        crossfade_duration = 2.0  # seconds between mood transitions
        volume_db = self.config.background_music_db  # -22dB default, same as Shorts
        volume_factor = 10 ** (volume_db / 20)

        try:
            # Check if we have per-section moods
            mood_segments = self._build_mood_segments(scenes, video_clip.duration) if scenes else []

            if mood_segments and len(mood_segments) > 1:
                # Per-section music: build a stitched audio track from mood-specific files
                music_clips = []
                for mood, start, end in mood_segments:
                    track_path = music_dir / f"{mood}.mp3"
                    if not track_path.exists():
                        track_path = music_path  # fallback to default

                    segment_duration = end - start
                    clip = AudioFileClip(str(track_path))

                    # Loop or trim to segment duration + crossfade overlap
                    target_dur = segment_duration + crossfade_duration
                    if clip.duration < target_dur:
                        # Loop by repeating
                        import math
                        repeats = math.ceil(target_dur / clip.duration)
                        clip = concatenate_audioclips([clip] * repeats)
                    clip = clip.subclipped(0, min(target_dur, clip.duration))

                    music_clips.append(clip)

                # Concatenate with crossfade
                if music_clips:
                    # Simple concatenation (crossfade handled via fade in/out per segment)
                    from moviepy.audio.fx import AudioFadeIn, AudioFadeOut
                    final_clips = []
                    for i, clip in enumerate(music_clips):
                        # Fade in at start of each segment (except first)
                        if i > 0:
                            clip = clip.with_effects([AudioFadeIn(crossfade_duration)])
                        # Fade out at end of each segment (except last)
                        if i < len(music_clips) - 1:
                            clip = clip.with_effects([AudioFadeOut(crossfade_duration)])
                        final_clips.append(clip)

                    music = concatenate_audioclips(final_clips)
                else:
                    music = AudioFileClip(str(music_path))
            else:
                # Single track loop (fallback)
                music = AudioFileClip(str(music_path))
                if music.duration < video_clip.duration:
                    import math
                    repeats = math.ceil(video_clip.duration / music.duration)
                    music = concatenate_audioclips([music] * repeats)

            # Trim to video duration
            music = music.subclipped(0, min(video_clip.duration, music.duration))

            # Apply volume reduction
            music = music.with_volume_scaled(volume_factor)

            # Fade in at start, fade out at end
            from moviepy.audio.fx import AudioFadeIn, AudioFadeOut
            music = music.with_effects([AudioFadeIn(3.0), AudioFadeOut(3.0)])

            # Composite with narration
            if video_clip.audio:
                final_audio = CompositeAudioClip([video_clip.audio, music])
                return video_clip.with_audio(final_audio)
            return video_clip.with_audio(music)

        except Exception as e:
            logger.warning(f"Could not mix background music: {e}")
            return video_clip

    def _build_mood_segments(self, scenes: list, total_duration: float) -> list[tuple[str, float, float]]:
        """Build a list of (mood, start_time, end_time) from scenes.

        Groups consecutive scenes with the same mood into segments.
        Defaults to 'neutral' for scenes without a music_mood tag.
        """
        if not scenes:
            return []

        segments: list[tuple[str, float, float]] = []
        current_mood = None
        segment_start = 0.0

        for scene in scenes:
            mood = scene.music_mood or "neutral"
            if mood != current_mood:
                if current_mood is not None:
                    segments.append((current_mood, segment_start, scene.start_time))
                current_mood = mood
                segment_start = scene.start_time

        # Close final segment
        if current_mood is not None:
            segments.append((current_mood, segment_start, total_duration))

        return segments

    def _normalize_loudness(self, video_path: Path) -> None:
        """Normalize audio to -14 LUFS (YouTube standard) with -1.0 dBTP ceiling.

        Uses FFmpeg's loudnorm filter in two-pass mode for accurate results.
        This ensures consistent perceived loudness across all videos and matches
        YouTube's internal normalization target, preventing platform-side compression.
        """
        import subprocess
        import tempfile

        try:
            # Two-pass loudnorm: first pass measures, second applies
            temp_output = video_path.with_suffix(".lufs_temp.mp4")

            # Single-pass loudnorm (linear mode for speed — close enough for our use case)
            cmd = [
                "ffmpeg", "-y", "-i", str(video_path),
                "-af", "loudnorm=I=-14:TP=-1.0:LRA=11:linear=true",
                "-c:v", "copy",  # Don't re-encode video — just fix audio
                "-c:a", "aac", "-b:a", "192k",
                str(temp_output),
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

            if result.returncode == 0 and temp_output.exists():
                # Replace original with normalized version
                temp_output.replace(video_path)
                logger.info(f"Audio normalized to -14 LUFS: {video_path.name}")
            else:
                logger.warning(f"LUFS normalization failed: {result.stderr[:200]}")
                temp_output.unlink(missing_ok=True)

        except subprocess.TimeoutExpired:
            logger.warning("LUFS normalization timed out")
        except Exception as e:
            logger.warning(f"LUFS normalization error: {e}")

    def burn_subtitles(self, video_path: Path, script, timings: list[NarrationResult]) -> Path:
        """Burn subtitles into video track.

        Args:
            video_path: Path to the source video.
            script: The Script object with narration text.
            timings: NarrationResult objects with timing info.

        Returns:
            Path to the subtitled video file.
        """
        # Subtitle burning is handled during assembly via text overlays.
        # This method provides a post-processing fallback using FFmpeg.
        output_path = video_path.with_name(video_path.stem + "_subtitled.mp4")

        # Generate SRT content from script and timings
        srt_path = video_path.with_suffix(".srt")
        self._generate_srt(script, timings, srt_path)

        # Use FFmpeg to burn subtitles
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-vf",
            f"subtitles={srt_path}",
            "-c:a",
            "copy",
            str(output_path),
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True)
            logger.info(f"Subtitles burned into {output_path}")
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            logger.warning(f"Could not burn subtitles via FFmpeg: {e}")
            return video_path

        return output_path

    def _generate_srt(self, script, timings: list[NarrationResult], output_path: Path) -> None:
        """Generate SRT subtitle file from script text and timing data."""
        lines: list[str] = []
        current_time = 0.0

        for i, timing in enumerate(timings, 1):
            start = self._format_srt_time(current_time)
            end = self._format_srt_time(current_time + timing.duration_seconds)
            # Get text (simplified - full implementation would chunk by sentence)
            text = f"Scene {timing.scene_id}"
            lines.append(f"{i}")
            lines.append(f"{start} --> {end}")
            lines.append(text)
            lines.append("")
            current_time += timing.duration_seconds

        output_path.write_text("\n".join(lines), encoding="utf-8")

    @staticmethod
    def _format_srt_time(seconds: float) -> str:
        """Format seconds as SRT timestamp (HH:MM:SS,mmm)."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
