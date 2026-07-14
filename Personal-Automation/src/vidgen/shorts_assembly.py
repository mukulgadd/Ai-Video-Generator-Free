"""Shorts assembler — composes vertical video (1080x1920) with bold text overlays and music.

Designed for YouTube Shorts: fast-paced image changes, large readable text for
muted viewing, background music, and channel branding.

Architecture: All visual rendering (Ken Burns + text overlays + logo) is done
in a single make_frame function using Pillow. This avoids MoviePy's
CompositeVideoClip which segfaults at 1080x1920 for 40s+ durations with
multiple TextClip layers.
"""

import logging
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from vidgen.config import ShortsConfig
from vidgen.models import ImageCue, OverlayCue

logger = logging.getLogger(__name__)

# Text style definitions
TEXT_STYLES = {
    "impact": {"font_size": 96, "color": (255, 255, 255), "stroke_width": 5, "stroke_color": (0, 0, 0)},
    "normal": {"font_size": 72, "color": (255, 255, 255), "stroke_width": 3, "stroke_color": (0, 0, 0)},
    "danger": {"font_size": 80, "color": (233, 69, 96), "stroke_width": 4, "stroke_color": (0, 0, 0)},
}

FONT_FALLBACKS = [
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/SFNSDisplay.ttf",
    "/Library/Fonts/Arial Bold.ttf",
]


class ShortsAssembler:
    """Assembles vertical YouTube Shorts from images, narration, overlays, and music.

    Output: 1080x1920 MP4, 24fps, H.264 CRF 18, AAC audio.
    """

    def __init__(self, config: ShortsConfig) -> None:
        self.config = config
        self._font_path = self._resolve_font()

    def assemble(
        self,
        narration_path: Path,
        image_paths: list[Path],
        image_cues: list[ImageCue],
        overlay_cues: list[OverlayCue],
        hook_text: str,
        output_path: Path,
        music_path: Path | None = None,
        logo_path: Path | None = None,
        duration_target: float | None = None,
    ) -> Path:
        """Assemble a complete YouTube Short.

        Uses a single VideoClip(make_frame) that renders Ken Burns + text overlays
        + logo via Pillow. No CompositeVideoClip — avoids MoviePy 2.x segfaults.
        
        If sentence boundaries file exists alongside narration, uses it for
        precise overlay timing sync. Otherwise falls back to proportional scaling.
        """
        from moviepy import AudioFileClip, VideoClip

        output_path.parent.mkdir(parents=True, exist_ok=True)
        WIDTH, HEIGHT = self.config.image_output_width, self.config.image_output_height

        # Load narration to get actual duration
        narration_audio = AudioFileClip(str(narration_path))
        total_duration = narration_audio.duration

        # Detect looped short: narration ends with "..." = mid-sentence loop
        # Looped shorts have no end card (audio runs edge-to-edge for seamless loop)
        is_looped = hook_text.endswith("...")
        end_card_duration = 0.0 if is_looped else 3.0
        final_duration = total_duration + end_card_duration

        # Calculate time scaling factor: overlay times in script assume duration_target,
        # but actual narration may be shorter/longer. Scale proportionally.
        if duration_target and duration_target > 0:
            time_scale = total_duration / duration_target
        else:
            time_scale = 1.0

        # Pre-load all images at 110% size for Ken Burns headroom
        zoom = self.config.ken_burns_zoom
        num_images = min(len(image_paths), len(image_cues))
        if num_images == 0:
            raise RuntimeError("No valid image clips to assemble")

        duration_per_image = final_duration / num_images
        image_arrays = []
        for i in range(num_images):
            img_path = image_paths[i]
            if not img_path.exists():
                # Use a black frame as fallback
                image_arrays.append(np.zeros((int(HEIGHT * (1 + zoom)), int(WIDTH * (1 + zoom)), 3), dtype=np.uint8))
                continue
            img = Image.open(img_path).convert("RGB").resize(
                (int(WIDTH * (1 + zoom)), int(HEIGHT * (1 + zoom))), Image.LANCZOS
            )
            image_arrays.append(np.array(img))

        img_h, img_w = image_arrays[0].shape[:2]

        # Pre-load logo if available
        logo_img = None
        if logo_path and logo_path.exists():
            try:
                logo_size = self.config.logo_size
                logo_img = Image.open(logo_path).convert("RGBA").resize(
                    (logo_size, logo_size), Image.LANCZOS
                )
            except Exception:
                pass

        # Build overlay schedule: list of (start, end, text, style_def, font_size)
        # Use sentence boundaries for precise sync if available, else proportional scaling
        PREEMPTIVE_OFFSET = 0.3
        
        # Try to load sentence boundaries from narration
        import json as _json
        boundaries_path = narration_path.with_suffix(".boundaries.json")
        sentence_boundaries = None
        if boundaries_path.exists():
            try:
                sentence_boundaries = _json.loads(boundaries_path.read_text())
                logger.info(f"Using {len(sentence_boundaries)} sentence boundaries for overlay sync")
            except Exception:
                pass

        overlay_schedule = []
        if hook_text:
            overlay_schedule.append((0.0, 3.0, hook_text, TEXT_STYLES["impact"], 110))
        
        if sentence_boundaries and overlay_cues:
            # PRECISE MODE: match each overlay to its sentence
            for cue in overlay_cues:
                fallback = cue.start_time * time_scale
                best_time = self._match_overlay_to_sentence(cue.text, sentence_boundaries, fallback_time=fallback)
                if best_time is not None:
                    scaled_start = max(0.0, best_time - PREEMPTIVE_OFFSET)
                else:
                    scaled_start = max(0.0, fallback - PREEMPTIVE_OFFSET)
                scaled_end = min(scaled_start + 3.5, final_duration)  # Max 3.5s display
                if scaled_start >= final_duration:
                    continue
                style_def = TEXT_STYLES.get(cue.style, TEXT_STYLES["normal"])
                overlay_schedule.append((scaled_start, scaled_end, cue.text, style_def, style_def["font_size"]))
        else:
            # FALLBACK: proportional time scaling
            for cue in overlay_cues:
                scaled_start = max(0.0, cue.start_time * time_scale - PREEMPTIVE_OFFSET)
                scaled_end = min((cue.start_time + cue.duration) * time_scale, final_duration)
                if scaled_start >= final_duration:
                    continue
                style_def = TEXT_STYLES.get(cue.style, TEXT_STYLES["normal"])
                overlay_schedule.append((scaled_start, scaled_end, cue.text, style_def, style_def["font_size"]))

        # Clamp: each overlay ends when next starts (no overlap)
        for i in range(len(overlay_schedule) - 1):
            s_i, e_i, t_i, st_i, sz_i = overlay_schedule[i]
            s_next = overlay_schedule[i + 1][0]
            if e_i > s_next:
                overlay_schedule[i] = (s_i, s_next, t_i, st_i, sz_i)

        # End card (only for non-looped shorts — looped shorts run edge-to-edge)
        if not is_looped:
            overlay_schedule.append((total_duration, final_duration, "Full breakdown ↓",
                                     TEXT_STYLES["normal"], 64))

        # Pre-load fonts for each size
        font_cache = {}
        for _, _, _, style_def, size in overlay_schedule:
            if size not in font_cache and self._font_path:
                try:
                    font_cache[size] = ImageFont.truetype(self._font_path, size)
                except (OSError, IOError):
                    font_cache[size] = ImageFont.load_default()

        # The single make_frame function — renders everything via Pillow
        def make_frame(t):
            # Determine which image to show
            img_idx = min(int(t / duration_per_image), num_images - 1)
            img_array = image_arrays[img_idx]
            local_t = t - img_idx * duration_per_image

            # Ken Burns: crop from pre-scaled image
            direction = "zoom-in" if img_idx % 2 == 0 else "zoom-out"
            progress = local_t / duration_per_image if duration_per_image > 0 else 0
            if direction == "zoom-in":
                scale = 1.0 / (1.0 + zoom * progress)
            else:
                scale = 1.0 / ((1.0 + zoom) - zoom * progress)

            # Pattern interrupt: punch-in at midpoint of each image (1.5s smoothstep arc)
            # 20% zoom, slow entry + slow exit (zero-jerk easing)
            PUNCH_DURATION = 1.5  # seconds (cinematic, unhurried)
            PUNCH_STRENGTH = 0.20  # 20% zoom
            mid_point = duration_per_image / 2
            punch_start = mid_point - PUNCH_DURATION / 2
            punch_end = mid_point + PUNCH_DURATION / 2
            if punch_start <= local_t <= punch_end:
                # Smoothstep easing: 3t² - 2t³ (zero velocity at start and end)
                punch_progress = (local_t - punch_start) / PUNCH_DURATION
                # Smoothstep in, then smoothstep out (symmetric bell)
                if punch_progress < 0.5:
                    t_norm = punch_progress * 2  # 0→1 for first half
                    punch_factor = (3 * t_norm * t_norm - 2 * t_norm * t_norm * t_norm) * PUNCH_STRENGTH
                else:
                    t_norm = (1.0 - punch_progress) * 2  # 1→0 for second half
                    punch_factor = (3 * t_norm * t_norm - 2 * t_norm * t_norm * t_norm) * PUNCH_STRENGTH
                scale *= (1.0 - punch_factor)

            crop_w = min(int(WIDTH / scale), img_w)
            crop_h = min(int(HEIGHT / scale), img_h)
            x = (img_w - crop_w) // 2
            y = (img_h - crop_h) // 2

            cropped = img_array[y:y + crop_h, x:x + crop_w]
            frame_img = Image.fromarray(cropped).resize((WIDTH, HEIGHT), Image.BILINEAR)

            # Draw text overlays active at time t (show only the LAST active one to avoid stacking)
            draw = None
            active_overlay = None
            for start, end, text, style_def, font_size in overlay_schedule:
                if start <= t < end:
                    active_overlay = (text, style_def, font_size, start)

            if active_overlay:
                text, style_def, font_size, start = active_overlay
                draw = ImageDraw.Draw(frame_img)
                font = font_cache.get(font_size)
                color = style_def["color"]
                stroke_color = style_def["stroke_color"]
                stroke_width = style_def["stroke_width"]

                # 200ms fade-in: interpolate alpha from 0→1 over first 0.2s
                FADE_DURATION = 0.2
                elapsed_since_start = t - start
                if elapsed_since_start < FADE_DURATION:
                    alpha = elapsed_since_start / FADE_DURATION
                else:
                    alpha = 1.0

                # Word-wrap text to fit within 85% of frame width
                max_text_width = int(WIDTH * 0.85)
                wrapped_text = _wrap_text(text, font, max_text_width, draw)

                # Measure wrapped text
                if font:
                    bbox = draw.multiline_textbbox((0, 0), wrapped_text, font=font)
                    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                else:
                    tw, th = len(text) * font_size // 2, font_size

                tx = (WIDTH - tw) // 2
                ty = int(HEIGHT * 0.70) - th // 2

                # Hook text position is higher
                if start == 0.0 and font_size >= 100:
                    ty = int(HEIGHT * 0.55) - th // 2

                # End card position is centered
                if start >= total_duration:
                    ty = int(HEIGHT * 0.45) - th // 2

                if font:
                    # Render text with fade-in alpha via composite layer
                    if alpha < 1.0:
                        # Create transparent overlay for fade effect
                        txt_layer = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
                        txt_draw = ImageDraw.Draw(txt_layer)
                        # Convert color tuple to RGBA with alpha
                        rgba_color = color + (int(255 * alpha),)
                        rgba_stroke = stroke_color + (int(255 * alpha),)
                        txt_draw.multiline_text((tx, ty), wrapped_text, font=font, fill=rgba_color,
                                                stroke_width=stroke_width, stroke_fill=rgba_stroke,
                                                align="center")
                        frame_img = Image.alpha_composite(frame_img.convert("RGBA"), txt_layer).convert("RGB")
                    else:
                        draw.multiline_text((tx, ty), wrapped_text, font=font, fill=color,
                                            stroke_width=stroke_width, stroke_fill=stroke_color,
                                            align="center")
                else:
                    draw.text((tx, ty), wrapped_text, fill=color)

            # Draw logo
            if logo_img is not None:
                margin = 20
                logo_pos = (margin, HEIGHT - self.config.logo_size - margin)
                frame_img.paste(logo_img, logo_pos, logo_img)

            return np.array(frame_img)

        # Create the video clip
        video = VideoClip(make_frame, duration=final_duration).with_fps(24)

        # Attach audio
        final_audio = self._build_audio(narration_audio, music_path, final_duration)
        video = video.with_audio(final_audio)

        # Write output
        video.write_videofile(
            str(output_path),
            fps=24,
            codec="libx264",
            audio_codec="aac",
            audio_bitrate="192k",
            logger=None,
            ffmpeg_params=["-crf", "18", "-preset", "veryfast"],
        )

        # Cleanup
        duration = video.duration
        video.close()
        narration_audio.close()

        logger.info(f"Short assembled: {output_path} ({duration:.1f}s)")
        return output_path

    def _match_overlay_to_sentence(self, overlay_text: str, boundaries: list[dict], fallback_time: float = 0.0) -> float | None:
        """Match an overlay text to the sentence it references.
        
        Uses word overlap scoring with ±3s guard rail: finds the sentence with
        the most shared keywords, but only accepts it if within 3s of the
        expected time (prevents wild jumps to wrong sentences).
        
        Returns the start time of the best matching sentence, or None if no match.
        """
        overlay_words = set(overlay_text.lower().replace("$", "").replace("%", "").split())
        # Remove very short words
        overlay_words = {w for w in overlay_words if len(w) > 2}
        
        if not overlay_words:
            return None

        best_score = 0
        best_time = None
        
        for boundary in boundaries:
            sentence_words = set(boundary["text"].lower().replace("$", "").replace("%", "").split())
            overlap = len(overlay_words & sentence_words)
            if overlap > best_score:
                best_score = overlap
                best_time = boundary["start"]
        
        # Require at least 2 word match (prevents false matches on generic words)
        if best_score < 2:
            return None
        
        # ±3s guard rail: only accept if within 3s of expected position
        if best_time is not None and abs(best_time - fallback_time) > 3.0:
            return None
            
        return best_time

    def _build_audio(self, narration_audio, music_path: Path | None, duration: float):
        """Build final audio track: narration + optional background music."""
        from moviepy import AudioFileClip, CompositeAudioClip, concatenate_audioclips
        from moviepy.audio.fx import AudioFadeIn, AudioFadeOut
        import math

        if not music_path or not music_path.exists():
            return narration_audio

        try:
            music = AudioFileClip(str(music_path))

            # Loop by concatenating copies if shorter than video
            if music.duration < duration:
                repeats = math.ceil(duration / music.duration)
                music = concatenate_audioclips([music] * repeats)

            # Trim to exact duration
            music = music.subclipped(0, duration)

            # Apply volume reduction (dB to linear)
            db = self.config.music_volume_db
            volume_factor = 10 ** (db / 20)
            music = music.with_volume_scaled(volume_factor)

            # Apply fades via MoviePy 2.x effects API
            music = music.with_effects([
                AudioFadeIn(self.config.music_fade_in),
                AudioFadeOut(self.config.music_fade_out),
            ])

            # Composite narration + music
            return CompositeAudioClip([narration_audio, music])
        except Exception as e:
            logger.warning(f"Could not mix background music: {e}")
            return narration_audio

    def _resolve_font(self) -> str | None:
        """Find the best available bold font."""
        for path in FONT_FALLBACKS:
            try:
                ImageFont.truetype(path, 48)
                return path
            except (OSError, IOError):
                continue
        return None


def _wrap_text(text: str, font, max_width: int, draw: ImageDraw.ImageDraw) -> str:
    """Word-wrap text to fit within max_width pixels."""
    if font is None:
        return text

    # Check if text already fits
    bbox = draw.textbbox((0, 0), text, font=font)
    if bbox[2] - bbox[0] <= max_width:
        return text

    # Word wrap
    words = text.split()
    lines: list[str] = []
    current_line: list[str] = []

    for word in words:
        test_line = " ".join(current_line + [word])
        bbox = draw.textbbox((0, 0), test_line, font=font)
        if bbox[2] - bbox[0] <= max_width:
            current_line.append(word)
        else:
            if current_line:
                lines.append(" ".join(current_line))
            current_line = [word]

    if current_line:
        lines.append(" ".join(current_line))

    return "\n".join(lines)
