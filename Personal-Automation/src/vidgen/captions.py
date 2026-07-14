"""SRT caption generator — converts sentence boundary data to SubRip subtitle files.

Reads .boundaries.json files produced by edge-tts during narration and assembles
them into a single .srt file with proper inter-scene timing (including 1s gaps).

Usage:
    from vidgen.captions import generate_srt
    generate_srt(narration_dir=Path("jobs/.../narration"), output_path=Path("output/.../captions.srt"))
"""

import json
import logging
import wave
from pathlib import Path

logger = logging.getLogger(__name__)


def _format_srt_time(seconds: float) -> str:
    """Convert seconds to SRT timestamp format: HH:MM:SS,mmm"""
    if seconds < 0:
        seconds = 0
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _get_wav_duration(wav_path: Path) -> float:
    """Get duration of a WAV file in seconds."""
    try:
        with wave.open(str(wav_path), "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            return frames / rate
    except Exception:
        return 0.0


def generate_srt(
    narration_dir: Path,
    output_path: Path,
    gap_seconds: float = 1.0,
) -> Path | None:
    """Generate an SRT caption file from sentence boundary data.

    Reads all scene_NNN.boundaries.json files in order, accumulates
    timing offsets (including inter-scene gaps), and writes a single
    .srt file suitable for YouTube upload.

    Args:
        narration_dir: Directory containing scene_NNN.wav and scene_NNN.boundaries.json files.
        output_path: Where to write the .srt file.
        gap_seconds: Gap between scenes (must match assembly gap, default 1.0s).

    Returns:
        Path to the generated .srt file, or None if no boundary data found.
    """
    # Find all boundary files in scene order
    boundary_files = sorted(narration_dir.glob("scene_*.boundaries.json"))

    if not boundary_files:
        logger.warning(f"No .boundaries.json files found in {narration_dir}")
        return None

    entries: list[tuple[float, float, str]] = []  # (start, end, text)
    running_offset = 0.0

    for bf in boundary_files:
        # Get corresponding WAV for actual scene duration
        wav_path = bf.with_suffix("").with_suffix(".wav")  # scene_001.boundaries.json → scene_001.wav

        try:
            boundaries = json.loads(bf.read_text())
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Could not read {bf.name}: {e}")
            # Still advance offset by WAV duration if available
            if wav_path.exists():
                running_offset += _get_wav_duration(wav_path) + gap_seconds
            continue

        for b in boundaries:
            text = b.get("text", "").strip()
            start = b.get("start", 0.0)
            duration = b.get("duration", 0.0)

            # Skip pause markers
            if not text or text == "...":
                continue

            abs_start = running_offset + start
            abs_end = running_offset + start + duration

            entries.append((abs_start, abs_end, text))

        # Advance offset by actual WAV duration + gap
        if wav_path.exists():
            scene_duration = _get_wav_duration(wav_path)
        else:
            # Fallback: use last boundary end time
            if boundaries:
                last = boundaries[-1]
                scene_duration = last.get("start", 0) + last.get("duration", 0)
            else:
                scene_duration = 0.0

        running_offset += scene_duration + gap_seconds

    if not entries:
        logger.warning("No caption entries generated")
        return None

    # Write SRT
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    for idx, (start, end, text) in enumerate(entries, 1):
        lines.append(str(idx))
        lines.append(f"{_format_srt_time(start)} --> {_format_srt_time(end)}")
        lines.append(text)
        lines.append("")  # Blank line separator

    output_path.write_text("\n".join(lines), encoding="utf-8")

    logger.info(f"SRT generated: {output_path.name} ({len(entries)} entries, {_format_srt_time(entries[-1][1])} total)")
    return output_path
