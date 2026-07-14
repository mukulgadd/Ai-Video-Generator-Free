"""Parser for YouTube Shorts script markdown files.

Parses markdown with YAML frontmatter into ShortScript models.
Format: YAML front matter + ## Narration + ## Text Overlays + ## Images sections.
"""

import logging
import re
from pathlib import Path

import yaml

from vidgen.models import ImageCue, OverlayCue, ShortScript

logger = logging.getLogger(__name__)


class ShortParseError(Exception):
    """Raised when a Short script cannot be parsed."""
    pass


def parse_short_script(path: Path) -> ShortScript:
    """Parse a Short script markdown file into a ShortScript model.

    Expected format:
    ---
    title: "..."
    duration_target: 35
    source_video: "003"
    music_mood: "suspense"
    style_prefix: "..."
    hook_text: "..."
    ---
    ## Narration
    [narration text]

    ## Text Overlays
    | time | text | style |
    |------|------|-------|
    | 0 | Hook text | impact |
    ...

    ## Images
    - prompt 1
    - prompt 2
    ...

    Args:
        path: Path to the .md file.

    Returns:
        Validated ShortScript model.

    Raises:
        ShortParseError: If format is invalid or required fields are missing.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, IOError) as e:
        raise ShortParseError(f"Cannot read Short script file: {e}") from e

    if not text.strip():
        raise ShortParseError("Short script file is empty")

    # Split YAML frontmatter from body
    frontmatter, body = _split_frontmatter(text)

    # Parse YAML
    try:
        meta = yaml.safe_load(frontmatter)
    except yaml.YAMLError as e:
        raise ShortParseError(f"Invalid YAML frontmatter: {e}") from e

    if not isinstance(meta, dict):
        raise ShortParseError("YAML frontmatter must be a mapping")

    # Parse sections
    narration = _parse_narration_section(body)
    overlay_cues = _parse_overlay_section(body)
    image_cues = _parse_images_section(body, meta.get("duration_target", 35))

    # Clean narration text
    narration = _clean_narration(narration)

    # Build ShortScript
    try:
        script = ShortScript(
            title=meta.get("title", ""),
            duration_target=meta.get("duration_target", 35),
            source_video=meta.get("source_video", ""),
            music_mood=meta.get("music_mood", "neutral"),
            music_track=meta.get("music_track"),
            style_prefix=meta.get("style_prefix", ""),
            hook_text=meta.get("hook_text", ""),
            narration_text=narration,
            overlay_cues=overlay_cues,
            image_cues=image_cues,
        )
    except Exception as e:
        raise ShortParseError(f"Short script validation failed: {e}") from e

    return script


def validate_short_script(script: ShortScript) -> list[str]:
    """Validate timing and consistency of a parsed ShortScript.

    Returns list of error strings. Empty list = valid.
    """
    errors: list[str] = []

    # Check overlay timing doesn't exceed duration
    for cue in script.overlay_cues:
        if cue.start_time + cue.duration > script.duration_target + 5:
            errors.append(
                f"Overlay '{cue.text}' ends at {cue.start_time + cue.duration}s "
                f"but duration target is {script.duration_target}s"
            )

    # Check image timing doesn't exceed duration
    total_image_time = sum(c.duration for c in script.image_cues)
    if total_image_time < script.duration_target * 0.7:
        errors.append(
            f"Total image time ({total_image_time:.1f}s) covers less than 70% "
            f"of target duration ({script.duration_target}s)"
        )

    # Check word count vs duration target
    word_count = len(script.narration_text.split())
    expected_duration = word_count / 150 * 60  # at 150 WPM
    if expected_duration > script.duration_target + 10:
        errors.append(
            f"Narration ({word_count} words, ~{expected_duration:.0f}s) "
            f"likely exceeds duration target ({script.duration_target}s)"
        )

    return errors


def _split_frontmatter(text: str) -> tuple[str, str]:
    """Split YAML frontmatter from markdown body."""
    # Match --- delimited frontmatter
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)", text, re.DOTALL)
    if not match:
        raise ShortParseError(
            "No YAML frontmatter found. File must start with '---'"
        )
    return match.group(1), match.group(2)


def _parse_narration_section(body: str) -> str:
    """Extract narration text from ## Narration section."""
    match = re.search(
        r"##\s*Narration\s*\n(.*?)(?=\n##|\Z)", body, re.DOTALL | re.IGNORECASE
    )
    if not match:
        raise ShortParseError("Missing '## Narration' section")
    return match.group(1).strip()


def _parse_overlay_section(body: str) -> list[OverlayCue]:
    """Extract overlay cues from ## Text Overlays table."""
    match = re.search(
        r"##\s*Text Overlays\s*\n(.*?)(?=\n##|\Z)", body, re.DOTALL | re.IGNORECASE
    )
    if not match:
        return []  # Optional section

    table_text = match.group(1).strip()
    cues: list[OverlayCue] = []

    for line in table_text.split("\n"):
        line = line.strip()
        # Skip header rows and separator
        if not line or line.startswith("|--") or line.startswith("| time"):
            continue
        if "|" not in line:
            continue

        parts = [p.strip() for p in line.split("|") if p.strip()]
        if len(parts) < 2:
            continue

        try:
            time_val = float(parts[0].replace(":", ""))
            text_val = parts[1]
            style_val = parts[2] if len(parts) > 2 else "normal"

            cues.append(OverlayCue(
                text=text_val,
                start_time=time_val,
                duration=4.0,
                style=style_val,
            ))
        except (ValueError, IndexError):
            continue

    # Clamp durations so each overlay disappears before the next one starts
    for i in range(len(cues) - 1):
        gap = cues[i + 1].start_time - cues[i].start_time
        if gap > 0 and cues[i].duration > gap:
            cues[i] = OverlayCue(
                text=cues[i].text,
                start_time=cues[i].start_time,
                duration=gap,
                style=cues[i].style,
            )

    return cues


def _parse_images_section(body: str, duration_target: int) -> list[ImageCue]:
    """Extract image prompts from ## Images bullet list."""
    match = re.search(
        r"##\s*Images\s*\n(.*?)(?=\n##|\Z)", body, re.DOTALL | re.IGNORECASE
    )
    if not match:
        raise ShortParseError("Missing '## Images' section")

    images_text = match.group(1).strip()
    prompts: list[str] = []

    for line in images_text.split("\n"):
        line = line.strip()
        if line.startswith("- ") or line.startswith("* "):
            prompts.append(line[2:].strip())

    if len(prompts) < 2:
        raise ShortParseError(
            f"Need at least 2 image prompts in ## Images, got {len(prompts)}"
        )

    # Auto-assign timing based on even distribution
    duration_per_image = duration_target / len(prompts)
    cues: list[ImageCue] = []
    for i, prompt in enumerate(prompts):
        # Check for data overlay syntax: [DATA:type|key=value|key=value] prompt
        visual_type = "scene"
        data_overlay = None
        data_match = re.match(r"\[DATA:([^\]]+)\]\s*(.*)", prompt)
        if data_match:
            visual_type = "data"
            data_parts = data_match.group(1).split("|")
            data_type = data_parts[0].strip()
            data_overlay = {"type": data_type}
            for part in data_parts[1:]:
                if "=" in part:
                    key, val = part.split("=", 1)
                    key = key.strip()
                    val = val.strip()
                    # Parse lists (comma-separated)
                    if "," in val:
                        data_overlay[key] = [v.strip() for v in val.split(",")]
                    else:
                        data_overlay[key] = val
            prompt = data_match.group(2).strip() or "dark tech interface with blurred dashboard elements, vertical composition, no text, no numbers, no readable labels"

        cues.append(ImageCue(
            prompt=prompt,
            start_time=i * duration_per_image,
            duration=duration_per_image,
            visual_type=visual_type,
            data_overlay=data_overlay,
        ))

    return cues


def _clean_narration(text: str) -> str:
    """Strip markdown formatting and scene markers from narration text."""
    # Remove bold markers
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    # Remove scene markers [SCENE: ...]
    text = re.sub(r"\[SCENE:.*?\]", "", text)
    # Remove markdown links
    text = re.sub(r"\[(.*?)\]\(.*?\)", r"\1", text)
    # Collapse multiple whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text
