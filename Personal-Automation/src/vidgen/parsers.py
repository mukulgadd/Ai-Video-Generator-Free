"""Parsers for script.md and scene_plan.json input files.

Handles parsing structured markdown scripts with YAML frontmatter and
JSON scene plans into validated Pydantic model instances.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import yaml

from vidgen.models import Scene, ScenePlan, Script, ScriptSection, TextOverlay


class ParseError(Exception):
    """Raised when an input file cannot be parsed into the expected format."""


# --- Script Parsing ---

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_SECTION_HEADER_RE = re.compile(r"^##\s+(.+)$", re.MULTILINE)
_SCENE_MARKER_RE = re.compile(r"^\[SCENE:\s*(.*?)\]\s*$", re.MULTILINE)
_EMPHASIS_RE = re.compile(r"\*\*(.+?)\*\*")


def _derive_section_id(title: str) -> str:
    """Derive a section ID from a header title.

    Examples:
        "Hook" -> "hook"
        "Introduction" -> "introduction"
        "Section 1: The Dashboard Fatigue Problem" -> "section_1"
        "Section 2: How AI Agents Change the Equation" -> "section_2"
        "Conclusion" -> "conclusion"
    """
    lower = title.strip().lower()
    if lower == "hook":
        return "hook"
    if lower == "introduction":
        return "introduction"
    if lower == "conclusion":
        return "conclusion"
    # Match "Section N: ..." pattern
    section_match = re.match(r"section\s+(\d+)", lower)
    if section_match:
        return f"section_{section_match.group(1)}"
    # Fallback: slugify the title
    slug = re.sub(r"[^a-z0-9]+", "_", lower).strip("_")
    return slug


def _extract_emphasis_markers(text: str) -> list[tuple[int, int]]:
    """Extract emphasis marker positions from text after removing markdown formatting.

    Returns positions (start, end) of emphasized text in the cleaned (no markdown) version.
    """
    markers: list[tuple[int, int]] = []
    # We need to compute positions in the *cleaned* text (with ** removed)
    clean_parts: list[str] = []
    last_end = 0
    for match in _EMPHASIS_RE.finditer(text):
        # Text before this match
        clean_parts.append(text[last_end : match.start()])
        start_pos = sum(len(p) for p in clean_parts)
        emphasized_text = match.group(1)
        end_pos = start_pos + len(emphasized_text)
        markers.append((start_pos, end_pos))
        clean_parts.append(emphasized_text)
        last_end = match.end()
    return markers


def _clean_emphasis(text: str) -> str:
    """Remove markdown bold markers from text."""
    return _EMPHASIS_RE.sub(r"\1", text)


def _parse_sections(body: str) -> list[tuple[str, str]]:
    """Split markdown body into (header, content) tuples by ## headers."""
    splits = _SECTION_HEADER_RE.split(body)
    # splits[0] is text before first header (usually empty), then alternating title/content
    sections: list[tuple[str, str]] = []
    # Start from index 1 (first header)
    for i in range(1, len(splits), 2):
        title = splits[i].strip()
        content = splits[i + 1] if i + 1 < len(splits) else ""
        sections.append((title, content.strip()))
    return sections


def parse_script(path: Path) -> Script:
    """Parse a script.md file into a Script object.

    Raises ParseError if the file is empty, missing frontmatter, or has
    an invalid structure.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, IOError) as e:
        raise ParseError(f"Cannot read script file: {e}") from e

    if not text.strip():
        raise ParseError("Script file is empty")

    # Parse YAML frontmatter
    fm_match = _FRONTMATTER_RE.match(text)
    if not fm_match:
        raise ParseError("Script file is missing YAML frontmatter (expected --- delimiters)")

    try:
        frontmatter = yaml.safe_load(fm_match.group(1))
    except yaml.YAMLError as e:
        raise ParseError(f"Invalid YAML frontmatter: {e}") from e

    if not isinstance(frontmatter, dict):
        raise ParseError("YAML frontmatter must be a mapping")

    title = frontmatter.get("title", "")
    if not title:
        raise ParseError("Script frontmatter missing 'title' field")

    # Parse body sections
    body = text[fm_match.end() :]
    raw_sections = _parse_sections(body)

    if not raw_sections:
        raise ParseError("Script has no sections (expected ## headers)")

    # Build ScriptSection objects
    hook: ScriptSection | None = None
    introduction: ScriptSection | None = None
    conclusion: ScriptSection | None = None
    body_sections: list[ScriptSection] = []

    for section_title, content in raw_sections:
        section_id = _derive_section_id(section_title)

        # Extract scene marker (first one for metadata)
        scene_match = _SCENE_MARKER_RE.search(content)
        scene_marker = scene_match.group(1) if scene_match else ""

        # Narration text: remove ALL scene marker lines (not just the first)
        narration_text = _SCENE_MARKER_RE.sub("", content).strip()

        # Extract emphasis markers BEFORE removing bold syntax
        emphasis_markers = _extract_emphasis_markers(narration_text)

        # Remove markdown bold markers (would be read aloud by TTS)
        narration_text = _clean_emphasis(narration_text)

        section = ScriptSection(
            id=section_id,
            title=section_title,
            narration_text=narration_text,
            scene_marker=scene_marker,
            emphasis_markers=emphasis_markers,
        )

        if section_id == "hook":
            hook = section
        elif section_id == "introduction":
            introduction = section
        elif section_id == "conclusion":
            conclusion = section
        else:
            body_sections.append(section)

    if hook is None:
        raise ParseError("Script missing required 'Hook' section")
    if introduction is None:
        raise ParseError("Script missing required 'Introduction' section")
    if conclusion is None:
        raise ParseError("Script missing required 'Conclusion' section")
    if not body_sections:
        raise ParseError("Script must have at least one body section")

    # Compute total word count from all narration text (cleaned of markdown)
    all_narration = " ".join(
        _clean_emphasis(s.narration_text)
        for s in [hook, introduction, *body_sections, conclusion]
    )
    total_word_count = len(all_narration.split())

    return Script(
        title=title,
        hook=hook,
        introduction=introduction,
        body_sections=body_sections,
        conclusion=conclusion,
        total_word_count=total_word_count,
    )


# --- Script Serialization ---


def serialize_script(script: Script) -> str:
    """Serialize a Script object back to markdown format with YAML frontmatter.

    Produces output that round-trips cleanly: parse_script(serialize_script(s)) == s
    """
    lines: list[str] = []

    # YAML frontmatter
    lines.append("---")
    lines.append(f'title: "{script.title}"')
    lines.append("---")
    lines.append("")

    # Sections in order
    all_sections = [
        script.hook,
        script.introduction,
        *script.body_sections,
        script.conclusion,
    ]

    for section in all_sections:
        lines.append(f"## {section.title}")
        lines.append("")
        if section.scene_marker:
            lines.append(f"[SCENE: {section.scene_marker}]")
            lines.append("")
        lines.append(section.narration_text)
        lines.append("")

    return "\n".join(lines)


# --- Scene Plan Parsing ---


def parse_scene_plan(path: Path) -> ScenePlan:
    """Parse a scene_plan.json file into a ScenePlan object.

    Raises ParseError if the file is empty, not valid JSON, or doesn't
    conform to the expected schema.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, IOError) as e:
        raise ParseError(f"Cannot read scene plan file: {e}") from e

    if not text.strip():
        raise ParseError("Scene plan file is empty")

    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise ParseError(f"Invalid JSON in scene plan: {e}") from e

    if not isinstance(data, dict):
        raise ParseError("Scene plan JSON must be an object")

    try:
        scenes: list[Scene] = []
        style_prefix = data.get("style_prefix", "")

        for scene_data in data.get("scenes", []):
            # Map JSON field names to model field names
            # Support both text_overlay (single, legacy) and text_overlays (list, new)
            text_overlay_data = scene_data.get("text_overlay")
            text_overlay: TextOverlay | None = None
            if text_overlay_data is not None:
                text_overlay = TextOverlay(
                    text=text_overlay_data["text"],
                    position=text_overlay_data["position"],
                    appear_at=text_overlay_data["appear_at"],
                    duration=text_overlay_data["duration"],
                )

            # Parse text_overlays list (multiple overlays per scene)
            text_overlays_data = scene_data.get("text_overlays", [])
            text_overlays: list[TextOverlay] = []
            for ovl_data in text_overlays_data:
                text_overlays.append(TextOverlay(
                    text=ovl_data["text"],
                    position=ovl_data.get("position", "bottom"),
                    appear_at=ovl_data["appear_at"],
                    duration=ovl_data.get("duration", 5.0),
                ))

            scene = Scene(
                id=scene_data["id"],
                section_id=scene_data["section_id"],
                image_prompt=scene_data["image_prompt"],
                image_prompts=scene_data.get("image_prompts", []),
                style_prefix=style_prefix,
                ken_burns_direction=scene_data["ken_burns"],
                start_time=scene_data["start_time"],
                duration=scene_data["duration"],
                transition_type=scene_data["transition"],
                text_overlay=text_overlay,
                text_overlays=text_overlays,
                music_mood=scene_data.get("music_mood"),
                visual_type=scene_data.get("visual_type", "scene"),
                data_overlay=scene_data.get("data_overlay"),
            )
            scenes.append(scene)

        scene_plan = ScenePlan(
            video_title=data.get("video_title", ""),
            topic_slug=data.get("topic_slug", ""),
            style_prefix=style_prefix,
            scenes=scenes,
            total_duration=data.get("total_duration_seconds", 0.0),
            thumbnail_text=data.get("thumbnail_text"),
            thumbnail_accent_word=data.get("thumbnail_accent_word"),
            thumbnail_accent_color=data.get("thumbnail_accent_color"),
        )
    except (KeyError, TypeError, ValueError) as e:
        raise ParseError(f"Scene plan structure error: {e}") from e

    return scene_plan


# --- Scene Plan Serialization ---


def serialize_scene_plan(plan: ScenePlan) -> str:
    """Serialize a ScenePlan object back to JSON format.

    Produces output that round-trips cleanly: parse_scene_plan(serialize_scene_plan(p)) == p
    """
    scenes_data: list[dict] = []
    for scene in plan.scenes:
        scene_dict: dict = {
            "id": scene.id,
            "section_id": scene.section_id,
            "image_prompt": scene.image_prompt,
            "ken_burns": scene.ken_burns_direction,
            "start_time": scene.start_time,
            "duration": scene.duration,
            "transition": scene.transition_type,
        }
        # Serialize text_overlays (list, preferred) or text_overlay (single, legacy)
        # Use singular form when exactly 1 overlay (round-trip compat with existing plans)
        if scene.text_overlays and len(scene.text_overlays) > 1:
            scene_dict["text_overlays"] = [
                {
                    "text": ovl.text,
                    "position": ovl.position,
                    "appear_at": ovl.appear_at,
                    "duration": ovl.duration,
                }
                for ovl in scene.text_overlays
            ]
        elif scene.text_overlays:
            # Single overlay — write as text_overlay for backward compat
            ovl = scene.text_overlays[0]
            scene_dict["text_overlay"] = {
                "text": ovl.text,
                "position": ovl.position,
                "appear_at": ovl.appear_at,
                "duration": ovl.duration,
            }
        elif scene.text_overlay is not None:
            scene_dict["text_overlay"] = {
                "text": scene.text_overlay.text,
                "position": scene.text_overlay.position,
                "appear_at": scene.text_overlay.appear_at,
                "duration": scene.text_overlay.duration,
            }
        scenes_data.append(scene_dict)

    data = {
        "video_title": plan.video_title,
        "topic_slug": plan.topic_slug,
        "style_prefix": plan.style_prefix,
        "total_duration_seconds": plan.total_duration,
        "scenes": scenes_data,
    }

    # Include thumbnail hints if set
    if plan.thumbnail_text:
        data["thumbnail_text"] = plan.thumbnail_text
    if plan.thumbnail_accent_word:
        data["thumbnail_accent_word"] = plan.thumbnail_accent_word
    if plan.thumbnail_accent_color:
        data["thumbnail_accent_color"] = plan.thumbnail_accent_color

    return json.dumps(data, indent=2)


# --- Alignment Validation ---


def validate_script_scene_alignment(script: Script, plan: ScenePlan) -> list[str]:
    """Check that all scenes in the plan reference valid script section IDs.

    Returns a list of error strings. An empty list means alignment is valid.
    """
    # Collect valid section IDs from the script
    valid_ids = {script.hook.id, script.introduction.id, script.conclusion.id}
    for section in script.body_sections:
        valid_ids.add(section.id)

    errors: list[str] = []
    for scene in plan.scenes:
        if scene.section_id not in valid_ids:
            errors.append(
                f"Scene '{scene.id}' references section_id '{scene.section_id}' "
                f"which does not exist in the script. "
                f"Valid section IDs: {sorted(valid_ids)}"
            )

    return errors
