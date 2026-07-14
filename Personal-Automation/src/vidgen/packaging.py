"""Output packaging - organizes final artifacts into upload-ready directory structure."""

import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path

from vidgen.config import PipelineConfig
from vidgen.models import Script, ScenePlan, VideoMetadata

logger = logging.getLogger(__name__)


def package_output(
    job_dir: Path,
    script: Script,
    scene_plan: ScenePlan,
    config: PipelineConfig,
) -> Path:
    """Package all generated artifacts into the final output directory.

    Creates structure: output/{date}_{topic_slug}/
        - video.mp4 (horizontal 1920x1080)
        - thumbnail.png
        - metadata.json
        - shorts/
            - short_1.mp4 + short_1_metadata.json
            - ...

    Args:
        job_dir: The job working directory with generated artifacts.
        script: The parsed script (for metadata generation).
        scene_plan: The scene plan (for topic_slug).
        config: Pipeline config (for output_dir).

    Returns:
        Path to the final output directory.
    """
    date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
    topic_slug = scene_plan.topic_slug or "untitled"
    output_dir = config.output_dir / f"{date_str}_{topic_slug}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Copy video
    video_src = job_dir / "assembly" / "video_raw.mp4"
    video_dst = output_dir / "video.mp4"
    if video_src.exists():
        shutil.copy2(video_src, video_dst)

    # Copy best thumbnail (first variant)
    thumb_dir = job_dir / "thumbnails"
    if thumb_dir.exists():
        thumbs = sorted(thumb_dir.glob("variant_*.png"))
        if thumbs:
            shutil.copy2(thumbs[0], output_dir / "thumbnail.png")

    # Generate metadata
    metadata = generate_metadata(script, scene_plan, video_dst)
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata.model_dump(), indent=2, default=str),
        encoding="utf-8",
    )

    # Copy shorts
    shorts_src = job_dir / "shorts"
    if shorts_src.exists():
        shorts_dst = output_dir / "shorts"
        shorts_dst.mkdir(exist_ok=True)
        for short_file in sorted(shorts_src.glob("*.mp4")):
            shutil.copy2(short_file, shorts_dst / short_file.name)
            # Generate per-short metadata
            short_meta = {
                "title": f"{script.title} - Clip"[:100],
                "description": f"From: {script.title}",
                "tags": _generate_tags(script, scene_plan),
            }
            meta_file = shorts_dst / f"{short_file.stem}_metadata.json"
            meta_file.write_text(json.dumps(short_meta, indent=2), encoding="utf-8")

    logger.info(f"Output packaged to: {output_dir}")
    return output_dir


def generate_metadata(
    script: Script, scene_plan: ScenePlan, video_path: Path
) -> VideoMetadata:
    """Generate YouTube-optimized metadata for the video.

    Args:
        script: The source script.
        scene_plan: The scene plan.
        video_path: Path to the final video file.

    Returns:
        VideoMetadata with title, description, tags, and chapters.
    """
    # Use scene_plan video_title (often more optimized) or fall back to script title
    title = scene_plan.video_title or script.title

    # Title length warning (mobile truncates at ~60 chars)
    if len(title) > 60:
        logger.warning(
            f"Title exceeds 60 chars ({len(title)} chars): '{title}'. "
            f"First 60: '{title[:60]}...' — consider shortening for mobile visibility."
        )

    # Generate chapter timestamps from script sections
    chapters = _generate_chapters(script, scene_plan)

    # Generate description with chapters
    description = _generate_description(script, title, chapters)

    # Generate tags
    tags = _generate_tags(script, scene_plan)

    # Get video duration from scene plan
    duration = scene_plan.total_duration

    return VideoMetadata(
        title=title[:100],
        description=description[:5000],
        tags=tags,
        chapters=chapters,
        duration_seconds=duration,
        resolution="1920x1080",
        file_path=str(video_path),
    )


def _generate_chapters(script: Script, scene_plan: ScenePlan) -> list[tuple[str, str]]:
    """Generate chapter timestamps from script section structure."""
    chapters: list[tuple[str, str]] = []

    # Group scenes by section to get timing
    section_times: dict[str, float] = {}
    for scene in scene_plan.scenes:
        if scene.section_id not in section_times:
            section_times[scene.section_id] = scene.start_time

    # Build chapters from sections
    if script.hook.id in section_times:
        chapters.append((_format_timestamp(section_times[script.hook.id]), script.hook.title if script.hook.title != "Hook" else "The AI Crash is Coming"))
    if script.introduction.id in section_times:
        chapters.append(
            (_format_timestamp(section_times[script.introduction.id]), script.introduction.title if script.introduction.title != "Introduction" else "The Playbook")
        )
    for section in script.body_sections:
        if section.id in section_times:
            # Clean "Section N:" prefix
            import re
            clean_title = re.sub(r"^Section\s+\d+:\s*", "", section.title)
            chapters.append(
                (_format_timestamp(section_times[section.id]), clean_title)
            )
    if script.conclusion.id in section_times:
        chapters.append(
            (_format_timestamp(section_times[script.conclusion.id]), script.conclusion.title if script.conclusion.title != "Conclusion" else "The Survival Framework")
        )

    return chapters


def _generate_description(script: Script, title: str, chapters: list[tuple[str, str]]) -> str:
    """Generate YouTube description matching Token Economy AI format.

    Structure:
    1. Engaging opening question/hook (from script hook, 2-3 sentences)
    2. Value proposition paragraph (what the video covers specifically)
    3. Subscribe CTA with channel link
    4. Social links block
    5. Chapters
    6. Hashtags
    """
    import re

    lines: list[str] = []

    # 1. Opening hook — extract first 2 sentences from hook narration (engaging question style)
    hook_text = re.sub(r"\[SCENE:.*?\]", "", script.hook.narration_text).strip()
    hook_text = re.sub(r"\*\*(.*?)\*\*", r"\1", hook_text)
    hook_text = re.sub(r"\s+", " ", hook_text)
    # Take first 2-3 sentences
    hook_sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', hook_text)
    opening = " ".join(hook_sentences[:3])
    if len(opening) > 400:
        opening = " ".join(hook_sentences[:2])
    lines.append(opening)
    lines.append("")

    # 2. Value proposition — what the viewer will learn (from intro narration)
    intro_text = re.sub(r"\[SCENE:.*?\]", "", script.introduction.narration_text).strip()
    intro_text = re.sub(r"\*\*(.*?)\*\*", r"\1", intro_text)
    intro_text = re.sub(r"\s+", " ", intro_text)
    # Rewrite as "In this video, we..." if not already
    if not intro_text.lower().startswith("in this video"):
        intro_text = f"In this video, {intro_text[0].lower()}{intro_text[1:]}"
    lines.append(intro_text)
    lines.append("")

    # 3. Subscribe CTA
    lines.append(
        "🔔 Subscribe for weekly deep dives into the AI economy, business strategy, "
        "and the future of tech: https://www.youtube.com/@TokenEconomyAI"
    )
    lines.append("")

    # 4. Free resource / lead magnet (always present — update URL when product ready)
    lines.append("📊 Free AI Business Framework: https://tokeneconomyai.substack.com")
    lines.append("")

    # 5. Social links
    lines.append("📱 Follow for daily insights:")
    lines.append("X/Twitter: https://x.com/TokenEconomyAI")
    lines.append("LinkedIn: https://linkedin.com/company/tokeneconomyai")
    lines.append("Newsletter: https://substack.com/@tokeneconomyai")
    lines.append("")

    # 6. Chapters
    if chapters:
        lines.append("⏱️ Chapters:")
        for timestamp, chapter_title in chapters:
            # Clean "Section N:" prefix from chapter names
            clean_title = re.sub(r"^Section\s+\d+:\s*", "", chapter_title)
            lines.append(f"{timestamp} — {clean_title}")
        lines.append("")

    # 6. Hashtags
    lines.append("#ai #artificialintelligence #techbusiness #tokeneconomy")

    return "\n".join(lines)


def _generate_tags(script: Script, scene_plan: ScenePlan) -> list[str]:
    """Generate SEO-optimized tags from script content and topic."""
    # Core channel tags (always present)
    channel_tags = [
        "Token Economy",
        "AI business",
        "artificial intelligence",
        "tech analysis",
    ]

    # Topic-specific tags derived from title keywords (meaningful multi-word phrases)
    title = scene_plan.video_title or script.title
    # Extract meaningful phrases from title (words > 3 chars, skip common filler)
    skip_words = {
        "the", "and", "how", "why", "what", "for", "are", "will", "that",
        "this", "with", "from", "into", "than", "more", "most", "been",
        "being", "their", "them", "they", "have", "does", "every",
    }
    title_words = [
        w.strip("—:()") for w in title.lower().split()
        if len(w.strip("—:()")) > 3 and w.lower().strip("—:()") not in skip_words
    ]
    # Take meaningful title keywords
    topic_tags = list(dict.fromkeys(title_words))[:8]

    # Section-derived tags (from body section titles)
    section_tags = []
    for section in script.body_sections:
        # Extract key phrases from section titles
        words = [
            w.strip("—:()") for w in section.title.lower().split()
            if len(w.strip("—:()")) > 3 and w.lower().strip("—:()") not in skip_words
        ]
        section_tags.extend(words[:2])
    section_tags = list(dict.fromkeys(section_tags))[:6]

    # Broader discovery tags
    broad_tags = [
        "technology",
        "startup",
        "business strategy",
        "future of AI",
        "machine learning",
    ]

    # Combine, deduplicate, cap at 20
    all_tags = channel_tags + topic_tags + section_tags + broad_tags
    seen: set[str] = set()
    unique: list[str] = []
    for tag in all_tags:
        tag_lower = tag.lower()
        if tag_lower not in seen and tag:
            seen.add(tag_lower)
            unique.append(tag)
    return unique[:20]


def _format_timestamp(seconds: float) -> str:
    """Format seconds as M:SS or H:MM:SS timestamp."""
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    if mins >= 60:
        hours = mins // 60
        mins = mins % 60
        return f"{hours}:{mins:02d}:{secs:02d}"
    return f"{mins}:{secs:02d}"
