"""Upload package generator — creates per-platform metadata files for distribution.

Generates a metadata/ folder with separate files for each platform:
- youtube.md  — Long-form (titles, description, tags, chapters) + Shorts metadata
- twitter.md  — Full X/Twitter thread (copy-paste ready)
- linkedin.md — LinkedIn post with hashtags
- substack.md — Newsletter article
- community.md — 3 community tab posts (2 image + 1 poll)
"""

import logging
import re
from pathlib import Path

from vidgen.distribution import generate_linkedin_post, generate_newsletter, generate_thread
from vidgen.models import Script, ScenePlan
from vidgen.packaging import _generate_chapters, _generate_tags
from vidgen.parsers import parse_script

logger = logging.getLogger(__name__)

# Platform links (constant)
CHANNEL_URL = "https://www.youtube.com/@TokenEconomyAI"
X_URL = "https://x.com/TokenEconomyAI"
LINKEDIN_URL = "https://linkedin.com/company/tokeneconomyai"
SUBSTACK_URL = "https://substack.com/@tokeneconomyai"
LEAD_MAGNET_URL = "https://tokeneconomyai.substack.com"


def generate_upload_package(
    script: Script,
    scene_plan: ScenePlan,
    video_url: str,
    short_scripts: list[Path] | None = None,
    output_path: Path | None = None,
) -> str:
    """Generate a complete upload package as separate per-platform files.

    Creates a metadata/ folder with youtube.md, twitter.md, linkedin.md,
    substack.md, and community.md.

    Args:
        script: Parsed video script.
        scene_plan: Parsed scene plan.
        video_url: YouTube video URL.
        short_scripts: List of Short script .md paths for this video.
        output_path: Directory where the metadata/ folder will be created.
            If it ends in .md (legacy path), the parent dir is used.

    Returns:
        The youtube.md content as a string (for backward compatibility).
    """
    # Resolve output directory — handle legacy single-file paths
    if output_path is not None:
        if output_path.suffix == ".md":
            metadata_dir = output_path.parent / "metadata"
        else:
            metadata_dir = output_path / "metadata" if output_path.name != "metadata" else output_path
    else:
        metadata_dir = Path("output") / "metadata"

    metadata_dir.mkdir(parents=True, exist_ok=True)

    title = scene_plan.video_title or script.title

    # Generate each platform file
    youtube_content = _generate_youtube_md(script, scene_plan, video_url, short_scripts)
    twitter_content = _generate_twitter_md(script, video_url)
    linkedin_content = _generate_linkedin_md(script, video_url)
    substack_content = _generate_substack_md(script, video_url)
    community_content = _generate_community_md(script, video_url)

    # Write all files
    (metadata_dir / "youtube.md").write_text(youtube_content)
    (metadata_dir / "twitter.md").write_text(twitter_content)
    (metadata_dir / "linkedin.md").write_text(linkedin_content)
    (metadata_dir / "substack.md").write_text(substack_content)
    (metadata_dir / "community.md").write_text(community_content)

    logger.info(f"Upload package: {metadata_dir}/")
    logger.info(f"  youtube.md, twitter.md, linkedin.md, substack.md, community.md")

    return youtube_content


# --- YouTube (Long-form + Shorts) ---


def _generate_youtube_md(
    script: Script,
    scene_plan: ScenePlan,
    video_url: str,
    short_scripts: list[Path] | None = None,
) -> str:
    """Generate youtube.md with titles, description, tags, chapters, and Shorts metadata."""
    sections: list[str] = []

    title = scene_plan.video_title or script.title

    # Header
    sections.append(f"# YouTube Upload: {title}\n")
    sections.append(f"**Video URL:** {video_url}")
    sections.append(f"**Category:** Science & Technology\n")

    # --- A/B/C Titles ---
    sections.append("## A/B/C Titles\n")
    titles = _generate_ab_titles(script, scene_plan)
    for label, t in titles:
        char_count = len(t)
        warning = " ⚠️ >60 chars" if char_count > 60 else ""
        sections.append(f"**{label}:** {t} ({char_count} chars){warning}")
    sections.append("")

    # --- Description ---
    sections.append("## Description\n")
    sections.append("```")
    desc = _generate_full_description(script, scene_plan, video_url)
    sections.append(desc)
    sections.append("```\n")

    # --- Tags ---
    sections.append("## Tags\n")
    tags = _generate_tags(script, scene_plan)
    sections.append("```")
    sections.append(", ".join(tags))
    sections.append("```\n")

    # --- Chapters ---
    sections.append("## Chapters\n")
    chapters = _generate_chapters(script, scene_plan)
    if chapters:
        for timestamp, chapter_title in chapters:
            clean_title = re.sub(r"^Section\s+\d+:\s*", "", chapter_title)
            sections.append(f"- {timestamp} — {clean_title}")
    else:
        sections.append("*No chapters generated (missing scene durations)*")
    sections.append("")

    # --- Shorts Metadata ---
    if short_scripts:
        sections.append("---\n")
        sections.append("## Shorts Metadata\n")
        for i, short_path in enumerate(short_scripts, 1):
            short_meta = _generate_short_metadata(short_path, video_url, i)
            sections.append(short_meta)

    return "\n".join(sections)


# --- Twitter/X Thread ---


def _generate_twitter_md(script: Script, video_url: str) -> str:
    """Generate twitter.md with the full thread, copy-paste ready."""
    sections: list[str] = []

    sections.append("# X/Twitter Thread\n")
    sections.append("Copy-paste each tweet. Thread connector 🧵 is included.\n")

    thread = generate_thread(script, video_url=video_url)
    sections.append(thread)

    return "\n".join(sections)


# --- LinkedIn ---


def _generate_linkedin_md(script: Script, video_url: str) -> str:
    """Generate linkedin.md with post text and hashtags."""
    sections: list[str] = []

    sections.append("# LinkedIn Post\n")
    sections.append("Copy-paste into LinkedIn company page post.\n")

    linkedin = generate_linkedin_post(script, video_url=video_url)
    sections.append(linkedin)

    return "\n".join(sections)


# --- Substack Newsletter ---


def _generate_substack_md(script: Script, video_url: str) -> str:
    """Generate substack.md with the full newsletter article + Notes teaser."""
    sections: list[str] = []

    sections.append("# Substack Newsletter\n")
    sections.append(f"**Video embed:** {video_url}\n")

    # Substack Notes teaser (post Tuesday morning before video drops)
    sections.append("---\n")
    sections.append("## Substack Notes Teaser (Post Tuesday morning)\n")
    hook_text = re.sub(r"\[SCENE:.*?\]", "", script.hook.narration_text).strip()
    hook_text = re.sub(r"\*\*(.*?)\*\*", r"\1", hook_text)
    hook_sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', hook_text)
    teaser = " ".join(hook_sentences[:2])
    if len(teaser) > 200:
        teaser = teaser[:200].rsplit(" ", 1)[0] + "..."
    sections.append(f"```\n{teaser}\n\nFull analysis drops Thursday. Subscribe to not miss it.\n```\n")

    sections.append("---\n")

    newsletter = generate_newsletter(script, video_url=video_url)
    sections.append(newsletter)

    return "\n".join(sections)


# --- Community Tab ---


def _generate_community_md(script: Script, video_url: str) -> str:
    """Generate community.md with 3 posts: image + poll + image."""
    sections: list[str] = []

    sections.append("# Community Tab Posts\n")
    sections.append("Schedule: Post 1 (Day +1), Post 2 (Day +3), Post 3 (Day +5)\n")

    # Post 1: Image + engagement hook (Day +1)
    sections.append("## Post 1 — Image Post (Day +1)\n")
    sections.append("**Image:** `community/post_1.png`\n")
    post1_text = _generate_image_post_text(script, video_url, variant="hook")
    sections.append(f"**Text ({len(post1_text)} chars):**")
    sections.append(f"```\n{post1_text}\n```\n")

    # Post 2: Poll (Day +3)
    sections.append("## Post 2 — Poll (Day +3)\n")
    poll_question, poll_options = _generate_poll(script)
    sections.append(f"**Question:** {poll_question}\n")
    sections.append("**Options:**")
    for i, option in enumerate(poll_options, 1):
        sections.append(f"{i}. {option}")
    sections.append("")
    teaser = _generate_poll_teaser(script, video_url)
    sections.append(f"**Teaser text ({len(teaser)} chars):**")
    sections.append(f"```\n{teaser}\n```\n")

    # Post 3: Image + data-stat hook (Day +5)
    sections.append("## Post 3 — Image Post (Day +5)\n")
    sections.append("**Image:** `community/post_2.png`\n")
    post3_text = _generate_image_post_text(script, video_url, variant="stat")
    sections.append(f"**Text ({len(post3_text)} chars):**")
    sections.append(f"```\n{post3_text}\n```\n")

    return "\n".join(sections)


# --- Shared Generators ---


def _generate_ab_titles(script: Script, scene_plan: ScenePlan) -> list[tuple[str, str]]:
    """Generate 3 title variants: Threat, Opportunity, Curiosity."""
    base_title = scene_plan.video_title or script.title

    # Extract key elements from title for reframing
    short = re.split(r'[—\-:]', base_title)[0].strip()
    words = short.split()

    # Variant A: Threat (negative framing, urgency)
    threat_title = base_title[:60] if len(base_title) <= 60 else short[:60]

    # Variant B: Opportunity (positive framing)
    alt_text = getattr(scene_plan, 'thumbnail_text_alt', None) or ''
    if alt_text:
        opp_title = f"{short}: {alt_text.title()}"[:60]
    else:
        opp_title = f"How to Profit from {' '.join(words[:4])}"[:60]

    # Variant C: Curiosity (question/reveal framing)
    curiosity_title = f"What {' '.join(words[1:4])} Means for Your Business in 2026"[:60]

    return [
        ("A (Threat)", threat_title),
        ("B (Opportunity)", opp_title),
        ("C (Curiosity)", curiosity_title),
    ]


def _generate_full_description(
    script: Script, scene_plan: ScenePlan, video_url: str
) -> str:
    """Generate the full YouTube description with all cross-platform links."""
    lines: list[str] = []

    # 1. Hook paragraph (2-3 sentences)
    hook_text = re.sub(r"\[SCENE:.*?\]", "", script.hook.narration_text).strip()
    hook_text = re.sub(r"\*\*(.*?)\*\*", r"\1", hook_text)
    hook_text = re.sub(r"\s+", " ", hook_text)
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', hook_text)
    opening = " ".join(sentences[:3])[:400]
    lines.append(opening)
    lines.append("")

    # 2. Value proposition
    intro_text = re.sub(r"\[SCENE:.*?\]", "", script.introduction.narration_text).strip()
    intro_text = re.sub(r"\*\*(.*?)\*\*", r"\1", intro_text)
    intro_text = re.sub(r"\s+", " ", intro_text)
    if not intro_text.lower().startswith("in this video"):
        intro_text = f"In this video, {intro_text[0].lower()}{intro_text[1:]}"
    lines.append(intro_text)
    lines.append("")

    # 3. Lead magnet
    lines.append(f"📊 Free AI Business Framework: {LEAD_MAGNET_URL}")
    lines.append("")

    # 4. Subscribe CTA
    lines.append(
        f"🔔 Subscribe for weekly deep dives into the AI economy, "
        f"business strategy, and the future of tech: {CHANNEL_URL}"
    )
    lines.append("")

    # 5. Social links
    lines.append("📱 Follow for daily insights:")
    lines.append(f"X/Twitter: {X_URL}")
    lines.append(f"LinkedIn: {LINKEDIN_URL}")
    lines.append(f"Newsletter: {SUBSTACK_URL}")
    lines.append("")

    # 6. Chapters
    chapters = _generate_chapters(script, scene_plan)
    if chapters:
        lines.append("⏱️ Chapters:")
        for timestamp, chapter_title in chapters:
            clean_title = re.sub(r"^Section\s+\d+:\s*", "", chapter_title)
            lines.append(f"{timestamp} — {clean_title}")
        lines.append("")

    # 7. Hashtags
    lines.append("#ai #artificialintelligence #techbusiness #tokeneconomy")

    return "\n".join(lines)


def _generate_short_metadata(short_path: Path, parent_video_url: str, index: int) -> str:
    """Generate copy-paste metadata for a single Short."""
    from vidgen.shorts_parser import parse_short_script

    try:
        script = parse_short_script(short_path)
    except Exception:
        return f"### Short {index}: {short_path.name}\n**ERROR:** Could not parse\n"

    lines: list[str] = []
    lines.append(f"### Short {index}: {short_path.stem}.mp4")
    lines.append(f"**File:** `output/shorts/{short_path.stem}.mp4`\n")

    # Title (under 100 chars with hashtags)
    title = f"{script.title} #ai #shorts"
    lines.append(f"**Title:** {title}\n")

    # Description
    lines.append("**Description:**")
    lines.append("```")
    first_sentence = script.narration_text.split('.')[0] + '.'
    if len(first_sentence) > 100:
        first_sentence = ' '.join(first_sentence.split()[:12]) + '...'
    lines.append(first_sentence)
    lines.append("")
    lines.append(f"Full breakdown: {parent_video_url}")
    lines.append("")
    lines.append(f"Subscribe @TokenEconomyAI")
    lines.append(f"{X_URL}")
    lines.append(f"{SUBSTACK_URL}")
    lines.append("```\n")

    # Related video pointer with YouTube Studio instructions
    lines.append(f"**⚡ Related Video:** {parent_video_url}")
    lines.append("*YouTube Studio → Content → select Short → Details → Show More → Related video*\n")
    lines.append("---\n")

    return "\n".join(lines)


# --- Community Helpers ---


def _generate_image_post_text(script: Script, video_url: str, variant: str = "hook") -> str:
    """Generate ≤280 char post text for a Community image post.

    variant="hook" — engagement hook from the script opening.
    variant="stat" — data/stat hook from body sections.
    """
    if variant == "stat":
        # Find a compelling stat from body sections
        for section in script.body_sections:
            text = re.sub(r"\[SCENE:.*?\]", "", section.narration_text).strip()
            text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
            sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
            for sent in sentences:
                if re.search(r'\d+\s*%|\$\d+|\d+\s*billion|\d+\s*million', sent, re.IGNORECASE):
                    post = f"{sent}\n\nNew analysis just dropped 👇\n{video_url}"
                    return post[:280]
        # Fallback: use first body sentence
        text = re.sub(r"\[SCENE:.*?\]", "", script.body_sections[0].narration_text).strip()
        text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
        fallback = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)[0]
        post = f"{fallback}\n\nFull breakdown in our latest video 👇\n{video_url}"
        return post[:280]
    else:
        # Hook variant — use the script opening
        hook_text = re.sub(r"\[SCENE:.*?\]", "", script.hook.narration_text).strip()
        hook_text = re.sub(r"\*\*(.*?)\*\*", r"\1", hook_text)
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', hook_text)
        first = sentences[0] if sentences else hook_text[:200]
        post = f"{first}\n\nDo you agree? New analysis just dropped 👇\n{video_url}"
        return post[:280]


def _generate_poll(script: Script) -> tuple[str, list[str]]:
    """Generate a poll question + 4 options from body section titles.

    Returns:
        Tuple of (question, [option1, option2, option3, option4])
    """
    # Build poll question from the video topic
    title = script.title
    short_title = re.split(r'[—\-:(\[]', title)[0].strip()
    question = f"Which factor matters most for {short_title.lower()}?"

    # Options from body section titles (up to 4)
    options: list[str] = []
    for section in script.body_sections[:4]:
        # Clean section title
        clean = re.sub(r"^Section\s+\d+:\s*", "", section.title)
        clean = re.split(r'[—\-:]', clean)[0].strip()
        if len(clean) <= 60:
            options.append(clean)

    # Pad to 4 if needed
    fallback_options = ["Technology", "Market timing", "Team execution", "Funding"]
    while len(options) < 4:
        options.append(fallback_options[len(options)])

    return question[:280], options[:4]


def _generate_poll_teaser(script: Script, video_url: str) -> str:
    """Generate ≤280 char teaser text to accompany the poll."""
    title = script.title
    short_title = re.split(r'[—\-:(\[]', title)[0].strip()
    teaser = f"We broke down the data on {short_title.lower()}. The results surprised us.\n\nVote, then watch the full analysis 👇\n{video_url}"
    return teaser[:280]
