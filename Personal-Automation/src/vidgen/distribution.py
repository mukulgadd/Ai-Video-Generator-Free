"""Content distribution generators — X threads, Substack newsletters, LinkedIn posts.

Transforms video scripts into platform-specific content formats for
multi-platform distribution. All generators work from the parsed Script model.
"""

import re
import textwrap
from pathlib import Path

from vidgen.models import Script


# --- X/Twitter Thread Generator ---


def generate_thread(script: Script, video_url: str = "") -> str:
    """Generate a 6-tweet X/Twitter thread optimized for 2026 algorithm.

    2026 X algo context:
    - Bookmarks are worth 10x a like in distribution
    - Replies are worth 13.5x a like
    - Threads generate 3x more engagement than single tweets
    - Tweet 2 should be "save-worthy" (data list, framework, tool stack)

    Structure:
    1. Hook tweet — attention-grabbing stat (from script hook)
    2. The Bookmark Tweet — save-worthy data list/framework from body
    3-5. Key insight tweets — one per remaining body section
    6. CTA tweet — link + subscribe

    Each tweet stays under 280 characters. Thread uses 🧵 connector.

    Args:
        script: Parsed Script model.
        video_url: Optional YouTube video URL to include in CTA.

    Returns:
        Formatted thread as a single string with tweet separators.
    """
    tweets: list[str] = []

    # Tweet 1: Hook (strongest opening line from hook section)
    hook_text = _extract_hook_line(script.hook.narration_text)
    tweets.append(hook_text + "\n\n🧵 Thread:")

    # Tweet 2: THE BOOKMARK TWEET — save-worthy list/framework
    bookmark_tweet = _build_bookmark_tweet(script)
    tweets.append(bookmark_tweet)

    # Tweets 3-5: One key insight per remaining body section
    start_idx = 1  # Skip first body section (used for bookmark tweet data)
    for section in script.body_sections[start_idx:start_idx + 3]:
        insight = _extract_section_insight(section.narration_text, section.title)
        if insight:
            tweets.append(insight)

    # Tweet 6: CTA
    cta = _build_cta_tweet(script.title, video_url)
    tweets.append(cta)

    # Enforce 280 char limit
    tweets = [_trim_to_limit(t, 280) for t in tweets]

    # Format as numbered thread
    lines = []
    for i, tweet in enumerate(tweets, 1):
        lines.append(f"--- Tweet {i}/{len(tweets)} ---")
        lines.append(tweet)
        lines.append("")

    return "\n".join(lines)


def _build_bookmark_tweet(script: Script) -> str:
    """Build Tweet 2 — the bookmark-worthy data/framework tweet.

    Extracts 3-5 key stats or framework points from the script body,
    formatted as a numbered list that users want to save for reference.
    """
    # Collect stats/numbers from all body sections
    stats: list[str] = []
    for section in script.body_sections:
        text = _clean_narration(section.narration_text)
        sentences = _split_sentences(text)
        for sent in sentences:
            if re.search(r'\d+\s*%|\$\d+|\d+x|\d+\s*billion|\d+\s*million', sent, re.IGNORECASE):
                # Extract a short, punchy version (< 50 chars ideal)
                short = _shorten_stat(sent)
                if short and short not in stats:
                    stats.append(short)
                    if len(stats) >= 5:
                        break
        if len(stats) >= 5:
            break

    if len(stats) >= 3:
        # Format as numbered list
        header = "The key numbers:\n\n"
        items = "\n".join(f"→ {s}" for s in stats[:5])
        tweet = header + items + "\n\n(bookmark this)"
        return _trim_to_limit(tweet, 280)
    else:
        # Fallback: use section titles as a framework list
        header = "The framework:\n\n"
        titles = []
        for section in script.body_sections[:5]:
            clean = re.sub(r"^Section\s+\d+:\s*", "", section.title).strip()
            clean = re.split(r'[—\-:]', clean)[0].strip()
            titles.append(f"→ {clean}")
        items = "\n".join(titles)
        tweet = header + items + "\n\n(save for later)"
        return _trim_to_limit(tweet, 280)


def _shorten_stat(sentence: str) -> str:
    """Shorten a stat sentence to a punchy <50 char bullet point."""
    # Try to extract the core stat (number + context)
    # "Revenue growth at heavily automated firms averaged twenty-three percent annually"
    # → "23% revenue growth at automated firms"
    sentence = sentence.strip().rstrip('.')
    if len(sentence) <= 50:
        return sentence
    # Take first 50 chars at a word boundary
    words = sentence.split()
    result = ""
    for word in words:
        candidate = result + " " + word if result else word
        if len(candidate) > 50:
            break
        result = candidate
    return result if len(result) > 15 else ""


def _extract_hook_line(hook_text: str) -> str:
    """Extract the strongest opening line from hook narration.

    Picks the first sentence that contains a number or a strong claim.
    Falls back to first sentence if no stat found.
    """
    # Clean scene markers and emphasis
    text = _clean_narration(hook_text)
    sentences = _split_sentences(text)

    # Prefer sentences with numbers (stats hit harder on X)
    for sent in sentences[:3]:
        if re.search(r'\d', sent) and len(sent) <= 250:
            return sent

    # Fallback: first sentence
    return sentences[0] if sentences else text[:250]


def _extract_section_insight(narration: str, title: str) -> str:
    """Extract the single best insight from a body section.

    Prioritizes: sentences with specific numbers > contrarian claims > definitions.
    """
    text = _clean_narration(narration)
    sentences = _split_sentences(text)

    # Strategy: find the most tweetable sentence
    # Priority 1: Short sentence with a specific number/percentage
    for sent in sentences:
        if re.search(r'\d+\s*percent|\d+%|\$\d+|\d+\s*billion|\d+\s*million', sent, re.IGNORECASE):
            if 60 <= len(sent) <= 270:
                return sent

    # Priority 1b: Two short sentences combined if they have numbers
    for i, sent in enumerate(sentences[:-1]):
        if re.search(r'\d', sent) and len(sent) < 140:
            combined = sent + " " + sentences[i + 1]
            if len(combined) <= 270:
                return combined

    # Priority 2: Sentence with a strong framing ("The problem is...", "This means...")
    strong_patterns = [
        r"the (?:problem|lesson|pattern|key|result|data|truth) is",
        r"this (?:means|creates|explains|shows)",
        r"if .+ then",
        r"the survival test",
    ]
    for sent in sentences:
        for pattern in strong_patterns:
            if re.search(pattern, sent, re.IGNORECASE) and 60 <= len(sent) <= 260:
                return sent

    # Priority 3: First sentence that's a reasonable length
    for sent in sentences[:5]:
        if 60 <= len(sent) <= 260:
            return sent

    # Fallback: construct from title
    return f"Pattern: {title}."


def _build_cta_tweet(title: str, video_url: str) -> str:
    """Build the final CTA tweet."""
    parts = []
    parts.append("Full breakdown with data, sources, and frameworks:")
    if video_url:
        parts.append(f"\n{video_url}")
    else:
        parts.append("\n[link in bio]")
    parts.append("\nSubscribe @TokenEconomyAI for weekly AI business analysis.")
    parts.append("\n📩 Newsletter: https://substack.com/@tokeneconomyai")
    return "".join(parts)


# --- Substack Newsletter Formatter ---


def generate_newsletter(script: Script, video_url: str = "") -> str:
    """Generate a Substack newsletter from a video script.

    Structure:
    - Title + subtitle
    - Video embed
    - TL;DR (3 bullet points)
    - Expanded body sections with headers
    - Key data callouts
    - Subscribe CTA

    The newsletter expands on the video — not a transcript. It restructures
    the analysis into a readable article format.

    Args:
        script: Parsed Script model.
        video_url: Optional YouTube video URL for embed.

    Returns:
        Markdown-formatted newsletter.
    """
    sections: list[str] = []

    # Title
    sections.append(f"# {script.title}\n")

    # Video embed
    if video_url:
        sections.append(f"> 📺 **Watch the full video analysis:** [{script.title}]({video_url})\n")

    # TL;DR
    tldr = _generate_tldr(script)
    sections.append("## TL;DR\n")
    sections.append(tldr + "\n")

    # Opening (from hook — rewritten as article intro)
    intro = _format_newsletter_intro(script.hook.narration_text)
    sections.append(intro + "\n")

    # Body sections
    for section in script.body_sections:
        header = _clean_section_title(section.title)
        sections.append(f"## {header}\n")
        body = _format_section_for_newsletter(section.narration_text)
        sections.append(body + "\n")

    # Conclusion
    sections.append("## The Bottom Line\n")
    conclusion = _format_section_for_newsletter(script.conclusion.narration_text)
    sections.append(conclusion + "\n")

    # CTA
    sections.append("---\n")
    sections.append(
        "*If this analysis was useful, subscribe for weekly deep dives into "
        "the economics of AI. Every Tuesday and Friday — data, frameworks, "
        "and pattern recognition applied to the fastest-moving market in history.*\n"
    )
    sections.append("[Subscribe to Token Economy AI](https://tokeneconomyai.substack.com)\n")
    sections.append(
        "📺 YouTube: https://www.youtube.com/@TokenEconomyAI\n"
        "🐦 X/Twitter: https://x.com/TokenEconomyAI\n"
        "💼 LinkedIn: https://linkedin.com/company/tokeneconomyai\n"
    )

    return "\n".join(sections)


def _generate_tldr(script: Script) -> str:
    """Generate 3-bullet TL;DR from body sections."""
    bullets: list[str] = []

    for section in script.body_sections[:5]:
        text = _clean_narration(section.narration_text)
        sentences = _split_sentences(text)
        # Pick the most informative sentence (preferring ones with numbers)
        best = None
        for sent in sentences:
            if re.search(r'\d', sent) and 30 <= len(sent) <= 200:
                best = sent
                break
        if not best and sentences:
            best = sentences[0]
        if best and len(bullets) < 3:
            bullets.append(f"- {best}")

    return "\n".join(bullets) if bullets else "- Key analysis inside."


def _format_newsletter_intro(hook_text: str) -> str:
    """Convert hook narration into an article-style opening paragraph."""
    text = _clean_narration(hook_text)
    # Take first 2-3 sentences as the opening
    sentences = _split_sentences(text)
    intro_sentences = sentences[:3]
    return " ".join(intro_sentences)


def _format_section_for_newsletter(narration: str) -> str:
    """Format a narration section as readable article paragraphs."""
    text = _clean_narration(narration)
    sentences = _split_sentences(text)

    # Group into paragraphs of 3-4 sentences
    paragraphs: list[str] = []
    current: list[str] = []

    for sent in sentences:
        current.append(sent)
        if len(current) >= 4:
            paragraphs.append(" ".join(current))
            current = []

    if current:
        paragraphs.append(" ".join(current))

    return "\n\n".join(paragraphs)


def _clean_section_title(title: str) -> str:
    """Clean a section title for use as newsletter header."""
    # Remove "Section N:" prefix
    title = re.sub(r"^Section\s+\d+:\s*", "", title)
    return title.strip()


# --- LinkedIn Post Generator ---


def generate_linkedin_post(script: Script, video_url: str = "") -> str:
    """Generate a LinkedIn post optimized for the 2026 algorithm.

    Key 2026 LinkedIn algo rules:
    - Dwell time is the primary signal (one-sentence paragraphs)
    - External links suppress reach by ~60% (link goes in first comment)
    - "Saves" are the highest-value engagement metric
    - First 60-90 minutes of engagement velocity determines reach

    Format: Hook → single-sentence paragraphs → CTA → comment instruction.
    Stays under 250 words.

    Args:
        script: Parsed Script model.
        video_url: Optional video URL (placed in comment instruction).

    Returns:
        LinkedIn post text with comment instruction at bottom.
    """
    parts: list[str] = []

    # Opening hook — strongest stat from the script (grabs scroll-stoppers)
    hook = _extract_hook_line(script.hook.narration_text)
    parts.append(hook)
    parts.append("")

    # Key insights as ONE-SENTENCE PARAGRAPHS (maximizes dwell time on mobile)
    for section in script.body_sections[:3]:
        insight = _extract_section_insight(section.narration_text, section.title)
        if insight:
            # Break multi-sentence insights into single sentences
            sentences = _split_sentences(insight)
            for sent in sentences[:2]:
                parts.append(sent)
                parts.append("")

    # Business angle from conclusion (one strong sentence)
    conclusion = _clean_narration(script.conclusion.narration_text)
    conclusion_sentences = _split_sentences(conclusion)
    if conclusion_sentences:
        parts.append(conclusion_sentences[0])
        parts.append("")

    # CTA — no outbound link in body (algo penalty)
    parts.append("Save this for reference. ♻️ Repost if your network needs to see this.")
    parts.append("")
    parts.append("Full breakdown with the data → first comment ⬇️")
    parts.append("")
    parts.append("#AI #ArtificialIntelligence #Business #Technology #TokenEconomyAI")

    # Comment instruction (not part of the post body on LinkedIn)
    parts.append("")
    parts.append("---")
    parts.append("⬇️ FIRST COMMENT (post immediately after publishing):")
    if video_url:
        parts.append(f"Full analysis with data and frameworks: {video_url}")
    else:
        parts.append("Full analysis: [paste YouTube URL here]")
    parts.append(f"📩 Newsletter: https://substack.com/@tokeneconomyai")
    parts.append(f"🐦 X/Twitter: https://x.com/TokenEconomyAI")

    return "\n".join(parts)


# --- Shared Utilities ---


def _clean_narration(text: str) -> str:
    """Strip scene markers, bold markers, and excess whitespace."""
    text = re.sub(r"\[SCENE:.*?\]", "", text)
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences, handling common abbreviations."""
    # Simple sentence splitter — split on period/exclamation/question followed by space + capital
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    return [s.strip() for s in sentences if s.strip()]


def _trim_to_limit(text: str, limit: int) -> str:
    """Trim text to character limit, breaking at word boundary."""
    if len(text) <= limit:
        return text
    trimmed = text[:limit - 3]
    # Break at last space
    last_space = trimmed.rfind(" ")
    if last_space > limit // 2:
        trimmed = trimmed[:last_space]
    return trimmed + "..."
