"""Community Tab post generator — creates square image crops + engagement text.

Picks the most visually striking scene images, crops them to 1:1 (1080x1080)
for optimal Community Tab display, and generates short engagement posts.
"""

import logging
import random
from pathlib import Path

from PIL import Image

from vidgen.models import Script

logger = logging.getLogger(__name__)


def generate_community_posts(
    script: Script,
    scene_images: list[Path],
    output_dir: Path,
    count: int = 2,
    video_url: str = "",
) -> list[tuple[Path, str]]:
    """Generate Community Tab posts from scene images.

    Picks visually diverse images from the middle of the video (avoiding
    hook/conclusion which are less visually interesting), crops to 1:1
    square, and generates 280-char engagement text.

    Args:
        script: The video script (for generating post text).
        scene_images: All generated scene images.
        output_dir: Where to save the cropped images + text files.
        count: Number of community posts to generate (default 2).
        video_url: YouTube video URL for the post CTA.

    Returns:
        List of (image_path, post_text) tuples.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[tuple[Path, str]] = []

    if len(scene_images) < 3:
        logger.warning("Not enough scene images for community posts")
        return []

    # Pick images from the middle 60% of the video (skip first/last 20%)
    start_idx = max(1, len(scene_images) // 5)
    end_idx = len(scene_images) - start_idx
    middle_images = scene_images[start_idx:end_idx]

    # Select the largest files (more detail = more visually striking)
    middle_images_sorted = sorted(middle_images, key=lambda p: p.stat().st_size, reverse=True)
    selected = middle_images_sorted[:count]

    # Generate post text options
    post_texts = _generate_post_texts(script, video_url)

    for i, img_path in enumerate(selected):
        # Crop to 1:1 square (center crop from 1920x1080 → 1080x1080)
        cropped_path = output_dir / f"post_{i + 1}.png"
        _crop_square(img_path, cropped_path)

        # Get corresponding text
        text = post_texts[i] if i < len(post_texts) else post_texts[0]

        # Save text file alongside
        text_path = output_dir / f"post_{i + 1}.txt"
        text_path.write_text(text)

        results.append((cropped_path, text))
        logger.info(f"Community post {i + 1}: {cropped_path.name}")

    return results


def _crop_square(input_path: Path, output_path: Path, size: int = 1080) -> None:
    """Center-crop an image to 1:1 square and resize to target size."""
    img = Image.open(input_path)
    w, h = img.size

    # Center crop to square (use the shorter dimension)
    short_side = min(w, h)
    left = (w - short_side) // 2
    top = (h - short_side) // 2
    right = left + short_side
    bottom = top + short_side

    cropped = img.crop((left, top, right, bottom))
    cropped = cropped.resize((size, size), Image.LANCZOS)
    cropped.save(str(output_path), "PNG")


def _generate_post_texts(script: Script, video_url: str) -> list[str]:
    """Generate 2-3 engagement post options from the script.

    Each post is ≤280 chars (Community Tab optimal length).
    Mix of: poll-style questions, hot takes, and hook teasers.
    """
    import re

    title = script.title
    # Clean title for use in post
    short_title = re.split(r'[—\-:(\[]', title)[0].strip()

    posts: list[str] = []

    # Post 1: Poll/question style (drives comments)
    hook_text = re.sub(r"\[SCENE:.*?\]", "", script.hook.narration_text).strip()
    hook_text = re.sub(r"\*\*(.*?)\*\*", r"\1", hook_text)
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', hook_text)
    first_sentence = sentences[0] if sentences else hook_text[:200]

    poll_post = f"{first_sentence}\n\nDo you agree? New analysis just dropped 👇"
    if video_url:
        poll_post += f"\n{video_url}"
    posts.append(poll_post[:280])

    # Post 2: Hot take / one insight (curiosity gap)
    if script.body_sections:
        body = script.body_sections[0]
        body_text = re.sub(r"\[SCENE:.*?\]", "", body.narration_text).strip()
        body_text = re.sub(r"\*\*(.*?)\*\*", r"\1", body_text)
        body_sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', body_text)
        # Pick a sentence with a number (data-driven hook)
        data_sentence = None
        for s in body_sentences:
            if re.search(r'\d', s) and 40 <= len(s) <= 200:
                data_sentence = s
                break
        if data_sentence:
            insight_post = f"One stat that shocked me while researching this:\n\n{data_sentence}"
            if video_url:
                insight_post += f"\n\nFull breakdown: {video_url}"
            posts.append(insight_post[:280])

    # Post 3: Simple teaser
    teaser = f"New deep dive just dropped: {short_title}\n\nLink in the description of our latest video."
    posts.append(teaser[:280])

    return posts
