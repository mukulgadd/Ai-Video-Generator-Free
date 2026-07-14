"""Thumbnail generator — premium Token Economy style.

Design spec (from CHANNEL_PLAN.md):
- Dark background (navy/black gradient overlay on scene image)
- Bold 3-5 word white text (Helvetica Bold), uppercase
- One accent-colored keyword (amber #f59e0b or blue #3b82f6)
- NO face, NO arrows, NO emoji — clean and premium
- Output: 1280x720 PNG, 3 variants per video
"""

import logging
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

from vidgen.config import BrandingConfig

logger = logging.getLogger(__name__)

# Token Economy brand colors
NAVY_BG = (10, 15, 30)  # #0a0f1e
ACCENT_GOLD = "#f59e0b"
ACCENT_BLUE = "#3b82f6"
TEXT_WHITE = "#ffffff"
STROKE_COLOR = "#000000"

# Font settings
FONT_PRIMARY = "Helvetica-Bold"
FONT_FALLBACKS = [
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/SFNSDisplay.ttf",
    "/Library/Fonts/Arial Bold.ttf",
    "Helvetica",
]


class ThumbnailGenerator:
    """Generates YouTube thumbnails with dark gradient overlays and bold text.

    Uses existing scene images as atmospheric backgrounds, applies a dark
    gradient overlay for readability, and composites bold text with an
    accent-colored keyword. Produces 3 variants using different source
    images from the video's scenes.
    """

    def __init__(self, config: BrandingConfig, image_gen=None) -> None:
        """Initialize with branding config. image_gen is optional (unused in new design)."""
        self.config = config
        self._font_path = self._resolve_font()

    def generate_variants(
        self,
        title: str,
        style_prefix: str,
        output_dir: Path,
        count: int = 3,
        scene_images: list[Path] | None = None,
        thumbnail_text: str | None = None,
        accent_word: str | None = None,
        accent_color: str | None = None,
    ) -> list[Path]:
        """Generate multiple thumbnail variants using scene images as backgrounds.

        Args:
            title: Video title (used to extract thumbnail text if thumbnail_text not given).
            style_prefix: Unused (kept for interface compatibility).
            output_dir: Directory to write variant PNGs.
            count: Number of variants to generate (default 3).
            scene_images: List of scene image paths to use as backgrounds.
                Selects the most visually distinct ones. If None, generates solid gradient.
            thumbnail_text: Explicit 3-5 word text for the thumbnail. If None, extracted from title.
            accent_word: Which word to highlight in accent color. If None, auto-detected.
            accent_color: Hex color for accent word. Defaults to gold (#f59e0b).

        Returns:
            List of paths to generated thumbnail variants.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        paths: list[Path] = []

        # Determine text content
        text = thumbnail_text or self._extract_thumbnail_text(title)
        color = accent_color or ACCENT_GOLD

        # Select background images (spread across scenes for variety)
        backgrounds = self._select_backgrounds(scene_images, count)

        for i in range(count):
            final_path = output_dir / f"variant_{i + 1}.png"

            # Get background (scene image or solid gradient)
            if i < len(backgrounds) and backgrounds[i].exists():
                base = self._prepare_background(backgrounds[i])
            else:
                base = self._generate_gradient_background()

            # Apply dark gradient overlay
            base = self._apply_dark_overlay(base)

            # Render text with accent
            self._render_text(base, text, accent_word, color)

            # Feed Pop: boost contrast +15% and saturation +10% for YouTube dark mode visibility
            base = self._apply_feed_pop(base)

            # Save
            base.save(str(final_path), "PNG", quality=95)
            paths.append(final_path)
            logger.info(f"Thumbnail {i + 1}/{count}: {final_path.name}")

        return paths

    def apply_text_overlay(self, image_path: Path, text: str, output_path: Path) -> Path:
        """Apply bold text overlay on an existing image (legacy interface).

        Args:
            image_path: Path to the base thumbnail image.
            text: Text to overlay (3-5 words max).
            output_path: Where to write the final thumbnail.

        Returns:
            Path to the output thumbnail.
        """
        base = self._prepare_background(image_path)
        base = self._apply_dark_overlay(base)
        self._render_text(base, text.upper(), accent_word=None, accent_color=ACCENT_GOLD)
        base.save(str(output_path), "PNG", quality=95)
        return output_path

    # --- Private methods ---

    def _prepare_background(self, image_path: Path) -> Image.Image:
        """Load and crop/resize a scene image to 1280x720 thumbnail dimensions."""
        img = Image.open(str(image_path)).convert("RGB")

        # Crop to 16:9 if needed, then resize to 1280x720
        target_ratio = 1280 / 720
        img_ratio = img.width / img.height

        if abs(img_ratio - target_ratio) > 0.01:
            # Center crop to 16:9
            if img_ratio > target_ratio:
                # Too wide — crop sides
                new_width = int(img.height * target_ratio)
                left = (img.width - new_width) // 2
                img = img.crop((left, 0, left + new_width, img.height))
            else:
                # Too tall — crop top/bottom
                new_height = int(img.width / target_ratio)
                top = (img.height - new_height) // 2
                img = img.crop((0, top, img.width, top + new_height))

        # Resize to final thumbnail dimensions
        img = img.resize((1280, 720), Image.LANCZOS)

        # Slight blur to push image into background (text pops more)
        img = img.filter(ImageFilter.GaussianBlur(radius=1.5))

        return img

    def _generate_gradient_background(self) -> Image.Image:
        """Generate a solid navy gradient as fallback when no scene images available."""
        img = Image.new("RGB", (1280, 720))
        pixels = img.load()

        for y in range(720):
            t = y / 720
            # Navy to slightly lighter navy (subtle vertical gradient)
            r = int(10 + 15 * t)
            g = int(15 + 10 * t)
            b = int(30 + 25 * t)
            for x in range(1280):
                pixels[x, y] = (r, g, b)

        return img

    def _apply_dark_overlay(self, img: Image.Image) -> Image.Image:
        """Apply a dramatic dark gradient overlay for text readability.

        Creates a vignette-style darkening: strong at bottom (where text goes),
        moderate at top, with the center slightly lighter to show the image.
        """
        overlay = Image.new("RGBA", (1280, 720), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        # Bottom gradient: heavy dark (text area)
        for y in range(720):
            # Non-linear darkening curve — heavier at bottom
            if y < 200:
                # Top: moderate darkness
                alpha = int(140 + (200 - y) * 0.3)
            elif y < 400:
                # Middle: lightest area (show the image)
                alpha = int(100 + (y - 200) * 0.15)
            else:
                # Bottom: heavy darkness for text readability
                alpha = int(130 + (y - 400) * 0.4)

            alpha = min(alpha, 220)  # Cap at ~86% opacity
            draw.line([(0, y), (1279, y)], fill=(10, 15, 30, alpha))

        # Composite the overlay
        img = img.convert("RGBA")
        img = Image.alpha_composite(img, overlay)
        return img.convert("RGB")

    def _apply_feed_pop(self, img: Image.Image) -> Image.Image:
        """Boost contrast +15% and saturation +10% for YouTube feed visibility.

        Thumbnails need to pop against YouTube's dark mode UI on mobile.
        This enhancement is applied ONLY to thumbnail exports, not video frames.
        """
        from PIL import ImageEnhance

        # Contrast: makes darks darker and brights brighter (+15%)
        img = ImageEnhance.Contrast(img).enhance(1.15)

        # Saturation: makes accent colors (gold, red) more vivid (+10%)
        img = ImageEnhance.Color(img).enhance(1.10)

        return img

    def _render_text(
        self,
        img: Image.Image,
        text: str,
        accent_word: str | None,
        accent_color: str,
    ) -> None:
        """Render bold text with optional accent-colored keyword.

        Text is centered horizontally, positioned in the lower-center area.
        If accent_word is specified, that word renders in the accent color.
        All other words render in white with a black stroke.
        """
        draw = ImageDraw.Draw(img)
        text_upper = text.upper()

        # Determine font size (fill ~70% of width, respect 40% area constraint)
        font_size = self._calculate_optimal_font_size(draw, text_upper)
        font = self._load_font(font_size)

        words = text_upper.split()

        # Auto-detect accent word if not specified (pick the most "hook-worthy" word)
        if accent_word is None and len(words) > 1:
            accent_word = self._pick_accent_word(words)
        if accent_word:
            accent_word = accent_word.upper()

        # Calculate total text width for centering
        space_width = draw.textlength(" ", font=font)
        word_widths = [draw.textlength(w, font=font) for w in words]
        total_width = sum(word_widths) + space_width * (len(words) - 1)

        # Safe zone: keep text within center 70% to avoid YouTube UI overlays
        # YouTube places duration badge at bottom-right (~50x20px)
        # Safe area: 15% padding left/right, 20% padding top/bottom
        SAFE_LEFT = int(1280 * 0.15)
        SAFE_RIGHT = int(1280 * 0.85)
        SAFE_TOP = int(720 * 0.20)
        SAFE_BOTTOM = int(720 * 0.75)
        safe_width = SAFE_RIGHT - SAFE_LEFT

        # Clamp text within safe zone
        x_start = max(SAFE_LEFT, (1280 - total_width) / 2)
        if x_start + total_width > SAFE_RIGHT:
            x_start = SAFE_RIGHT - total_width

        # Position: centered in safe zone vertically (lower-center)
        y_pos = int((SAFE_TOP + SAFE_BOTTOM) / 2)  # Center of safe area

        # Render each word
        x_cursor = x_start
        stroke_width = max(4, font_size // 20)

        for word in words:
            color = accent_color if word == accent_word else TEXT_WHITE

            # Draw with stroke for contrast against any background
            draw.text(
                (x_cursor, y_pos),
                word,
                font=font,
                fill=color,
                stroke_width=stroke_width,
                stroke_fill=STROKE_COLOR,
            )
            x_cursor += draw.textlength(word, font=font) + space_width

    def _calculate_optimal_font_size(self, draw: ImageDraw.Draw, text: str) -> int:
        """Find the largest font size that fits within safe zone constraints."""
        # Safe zone: 70% of width (15% padding each side)
        max_width = int(1280 * 0.70)
        max_area = 0.30 * 1280 * 720  # 30% of image area

        for size in range(140, 36, -4):
            font = self._load_font(size)
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]

            if text_width <= max_width and (text_width * text_height) <= max_area:
                return size

        return 40  # Minimum fallback

    def _load_font(self, size: int) -> ImageFont.FreeTypeFont:
        """Load Helvetica Bold (or best available fallback) at given size."""
        if self._font_path:
            try:
                return ImageFont.truetype(self._font_path, size)
            except (OSError, IOError):
                pass

        # Try each fallback
        for path in FONT_FALLBACKS:
            try:
                return ImageFont.truetype(path, size)
            except (OSError, IOError):
                continue

        # Last resort
        return ImageFont.load_default()

    def _resolve_font(self) -> str | None:
        """Find the best available bold font on the system."""
        # Prefer the configured font
        configured = self.config.thumbnail_font
        if configured:
            # Try as-is first (covers both paths and font names on macOS)
            try:
                ImageFont.truetype(configured, 48)
                return configured
            except (OSError, IOError):
                pass

        # Try known bold fonts
        for path in FONT_FALLBACKS:
            try:
                ImageFont.truetype(path, 48)
                return path
            except (OSError, IOError):
                continue

        return None

    def _select_backgrounds(self, scene_images: list[Path] | None, count: int) -> list[Path]:
        """Select diverse background images from scene images.

        Spreads selections across scenes (beginning, middle, end) for variety.
        Prefers the first image of each scene (usually the most establishing shot).
        """
        if not scene_images:
            return []

        valid = [p for p in scene_images if p.exists()]
        if not valid:
            return []

        if len(valid) <= count:
            return valid

        # Spread evenly: pick from beginning, middle, and end
        step = len(valid) / count
        selected = []
        for i in range(count):
            idx = int(i * step)
            selected.append(valid[idx])

        return selected

    def _extract_thumbnail_text(self, title: str) -> str:
        """Extract 3-5 impactful words from the video title for thumbnail text."""
        # Remove common filler/connector words
        skip = {
            "the", "a", "an", "is", "are", "of", "in", "to", "and", "for",
            "how", "why", "what", "this", "that", "with", "from", "about",
        }

        words = title.split()

        # If title has parenthetical, extract content from parens (usually the hook)
        if "(" in title and ")" in title:
            paren_content = title[title.index("(") + 1:title.index(")")]
            paren_words = [w for w in paren_content.split() if w.lower() not in skip]
            if 2 <= len(paren_words) <= 5:
                return " ".join(paren_words).upper()

        # Filter to key words, take up to 4
        key_words = [w for w in words if w.lower() not in skip and len(w) > 2]
        if len(key_words) >= 3:
            return " ".join(key_words[:4]).upper()

        # Fallback: just take first 4 words
        return " ".join(words[:4]).upper()

    def _pick_accent_word(self, words: list[str]) -> str:
        """Pick the most visually impactful word to highlight in accent color.

        Prefers: numbers/percentages > emotional words > longest word.
        """
        # Numbers and percentages are always eye-catching
        for word in words:
            cleaned = word.strip("(),.:;!?")
            if any(c.isdigit() for c in cleaned):
                return word

        # Emotional/power words
        power_words = {
            "FAIL", "DEAD", "KILL", "WIN", "SAVE", "SAVED", "LOST", "SECRET",
            "WRONG", "TRUTH", "LIE", "LIES", "NEVER", "ALWAYS", "BILLION",
            "MILLION", "FREE", "FAST", "SLOW", "BROKE", "RICH", "WAR",
        }
        for word in words:
            if word in power_words:
                return word

        # Fallback: longest word (usually most descriptive)
        return max(words, key=len)
