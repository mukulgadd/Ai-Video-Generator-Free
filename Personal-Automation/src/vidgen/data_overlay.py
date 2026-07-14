"""Data overlay renderer — composites charts and stats onto MFLUX backgrounds.

Generates professional data visualizations by combining:
1. MFLUX atmospheric background (blurred for depth)
2. Matplotlib charts (transparent, brand-colored)
3. Pillow text (big stats, bullet lists)
4. Global grain texture (cohesion between layers)

Configurable via DataOverlayConfig — can be enabled/disabled per pipeline run.
"""

import io
import logging
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from vidgen.config import DataOverlayConfig

logger = logging.getLogger(__name__)

# Brand colors
BRAND_BLUE = "#3b82f6"
BRAND_GOLD = "#f59e0b"
BRAND_RED = "#e94560"
BRAND_WHITE = "#ffffff"
BRAND_NAVY = "#0a0f1e"

FONT_FALLBACKS = [
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/SFNSDisplay.ttf",
    "/Library/Fonts/Arial Bold.ttf",
]


class DataOverlayRenderer:
    """Renders data visualizations on MFLUX backgrounds.

    Supports 5 templates:
    - big_stat: Single large number/stat centered
    - bar_comparison: Side-by-side bars (before/after)
    - line_trend: Rising or declining trend line
    - comparison: Two-column with arrows
    - bullet_list: 3-5 items with check/cross marks

    Pipeline:
    1. Load MFLUX background
    2. Apply Gaussian blur (depth of field effect)
    3. Apply dark gradient overlay (readability)
    4. Render chart via Matplotlib or text via Pillow (transparent)
    5. Composite onto background
    6. Apply grain texture (visual cohesion)
    """

    def __init__(self, config: DataOverlayConfig) -> None:
        self.config = config
        self._font_path = self._resolve_font()

    def render(self, background_path: Path, overlay_data: dict, output_path: Path) -> Path:
        """Render a data overlay onto an MFLUX background.

        Args:
            background_path: Path to the MFLUX-generated background PNG.
            overlay_data: Dict with 'type' and type-specific fields.
            output_path: Where to save the final composited image.

        Returns:
            Path to the saved image.
        """
        if not self.config.enabled:
            # If disabled, just return the background as-is
            return background_path

        # Load background
        bg = Image.open(background_path).convert("RGBA")
        width, height = bg.size

        # Step 1: Gaussian blur (depth of field)
        if self.config.blur_radius > 0:
            bg_rgb = bg.convert("RGB")
            bg_rgb = bg_rgb.filter(ImageFilter.GaussianBlur(radius=self.config.blur_radius))
            bg = bg_rgb.convert("RGBA")

        # Step 2: Dark gradient overlay (readability)
        dark_overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw_overlay = ImageDraw.Draw(dark_overlay)
        # Center-focused darkening — darker at edges, slightly lighter center
        for y in range(height):
            # Radial-ish: darker near edges
            dist_from_center = abs(y - height // 2) / (height // 2)
            alpha = int(self.config.dark_overlay_base + dist_from_center * 40)
            alpha = min(alpha, 200)
            draw_overlay.line([(0, y), (width - 1, y)], fill=(10, 15, 30, alpha))
        bg = Image.alpha_composite(bg, dark_overlay)

        # Step 3: Render the data visualization
        viz_type = overlay_data.get("type", "big_stat")
        if viz_type == "bar_comparison":
            chart = self._render_bar_chart(overlay_data, width, height)
        elif viz_type == "line_trend":
            chart = self._render_line_trend(overlay_data, width, height)
        elif viz_type == "comparison":
            chart = self._render_comparison(overlay_data, width, height)
        elif viz_type == "bullet_list":
            chart = self._render_bullet_list(overlay_data, width, height)
        else:  # big_stat (default)
            chart = self._render_big_stat(overlay_data, width, height)

        # Step 4: Composite chart onto background
        if chart:
            bg = Image.alpha_composite(bg, chart)

        # Step 5: Apply grain texture (cohesion)
        if self.config.grain_opacity > 0:
            bg = self._apply_grain(bg)

        # Save
        output_path.parent.mkdir(parents=True, exist_ok=True)
        bg.convert("RGB").save(str(output_path), "PNG")
        logger.info(f"Data overlay rendered: {output_path.name} (type={viz_type})")
        return output_path

    def _render_big_stat(self, data: dict, width: int, height: int) -> Image.Image:
        """Render a single large stat number centered."""
        layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)

        stat_text = data.get("value", "$0")
        label_text = data.get("label", "")
        color = data.get("color", BRAND_GOLD)

        # Large stat
        stat_font = self._load_font(int(height * 0.15))
        if stat_font:
            bbox = draw.textbbox((0, 0), stat_text, font=stat_font)
            tw = bbox[2] - bbox[0]
            tx = (width - tw) // 2
            ty = int(height * 0.35)
            draw.text((tx, ty), stat_text, font=stat_font, fill=color,
                      stroke_width=3, stroke_fill="#000000")

        # Label below
        if label_text:
            label_font = self._load_font(int(height * 0.05))
            if label_font:
                bbox = draw.textbbox((0, 0), label_text, font=label_font)
                tw = bbox[2] - bbox[0]
                tx = (width - tw) // 2
                ty = int(height * 0.55)
                draw.text((tx, ty), label_text, font=label_font, fill=BRAND_WHITE,
                          stroke_width=2, stroke_fill="#000000")

        return layer

    def _render_bar_chart(self, data: dict, width: int, height: int) -> Image.Image:
        """Render bar comparison using Matplotlib (transparent)."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        labels = data.get("labels", ["Before", "After"])
        values = data.get("values", ["100", "10"])
        color = data.get("color", BRAND_RED)

        # Parse numeric values (strip $ and other chars)
        numeric_values = []
        for v in values:
            cleaned = ''.join(c for c in str(v) if c.isdigit() or c == '.')
            numeric_values.append(float(cleaned) if cleaned else 0)

        # Create Matplotlib figure with transparent background
        fig, ax = plt.subplots(figsize=(8, 5))
        fig.patch.set_alpha(0.0)
        ax.patch.set_alpha(0.0)

        bars = ax.bar(labels, numeric_values, color=color, width=0.5, edgecolor='white', linewidth=0.5)

        # Style
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#ffffff40')
        ax.spines['bottom'].set_color('#ffffff40')
        ax.tick_params(colors='white', labelsize=14)
        ax.set_ylabel('')

        # Add value labels on bars
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(numeric_values) * 0.02,
                    str(val), ha='center', va='bottom', color='white', fontsize=16, fontweight='bold')

        plt.tight_layout()

        # Render to transparent PNG in memory
        buf = io.BytesIO()
        plt.savefig(buf, format='png', transparent=True, dpi=150, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)

        chart_img = Image.open(buf).convert("RGBA")

        # Center on canvas
        layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        # Scale chart to fit ~70% of frame
        target_w = int(width * 0.7)
        target_h = int(height * 0.5)
        chart_img = chart_img.resize((target_w, target_h), Image.LANCZOS)
        x = (width - target_w) // 2
        y = (height - target_h) // 2
        layer.paste(chart_img, (x, y), chart_img)

        return layer

    def _render_line_trend(self, data: dict, width: int, height: int) -> Image.Image:
        """Render a simple trend line (rising or declining)."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        direction = data.get("direction", "down")  # "up" or "down"
        label = data.get("label", "Cost")
        color = data.get("color", BRAND_RED if direction == "down" else BRAND_GOLD)

        # Generate trend data
        x = list(range(10))
        if direction == "down":
            y = [100 - i * 10 + np.random.randint(-3, 3) for i in x]
        else:
            y = [10 + i * 10 + np.random.randint(-3, 3) for i in x]

        fig, ax = plt.subplots(figsize=(8, 4))
        fig.patch.set_alpha(0.0)
        ax.patch.set_alpha(0.0)

        ax.plot(x, y, color=color, linewidth=3, marker='o', markersize=4)
        ax.fill_between(x, y, alpha=0.2, color=color)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#ffffff40')
        ax.spines['bottom'].set_color('#ffffff40')
        ax.tick_params(colors='white', labelsize=10)
        ax.set_title(label, color='white', fontsize=16, pad=10)

        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png', transparent=True, dpi=150, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)

        chart_img = Image.open(buf).convert("RGBA")

        layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        target_w = int(width * 0.7)
        target_h = int(height * 0.4)
        chart_img = chart_img.resize((target_w, target_h), Image.LANCZOS)
        x_pos = (width - target_w) // 2
        y_pos = (height - target_h) // 2
        layer.paste(chart_img, (x_pos, y_pos), chart_img)

        return layer

    def _render_comparison(self, data: dict, width: int, height: int) -> Image.Image:
        """Render a two-column before/after comparison with arrow."""
        layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)

        left_value = data.get("left_value", "$42")
        right_value = data.get("right_value", "$1")
        left_label = data.get("left_label", "Before")
        right_label = data.get("right_label", "After AI")
        color_left = data.get("color_left", BRAND_RED)
        color_right = data.get("color_right", BRAND_GOLD)

        # Fonts
        value_font = self._load_font(int(height * 0.12))
        label_font = self._load_font(int(height * 0.04))

        center_x = width // 2
        y_center = height // 2

        if value_font and label_font:
            # Left value
            draw.text((center_x - int(width * 0.25), y_center - int(height * 0.08)),
                      left_value, font=value_font, fill=color_left,
                      stroke_width=3, stroke_fill="#000000", anchor="mm")
            draw.text((center_x - int(width * 0.25), y_center + int(height * 0.05)),
                      left_label, font=label_font, fill=BRAND_WHITE, anchor="mm")

            # Arrow
            arrow_font = self._load_font(int(height * 0.08))
            if arrow_font:
                draw.text((center_x, y_center - int(height * 0.04)),
                          "→", font=arrow_font, fill=BRAND_WHITE, anchor="mm")

            # Right value
            draw.text((center_x + int(width * 0.25), y_center - int(height * 0.08)),
                      right_value, font=value_font, fill=color_right,
                      stroke_width=3, stroke_fill="#000000", anchor="mm")
            draw.text((center_x + int(width * 0.25), y_center + int(height * 0.05)),
                      right_label, font=label_font, fill=BRAND_WHITE, anchor="mm")

        return layer

    def _render_bullet_list(self, data: dict, width: int, height: int) -> Image.Image:
        """Render a bullet list with check/cross marks."""
        layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)

        items = data.get("items", [])
        title = data.get("title", "")

        font = self._load_font(int(height * 0.045))
        title_font = self._load_font(int(height * 0.06))

        if not font:
            return layer

        y_start = int(height * 0.25)
        x_start = int(width * 0.15)
        line_height = int(height * 0.08)

        # Title
        if title and title_font:
            draw.text((x_start, y_start - line_height * 1.5), title,
                      font=title_font, fill=BRAND_WHITE,
                      stroke_width=2, stroke_fill="#000000")

        # Items
        for i, item in enumerate(items[:6]):
            y = y_start + i * line_height
            # Determine icon
            if isinstance(item, dict):
                text = item.get("text", "")
                positive = item.get("positive", True)
            else:
                text = str(item)
                positive = True

            icon = "✓" if positive else "✗"
            icon_color = BRAND_GOLD if positive else BRAND_RED

            draw.text((x_start, y), icon, font=font, fill=icon_color)
            draw.text((x_start + int(width * 0.05), y), text, font=font, fill=BRAND_WHITE,
                      stroke_width=1, stroke_fill="#00000080")

        return layer

    def _apply_grain(self, img: Image.Image) -> Image.Image:
        """Apply subtle film grain texture for visual cohesion."""
        width, height = img.size
        # Generate random noise
        noise = np.random.randint(0, 255, (height, width), dtype=np.uint8)
        noise_img = Image.fromarray(noise, mode="L").convert("RGBA")

        # Make it very subtle (5% opacity)
        opacity = int(255 * self.config.grain_opacity)
        noise_rgba = Image.new("RGBA", (width, height), (128, 128, 128, 0))
        noise_data = np.array(noise_rgba)
        noise_data[:, :, 0] = noise  # R
        noise_data[:, :, 1] = noise  # G
        noise_data[:, :, 2] = noise  # B
        noise_data[:, :, 3] = opacity  # A
        noise_img = Image.fromarray(noise_data, mode="RGBA")

        # Blend
        return Image.alpha_composite(img, noise_img)

    def _load_font(self, size: int) -> ImageFont.FreeTypeFont | None:
        """Load font at given size."""
        if self._font_path:
            try:
                return ImageFont.truetype(self._font_path, size)
            except (OSError, IOError):
                pass
        return None

    def _resolve_font(self) -> str | None:
        """Find available font."""
        for path in FONT_FALLBACKS:
            try:
                ImageFont.truetype(path, 48)
                return path
            except (OSError, IOError):
                continue
        return None
