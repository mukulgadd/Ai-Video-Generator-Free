"""Tests for the shorts assembly module — video composition with Ken Burns, overlays, music."""

import os
import subprocess
import tempfile
from pathlib import Path

import pytest
from PIL import Image

from vidgen.config import ShortsConfig
from vidgen.models import ImageCue, OverlayCue
from vidgen.shorts_assembly import ShortsAssembler


@pytest.fixture
def config() -> ShortsConfig:
    return ShortsConfig()


@pytest.fixture
def temp_assets(tmp_path: Path) -> dict:
    """Create placeholder assets for assembly testing."""
    # Silent WAV narration (10s)
    narration = tmp_path / "narration.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono",
         "-t", "10", str(narration)],
        capture_output=True, check=True,
    )

    # Music (30s sine tone)
    music = tmp_path / "music.mp3"
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "sine=frequency=220:duration=30",
         str(music)],
        capture_output=True, check=True,
    )

    # Vertical images (1080x1920)
    images = []
    for i in range(4):
        img_path = tmp_path / f"img_{i+1}.png"
        img = Image.new("RGB", (1080, 1920), (30 + i * 50, 40, 120 - i * 20))
        img.save(img_path)
        images.append(img_path)

    # Logo
    logo = tmp_path / "logo.png"
    logo_img = Image.new("RGBA", (60, 60), (255, 200, 0, 180))
    logo_img.save(logo)

    return {
        "narration": narration,
        "music": music,
        "images": images,
        "logo": logo,
        "output": tmp_path / "output.mp4",
    }


@pytest.fixture
def image_cues() -> list[ImageCue]:
    return [
        ImageCue(prompt="test 1", start_time=0.0, duration=3.25),
        ImageCue(prompt="test 2", start_time=3.25, duration=3.25),
        ImageCue(prompt="test 3", start_time=6.5, duration=3.25),
        ImageCue(prompt="test 4", start_time=9.75, duration=3.25),
    ]


@pytest.fixture
def overlay_cues() -> list[OverlayCue]:
    return [
        OverlayCue(text="Impact text", start_time=1.0, duration=3.0, style="impact"),
        OverlayCue(text="Normal text", start_time=5.0, duration=3.0, style="normal"),
        OverlayCue(text="DANGER", start_time=9.0, duration=2.0, style="danger"),
    ]


class TestShortsAssemblerInit:
    def test_instantiation_default_config(self, config: ShortsConfig):
        assembler = ShortsAssembler(config)
        assert assembler.config == config

    def test_font_resolved(self, config: ShortsConfig):
        assembler = ShortsAssembler(config)
        # On macOS, at least one font fallback should resolve
        assert assembler._font_path is not None


class TestShortsAssemblerAssemble:
    """Integration tests for the full assemble() method."""

    def test_basic_assembly_produces_mp4(
        self, config, temp_assets, image_cues, overlay_cues
    ):
        """Minimal assembly: images + narration + overlays."""
        assembler = ShortsAssembler(config)
        result = assembler.assemble(
            narration_path=temp_assets["narration"],
            image_paths=temp_assets["images"],
            image_cues=image_cues,
            overlay_cues=overlay_cues,
            hook_text="HOOK",
            output_path=temp_assets["output"],
        )
        assert result == temp_assets["output"]
        assert result.exists()
        assert result.stat().st_size > 10000

    def test_assembly_with_music(
        self, config, temp_assets, image_cues, overlay_cues
    ):
        """Assembly with background music mixing."""
        assembler = ShortsAssembler(config)
        result = assembler.assemble(
            narration_path=temp_assets["narration"],
            image_paths=temp_assets["images"],
            image_cues=image_cues,
            overlay_cues=overlay_cues,
            hook_text="HOOK",
            output_path=temp_assets["output"],
            music_path=temp_assets["music"],
        )
        assert result.exists()
        assert result.stat().st_size > 10000

    def test_assembly_with_logo(
        self, config, temp_assets, image_cues, overlay_cues
    ):
        """Assembly with logo watermark."""
        assembler = ShortsAssembler(config)
        result = assembler.assemble(
            narration_path=temp_assets["narration"],
            image_paths=temp_assets["images"],
            image_cues=image_cues,
            overlay_cues=overlay_cues,
            hook_text="HOOK",
            output_path=temp_assets["output"],
            logo_path=temp_assets["logo"],
        )
        assert result.exists()

    def test_full_assembly_all_features(
        self, config, temp_assets, image_cues, overlay_cues
    ):
        """Full assembly with everything: KB, overlays, music, logo, end card."""
        assembler = ShortsAssembler(config)
        result = assembler.assemble(
            narration_path=temp_assets["narration"],
            image_paths=temp_assets["images"],
            image_cues=image_cues,
            overlay_cues=overlay_cues,
            hook_text="Testing full pipeline",
            output_path=temp_assets["output"],
            music_path=temp_assets["music"],
            logo_path=temp_assets["logo"],
        )
        assert result.exists()
        # Verify it's a valid video with ffprobe
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries",
             "stream=width,height,codec_name,codec_type",
             "-of", "csv=p=0", str(result)],
            capture_output=True, text=True,
        )
        lines = probe.stdout.strip().split("\n")
        # Should have video and audio streams
        codecs = [l.strip() for l in lines if l.strip()]
        assert any("h264" in c for c in codecs)
        assert any("aac" in c for c in codecs)

    def test_output_resolution_is_vertical(
        self, config, temp_assets, image_cues, overlay_cues
    ):
        """Output must be 1080x1920 vertical."""
        assembler = ShortsAssembler(config)
        result = assembler.assemble(
            narration_path=temp_assets["narration"],
            image_paths=temp_assets["images"],
            image_cues=image_cues,
            overlay_cues=overlay_cues,
            hook_text="HOOK",
            output_path=temp_assets["output"],
        )
        # Check dimensions with ffprobe
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height",
             "-of", "csv=p=0:s=x", str(result)],
            capture_output=True, text=True,
        )
        dims = probe.stdout.strip()
        assert dims == "1080x1920"

    def test_output_duration_includes_end_card(
        self, config, temp_assets, image_cues, overlay_cues
    ):
        """Output duration should be narration + 3s end card."""
        assembler = ShortsAssembler(config)
        result = assembler.assemble(
            narration_path=temp_assets["narration"],
            image_paths=temp_assets["images"],
            image_cues=image_cues,
            overlay_cues=overlay_cues,
            hook_text="HOOK",
            output_path=temp_assets["output"],
        )
        # Check duration with ffprobe
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(result)],
            capture_output=True, text=True,
        )
        duration = float(probe.stdout.strip())
        # 10s narration + 3s end card = 13s (allow 0.5s tolerance)
        assert 12.5 <= duration <= 13.5

    def test_empty_image_list_raises(self, config, temp_assets, overlay_cues):
        """Should raise if no images provided."""
        assembler = ShortsAssembler(config)
        with pytest.raises(RuntimeError, match="No valid image clips"):
            assembler.assemble(
                narration_path=temp_assets["narration"],
                image_paths=[],
                image_cues=[],
                overlay_cues=overlay_cues,
                hook_text="HOOK",
                output_path=temp_assets["output"],
            )

    def test_missing_image_skipped_gracefully(
        self, config, temp_assets, image_cues, overlay_cues
    ):
        """Missing images are skipped, assembly continues with remaining."""
        # Replace one path with nonexistent
        images = temp_assets["images"].copy()
        images[1] = Path("/nonexistent/img.png")

        assembler = ShortsAssembler(config)
        result = assembler.assemble(
            narration_path=temp_assets["narration"],
            image_paths=images,
            image_cues=image_cues,
            overlay_cues=overlay_cues,
            hook_text="HOOK",
            output_path=temp_assets["output"],
        )
        assert result.exists()

    def test_nonexistent_music_ignored(
        self, config, temp_assets, image_cues, overlay_cues
    ):
        """Nonexistent music path should not crash assembly."""
        assembler = ShortsAssembler(config)
        result = assembler.assemble(
            narration_path=temp_assets["narration"],
            image_paths=temp_assets["images"],
            image_cues=image_cues,
            overlay_cues=overlay_cues,
            hook_text="HOOK",
            output_path=temp_assets["output"],
            music_path=Path("/nonexistent/music.mp3"),
        )
        assert result.exists()

    def test_nonexistent_logo_ignored(
        self, config, temp_assets, image_cues, overlay_cues
    ):
        """Nonexistent logo path should not crash assembly."""
        assembler = ShortsAssembler(config)
        result = assembler.assemble(
            narration_path=temp_assets["narration"],
            image_paths=temp_assets["images"],
            image_cues=image_cues,
            overlay_cues=overlay_cues,
            hook_text="HOOK",
            output_path=temp_assets["output"],
            logo_path=Path("/nonexistent/logo.png"),
        )
        assert result.exists()

    def test_output_directory_created(
        self, config, temp_assets, image_cues, overlay_cues
    ):
        """Output parent dirs should be created if they don't exist."""
        output = temp_assets["output"].parent / "nested" / "dir" / "short.mp4"
        assembler = ShortsAssembler(config)
        result = assembler.assemble(
            narration_path=temp_assets["narration"],
            image_paths=temp_assets["images"],
            image_cues=image_cues,
            overlay_cues=overlay_cues,
            hook_text="HOOK",
            output_path=output,
        )
        assert result.exists()


class TestKenBurnsClip:
    """Test the make_frame Ken Burns implementation."""

    def test_ken_burns_produces_correct_size(self, config, tmp_path: Path):
        """Ken Burns clip frames should be exactly WxH."""
        from moviepy import VideoClip

        img_path = tmp_path / "test.png"
        Image.new("RGB", (1080, 1920), (50, 100, 150)).save(img_path)

        assembler = ShortsAssembler(config)
        clip = assembler._make_ken_burns_clip(
            img_path, 3.0, "zoom-in", 1080, 1920
        )

        assert clip.duration == 3.0
        # Check a frame at t=0 and t=2
        frame_0 = clip.get_frame(0)
        frame_mid = clip.get_frame(1.5)
        assert frame_0.shape == (1920, 1080, 3)
        assert frame_mid.shape == (1920, 1080, 3)
        clip.close()

    def test_ken_burns_zoom_in_differs_from_zoom_out(self, config, tmp_path: Path):
        """Zoom-in and zoom-out should produce different frame sequences."""
        import numpy as np

        img_path = tmp_path / "test.png"
        # Create an image with a strong gradient so zoom differences are visible
        arr = np.zeros((1920, 1080, 3), dtype=np.uint8)
        for y in range(1920):
            arr[y, :, 0] = int(y / 1920 * 255)  # red gradient top-bottom
            arr[y, :, 1] = int((1920 - y) / 1920 * 255)  # green inverse
        for x in range(1080):
            arr[:, x, 2] = int(x / 1080 * 255)  # blue gradient left-right
        Image.fromarray(arr).save(img_path)

        assembler = ShortsAssembler(config)
        clip_in = assembler._make_ken_burns_clip(img_path, 3.0, "zoom-in", 1080, 1920)
        clip_out = assembler._make_ken_burns_clip(img_path, 3.0, "zoom-out", 1080, 1920)

        # At t=0, zoom-in starts wide and zoom-out starts cropped — should differ
        frame_in = clip_in.get_frame(0)
        frame_out = clip_out.get_frame(0)
        assert not np.array_equal(frame_in, frame_out)
        clip_in.close()
        clip_out.close()
