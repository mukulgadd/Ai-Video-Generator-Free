"""Tests for the shorts parser module — parsing markdown + YAML frontmatter."""

import tempfile
from pathlib import Path

import pytest

from vidgen.models import ImageCue, OverlayCue, ShortScript
from vidgen.shorts_parser import ShortParseError, parse_short_script, validate_short_script


VALID_SCRIPT = """\
---
title: "Test Short Title"
duration_target: 35
source_video: "003"
music_mood: "tension"
style_prefix: "dark cinematic, dramatic lighting"
hook_text: "Hook text here"
---

## Narration

This is the narration text for the test short video. It needs to be at least thirty words to pass validation. Here we add some more words to ensure we hit the minimum word count requirement for the narration text field in the model. These extra words help.

## Text Overlays

| time | text | style |
|------|------|-------|
| 0 | First overlay | impact |
| 5 | Second overlay | normal |
| 10 | Third overlay | danger |

## Images

- A dramatic scene with vertical composition, no text, no numbers, no readable labels
- A second dramatic scene with blue lighting, vertical composition, no text, no numbers, no readable labels
- A third scene with red accents, vertical composition, no text, no numbers, no readable labels
"""


@pytest.fixture
def valid_script_path(tmp_path: Path) -> Path:
    """Write a valid short script to a temp file."""
    path = tmp_path / "test_short.md"
    path.write_text(VALID_SCRIPT)
    return path


class TestParseShortScript:
    """Test parse_short_script() with valid and invalid inputs."""

    def test_parse_valid_script(self, valid_script_path: Path):
        script = parse_short_script(valid_script_path)
        assert script.title == "Test Short Title"
        assert script.duration_target == 35
        assert script.source_video == "003"
        assert script.music_mood == "tension"
        assert script.hook_text == "Hook text here"
        assert len(script.overlay_cues) == 3
        assert len(script.image_cues) == 3

    def test_overlay_cues_parsed_correctly(self, valid_script_path: Path):
        script = parse_short_script(valid_script_path)
        assert script.overlay_cues[0].text == "First overlay"
        assert script.overlay_cues[0].start_time == 0.0
        assert script.overlay_cues[0].style == "impact"
        assert script.overlay_cues[1].start_time == 5.0
        assert script.overlay_cues[2].style == "danger"

    def test_image_cues_auto_timed(self, valid_script_path: Path):
        script = parse_short_script(valid_script_path)
        # 3 images across 35s duration_target
        expected_dur = 35.0 / 3
        for i, cue in enumerate(script.image_cues):
            assert abs(cue.start_time - i * expected_dur) < 0.1
            assert abs(cue.duration - expected_dur) < 0.1

    def test_narration_cleaned(self, valid_script_path: Path):
        script = parse_short_script(valid_script_path)
        assert "**" not in script.narration_text
        assert "[SCENE:" not in script.narration_text
        assert len(script.narration_text.split()) >= 30

    def test_style_prefix_preserved(self, valid_script_path: Path):
        script = parse_short_script(valid_script_path)
        assert script.style_prefix == "dark cinematic, dramatic lighting"

    def test_missing_frontmatter_raises(self, tmp_path: Path):
        path = tmp_path / "bad.md"
        path.write_text("# No YAML here\nJust markdown")
        with pytest.raises(ShortParseError, match="frontmatter"):
            parse_short_script(path)

    def test_empty_file_raises(self, tmp_path: Path):
        path = tmp_path / "empty.md"
        path.write_text("")
        with pytest.raises(ShortParseError, match="empty"):
            parse_short_script(path)

    def test_missing_narration_section_raises(self, tmp_path: Path):
        path = tmp_path / "no_narration.md"
        path.write_text("""\
---
title: "X"
duration_target: 35
hook_text: "Y"
---

## Images

- A test prompt one, no text, no numbers, no readable labels
- A test prompt two, no text, no numbers, no readable labels
""")
        with pytest.raises(ShortParseError, match="Narration"):
            parse_short_script(path)

    def test_missing_images_section_raises(self, tmp_path: Path):
        path = tmp_path / "no_images.md"
        path.write_text("""\
---
title: "X"
duration_target: 35
hook_text: "Y"
---

## Narration

This is the narration text for the short. It needs enough words to pass the word count validation. We add padding words here to make sure.
""")
        with pytest.raises(ShortParseError, match="Images"):
            parse_short_script(path)

    def test_too_few_images_raises(self, tmp_path: Path):
        path = tmp_path / "one_image.md"
        path.write_text("""\
---
title: "X"
duration_target: 35
hook_text: "Y"
---

## Narration

This is the narration text for the short. It needs enough words to pass the word count validation. We add padding words here to make sure.

## Images

- Only one image prompt here, no text, no numbers, no readable labels
""")
        with pytest.raises(ShortParseError, match="at least 2"):
            parse_short_script(path)

    def test_invalid_yaml_raises(self, tmp_path: Path):
        path = tmp_path / "bad_yaml.md"
        path.write_text("""\
---
title: [unterminated
---

## Narration

Words words words words words words words words words words words words words words words words words words words words words words words words words words words words words words.

## Images

- A image one, no text, no numbers, no readable labels
- A image two, no text, no numbers, no readable labels
""")
        with pytest.raises(ShortParseError, match="YAML"):
            parse_short_script(path)

    def test_nonexistent_file_raises(self, tmp_path: Path):
        path = tmp_path / "nope.md"
        with pytest.raises(ShortParseError, match="Cannot read"):
            parse_short_script(path)

    def test_optional_overlays_section(self, tmp_path: Path):
        """Text Overlays is optional — script should parse without it."""
        path = tmp_path / "no_overlays.md"
        path.write_text("""\
---
title: "No Overlays"
duration_target: 35
hook_text: "Hook"
---

## Narration

This is the narration text for the short. It needs enough words to pass the word count validation. We add padding words here to make sure we get above thirty words total for the narration text.

## Images

- A image one, no text, no numbers, no readable labels
- A image two, no text, no numbers, no readable labels
- A image three, no text, no numbers, no readable labels
""")
        script = parse_short_script(path)
        assert script.overlay_cues == []

    def test_bold_markdown_stripped_from_narration(self, tmp_path: Path):
        path = tmp_path / "bold.md"
        path.write_text("""\
---
title: "Bold Test"
duration_target: 35
hook_text: "Hook"
---

## Narration

This has **bold text** in it and also some [SCENE: marker] stuff. We need thirty plus words here to pass validation so adding more context for the narration. Here are even more words to be safe and get well above the minimum count.

## Images

- A image one, no text, no numbers, no readable labels
- A image two, no text, no numbers, no readable labels
- A image three, no text, no numbers, no readable labels
""")
        script = parse_short_script(path)
        assert "**" not in script.narration_text
        assert "[SCENE:" not in script.narration_text
        assert "bold text" in script.narration_text


class TestValidateShortScript:
    """Test validate_short_script() timing and consistency checks."""

    def test_valid_script_no_errors(self, valid_script_path: Path):
        script = parse_short_script(valid_script_path)
        errors = validate_short_script(script)
        assert errors == []

    def test_overlay_exceeding_duration_warns(self):
        script = ShortScript(
            title="Test",
            duration_target=35,
            hook_text="Hook",
            narration_text=" ".join(["word"] * 50),
            overlay_cues=[
                OverlayCue(text="Late overlay", start_time=38.0, duration=4.0, style="normal"),
            ],
            image_cues=[
                ImageCue(prompt="p1", start_time=0, duration=12),
                ImageCue(prompt="p2", start_time=12, duration=12),
                ImageCue(prompt="p3", start_time=24, duration=11),
            ],
        )
        errors = validate_short_script(script)
        assert any("ends at" in e for e in errors)

    def test_insufficient_image_coverage_warns(self):
        script = ShortScript(
            title="Test",
            duration_target=35,
            hook_text="Hook",
            narration_text=" ".join(["word"] * 50),
            overlay_cues=[],
            image_cues=[
                ImageCue(prompt="p1", start_time=0, duration=5),
                ImageCue(prompt="p2", start_time=5, duration=5),
                ImageCue(prompt="p3", start_time=10, duration=5),
            ],
        )
        errors = validate_short_script(script)
        # 15s of images for 35s target = 43% < 70%
        assert any("covers less than 70%" in e for e in errors)


class TestParseRealScripts:
    """Test that the actual scripts in shorts/ directory parse correctly."""

    @pytest.mark.parametrize("filename", [
        "003_short_1_thin_wrapper.md",
        "003_short_2_capability_treadmill.md",
        "003_short_3_survival_signs.md",
    ])
    def test_real_script_parses(self, filename: str):
        path = Path("shorts") / filename
        if not path.exists():
            pytest.skip(f"Script not found: {path}")
        script = parse_short_script(path)
        assert script.title
        assert 30 <= script.duration_target <= 60
        assert len(script.image_cues) >= 2
        assert len(script.narration_text.split()) >= 30
