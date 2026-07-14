"""Unit tests for vidgen.parsers module."""

from pathlib import Path

import pytest

from vidgen.parsers import (
    ParseError,
    parse_scene_plan,
    parse_script,
    serialize_scene_plan,
    serialize_script,
    validate_script_scene_alignment,
)


# --- Fixtures ---


MINIMAL_SCRIPT = """\
---
title: "Test Video"
topic_slug: test-video
target_duration_minutes: 5
niche: ai-tech-business
created_at: 2025-01-20T09:00:00Z
---

## Hook

[SCENE: A dramatic opening]

This is the hook text.

## Introduction

[SCENE: Clean intro visual]

This is the introduction.

## Section 1: Main Topic

[SCENE: Main visual]

This is the main body content with **emphasis** on key points.

## Conclusion

[SCENE: Closing visual]

This is the conclusion.
"""

MINIMAL_SCENE_PLAN = """\
{
  "video_title": "Test Video",
  "topic_slug": "test-video",
  "style_prefix": "Professional style",
  "total_duration_seconds": 120.0,
  "scenes": [
    {
      "id": "scene_001",
      "section_id": "hook",
      "image_prompt": "A dramatic scene",
      "ken_burns": "zoom-in",
      "start_time": 0.0,
      "duration": 10.0,
      "transition": "crossfade",
      "text_overlay": {
        "text": "Key Stat",
        "position": "bottom",
        "appear_at": 2.0,
        "duration": 5.0
      }
    },
    {
      "id": "scene_002",
      "section_id": "section_1",
      "image_prompt": "Main content visual",
      "ken_burns": "pan-right",
      "start_time": 10.0,
      "duration": 20.0,
      "transition": "crossfade",
      "text_overlay": null
    }
  ]
}
"""


@pytest.fixture
def script_file(tmp_path: Path) -> Path:
    p = tmp_path / "script.md"
    p.write_text(MINIMAL_SCRIPT, encoding="utf-8")
    return p


@pytest.fixture
def scene_plan_file(tmp_path: Path) -> Path:
    p = tmp_path / "scene_plan.json"
    p.write_text(MINIMAL_SCENE_PLAN, encoding="utf-8")
    return p


# --- parse_script tests ---


class TestParseScript:
    def test_parse_minimal_script(self, script_file: Path):
        script = parse_script(script_file)
        assert script.title == "Test Video"
        assert script.hook.id == "hook"
        assert script.introduction.id == "introduction"
        assert script.conclusion.id == "conclusion"
        assert len(script.body_sections) == 1
        assert script.body_sections[0].id == "section_1"

    def test_parse_scene_markers(self, script_file: Path):
        script = parse_script(script_file)
        assert script.hook.scene_marker == "A dramatic opening"
        assert script.introduction.scene_marker == "Clean intro visual"
        assert script.body_sections[0].scene_marker == "Main visual"
        assert script.conclusion.scene_marker == "Closing visual"

    def test_parse_narration_text_excludes_scene_marker(self, script_file: Path):
        script = parse_script(script_file)
        assert "[SCENE:" not in script.hook.narration_text
        assert "This is the hook text." in script.hook.narration_text

    def test_parse_emphasis_markers(self, script_file: Path):
        script = parse_script(script_file)
        section = script.body_sections[0]
        # "emphasis" should be marked
        assert len(section.emphasis_markers) == 1
        start, end = section.emphasis_markers[0]
        cleaned = section.narration_text.replace("**", "")
        # The emphasis should be around "emphasis"
        assert "emphasis" in cleaned[start:end]

    def test_parse_word_count(self, script_file: Path):
        script = parse_script(script_file)
        assert script.total_word_count > 0

    def test_parse_empty_file_raises(self, tmp_path: Path):
        p = tmp_path / "empty.md"
        p.write_text("", encoding="utf-8")
        with pytest.raises(ParseError, match="empty"):
            parse_script(p)

    def test_parse_missing_frontmatter_raises(self, tmp_path: Path):
        p = tmp_path / "no_fm.md"
        p.write_text("## Hook\n\nSome text\n", encoding="utf-8")
        with pytest.raises(ParseError, match="frontmatter"):
            parse_script(p)

    def test_parse_missing_file_raises(self, tmp_path: Path):
        p = tmp_path / "nonexistent.md"
        with pytest.raises(ParseError, match="Cannot read"):
            parse_script(p)

    def test_parse_missing_hook_raises(self, tmp_path: Path):
        p = tmp_path / "no_hook.md"
        p.write_text(
            "---\ntitle: Test\n---\n\n## Introduction\n\nText\n\n## Section 1: X\n\nBody\n\n## Conclusion\n\nEnd\n",
            encoding="utf-8",
        )
        with pytest.raises(ParseError, match="Hook"):
            parse_script(p)

    def test_parse_missing_body_section_raises(self, tmp_path: Path):
        p = tmp_path / "no_body.md"
        p.write_text(
            "---\ntitle: Test\n---\n\n## Hook\n\nHook text\n\n## Introduction\n\nIntro\n\n## Conclusion\n\nEnd\n",
            encoding="utf-8",
        )
        with pytest.raises(ParseError, match="at least one body"):
            parse_script(p)

    def test_parse_template_file(self):
        """Parse the actual template file to verify compatibility."""
        template_path = Path("templates/script_template.md")
        if template_path.exists():
            script = parse_script(template_path)
            assert script.title == "How AI Agents Are Quietly Replacing Traditional SaaS"
            assert len(script.body_sections) == 3
            assert script.body_sections[0].id == "section_1"
            assert script.body_sections[1].id == "section_2"
            assert script.body_sections[2].id == "section_3"


# --- serialize_script tests ---


class TestSerializeScript:
    def test_serialize_produces_valid_markdown(self, script_file: Path):
        script = parse_script(script_file)
        output = serialize_script(script)
        assert output.startswith("---\n")
        assert "## Hook" in output
        assert "## Conclusion" in output

    def test_serialize_includes_scene_markers(self, script_file: Path):
        script = parse_script(script_file)
        output = serialize_script(script)
        assert "[SCENE: A dramatic opening]" in output

    def test_round_trip_preserves_structure(self, script_file: Path, tmp_path: Path):
        """parse -> serialize -> parse should give the same structure."""
        original = parse_script(script_file)
        serialized = serialize_script(original)

        round_trip_path = tmp_path / "round_trip.md"
        round_trip_path.write_text(serialized, encoding="utf-8")
        reparsed = parse_script(round_trip_path)

        assert reparsed.title == original.title
        assert reparsed.hook.id == original.hook.id
        assert reparsed.hook.scene_marker == original.hook.scene_marker
        assert reparsed.introduction.id == original.introduction.id
        assert reparsed.conclusion.id == original.conclusion.id
        assert len(reparsed.body_sections) == len(original.body_sections)
        for orig_s, rep_s in zip(original.body_sections, reparsed.body_sections):
            assert rep_s.id == orig_s.id
            assert rep_s.scene_marker == orig_s.scene_marker


# --- parse_scene_plan tests ---


class TestParseScenePlan:
    def test_parse_minimal_plan(self, scene_plan_file: Path):
        plan = parse_scene_plan(scene_plan_file)
        assert plan.video_title == "Test Video"
        assert plan.style_prefix == "Professional style"
        assert plan.total_duration == 120.0
        assert len(plan.scenes) == 2

    def test_parse_scene_fields(self, scene_plan_file: Path):
        plan = parse_scene_plan(scene_plan_file)
        scene = plan.scenes[0]
        assert scene.id == "scene_001"
        assert scene.section_id == "hook"
        assert scene.ken_burns_direction == "zoom-in"
        assert scene.start_time == 0.0
        assert scene.duration == 10.0
        assert scene.transition_type == "crossfade"

    def test_parse_text_overlay(self, scene_plan_file: Path):
        plan = parse_scene_plan(scene_plan_file)
        overlay = plan.scenes[0].text_overlay
        assert overlay is not None
        assert overlay.text == "Key Stat"
        assert overlay.position == "bottom"
        assert overlay.appear_at == 2.0
        assert overlay.duration == 5.0

    def test_parse_null_text_overlay(self, scene_plan_file: Path):
        plan = parse_scene_plan(scene_plan_file)
        assert plan.scenes[1].text_overlay is None

    def test_parse_empty_file_raises(self, tmp_path: Path):
        p = tmp_path / "empty.json"
        p.write_text("", encoding="utf-8")
        with pytest.raises(ParseError, match="empty"):
            parse_scene_plan(p)

    def test_parse_invalid_json_raises(self, tmp_path: Path):
        p = tmp_path / "bad.json"
        p.write_text("{invalid json", encoding="utf-8")
        with pytest.raises(ParseError, match="Invalid JSON"):
            parse_scene_plan(p)

    def test_parse_missing_file_raises(self, tmp_path: Path):
        p = tmp_path / "nonexistent.json"
        with pytest.raises(ParseError, match="Cannot read"):
            parse_scene_plan(p)

    def test_parse_template_file(self):
        """Parse the actual template file to verify compatibility."""
        template_path = Path("templates/scene_plan_template.json")
        if template_path.exists():
            plan = parse_scene_plan(template_path)
            assert plan.video_title == "How AI Agents Are Quietly Replacing Traditional SaaS"
            assert len(plan.scenes) == 8


# --- serialize_scene_plan tests ---


class TestSerializeScenePlan:
    def test_serialize_produces_valid_json(self, scene_plan_file: Path):
        plan = parse_scene_plan(scene_plan_file)
        output = serialize_scene_plan(plan)
        import json

        data = json.loads(output)
        assert data["video_title"] == "Test Video"
        assert len(data["scenes"]) == 2

    def test_round_trip_preserves_structure(self, scene_plan_file: Path, tmp_path: Path):
        """parse -> serialize -> parse should give the same structure."""
        original = parse_scene_plan(scene_plan_file)
        serialized = serialize_scene_plan(original)

        round_trip_path = tmp_path / "round_trip.json"
        round_trip_path.write_text(serialized, encoding="utf-8")
        reparsed = parse_scene_plan(round_trip_path)

        assert reparsed.video_title == original.video_title
        assert reparsed.style_prefix == original.style_prefix
        assert reparsed.total_duration == original.total_duration
        assert len(reparsed.scenes) == len(original.scenes)
        for orig_s, rep_s in zip(original.scenes, reparsed.scenes):
            assert rep_s.id == orig_s.id
            assert rep_s.section_id == orig_s.section_id
            assert rep_s.ken_burns_direction == orig_s.ken_burns_direction
            assert rep_s.text_overlay == orig_s.text_overlay


# --- validate_script_scene_alignment tests ---


class TestValidateAlignment:
    def test_valid_alignment_returns_empty(self, script_file: Path, scene_plan_file: Path):
        script = parse_script(script_file)
        plan = parse_scene_plan(scene_plan_file)
        errors = validate_script_scene_alignment(script, plan)
        assert errors == []

    def test_invalid_section_id_returns_error(self, script_file: Path, tmp_path: Path):
        script = parse_script(script_file)

        # Create a plan with an invalid section_id
        bad_plan_data = """\
{
  "video_title": "Test",
  "topic_slug": "test",
  "style_prefix": "Style",
  "total_duration_seconds": 60.0,
  "scenes": [
    {
      "id": "scene_001",
      "section_id": "nonexistent_section",
      "image_prompt": "Prompt",
      "ken_burns": "zoom-in",
      "start_time": 0.0,
      "duration": 10.0,
      "transition": "crossfade",
      "text_overlay": null
    }
  ]
}
"""
        plan_path = tmp_path / "bad_plan.json"
        plan_path.write_text(bad_plan_data, encoding="utf-8")
        plan = parse_scene_plan(plan_path)

        errors = validate_script_scene_alignment(script, plan)
        assert len(errors) == 1
        assert "nonexistent_section" in errors[0]

    def test_multiple_invalid_sections(self, script_file: Path, tmp_path: Path):
        script = parse_script(script_file)

        bad_plan_data = """\
{
  "video_title": "Test",
  "topic_slug": "test",
  "style_prefix": "Style",
  "total_duration_seconds": 60.0,
  "scenes": [
    {
      "id": "scene_001",
      "section_id": "bad_1",
      "image_prompt": "Prompt",
      "ken_burns": "zoom-in",
      "start_time": 0.0,
      "duration": 10.0,
      "transition": "crossfade",
      "text_overlay": null
    },
    {
      "id": "scene_002",
      "section_id": "bad_2",
      "image_prompt": "Prompt",
      "ken_burns": "pan-left",
      "start_time": 10.0,
      "duration": 10.0,
      "transition": "cut",
      "text_overlay": null
    }
  ]
}
"""
        plan_path = tmp_path / "bad_plan.json"
        plan_path.write_text(bad_plan_data, encoding="utf-8")
        plan = parse_scene_plan(plan_path)

        errors = validate_script_scene_alignment(script, plan)
        assert len(errors) == 2
