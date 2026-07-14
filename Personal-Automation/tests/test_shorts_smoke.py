"""Smoke tests for the shorts extractor module."""

from vidgen.shorts import ShortsExtractor
from vidgen.config import PipelineConfig
from vidgen.models import Script, ScenePlan, ScriptSection, Scene, ShortSegment


def test_shorts_extractor_instantiation():
    """Test that ShortsExtractor can be instantiated with default config."""
    config = PipelineConfig()
    extractor = ShortsExtractor(config)
    assert extractor.config == config


def test_identify_segments_body_section_within_bounds():
    """Test segment identification when a body section is 45-60s."""
    config = PipelineConfig()
    extractor = ShortsExtractor(config)

    hook = ScriptSection(
        id="hook", title="Hook",
        narration_text="Did you know AI is changing everything?",
        scene_marker="[SCENE: AI visual]",
    )
    intro = ScriptSection(
        id="intro", title="Introduction",
        narration_text="Welcome to our deep dive into AI trends.",
        scene_marker="[SCENE: Intro visual]",
    )
    body1 = ScriptSection(
        id="section_1", title="Trend One",
        narration_text="First trend is autonomous automation in business workflows.",
        scene_marker="[SCENE: Automation visual]",
    )
    body2 = ScriptSection(
        id="section_2", title="Trend Two",
        narration_text="Second trend is generative content creation tools.",
        scene_marker="[SCENE: Content visual]",
    )
    conclusion = ScriptSection(
        id="conclusion", title="Conclusion",
        narration_text="Thanks for watching and subscribe.",
        scene_marker="[SCENE: Outro]",
    )

    script = Script(
        title="AI Trends That Matter",
        hook=hook,
        introduction=intro,
        body_sections=[body1, body2],
        conclusion=conclusion,
        total_word_count=800,
    )

    scenes = [
        Scene(id="s001", section_id="hook", image_prompt="AI", ken_burns_direction="zoom-in", start_time=0.0, duration=10.0, transition_type="crossfade"),
        Scene(id="s002", section_id="intro", image_prompt="Intro", ken_burns_direction="pan-left", start_time=10.0, duration=15.0, transition_type="crossfade"),
        Scene(id="s003", section_id="section_1", image_prompt="Auto", ken_burns_direction="pan-right", start_time=25.0, duration=50.0, transition_type="crossfade"),
        Scene(id="s004", section_id="section_2", image_prompt="Content", ken_burns_direction="zoom-out", start_time=75.0, duration=55.0, transition_type="crossfade"),
        Scene(id="s005", section_id="conclusion", image_prompt="Outro", ken_burns_direction="zoom-in", start_time=130.0, duration=20.0, transition_type="cut"),
    ]

    plan = ScenePlan(
        video_title="AI Trends That Matter",
        style_prefix="professional tech illustration",
        scenes=scenes,
        total_duration=150.0,
    )

    segments = extractor.identify_segments(script, plan)

    # Should find body sections that are 45-60s
    assert len(segments) >= 1
    for seg in segments:
        assert 45 <= seg.duration <= 60
        assert seg.hook_caption  # non-empty caption
        assert seg.title  # non-empty title
        assert len(seg.tags) >= 15


def test_identify_segments_truncates_long_section():
    """Test that sections longer than 60s get truncated to max 60s."""
    config = PipelineConfig()
    extractor = ShortsExtractor(config)

    hook = ScriptSection(id="hook", title="Hook", narration_text="Hook text.", scene_marker="[SCENE: Hook]")
    intro = ScriptSection(id="intro", title="Intro", narration_text="Intro text.", scene_marker="[SCENE: Intro]")
    body1 = ScriptSection(id="section_1", title="Long Section", narration_text="This is a long section." * 20, scene_marker="[SCENE: Long]")
    conclusion = ScriptSection(id="conclusion", title="Conclusion", narration_text="End.", scene_marker="[SCENE: End]")

    script = Script(
        title="Testing Long Sections",
        hook=hook,
        introduction=intro,
        body_sections=[body1],
        conclusion=conclusion,
        total_word_count=500,
    )

    scenes = [
        Scene(id="s001", section_id="hook", image_prompt="Hook", ken_burns_direction="zoom-in", start_time=0.0, duration=8.0, transition_type="crossfade"),
        Scene(id="s002", section_id="intro", image_prompt="Intro", ken_burns_direction="pan-left", start_time=8.0, duration=12.0, transition_type="crossfade"),
        Scene(id="s003", section_id="section_1", image_prompt="Long1", ken_burns_direction="pan-right", start_time=20.0, duration=30.0, transition_type="crossfade"),
        Scene(id="s004", section_id="section_1", image_prompt="Long2", ken_burns_direction="zoom-out", start_time=50.0, duration=30.0, transition_type="crossfade"),
        Scene(id="s005", section_id="section_1", image_prompt="Long3", ken_burns_direction="pan-left", start_time=80.0, duration=30.0, transition_type="crossfade"),
        Scene(id="s006", section_id="conclusion", image_prompt="End", ken_burns_direction="zoom-in", start_time=110.0, duration=10.0, transition_type="cut"),
    ]

    plan = ScenePlan(
        video_title="Testing Long Sections",
        style_prefix="tech illustration",
        scenes=scenes,
        total_duration=120.0,
    )

    segments = extractor.identify_segments(script, plan)

    # Should extract a 60s subsection from the 90s body section
    for seg in segments:
        assert seg.duration <= 60


def test_identify_segments_returns_max_three():
    """Test that at most 3 segments are returned."""
    config = PipelineConfig()
    extractor = ShortsExtractor(config)

    hook = ScriptSection(id="hook", title="Hook", narration_text="Hook text.", scene_marker="[SCENE: Hook]")
    intro = ScriptSection(id="intro", title="Intro", narration_text="Intro text.", scene_marker="[SCENE: Intro]")
    # Create 5 body sections each exactly 50s
    body_sections = [
        ScriptSection(id=f"section_{i}", title=f"Section {i}", narration_text=f"Content for section {i}.", scene_marker=f"[SCENE: S{i}]")
        for i in range(1, 6)
    ]
    conclusion = ScriptSection(id="conclusion", title="Conclusion", narration_text="End.", scene_marker="[SCENE: End]")

    script = Script(
        title="Many Sections Test",
        hook=hook,
        introduction=intro,
        body_sections=body_sections,
        conclusion=conclusion,
        total_word_count=1000,
    )

    scenes = []
    t = 0.0
    scenes.append(Scene(id="s000", section_id="hook", image_prompt="Hook", ken_burns_direction="zoom-in", start_time=t, duration=10.0, transition_type="crossfade"))
    t += 10.0
    scenes.append(Scene(id="s001", section_id="intro", image_prompt="Intro", ken_burns_direction="pan-left", start_time=t, duration=15.0, transition_type="crossfade"))
    t += 15.0
    for i in range(1, 6):
        scenes.append(Scene(id=f"s{i+1:03d}", section_id=f"section_{i}", image_prompt=f"S{i}", ken_burns_direction="pan-right", start_time=t, duration=50.0, transition_type="crossfade"))
        t += 50.0
    scenes.append(Scene(id="s007", section_id="conclusion", image_prompt="End", ken_burns_direction="zoom-in", start_time=t, duration=10.0, transition_type="cut"))

    plan = ScenePlan(
        video_title="Many Sections Test",
        style_prefix="tech illustration",
        scenes=scenes,
        total_duration=t + 10.0,
    )

    segments = extractor.identify_segments(script, plan)
    assert len(segments) <= 3


def test_generate_hook_caption_truncation():
    """Test that hook captions are truncated to 50 chars."""
    config = PipelineConfig()
    extractor = ShortsExtractor(config)

    section = ScriptSection(
        id="test", title="Test",
        narration_text="This is a very long first sentence that exceeds fifty characters easily. And more text.",
        scene_marker="[SCENE: test]",
    )
    caption = extractor._generate_hook_caption(section)
    assert len(caption) <= 50


def test_generate_short_tags_count():
    """Test that generated tags are between 15-30."""
    config = PipelineConfig()
    extractor = ShortsExtractor(config)

    script = Script(
        title="AI Trends",
        hook=ScriptSection(id="hook", title="Hook", narration_text="Hook.", scene_marker="[SCENE: H]"),
        introduction=ScriptSection(id="intro", title="Intro", narration_text="Intro.", scene_marker="[SCENE: I]"),
        body_sections=[ScriptSection(id="s1", title="S1", narration_text="Body.", scene_marker="[SCENE: B]")],
        conclusion=ScriptSection(id="end", title="End", narration_text="End.", scene_marker="[SCENE: E]"),
        total_word_count=200,
    )

    tags = extractor._generate_short_tags(script)
    assert 15 <= len(tags) <= 30
    # All tags should be unique
    assert len(tags) == len(set(tags))
