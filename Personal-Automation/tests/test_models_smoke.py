"""Smoke test to validate all models instantiate correctly."""

from pathlib import Path

from vidgen.models import (
    VALID_JOB_STATUSES,
    VALID_KEN_BURNS,
    VALID_POSITIONS,
    VALID_TRANSITIONS,
    DEFAULT_PACE_WPM,
    AssemblyResult,
    ImageResult,
    JobState,
    NarrationResult,
    QualityCheck,
    QualityReport,
    QueueEntry,
    QueueState,
    Scene,
    ScenePlan,
    Script,
    ScriptSection,
    ShortResult,
    ShortSegment,
    TextOverlay,
    VideoMetadata,
)


def test_script_section():
    section = ScriptSection(id="hook", title="Hook", narration_text="Hello world", scene_marker="visual")
    assert section.id == "hook"
    assert section.emphasis_markers == []


def test_script_computed_duration():
    script = Script(
        title="Test Video",
        hook=ScriptSection(id="hook", title="Hook", narration_text="Opening", scene_marker="visual"),
        introduction=ScriptSection(id="intro", title="Intro", narration_text="Intro", scene_marker="visual"),
        body_sections=[ScriptSection(id="s1", title="S1", narration_text="Body", scene_marker="visual")],
        conclusion=ScriptSection(id="end", title="End", narration_text="Outro", scene_marker="visual"),
        total_word_count=750,
    )
    expected = 750 / DEFAULT_PACE_WPM * 60
    assert script.estimated_duration_seconds == expected


def test_text_overlay():
    overlay = TextOverlay(text="73% by 2025", position="bottom", appear_at=3.0, duration=5.0)
    assert overlay.position == "bottom"


def test_scene():
    scene = Scene(
        id="scene_001",
        section_id="hook",
        image_prompt="A dramatic scene",
        ken_burns_direction="zoom-in",
        start_time=0.0,
        duration=14.0,
        transition_type="crossfade",
    )
    assert scene.ken_burns_direction == "zoom-in"


def test_scene_plan_with_topic_slug():
    scene = Scene(
        id="scene_001",
        section_id="hook",
        image_prompt="prompt",
        ken_burns_direction="pan-left",
        start_time=0.0,
        duration=14.0,
        transition_type="cut",
    )
    plan = ScenePlan(
        video_title="Test", topic_slug="test-video", style_prefix="clean", scenes=[scene], total_duration=420.0
    )
    assert plan.topic_slug == "test-video"


def test_job_state():
    state = JobState(job_id="abc-123", status="queued")
    assert state.completed_stages == []
    assert state.artifacts == {}


def test_queue_entry():
    entry = QueueEntry(
        job_id="abc-123",
        script_path="/path/script.md",
        scene_plan_path="/path/plan.json",
        status="queued",
        created_at="2024-01-15T10:00:00Z",
    )
    assert entry.priority == 0


def test_queue_state():
    entry = QueueEntry(
        job_id="abc-123",
        script_path="/path/script.md",
        scene_plan_path="/path/plan.json",
        status="queued",
        created_at="2024-01-15T10:00:00Z",
    )
    queue = QueueState(entries=[entry], last_updated="2024-01-15T10:00:00Z")
    assert len(queue.entries) == 1


def test_video_metadata():
    meta = VideoMetadata(
        title="Short Title",
        description="A description",
        tags=["tag" + str(i) for i in range(15)],
        chapters=[("0:00", "Intro"), ("1:30", "Section 1")],
        duration_seconds=420.0,
        resolution="1920x1080",
        file_path="output/video.mp4",
    )
    assert len(meta.tags) == 15


def test_quality_report():
    check = QualityCheck(name="duration", passed=True, details="OK")
    report = QualityReport(passed=True, checks=[check], timestamp="2024-01-15T22:00:00Z")
    assert report.passed is True


def test_narration_result():
    narr = NarrationResult(
        scene_id="scene_001",
        audio_path=Path("narration/scene_001.wav"),
        duration_seconds=14.0,
        word_count=35,
        actual_wpm=150.0,
    )
    assert narr.audio_path == Path("narration/scene_001.wav")


def test_image_result():
    img = ImageResult(
        scene_id="scene_001",
        image_path=Path("images/scene_001.png"),
        width=1920,
        height=1080,
        generation_time_seconds=45.0,
        seed_used=42,
    )
    assert img.width == 1920


def test_assembly_result():
    asm = AssemblyResult(
        video_path=Path("assembly/video.mp4"),
        duration_seconds=420.0,
        resolution=(1920, 1080),
        fps=30,
        file_size_bytes=500_000_000,
    )
    assert asm.resolution == (1920, 1080)


def test_short_segment():
    seg = ShortSegment(
        start_time=88.0,
        end_time=148.0,
        duration=60.0,
        source_scenes=["scene_005", "scene_006"],
        hook_caption="AI agents are coming",
        title="Short Title",
        description="Short desc",
        tags=["ai", "tech"],
    )
    assert seg.duration == 60.0


def test_short_result():
    meta = VideoMetadata(
        title="Short Title",
        description="A description",
        tags=["tag" + str(i) for i in range(15)],
        duration_seconds=60.0,
        resolution="1080x1920",
        file_path="shorts/short_1.mp4",
    )
    short = ShortResult(video_path=Path("shorts/short_1.mp4"), metadata=meta, duration_seconds=60.0)
    assert short.duration_seconds == 60.0


def test_constants():
    assert VALID_KEN_BURNS == [
        "pan-left", "pan-right", "pan-up", "pan-down",
        "zoom-in", "zoom-out",
        "diagonal-tl", "diagonal-br",
    ]
    assert VALID_TRANSITIONS == ["crossfade", "cut"]
    assert VALID_POSITIONS == ["top", "center", "bottom"]
    assert VALID_JOB_STATUSES == ["queued", "in-progress", "completed", "failed", "timed-out"]
    assert DEFAULT_PACE_WPM == 150
