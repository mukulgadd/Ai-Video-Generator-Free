"""Core data models for the vidgen pipeline.

Uses Pydantic BaseModel for automatic validation. All models are immutable
by default (frozen=True) to prevent accidental mutation during pipeline stages.
"""

from pathlib import Path

from pydantic import BaseModel, Field, computed_field, field_validator, model_validator

# --- Constants ---

VALID_KEN_BURNS = [
    "pan-left", "pan-right", "pan-up", "pan-down",
    "zoom-in", "zoom-out",
    "diagonal-tl", "diagonal-br",
]
VALID_TRANSITIONS = ["crossfade", "cut"]
VALID_POSITIONS = ["top", "center", "bottom"]
VALID_JOB_STATUSES = ["queued", "in-progress", "completed", "failed", "timed-out"]
VALID_QUEUE_STATUSES = ["queued", "in-progress", "completed", "failed"]

DEFAULT_PACE_WPM = 150  # Words per minute for duration estimation


# --- Script Models ---


class ScriptSection(BaseModel):
    """A single section of a video script (hook, intro, body, conclusion)."""

    id: str
    title: str
    narration_text: str
    scene_marker: str = ""
    emphasis_markers: list[tuple[int, int]] = Field(default_factory=list)

    @field_validator("emphasis_markers")
    @classmethod
    def validate_emphasis_markers(cls, v: list[tuple[int, int]]) -> list[tuple[int, int]]:
        for start, end in v:
            if start < 0 or end < 0:
                raise ValueError(f"Emphasis marker positions must be non-negative, got ({start}, {end})")
            if start >= end:
                raise ValueError(f"Emphasis marker start must be less than end, got ({start}, {end})")
        return v


class Script(BaseModel):
    """A complete video script with all sections and metadata."""

    title: str
    hook: ScriptSection
    introduction: ScriptSection
    body_sections: list[ScriptSection]
    conclusion: ScriptSection
    total_word_count: int = Field(ge=0)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def estimated_duration_seconds(self) -> float:
        """Estimated narration duration based on word count and average pace."""
        return self.total_word_count / DEFAULT_PACE_WPM * 60

    @field_validator("body_sections")
    @classmethod
    def validate_body_sections(cls, v: list[ScriptSection]) -> list[ScriptSection]:
        if len(v) < 1:
            raise ValueError("Script must have at least 1 body section")
        return v


# --- Scene Planning Models ---


class TextOverlay(BaseModel):
    """Text overlay specification for a scene."""

    text: str
    position: str
    appear_at: float = Field(ge=0.0)
    duration: float = Field(gt=0.0)

    @field_validator("position")
    @classmethod
    def validate_position(cls, v: str) -> str:
        if v not in VALID_POSITIONS:
            raise ValueError(f"Position must be one of {VALID_POSITIONS}, got '{v}'")
        return v


class Scene(BaseModel):
    """A single scene in the video with image prompt, timing, and effects."""

    id: str
    section_id: str
    image_prompt: str
    image_prompts: list[str] = Field(default_factory=list)  # Multiple prompts for sub-images (if empty, image_prompt is repeated)
    style_prefix: str = ""
    ken_burns_direction: str
    start_time: float = Field(ge=0.0)
    duration: float = Field(gt=0.0)
    transition_type: str
    text_overlay: TextOverlay | None = None
    text_overlays: list[TextOverlay] = Field(default_factory=list)  # Multiple overlays per scene (distributed across sub-images)
    music_mood: str | None = None  # "tension", "momentum", "neutral", "resolve"
    visual_type: str = "scene"  # "scene" (normal MFLUX) or "data" (MFLUX bg + chart overlay)
    data_overlay: dict | None = None  # Data viz config when visual_type == "data"

    @model_validator(mode="after")
    def _normalize_overlays(self) -> "Scene":
        """Ensure text_overlays is populated from text_overlay if needed."""
        if not self.text_overlays and self.text_overlay:
            object.__setattr__(self, "text_overlays", [self.text_overlay])
        return self

    @field_validator("ken_burns_direction")
    @classmethod
    def validate_ken_burns(cls, v: str) -> str:
        if v not in VALID_KEN_BURNS:
            raise ValueError(f"ken_burns_direction must be one of {VALID_KEN_BURNS}, got '{v}'")
        return v

    @field_validator("transition_type")
    @classmethod
    def validate_transition(cls, v: str) -> str:
        if v not in VALID_TRANSITIONS:
            raise ValueError(f"transition_type must be one of {VALID_TRANSITIONS}, got '{v}'")
        return v


class ScenePlan(BaseModel):
    """Complete scene plan for a video, containing all scenes with timing."""

    video_title: str
    topic_slug: str = ""
    style_prefix: str
    scenes: list[Scene]
    total_duration: float = Field(gt=0.0)

    # Thumbnail hints (optional — auto-derived from title if not set)
    thumbnail_text: str | None = None
    thumbnail_accent_word: str | None = None
    thumbnail_accent_color: str | None = None

    # A/B variant (opportunity angle — auto-uses gold if not set)
    thumbnail_text_alt: str | None = None
    thumbnail_accent_word_alt: str | None = None
    thumbnail_accent_color_alt: str | None = None

    @field_validator("scenes")
    @classmethod
    def validate_scenes_not_empty(cls, v: list[Scene]) -> list[Scene]:
        if len(v) == 0:
            raise ValueError("ScenePlan must have at least one scene")
        return v


# --- Job State Models ---


class JobState(BaseModel):
    """Tracks the state of a pipeline job for persistence and resumability."""

    job_id: str
    status: str
    current_stage: str | None = None
    completed_stages: list[str] = Field(default_factory=list)
    stage_timings: dict[str, float] = Field(default_factory=dict)
    started_at: str | None = None
    completed_at: str | None = None
    error: str | None = None
    artifacts: dict[str, list[str]] = Field(default_factory=dict)
    estimated_total_seconds: float | None = None
    timeout_seconds: float | None = None

    @field_validator("status")
    @classmethod
    def validate_status(cls, v: str) -> str:
        if v not in VALID_JOB_STATUSES:
            raise ValueError(f"Job status must be one of {VALID_JOB_STATUSES}, got '{v}'")
        return v


class QueueEntry(BaseModel):
    """A single entry in the processing queue."""

    job_id: str
    script_path: str
    scene_plan_path: str
    priority: int = 0
    status: str
    created_at: str
    started_at: str | None = None
    completed_at: str | None = None
    error: str | None = None
    output_path: str | None = None

    @field_validator("status")
    @classmethod
    def validate_status(cls, v: str) -> str:
        if v not in VALID_QUEUE_STATUSES:
            raise ValueError(f"Queue entry status must be one of {VALID_QUEUE_STATUSES}, got '{v}'")
        return v


class QueueState(BaseModel):
    """Persisted state of the processing queue."""

    entries: list[QueueEntry] = Field(default_factory=list)
    last_updated: str


# --- Video Output Models ---


class VideoMetadata(BaseModel):
    """Metadata for a generated video, used in output packaging."""

    title: str = Field(max_length=100)
    description: str = Field(max_length=5000)
    tags: list[str]
    chapters: list[tuple[str, str]] = Field(default_factory=list)
    category: str = "Science & Technology"
    duration_seconds: float = Field(gt=0.0)
    resolution: str
    file_path: str
    publish_date: str | None = None

    @field_validator("tags")
    @classmethod
    def validate_tags_count(cls, v: list[str]) -> list[str]:
        if len(v) < 5 or len(v) > 30:
            raise ValueError(f"Tags must have 5-30 items, got {len(v)}")
        return v


class QualityCheck(BaseModel):
    """Result of a single quality check."""

    name: str
    passed: bool
    details: str
    affected_asset: str | None = None


class QualityReport(BaseModel):
    """Aggregated quality report from all checks."""

    passed: bool
    checks: list[QualityCheck]
    timestamp: str

    @model_validator(mode="after")
    def validate_passed_consistency(self) -> "QualityReport":
        """Ensure passed=True only if all individual checks passed."""
        all_passed = all(check.passed for check in self.checks)
        if self.passed and not all_passed:
            raise ValueError("Report cannot be marked passed when individual checks have failed")
        return self


# --- Generation Result Models ---


class NarrationResult(BaseModel):
    """Result of narrating a single scene."""

    scene_id: str
    audio_path: Path
    duration_seconds: float = Field(gt=0.0)
    word_count: int = Field(ge=0)
    actual_wpm: float = Field(ge=0.0)


class ImageResult(BaseModel):
    """Result of generating a single image."""

    scene_id: str | None = None
    image_path: Path
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    generation_time_seconds: float = Field(ge=0.0)
    seed_used: int


class AssemblyResult(BaseModel):
    """Result of video assembly."""

    video_path: Path
    duration_seconds: float = Field(gt=0.0)
    resolution: tuple[int, int]
    fps: int = Field(gt=0)
    file_size_bytes: int = Field(gt=0)


# --- Shorts Models ---


class ShortSegment(BaseModel):
    """A segment identified for extraction as a YouTube Short."""

    start_time: float = Field(ge=0.0)
    end_time: float = Field(gt=0.0)
    duration: float = Field(gt=0.0)
    source_scenes: list[str]
    hook_caption: str
    title: str
    description: str
    tags: list[str]

    @model_validator(mode="after")
    def validate_timing(self) -> "ShortSegment":
        """Ensure end_time > start_time and duration is consistent."""
        if self.end_time <= self.start_time:
            raise ValueError(
                f"end_time ({self.end_time}) must be greater than start_time ({self.start_time})"
            )
        return self


class ShortResult(BaseModel):
    """Result of extracting a single YouTube Short."""

    video_path: Path
    metadata: VideoMetadata
    duration_seconds: float = Field(gt=0.0)


# --- YouTube Shorts Pipeline Models ---

VALID_OVERLAY_STYLES = ["impact", "normal", "danger"]
VALID_MUSIC_MOODS = ["tension", "momentum", "neutral", "resolve"]


class OverlayCue(BaseModel):
    """A single bold text overlay appearing at a specific time in a Short."""

    text: str = Field(max_length=50)
    start_time: float = Field(ge=0.0)
    duration: float = Field(gt=0.0, default=4.0)
    style: str = "normal"

    @field_validator("style")
    @classmethod
    def validate_style(cls, v: str) -> str:
        if v not in VALID_OVERLAY_STYLES:
            raise ValueError(f"Style must be one of {VALID_OVERLAY_STYLES}, got '{v}'")
        return v


class ImageCue(BaseModel):
    """A single image to display at a specific time in a Short."""

    prompt: str
    start_time: float = Field(ge=0.0)
    duration: float = Field(gt=0.0)
    visual_type: str = "scene"  # "scene" or "data"
    data_overlay: dict | None = None  # Chart config when visual_type == "data"


class ShortScript(BaseModel):
    """Complete script for a YouTube Short (30-60s vertical video)."""

    title: str
    duration_target: int = Field(ge=30, le=60)
    source_video: str = ""
    music_mood: str = "neutral"
    music_track: str | None = None
    style_prefix: str = ""
    hook_text: str = Field(max_length=60)
    narration_text: str
    overlay_cues: list[OverlayCue] = Field(default_factory=list)
    image_cues: list[ImageCue] = Field(default_factory=list)

    @field_validator("music_mood")
    @classmethod
    def validate_music_mood(cls, v: str) -> str:
        if v not in VALID_MUSIC_MOODS:
            raise ValueError(f"music_mood must be one of {VALID_MUSIC_MOODS}, got '{v}'")
        return v

    @field_validator("narration_text")
    @classmethod
    def validate_narration_length(cls, v: str) -> str:
        word_count = len(v.split())
        if word_count < 30 or word_count > 200:
            raise ValueError(
                f"Narration text must be 30-200 words, got {word_count}"
            )
        return v

    @field_validator("image_cues")
    @classmethod
    def validate_image_count(cls, v: list[ImageCue]) -> list[ImageCue]:
        if len(v) < 2 or len(v) > 10:
            raise ValueError(f"Must have 2-10 image cues, got {len(v)}")
        return v

    @computed_field
    @property
    def estimated_duration(self) -> float:
        """Estimated narration duration at 150 WPM."""
        return len(self.narration_text.split()) / DEFAULT_PACE_WPM * 60


class ShortGenerationResult(BaseModel):
    """Result of generating a YouTube Short."""

    success: bool
    output_path: Path | None = None
    duration_seconds: float = 0.0
    error: str | None = None
