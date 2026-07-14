"""Tests for the narration engine."""

import wave
from pathlib import Path

import pytest

from vidgen.config import VoiceConfig
from vidgen.models import NarrationResult, Scene, Script, ScriptSection
from vidgen.narration import NarrationEngine


# --- Fixtures ---


@pytest.fixture
def voice_config() -> VoiceConfig:
    return VoiceConfig(pace_wpm=150, sample_rate=24000, sentence_pause_ms=400)


@pytest.fixture
def engine(voice_config: VoiceConfig) -> NarrationEngine:
    return NarrationEngine(config=voice_config)


@pytest.fixture
def sample_scene() -> Scene:
    return Scene(
        id="scene-01",
        section_id="hook",
        image_prompt="A futuristic cityscape",
        ken_burns_direction="zoom-in",
        start_time=0.0,
        duration=10.0,
        transition_type="crossfade",
    )


@pytest.fixture
def sample_script() -> Script:
    return Script(
        title="Test Video",
        hook=ScriptSection(
            id="hook",
            title="Hook",
            narration_text="This is a hook sentence. It grabs attention quickly.",
        ),
        introduction=ScriptSection(
            id="intro",
            title="Introduction",
            narration_text="Welcome to the video. Today we explore AI.",
        ),
        body_sections=[
            ScriptSection(
                id="body-1",
                title="Main Point",
                narration_text="The main point is that AI is transforming everything.",
            ),
        ],
        conclusion=ScriptSection(
            id="conclusion",
            title="Conclusion",
            narration_text="In conclusion, the future is bright.",
        ),
        total_word_count=40,
    )


# --- Unit Tests ---


class TestNarrationEngineInit:
    def test_creates_with_config(self, voice_config: VoiceConfig) -> None:
        engine = NarrationEngine(config=voice_config)
        assert engine.config == voice_config
        assert engine._model is None
        assert engine._model_loaded is False

    def test_lazy_loading_deferred(self, engine: NarrationEngine) -> None:
        assert engine._model_loaded is False


class TestGenerateSceneAudio:
    def test_generates_wav_file(
        self, engine: NarrationEngine, sample_scene: Scene, tmp_path: Path
    ) -> None:
        output_path = tmp_path / "narration" / "scene-01.wav"
        text = "Hello world, this is a test sentence."

        result = engine.generate_scene_audio(sample_scene, text, output_path)

        assert output_path.exists()
        assert isinstance(result, NarrationResult)
        assert result.scene_id == "scene-01"
        assert result.audio_path == output_path

    def test_wav_file_format(
        self, engine: NarrationEngine, sample_scene: Scene, tmp_path: Path
    ) -> None:
        output_path = tmp_path / "scene-01.wav"
        text = "Testing the audio format output."

        engine.generate_scene_audio(sample_scene, text, output_path)

        with wave.open(str(output_path), "r") as wav_file:
            assert wav_file.getnchannels() == 1  # Mono
            assert wav_file.getsampwidth() == 2  # 16-bit
            assert wav_file.getframerate() == 24000

    def test_word_count_correct(
        self, engine: NarrationEngine, sample_scene: Scene, tmp_path: Path
    ) -> None:
        output_path = tmp_path / "scene-01.wav"
        text = "one two three four five"

        result = engine.generate_scene_audio(sample_scene, text, output_path)

        assert result.word_count == 5

    def test_duration_positive(
        self, engine: NarrationEngine, sample_scene: Scene, tmp_path: Path
    ) -> None:
        output_path = tmp_path / "scene-01.wav"
        text = "Some narration text for the scene."

        result = engine.generate_scene_audio(sample_scene, text, output_path)

        assert result.duration_seconds > 0

    def test_actual_wpm_calculated(
        self, engine: NarrationEngine, sample_scene: Scene, tmp_path: Path
    ) -> None:
        output_path = tmp_path / "scene-01.wav"
        text = "Some narration text for the scene."

        result = engine.generate_scene_audio(sample_scene, text, output_path)

        assert result.actual_wpm > 0

    def test_creates_parent_directories(
        self, engine: NarrationEngine, sample_scene: Scene, tmp_path: Path
    ) -> None:
        output_path = tmp_path / "deep" / "nested" / "dir" / "scene-01.wav"
        text = "Testing directory creation."

        engine.generate_scene_audio(sample_scene, text, output_path)

        assert output_path.exists()


class TestGenerateAll:
    def test_generates_all_scenes(
        self, engine: NarrationEngine, sample_script: Script, tmp_path: Path
    ) -> None:
        scenes = [
            Scene(
                id="scene-hook",
                section_id="hook",
                image_prompt="hook image",
                ken_burns_direction="zoom-in",
                start_time=0.0,
                duration=5.0,
                transition_type="crossfade",
            ),
            Scene(
                id="scene-intro",
                section_id="intro",
                image_prompt="intro image",
                ken_burns_direction="pan-left",
                start_time=5.0,
                duration=5.0,
                transition_type="crossfade",
            ),
        ]

        results = engine.generate_all(sample_script, scenes, tmp_path)

        assert len(results) == 2
        assert results[0].scene_id == "scene-hook"
        assert results[1].scene_id == "scene-intro"

    def test_skips_scenes_without_text(
        self, engine: NarrationEngine, sample_script: Script, tmp_path: Path
    ) -> None:
        scenes = [
            Scene(
                id="scene-orphan",
                section_id="nonexistent-section",
                image_prompt="orphan",
                ken_burns_direction="zoom-out",
                start_time=0.0,
                duration=5.0,
                transition_type="cut",
            ),
        ]

        results = engine.generate_all(sample_script, scenes, tmp_path)

        assert len(results) == 0

    def test_output_files_named_by_scene_id(
        self, engine: NarrationEngine, sample_script: Script, tmp_path: Path
    ) -> None:
        scenes = [
            Scene(
                id="scene-body-1",
                section_id="body-1",
                image_prompt="body image",
                ken_burns_direction="pan-right",
                start_time=10.0,
                duration=8.0,
                transition_type="crossfade",
            ),
        ]

        results = engine.generate_all(sample_script, scenes, tmp_path)

        assert results[0].audio_path == tmp_path / "scene-body-1.wav"


class TestReleaseModel:
    def test_release_resets_state(self, engine: NarrationEngine) -> None:
        engine._model_loaded = True
        engine._model = "fake_model"

        engine.release_model()

        assert engine._model is None
        assert engine._model_loaded is False

    def test_release_when_not_loaded(self, engine: NarrationEngine) -> None:
        # Should not raise
        engine.release_model()
        assert engine._model_loaded is False


class TestBuildSectionTextMap:
    def test_maps_all_sections(
        self, engine: NarrationEngine, sample_script: Script
    ) -> None:
        text_map = engine._build_section_text_map(sample_script)

        assert "hook" in text_map
        assert "intro" in text_map
        assert "body-1" in text_map
        assert "conclusion" in text_map
        assert len(text_map) == 4
