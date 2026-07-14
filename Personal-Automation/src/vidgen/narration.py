"""Narration engine - wraps Qwen3-TTS via mlx-audio for local speech generation."""

import logging
import os
import struct
import time
import wave
from pathlib import Path

import numpy as np

from vidgen.config import VoiceConfig
from vidgen.models import NarrationResult, Scene, Script

logger = logging.getLogger(__name__)


class NarrationEngine:
    """Generates speech audio using Qwen3-TTS via mlx-audio on Apple Silicon.

    Runs natively on Metal via MLX framework. The model is loaded lazily
    on first use and released explicitly via release_model().
    """

    def __init__(self, config: VoiceConfig) -> None:
        self.config = config
        self._model = None
        self._model_loaded = False

    def _ensure_model_loaded(self) -> None:
        """Lazily load the TTS model (only needed for qwen3 engine)."""
        if self._model_loaded:
            return

        # In placeholder mode (tests), skip real model loading
        if os.environ.get("VIDGEN_PLACEHOLDER", "0") == "1":
            self._model_loaded = True
            logger.info("TTS model: placeholder mode (VIDGEN_PLACEHOLDER=1)")
            return

        # edge-tts is cloud-based, no local model needed
        if self.config.engine == "edge-tts":
            self._model_loaded = True
            logger.info(f"TTS engine: edge-tts (voice: {self.config.voice}, rate: {self.config.rate})")
            return

        logger.info(f"Loading TTS model: {self.config.model}")
        try:
            from mlx_audio.tts.utils import load_model
            self._model = load_model(self.config.model)
            self._model_loaded = True
            logger.info(f"TTS model loaded. Speaker: {self.config.speaker}")
        except ImportError:
            logger.warning("mlx-audio not installed — falling back to placeholder")
            self._model_loaded = True
        except Exception as e:
            logger.error(f"Failed to load TTS model: {e}")
            self._model_loaded = True  # Mark loaded to avoid retry loops

    def generate_scene_audio(
        self, scene: Scene, text: str, output_path: Path
    ) -> NarrationResult:
        """Generate audio for a single scene.

        Args:
            scene: The scene metadata (for ID and timing context).
            text: The narration text to synthesize.
            output_path: Where to write the .wav file.

        Returns:
            NarrationResult with audio path, duration, and WPM stats.
        """
        self._ensure_model_loaded()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        start_time = time.time()
        word_count = len(text.split())

        # Generate audio using the TTS model
        duration_seconds = self._synthesize(text, output_path)

        generation_time = time.time() - start_time
        actual_wpm = word_count / duration_seconds * 60 if duration_seconds > 0 else 0

        logger.info(
            f"Generated narration for {scene.id}: "
            f"{duration_seconds:.1f}s, {actual_wpm:.0f} WPM, "
            f"took {generation_time:.1f}s"
        )

        return NarrationResult(
            scene_id=scene.id,
            audio_path=output_path,
            duration_seconds=duration_seconds,
            word_count=word_count,
            actual_wpm=actual_wpm,
        )

    def generate_all(
        self, script: Script, scenes: list[Scene], output_dir: Path
    ) -> list[NarrationResult]:
        """Generate narration for all scenes sequentially.

        When multiple scenes share the same section_id, the section's narration
        text is split proportionally across those scenes based on their planned
        durations, so each scene gets a unique portion of the narration.

        Args:
            script: The full script (provides narration text per section).
            scenes: List of scenes to generate audio for.
            output_dir: Directory to write .wav files into.

        Returns:
            List of NarrationResult objects, one per scene.
        """
        self._ensure_model_loaded()
        output_dir.mkdir(parents=True, exist_ok=True)
        results: list[NarrationResult] = []

        # Build section ID -> narration text mapping
        section_texts = self._build_section_text_map(script)

        # Split text across scenes sharing the same section_id
        scene_texts = self._distribute_text_to_scenes(scenes, section_texts)

        for i, scene in enumerate(scenes):
            text = scene_texts.get(scene.id, "")
            if not text:
                logger.warning(
                    f"No narration text found for scene {scene.id} "
                    f"(section: {scene.section_id})"
                )
                continue

            output_path = output_dir / f"{scene.id}.wav"
            result = self.generate_scene_audio(scene, text, output_path)
            results.append(result)

            logger.info(f"Completed {i + 1}/{len(scenes)} narration segments")

        return results

    def release_model(self) -> None:
        """Explicitly release GPU memory used by the TTS model."""
        if self._model is not None:
            del self._model
            self._model = None
        self._model_loaded = False
        logger.info("TTS model released")

    def _synthesize(self, text: str, output_path: Path) -> float:
        """Synthesize text to audio file.

        Routes to edge-tts (cloud) or Qwen3-TTS (local) based on config.engine.
        Falls back to placeholder WAV when in test mode.
        """
        # Placeholder mode for tests
        if os.environ.get("VIDGEN_PLACEHOLDER", "0") == "1" or self._model is None and self.config.engine == "qwen3":
            if os.environ.get("VIDGEN_PLACEHOLDER", "0") == "1":
                return self._generate_placeholder(text, output_path)

        # Apply strategic pauses to text
        enhanced_text = self._add_strategic_pauses(text)

        if self.config.engine == "edge-tts":
            return self._synthesize_edge_tts(enhanced_text, output_path)
        else:
            return self._synthesize_qwen3(enhanced_text, output_path)

    def _synthesize_edge_tts(self, text: str, output_path: Path) -> float:
        """Synthesize using Microsoft edge-tts (cloud, fast, no GPU).
        
        Also captures sentence boundary timestamps and saves them as a JSON
        file alongside the WAV (used for overlay timing sync).
        """
        import asyncio
        import json
        import subprocess

        try:
            import edge_tts
        except ImportError:
            logger.error("edge-tts not installed — falling back to placeholder")
            return self._generate_placeholder(text, output_path)

        async def _generate():
            communicate = edge_tts.Communicate(
                text,
                self.config.voice,
                rate=self.config.rate,
            )
            # Stream to capture both audio and sentence boundaries
            mp3_path = output_path.with_suffix(".mp3")
            sentence_boundaries: list[dict] = []

            audio_chunks = []
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_chunks.append(chunk["data"])
                elif chunk["type"] == "SentenceBoundary":
                    sentence_boundaries.append({
                        "start": chunk["offset"] / 10_000_000,  # 100ns ticks → seconds
                        "duration": chunk["duration"] / 10_000_000,
                        "text": chunk["text"],
                    })

            # Write audio
            with open(mp3_path, "wb") as f:
                for chunk_data in audio_chunks:
                    f.write(chunk_data)

            return mp3_path, sentence_boundaries

        try:
            mp3_path, sentence_boundaries = asyncio.run(_generate())

            # Save sentence boundaries as JSON (for overlay sync)
            boundaries_path = output_path.with_suffix(".boundaries.json")
            boundaries_path.write_text(json.dumps(sentence_boundaries, indent=2))
            logger.info(f"Captured {len(sentence_boundaries)} sentence boundaries")

            # Convert MP3 → WAV using ffmpeg
            result = subprocess.run(
                ["ffmpeg", "-y", "-i", str(mp3_path), "-ar", str(self.config.sample_rate),
                 "-ac", "1", str(output_path)],
                capture_output=True, timeout=30,
            )
            mp3_path.unlink(missing_ok=True)

            if result.returncode != 0:
                logger.error(f"ffmpeg MP3→WAV conversion failed: {result.stderr[:200]}")
                return self._generate_placeholder(text, output_path)

            # Get duration from WAV
            import wave
            with wave.open(str(output_path), "r") as wf:
                frames = wf.getnframes()
                rate = wf.getframerate()
                duration = frames / rate
            return duration

        except Exception as e:
            logger.error(f"edge-tts synthesis failed: {e} — using placeholder")
            return self._generate_placeholder(text, output_path)

    def _synthesize_qwen3(self, text: str, output_path: Path) -> float:
        """Synthesize using Qwen3-TTS (local, GPU, MLX)."""
        if self._model is None:
            return self._generate_placeholder(text, output_path)

        try:
            import soundfile as sf

            gen_kwargs = {
                "text": text,
                "speaker": self.config.speaker,
                "language": "en",
                "verbose": False,
            }

            if self.config.instruct:
                gen_kwargs["instruct"] = self.config.instruct

            results = list(self._model.generate_custom_voice(**gen_kwargs))

            if not results or not hasattr(results[0], "audio"):
                logger.error("TTS generation returned no audio — using placeholder")
                return self._generate_placeholder(text, output_path)

            audio = np.array(results[0].audio)
            sample_rate = self._model.sample_rate

            sf.write(str(output_path), audio, sample_rate)
            duration_seconds = len(audio) / sample_rate
            return duration_seconds

        except Exception as e:
            logger.error(f"Qwen3 TTS synthesis failed: {e} — using placeholder")
            return self._generate_placeholder(text, output_path)

    def _generate_placeholder(self, text: str, output_path: Path) -> float:
        """Generate a silent WAV as placeholder. Returns estimated duration."""
        word_count = len(text.split())
        duration_seconds = word_count / self.config.pace_wpm * 60

        # Add pauses (approximate: one sentence pause per sentence boundary)
        sentence_count = text.count(".") + text.count("!") + text.count("?")
        pause_time = sentence_count * (self.config.sentence_pause_ms / 1000)
        duration_seconds += pause_time

        sample_rate = self.config.sample_rate
        num_samples = int(duration_seconds * sample_rate)

        with wave.open(str(output_path), "w") as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(struct.pack(f"<{num_samples}h", *([0] * num_samples)))

        return duration_seconds

    def _add_strategic_pauses(self, text: str) -> str:
        """Insert subtle pauses after statistics and key revelations.

        Qwen3-TTS respects ellipsis '...' as a natural breath/pause point.
        This makes stats land harder and gives the listener time to absorb numbers.
        """
        import re

        # Add pause after dollar amounts: "$125 million." → "$125 million..."
        text = re.sub(
            r'(\$[\d,]+\s*(?:million|billion|thousand|dollars|a year|per year|annually))',
            r'\1...',
            text,
        )

        # Add pause after percentages: "forty percent." → "forty percent..."
        text = re.sub(
            r'(\b\w+\s+percent\b)',
            r'\1...',
            text,
        )

        # Add pause after time revelations: "eighteen months later" → "eighteen months later..."
        text = re.sub(
            r'((?:months|years|weeks|days)\s+later)',
            r'\1...',
            text,
        )

        # Add pause before contrast words (signals a shift)
        text = re.sub(
            r'\.\s+(But |However |Instead |The problem is|Then )',
            r'. ... \1',
            text,
        )

        # Clean up double/triple ellipses from overlapping rules
        text = re.sub(r'\.{4,}', '...', text)
        text = re.sub(r'\.\.\.\s*\.\.\.', '...', text)

        return text

    def _build_section_text_map(self, script: Script) -> dict[str, str]:
        """Build a mapping from section_id to narration text."""
        text_map: dict[str, str] = {}
        text_map[script.hook.id] = script.hook.narration_text
        text_map[script.introduction.id] = script.introduction.narration_text
        for section in script.body_sections:
            text_map[section.id] = section.narration_text
        text_map[script.conclusion.id] = script.conclusion.narration_text
        return text_map

    def _distribute_text_to_scenes(
        self, scenes: list, section_texts: dict[str, str]
    ) -> dict[str, str]:
        """Split section text proportionally across scenes sharing the same section_id.

        If only one scene uses a section_id, it gets the full text.
        If multiple scenes share a section_id, the text is split by sentences
        proportional to each scene's planned duration.

        Returns:
            Mapping from scene.id to its portion of narration text.
        """
        from collections import defaultdict

        # Group scenes by section_id
        section_scenes: dict[str, list] = defaultdict(list)
        for scene in scenes:
            section_scenes[scene.section_id].append(scene)

        scene_texts: dict[str, str] = {}

        for section_id, section_scene_list in section_scenes.items():
            full_text = section_texts.get(section_id, "")
            if not full_text:
                continue

            if len(section_scene_list) == 1:
                # Single scene gets full text
                scene_texts[section_scene_list[0].id] = full_text
            else:
                # Split by sentences proportional to duration
                sentences = [s.strip() for s in full_text.replace("\n\n", "\n").split("\n") if s.strip()]
                if not sentences:
                    # Fallback: split by periods
                    import re
                    sentences = [s.strip() + "." for s in re.split(r'(?<=[.!?])\s+', full_text) if s.strip()]

                total_duration = sum(s.duration for s in section_scene_list)
                if total_duration <= 0:
                    # Equal split fallback
                    chunk_size = max(1, len(sentences) // len(section_scene_list))
                    for idx, scene in enumerate(section_scene_list):
                        start = idx * chunk_size
                        end = start + chunk_size if idx < len(section_scene_list) - 1 else len(sentences)
                        scene_texts[scene.id] = " ".join(sentences[start:end])
                else:
                    # Proportional split by planned duration
                    sentence_idx = 0
                    for idx, scene in enumerate(section_scene_list):
                        proportion = scene.duration / total_duration
                        num_sentences = max(1, round(len(sentences) * proportion))

                        if idx == len(section_scene_list) - 1:
                            # Last scene gets remaining sentences
                            scene_texts[scene.id] = " ".join(sentences[sentence_idx:])
                        else:
                            end_idx = min(sentence_idx + num_sentences, len(sentences))
                            scene_texts[scene.id] = " ".join(sentences[sentence_idx:end_idx])
                            sentence_idx = end_idx

        return scene_texts
