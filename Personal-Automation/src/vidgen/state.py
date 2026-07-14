"""Job state tracker for persistence and resumability.

Tracks per-job progress through pipeline stages, enabling resume after
crash/timeout and providing timing data for monitoring.
"""

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from vidgen.models import JobState


class JobStateTracker:
    """Tracks per-job progress for persistence and resumability."""

    def __init__(self, state_file: Path) -> None:
        self.state_file = state_file
        self._stage_start_time: float | None = None

    def load(self) -> JobState:
        """Load state from disk. Returns fresh state if file doesn't exist."""
        if self.state_file.exists():
            data = json.loads(self.state_file.read_text(encoding="utf-8"))
            return JobState(**data)
        return JobState(job_id=str(uuid.uuid4()), status="queued")

    def save(self, state: JobState) -> None:
        """Persist state to disk atomically (write temp, rename)."""
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(state.model_dump(), indent=2, default=str), encoding="utf-8")
        tmp.rename(self.state_file)

    def mark_stage_started(self, stage: str) -> None:
        """Record stage start. Updates current_stage and status."""
        state = self.load()
        state = state.model_copy(
            update={
                "current_stage": stage,
                "status": "in-progress",
                "started_at": state.started_at or datetime.now(timezone.utc).isoformat(),
            }
        )
        self._stage_start_time = time.time()
        self.save(state)

    def mark_stage_completed(self, stage: str, artifacts: list[str]) -> None:
        """Record stage completion with timing and artifacts."""
        state = self.load()
        completed = list(state.completed_stages)
        if stage not in completed:
            completed.append(stage)

        timings = dict(state.stage_timings)
        if self._stage_start_time is not None:
            timings[stage] = round(time.time() - self._stage_start_time, 2)
            self._stage_start_time = None

        arts = dict(state.artifacts)
        arts[stage] = artifacts

        state = state.model_copy(
            update={
                "completed_stages": completed,
                "stage_timings": timings,
                "artifacts": arts,
                "current_stage": None,
            }
        )
        self.save(state)

    def mark_failed(self, error: str) -> None:
        """Record job failure."""
        state = self.load()
        state = state.model_copy(
            update={
                "status": "failed",
                "error": error,
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "current_stage": None,
            }
        )
        self.save(state)

    def mark_timed_out(self, error: str) -> None:
        """Record job timeout."""
        state = self.load()
        state = state.model_copy(
            update={
                "status": "timed-out",
                "error": error,
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "current_stage": None,
            }
        )
        self.save(state)

    def mark_completed(self) -> None:
        """Mark the entire job as completed."""
        state = self.load()
        state = state.model_copy(
            update={
                "status": "completed",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "current_stage": None,
            }
        )
        self.save(state)

    def is_stage_completed(self, stage: str) -> bool:
        """Check if a stage was previously completed (for resume)."""
        state = self.load()
        return stage in state.completed_stages
