"""Queue manager for batch video generation job processing.

Manages a persistent job queue stored as queue.json on disk, enabling
sequential processing of multiple video generation jobs with failure
isolation and resume support after system restarts.
"""

import json
import logging
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field

from vidgen.config import PipelineConfig
from vidgen.models import QueueEntry, QueueState

logger = logging.getLogger(__name__)


# --- Result Model ---


class QueueSummary(BaseModel):
    """Summary report produced after queue processing completes."""

    total_jobs: int
    completed: int
    failed: int
    total_duration_seconds: float
    results: list[QueueEntry]


# --- Queue Manager ---


class QueueManager:
    """Manages batch job processing with disk-persisted state.

    Jobs are processed sequentially to respect system memory constraints.
    Failed jobs are marked with error context but do not block remaining jobs.
    Queue state persists to disk for resume after restart.
    """

    def __init__(self, queue_file: Path, jobs_dir: Path) -> None:
        """Initialize the queue manager.

        Args:
            queue_file: Path to queue.json for state persistence.
            jobs_dir: Base directory where per-job working directories are created.
        """
        self.queue_file = queue_file
        self.jobs_dir = jobs_dir

    def add_job(
        self, script_path: Path, scene_plan_path: Path, priority: int = 0
    ) -> str:
        """Add a job to the processing queue.

        Args:
            script_path: Path to the script.md file.
            scene_plan_path: Path to the scene_plan.json file.
            priority: Job priority (higher = processed first). Default 0.

        Returns:
            The unique job_id (UUID) assigned to this job.
        """
        job_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        entry = QueueEntry(
            job_id=job_id,
            script_path=str(script_path),
            scene_plan_path=str(scene_plan_path),
            priority=priority,
            status="queued",
            created_at=now,
        )

        state = self.load_state()
        state.entries.append(entry)
        state = state.model_copy(update={"last_updated": now})
        self.save_state(state)

        logger.info(f"Added job {job_id} to queue (priority={priority})")
        return job_id

    def process_queue(self, config: PipelineConfig) -> QueueSummary:
        """Process all queued jobs sequentially.

        Jobs are sorted by priority (descending) then creation time (ascending).
        Failed jobs are marked with error context; remaining jobs continue.
        Progress is logged for each job.

        Args:
            config: Pipeline configuration for job execution.

        Returns:
            QueueSummary with counts and per-job results.
        """
        from vidgen.models import ScenePlan, Script
        from vidgen.parsers import parse_scene_plan, parse_script
        from vidgen.pipeline import PipelineOrchestrator

        state = self.load_state()
        queued_entries = [e for e in state.entries if e.status == "queued"]

        # Sort by priority (descending), then created_at (ascending)
        queued_entries.sort(key=lambda e: (-e.priority, e.created_at))

        total_jobs = len(queued_entries)
        completed = 0
        failed = 0
        queue_start = time.time()

        logger.info(f"Processing queue: {total_jobs} jobs pending")

        for idx, entry in enumerate(queued_entries, start=1):
            logger.info(
                f"Processing job {idx}/{total_jobs}: {entry.job_id} "
                f"(priority={entry.priority})"
            )

            # Mark job as in-progress
            self.mark_job(entry.job_id, "in-progress")

            job_start = time.time()
            try:
                # Parse inputs
                script = parse_script(Path(entry.script_path))
                scene_plan = parse_scene_plan(Path(entry.scene_plan_path))

                # Set up job directory using script name (not UUID)
                job_name = scene_plan.topic_slug or Path(entry.script_path).stem
                job_dir = self.jobs_dir / job_name
                job_dir.mkdir(parents=True, exist_ok=True)

                # Run pipeline
                orchestrator = PipelineOrchestrator(config, job_dir)
                result = orchestrator.run(script, scene_plan)

                if result.success:
                    self.mark_job(entry.job_id, "completed")
                    # Update output path
                    state = self.load_state()
                    for e in state.entries:
                        if e.job_id == entry.job_id:
                            e.output_path = str(result.output_dir) if result.output_dir else None
                            break
                    self.save_state(state)
                    completed += 1
                    logger.info(
                        f"Job {entry.job_id} completed in "
                        f"{time.time() - job_start:.1f}s"
                    )
                else:
                    error_msg = result.error or "Pipeline execution failed"
                    self.mark_job(entry.job_id, "failed", error=error_msg)
                    failed += 1
                    logger.warning(
                        f"Job {entry.job_id} failed: {error_msg}"
                    )

            except Exception as e:
                error_msg = str(e)
                self.mark_job(entry.job_id, "failed", error=error_msg)
                failed += 1
                logger.error(
                    f"Job {entry.job_id} failed with exception: {error_msg}"
                )

        total_duration = time.time() - queue_start

        # Reload final state for results
        final_state = self.load_state()
        logger.info(
            f"Queue processing complete: {completed} completed, "
            f"{failed} failed, {total_duration:.1f}s total"
        )

        return QueueSummary(
            total_jobs=total_jobs,
            completed=completed,
            failed=failed,
            total_duration_seconds=total_duration,
            results=final_state.entries,
        )

    def get_status(self) -> QueueState:
        """Return current queue state from disk.

        Returns:
            The current QueueState with all entries and last_updated timestamp.
        """
        return self.load_state()

    def mark_job(
        self, job_id: str, status: str, error: str | None = None
    ) -> None:
        """Update job status on disk.

        Args:
            job_id: The UUID of the job to update.
            status: New status (queued, in-progress, completed, failed).
            error: Optional error message (used when status is 'failed').

        Raises:
            ValueError: If job_id is not found in the queue.
        """
        state = self.load_state()
        now = datetime.now(timezone.utc).isoformat()
        found = False

        for entry in state.entries:
            if entry.job_id == job_id:
                entry.status = status
                entry.error = error
                if status == "in-progress":
                    entry.started_at = now
                elif status in ("completed", "failed"):
                    entry.completed_at = now
                found = True
                break

        if not found:
            raise ValueError(f"Job {job_id} not found in queue")

        state = state.model_copy(update={"last_updated": now})
        self.save_state(state)

    def load_state(self) -> QueueState:
        """Load queue state from disk file.

        Returns a fresh empty QueueState if the file doesn't exist.

        Returns:
            The persisted QueueState.
        """
        if self.queue_file.exists():
            data = json.loads(self.queue_file.read_text(encoding="utf-8"))
            return QueueState(**data)
        return QueueState(
            entries=[],
            last_updated=datetime.now(timezone.utc).isoformat(),
        )

    def save_state(self, state: QueueState) -> None:
        """Persist queue state to disk atomically.

        Writes to a temporary file then renames to prevent corruption
        on crash during write.

        Args:
            state: The QueueState to persist.
        """
        self.queue_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.queue_file.with_suffix(".tmp")
        data = state.model_dump()
        tmp.write_text(
            json.dumps(data, indent=2, default=str), encoding="utf-8"
        )
        tmp.rename(self.queue_file)
