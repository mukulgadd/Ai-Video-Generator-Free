"""Smoke tests for the QueueManager implementation."""

import tempfile
from pathlib import Path

import pytest

from vidgen.queue import QueueManager, QueueSummary
from vidgen.models import QueueState


class TestQueueManagerBasic:
    """Basic lifecycle tests for QueueManager."""

    def setup_method(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        tmpdir = Path(self._tmpdir.name)
        self.queue_file = tmpdir / "queue.json"
        self.jobs_dir = tmpdir / "jobs"
        self.mgr = QueueManager(self.queue_file, self.jobs_dir)

    def teardown_method(self):
        self._tmpdir.cleanup()

    def test_load_fresh_state(self):
        """Loading state when no file exists returns empty queue."""
        state = self.mgr.load_state()
        assert state.entries == []
        assert state.last_updated is not None

    def test_add_job_returns_uuid(self):
        """add_job returns a valid UUID string."""
        job_id = self.mgr.add_job(
            Path("/tmp/script.md"), Path("/tmp/scene_plan.json")
        )
        assert len(job_id) == 36
        assert job_id.count("-") == 4

    def test_add_job_persists_to_disk(self):
        """Added job is persisted and loadable."""
        job_id = self.mgr.add_job(
            Path("/tmp/script.md"), Path("/tmp/plan.json"), priority=5
        )
        state = self.mgr.load_state()
        assert len(state.entries) == 1
        entry = state.entries[0]
        assert entry.job_id == job_id
        assert entry.script_path == "/tmp/script.md"
        assert entry.scene_plan_path == "/tmp/plan.json"
        assert entry.priority == 5
        assert entry.status == "queued"
        assert entry.created_at is not None

    def test_mark_job_in_progress(self):
        """mark_job updates status and sets started_at."""
        job_id = self.mgr.add_job(Path("/tmp/s.md"), Path("/tmp/p.json"))
        self.mgr.mark_job(job_id, "in-progress")
        state = self.mgr.load_state()
        assert state.entries[0].status == "in-progress"
        assert state.entries[0].started_at is not None

    def test_mark_job_failed_with_error(self):
        """mark_job with failed status records error and completed_at."""
        job_id = self.mgr.add_job(Path("/tmp/s.md"), Path("/tmp/p.json"))
        self.mgr.mark_job(job_id, "failed", error="something broke")
        state = self.mgr.load_state()
        assert state.entries[0].status == "failed"
        assert state.entries[0].error == "something broke"
        assert state.entries[0].completed_at is not None

    def test_mark_job_completed(self):
        """mark_job with completed status sets completed_at."""
        job_id = self.mgr.add_job(Path("/tmp/s.md"), Path("/tmp/p.json"))
        self.mgr.mark_job(job_id, "completed")
        state = self.mgr.load_state()
        assert state.entries[0].status == "completed"
        assert state.entries[0].completed_at is not None

    def test_mark_job_unknown_raises(self):
        """mark_job with unknown job_id raises ValueError."""
        with pytest.raises(ValueError, match="not found"):
            self.mgr.mark_job("nonexistent-id", "failed")

    def test_get_status(self):
        """get_status returns current QueueState."""
        self.mgr.add_job(Path("/tmp/s.md"), Path("/tmp/p.json"))
        status = self.mgr.get_status()
        assert isinstance(status, QueueState)
        assert len(status.entries) == 1

    def test_multiple_jobs(self):
        """Multiple jobs can be added and retrieved."""
        id1 = self.mgr.add_job(Path("/tmp/s1.md"), Path("/tmp/p1.json"), priority=1)
        id2 = self.mgr.add_job(Path("/tmp/s2.md"), Path("/tmp/p2.json"), priority=2)
        id3 = self.mgr.add_job(Path("/tmp/s3.md"), Path("/tmp/p3.json"), priority=0)
        state = self.mgr.load_state()
        assert len(state.entries) == 3
        ids = {e.job_id for e in state.entries}
        assert ids == {id1, id2, id3}

    def test_save_load_roundtrip(self):
        """State survives a full save/load cycle."""
        self.mgr.add_job(Path("/tmp/s.md"), Path("/tmp/p.json"), priority=3)
        self.mgr.add_job(Path("/tmp/s2.md"), Path("/tmp/p2.json"), priority=1)

        # Create a new manager pointing to same file (simulates restart)
        mgr2 = QueueManager(self.queue_file, self.jobs_dir)
        state = mgr2.load_state()
        assert len(state.entries) == 2
        assert state.entries[0].priority == 3
        assert state.entries[1].priority == 1

    def test_queue_summary_model(self):
        """QueueSummary can be constructed with expected fields."""
        from vidgen.models import QueueEntry

        entry = QueueEntry(
            job_id="test-id",
            script_path="/tmp/s.md",
            scene_plan_path="/tmp/p.json",
            priority=0,
            status="completed",
            created_at="2024-01-01T00:00:00Z",
        )
        summary = QueueSummary(
            total_jobs=1,
            completed=1,
            failed=0,
            total_duration_seconds=42.5,
            results=[entry],
        )
        assert summary.total_jobs == 1
        assert summary.completed == 1
        assert summary.total_duration_seconds == 42.5
