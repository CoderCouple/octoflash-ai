"""
Job state persistence using JSON files.

Each job is stored as storage/renders/{job_id}/job.json.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

STORAGE_DIR = Path(__file__).resolve().parent.parent.parent / "storage"


def _job_path(job_id: str) -> Path:
    return STORAGE_DIR / "renders" / job_id / "job.json"


def create_job(job_id: str) -> dict:
    """Create a new job with pending status."""
    job = {
        "job_id": job_id,
        "status": "pending",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "completed_at": None,
        "error": None,
        "landscape_video": None,
        "portrait_video": None,
        "scene_file": None,
    }
    path = _job_path(job_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(job, indent=2))
    return job


def update_job(job_id: str, **fields) -> dict:
    """Update specific fields on an existing job."""
    job = get_job(job_id)
    if job is None:
        raise FileNotFoundError(f"Job {job_id} not found")
    job.update(fields)
    _job_path(job_id).write_text(json.dumps(job, indent=2))
    return job


def get_job(job_id: str) -> dict | None:
    """Read job state from disk. Returns None if not found."""
    path = _job_path(job_id)
    if not path.exists():
        return None
    return json.loads(path.read_text())
