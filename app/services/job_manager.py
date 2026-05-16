"""
Job state persistence using JSON files + optional Firebase dual-write.

Each job is stored as storage/renders/{job_id}/job.json.
When a video_id is associated, status changes are synced to Firebase.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

STORAGE_DIR = Path(__file__).resolve().parent.parent.parent / "storage"

# Maps job_id → firebase video_id for dual-write
_job_video_map: dict[str, str] = {}


def _job_path(job_id: str) -> Path:
    return STORAGE_DIR / "renders" / job_id / "job.json"


def create_job(job_id: str, video_id: str | None = None) -> dict:
    """Create a new job with pending status. Optionally links to a Firebase video."""
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

    if video_id:
        _job_video_map[job_id] = video_id
        _sync_to_firebase(video_id, status="generating", job_id=job_id)

    return job


def update_job(job_id: str, **fields) -> dict:
    """Update specific fields on an existing job. Syncs to Firebase if linked."""
    job = get_job(job_id)
    if job is None:
        raise FileNotFoundError(f"Job {job_id} not found")
    job.update(fields)
    _job_path(job_id).write_text(json.dumps(job, indent=2))

    # Dual-write to Firebase
    video_id = fields.pop("_firebase_video_id", None) or _job_video_map.get(job_id)
    if video_id:
        _sync_to_firebase(video_id, **fields)

    return job


def get_job(job_id: str) -> dict | None:
    """Read job state from disk. Returns None if not found."""
    path = _job_path(job_id)
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _sync_to_firebase(video_id: str, **fields):
    """Best-effort sync of job fields to Firebase video document."""
    try:
        from app.services.video_manager import update_video

        # Map job status to video status
        sync_fields = {}
        status = fields.get("status")
        if status == "rendering":
            sync_fields["status"] = "generating"
        elif status == "completed":
            sync_fields["status"] = "generated"
            sync_fields["generated_at"] = datetime.now(timezone.utc)
        elif status == "failed":
            sync_fields["status"] = "failed"

        # Forward relevant fields
        for key in ["error", "landscape_video", "portrait_video", "scene_file", "script_file", "job_id", "render_method"]:
            if key in fields:
                sync_fields[key] = fields[key]

        if sync_fields:
            update_video(video_id, **sync_fields)
    except Exception as e:
        logger.warning("Firebase sync failed for video %s: %s", video_id, e)
