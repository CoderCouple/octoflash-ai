"""
CRUD operations for the 'videos' Firestore collection.

Each document represents a video workflow from queue → analyze → generate → publish.
"""

import logging
import uuid
from datetime import datetime, timezone

from app.services.firebase_client import videos_collection, is_firebase_available

logger = logging.getLogger(__name__)


def create_video(
    source_url: str,
    source: str = "web",
    channel_id: str | None = None,
    source_short_youtube_id: str | None = None,
) -> dict | None:
    """Create a new video document with status='queued'. Returns the video dict or None."""
    col = videos_collection()
    if col is None:
        logger.warning("Firebase not available — cannot create video")
        return None

    video_id = uuid.uuid4().hex[:12]
    now = datetime.now(timezone.utc)

    # Extract YouTube ID from URL
    youtube_id = _extract_youtube_id(source_url)

    doc = {
        "video_id": video_id,
        "source_url": source_url,
        "source_urls": [source_url],
        "youtube_id": youtube_id,
        "channel_id": channel_id,
        "source_short_youtube_id": source_short_youtube_id or youtube_id,
        "status": "queued",
        "error": None,
        "created_at": now,
        "updated_at": now,
        "analyzed_at": None,
        "generated_at": None,
        "published_at": None,
        "duration_seconds": None,
        "frame_paths": [],
        "frame_count": 0,
        "transcript": None,
        "description": None,
        "manin_prompt": None,
        "title": None,
        "orientation": "portrait",
        "quality": "qm",
        "voiceover": True,
        "job_id": None,
        "landscape_video": None,
        "portrait_video": None,
        "scene_file": None,
        "script_file": None,
        "publish": {},
        "source": source,
    }

    col.document(video_id).set(doc)
    logger.info("Created video %s (status=queued, source=%s, url=%s)", video_id, source, source_url)
    return doc


def get_video(video_id: str) -> dict | None:
    """Get a single video document by ID."""
    col = videos_collection()
    if col is None:
        return None

    doc = col.document(video_id).get()
    if not doc.exists:
        return None

    data = doc.to_dict()
    # Convert Firestore timestamps to ISO strings for JSON serialization
    return _serialize_timestamps(data)


def update_video(video_id: str, **fields) -> dict | None:
    """Merge fields into an existing video document. Auto-sets updated_at."""
    col = videos_collection()
    if col is None:
        return None

    fields["updated_at"] = datetime.now(timezone.utc)
    col.document(video_id).update(fields)
    logger.debug("Updated video %s: %s", video_id, list(fields.keys()))
    return get_video(video_id)


def list_videos(limit: int = 50) -> list[dict]:
    """List videos ordered by created_at DESC."""
    col = videos_collection()
    if col is None:
        return []

    docs = col.order_by("created_at", direction="DESCENDING").limit(limit).stream()
    return [_serialize_timestamps(doc.to_dict()) for doc in docs]


def get_queued_videos(limit: int = 5) -> list[dict]:
    """Get videos with status='queued', ordered by created_at ASC."""
    col = videos_collection()
    if col is None:
        return []

    from google.cloud.firestore_v1.base_query import FieldFilter
    docs = (
        col.where(filter=FieldFilter("status", "==", "queued"))
        .order_by("created_at")
        .limit(limit)
        .stream()
    )
    return [_serialize_timestamps(doc.to_dict()) for doc in docs]


def _serialize_timestamps(data: dict) -> dict:
    """Convert Firestore DatetimeWithNanoseconds to ISO strings."""
    for key in ["created_at", "updated_at", "analyzed_at", "generated_at", "published_at"]:
        val = data.get(key)
        if val is not None and hasattr(val, "isoformat"):
            data[key] = val.isoformat()
    return data


def _extract_youtube_id(url: str) -> str | None:
    """Extract YouTube video ID from URL."""
    import re
    patterns = [
        r"youtube\.com/watch\?v=([\w-]+)",
        r"youtube\.com/shorts/([\w-]+)",
        r"youtu\.be/([\w-]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None
