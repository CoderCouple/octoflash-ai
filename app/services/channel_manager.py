"""
CRUD operations for the 'channels' Firestore collection.

Tracks YouTube channels that produce similar content (for inspiration/reference).
"""

import logging
import re
import uuid
from datetime import datetime, timezone

from app.services.firebase_client import get_db

logger = logging.getLogger(__name__)


def _channels_collection():
    db = get_db()
    if db is None:
        return None
    return db.collection("channels")


def create_channel(url: str, name: str = "", notes: str = "") -> dict | None:
    """Create a channel doc. Returns the channel dict or None."""
    col = _channels_collection()
    if col is None:
        return None

    channel_id = uuid.uuid4().hex[:12]
    now = datetime.now(timezone.utc)
    handle = _extract_handle(url)

    doc = {
        "channel_id": channel_id,
        "url": url,
        "handle": handle,
        "name": name or handle or url,
        "notes": notes,
        "created_at": now,
        "updated_at": now,
    }
    col.document(channel_id).set(doc)
    logger.info("Created channel %s (%s)", channel_id, doc["name"])
    return _serialize(doc)


def list_channels(limit: int = 100) -> list[dict]:
    col = _channels_collection()
    if col is None:
        return []
    docs = col.order_by("created_at", direction="DESCENDING").limit(limit).stream()
    return [_serialize(d.to_dict()) for d in docs]


def delete_channel(channel_id: str) -> bool:
    col = _channels_collection()
    if col is None:
        return False
    col.document(channel_id).delete()
    return True


def update_channel(channel_id: str, **fields) -> dict | None:
    col = _channels_collection()
    if col is None:
        return None
    fields["updated_at"] = datetime.now(timezone.utc)
    col.document(channel_id).update(fields)
    doc = col.document(channel_id).get()
    return _serialize(doc.to_dict()) if doc.exists else None


def _serialize(data: dict) -> dict:
    for key in ["created_at", "updated_at"]:
        v = data.get(key)
        if v is not None and hasattr(v, "isoformat"):
            data[key] = v.isoformat()
    return data


def _extract_handle(url: str) -> str:
    """Extract @handle or channel name from a YouTube URL."""
    patterns = [
        r"youtube\.com/@([\w.-]+)",
        r"youtube\.com/c/([\w.-]+)",
        r"youtube\.com/user/([\w.-]+)",
        r"youtube\.com/channel/([\w-]+)",
    ]
    for pat in patterns:
        m = re.search(pat, url)
        if m:
            prefix = "@" if "@" in pat else ""
            return prefix + m.group(1)
    return ""
