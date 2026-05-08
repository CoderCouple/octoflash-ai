"""
Firebase Admin SDK initialization.

Lazy-inits from firebase-service-account.json at project root.
Exports get_db() and videos_collection() helpers.
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_db = None
_initialized = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SERVICE_ACCOUNT_PATH = PROJECT_ROOT / "octoflash-ai-firebase-service-account.json"


def _init_firebase():
    """Initialize Firebase Admin SDK (called once lazily)."""
    global _db, _initialized

    if _initialized:
        return

    try:
        import firebase_admin
        from firebase_admin import credentials, firestore

        if not SERVICE_ACCOUNT_PATH.exists():
            logger.warning(
                "Firebase service account not found at %s — Firebase features disabled",
                SERVICE_ACCOUNT_PATH,
            )
            _initialized = True
            return

        cred = credentials.Certificate(str(SERVICE_ACCOUNT_PATH))
        firebase_admin.initialize_app(cred)
        _db = firestore.client()
        _initialized = True
        logger.info("Firebase initialized successfully")
    except Exception as e:
        logger.error("Firebase initialization failed: %s", e)
        _initialized = True


def get_db():
    """Get Firestore client. Returns None if Firebase is not configured."""
    if not _initialized:
        _init_firebase()
    return _db


def videos_collection():
    """Get the 'videos' collection reference. Returns None if Firebase is not configured."""
    db = get_db()
    if db is None:
        return None
    return db.collection("videos")


def is_firebase_available() -> bool:
    """Check if Firebase is properly configured and available."""
    return get_db() is not None
