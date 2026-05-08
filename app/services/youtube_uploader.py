"""
YouTube OAuth2 + upload service.

Connect once via OAuth, then upload videos to YouTube Data API v3.
Token is persisted to disk and auto-refreshed.
"""

import json
import logging
import os
from pathlib import Path

import google_auth_httplib2
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

logger = logging.getLogger(__name__)

SCOPES = [
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtube",
]
TOKEN_DIR = Path(__file__).resolve().parent.parent.parent / "storage" / ".tokens"
TOKEN_PATH = TOKEN_DIR / "youtube.json"

CLIENT_ID = os.getenv("YOUTUBE_CLIENT_ID", "")
CLIENT_SECRET = os.getenv("YOUTUBE_CLIENT_SECRET", "")
REDIRECT_URI = os.getenv("YOUTUBE_REDIRECT_URI", "http://localhost:8000/oauth/youtube/callback")


def _ensure_token_dir():
    TOKEN_DIR.mkdir(parents=True, exist_ok=True)


_CLIENT_CONFIG = {
    "web": {
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
        "redirect_uris": [REDIRECT_URI],
    }
}

_VERIFIER_PATH = TOKEN_DIR / "_code_verifier"


def get_youtube_auth_url() -> str:
    """Return Google OAuth2 consent URL."""
    from google_auth_oauthlib.flow import Flow

    if not CLIENT_ID or not CLIENT_SECRET:
        raise RuntimeError("YOUTUBE_CLIENT_ID and YOUTUBE_CLIENT_SECRET must be set in .env")

    flow = Flow.from_client_config(
        _CLIENT_CONFIG,
        scopes=SCOPES,
        redirect_uri=REDIRECT_URI,
    )
    auth_url, state = flow.authorization_url(
        access_type="offline",
        prompt="consent",
        include_granted_scopes="true",
    )

    # Persist the code_verifier so the callback can use it
    _ensure_token_dir()
    _VERIFIER_PATH.write_text(flow.code_verifier or "")

    return auth_url


def handle_youtube_callback(code: str) -> dict:
    """Exchange authorization code for tokens, save to disk."""
    from google_auth_oauthlib.flow import Flow

    flow = Flow.from_client_config(
        _CLIENT_CONFIG,
        scopes=SCOPES,
        redirect_uri=REDIRECT_URI,
    )

    # Restore the code_verifier from the auth step
    if _VERIFIER_PATH.exists():
        flow.code_verifier = _VERIFIER_PATH.read_text() or None
        _VERIFIER_PATH.unlink(missing_ok=True)

    flow.fetch_token(code=code)
    creds = flow.credentials

    _ensure_token_dir()
    token_data = {
        "token": creds.token,
        "refresh_token": creds.refresh_token,
        "token_uri": creds.token_uri,
        "client_id": creds.client_id,
        "client_secret": creds.client_secret,
        "scopes": list(creds.scopes or []),
    }
    TOKEN_PATH.write_text(json.dumps(token_data, indent=2))
    logger.info("YouTube OAuth tokens saved")
    return {"connected": True}


def is_youtube_connected() -> bool:
    """Check if a valid YouTube token file exists."""
    if not TOKEN_PATH.exists():
        return False
    try:
        data = json.loads(TOKEN_PATH.read_text())
        return bool(data.get("refresh_token"))
    except Exception:
        return False


def _make_unverified_session():
    """Create a requests.Session that skips SSL verification."""
    import requests as _requests
    session = _requests.Session()
    session.verify = False
    return session


def get_youtube_credentials():
    """Load credentials from disk and auto-refresh if expired."""
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request

    if not TOKEN_PATH.exists():
        raise RuntimeError("YouTube not connected. Please connect via OAuth first.")

    data = json.loads(TOKEN_PATH.read_text())
    creds = Credentials(
        token=data.get("token"),
        refresh_token=data.get("refresh_token"),
        token_uri=data.get("token_uri", "https://oauth2.googleapis.com/token"),
        client_id=data.get("client_id", CLIENT_ID),
        client_secret=data.get("client_secret", CLIENT_SECRET),
        scopes=data.get("scopes", SCOPES),
    )

    if creds.expired and creds.refresh_token:
        creds.refresh(Request(session=_make_unverified_session()))
        # Save refreshed token
        data["token"] = creds.token
        _ensure_token_dir()
        TOKEN_PATH.write_text(json.dumps(data, indent=2))

    return creds


def _get_youtube_service():
    """Build an authenticated YouTube API service."""
    from googleapiclient.discovery import build
    import httplib2

    creds = get_youtube_credentials()
    http = httplib2.Http(disable_ssl_certificate_validation=True)
    http = google_auth_httplib2.AuthorizedHttp(creds, http=http)
    return build("youtube", "v3", http=http)


def _find_or_create_playlist(youtube, playlist_name: str, privacy: str = "public") -> str:
    """Find an existing playlist by name, or create it. Returns playlist ID."""
    # Search existing playlists
    request = youtube.playlists().list(part="snippet", mine=True, maxResults=50)
    response = request.execute()

    for item in response.get("items", []):
        if item["snippet"]["title"].strip().lower() == playlist_name.strip().lower():
            logger.info("Found existing playlist: %s (%s)", playlist_name, item["id"])
            return item["id"]

    # Create new playlist
    body = {
        "snippet": {
            "title": playlist_name,
            "description": f"Videos from Octoflash — {playlist_name}",
        },
        "status": {
            "privacyStatus": privacy if privacy in ("public", "unlisted", "private") else "public",
        },
    }
    response = youtube.playlists().insert(part="snippet,status", body=body).execute()
    playlist_id = response["id"]
    logger.info("Created new playlist: %s (%s)", playlist_name, playlist_id)
    return playlist_id


def list_playlists() -> list[dict]:
    """List the authenticated user's YouTube playlists."""
    if not is_youtube_connected():
        return []
    try:
        youtube = _get_youtube_service()
        playlists = []
        request = youtube.playlists().list(part="snippet", mine=True, maxResults=50)
        while request:
            response = request.execute()
            for item in response.get("items", []):
                playlists.append({
                    "id": item["id"],
                    "title": item["snippet"]["title"],
                })
            request = youtube.playlists().list_next(request, response)
        return playlists
    except Exception as e:
        logger.warning("Failed to list playlists: %s", e)
        return []


def upload_to_youtube(
    video_path: str,
    title: str,
    description: str,
    tags: str,
    privacy: str = "public",
    video_type: str = "shorts",
    playlist: str = "",
) -> dict:
    """Upload a video to YouTube. Returns {"video_id": ..., "url": ...}."""
    from googleapiclient.http import MediaFileUpload

    if not Path(video_path).exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    youtube = _get_youtube_service()

    # Sanitize tags — YouTube rejects <, >, quotes; total character count must be ≤500
    raw_tags = [t.strip() for t in tags.split(",") if t.strip()] if tags else []
    tags_list = []
    total_chars = 0
    for t in raw_tags:
        # Strip leading # and disallowed characters
        clean = t.lstrip("#").replace("<", "").replace(">", "").replace('"', "").replace("'", "").strip()
        if not clean or len(clean) > 100:
            continue
        # YouTube counts tags with spaces with extra +2 (treated as quoted)
        cost = len(clean) + (2 if " " in clean else 0)
        if total_chars + cost + (1 if tags_list else 0) > 480:  # leave headroom for #Shorts
            break
        tags_list.append(clean)
        total_chars += cost + (1 if len(tags_list) > 1 else 0)

    # For Shorts: add #Shorts tag and ensure title has #Shorts
    if video_type == "shorts":
        if "Shorts" not in tags_list and "shorts" not in tags_list:
            tags_list.insert(0, "Shorts")
        if "#shorts" not in title.lower():
            title = f"{title} #Shorts"

    body = {
        "snippet": {
            "title": title[:100],
            "description": description[:5000],
            "tags": tags_list,
            "categoryId": "28",  # Science & Technology
        },
        "status": {
            "privacyStatus": privacy if privacy in ("public", "unlisted", "private") else "public",
            "selfDeclaredMadeForKids": False,
        },
    }

    media = MediaFileUpload(video_path, mimetype="video/mp4", resumable=True)

    request = youtube.videos().insert(
        part="snippet,status",
        body=body,
        media_body=media,
    )

    logger.info("Starting YouTube upload (%s): %s", video_type, title)
    response = None
    while response is None:
        status, response = request.next_chunk()
        if status:
            logger.info("Upload progress: %d%%", int(status.progress() * 100))

    video_id = response["id"]

    # Add to playlist if specified
    if playlist.strip():
        try:
            playlist_id = _find_or_create_playlist(youtube, playlist.strip(), privacy)
            youtube.playlistItems().insert(
                part="snippet",
                body={
                    "snippet": {
                        "playlistId": playlist_id,
                        "resourceId": {
                            "kind": "youtube#video",
                            "videoId": video_id,
                        },
                    }
                },
            ).execute()
            logger.info("Added video to playlist: %s", playlist.strip())
        except Exception as e:
            logger.warning("Failed to add to playlist: %s", e)

    url = f"https://www.youtube.com/watch?v={video_id}"
    if video_type == "shorts":
        url = f"https://www.youtube.com/shorts/{video_id}"

    logger.info("YouTube upload complete: %s", url)
    return {"video_id": video_id, "url": url}


def disconnect_youtube() -> dict:
    """Delete stored YouTube tokens."""
    if TOKEN_PATH.exists():
        TOKEN_PATH.unlink()
        logger.info("YouTube tokens deleted")
    return {"connected": False}
