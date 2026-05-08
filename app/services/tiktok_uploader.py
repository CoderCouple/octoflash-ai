"""
TikTok OAuth2 + video upload via Content Posting API.

Setup:
  1. Register an app at https://developers.tiktok.com
  2. Add the "Content Posting API" product (sandbox is fine for testing)
  3. Add `TIKTOK_CLIENT_KEY`, `TIKTOK_CLIENT_SECRET`, `TIKTOK_REDIRECT_URI` to .env
  4. Sandbox apps can post only to the developer's own account.

Upload flow (FILE_UPLOAD):
  1. POST /post/publish/inbox/video/init/  → returns upload_url + publish_id
  2. PUT video bytes to upload_url
  3. Poll /post/publish/status/fetch/ until PUBLISH_COMPLETE
"""

import json
import logging
import os
import urllib.parse
import urllib.request
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

logger = logging.getLogger(__name__)

CLIENT_KEY = os.getenv("TIKTOK_CLIENT_KEY", "")
CLIENT_SECRET = os.getenv("TIKTOK_CLIENT_SECRET", "")
REDIRECT_URI = os.getenv("TIKTOK_REDIRECT_URI", "http://localhost:8000/oauth/tiktok/callback")
SCOPES = "user.info.basic,video.upload,video.publish"

TOKEN_DIR = Path(__file__).resolve().parent.parent.parent / "storage" / ".tokens"
TOKEN_PATH = TOKEN_DIR / "tiktok.json"


def _save_token(data: dict):
    TOKEN_DIR.mkdir(parents=True, exist_ok=True)
    TOKEN_PATH.write_text(json.dumps(data))


def _load_token() -> dict | None:
    if not TOKEN_PATH.exists():
        return None
    return json.loads(TOKEN_PATH.read_text())


def get_tiktok_auth_url() -> str:
    if not CLIENT_KEY:
        raise RuntimeError("TIKTOK_CLIENT_KEY not configured in .env")
    params = {
        "client_key": CLIENT_KEY,
        "scope": SCOPES,
        "response_type": "code",
        "redirect_uri": REDIRECT_URI,
        "state": "octoflash",
    }
    return "https://www.tiktok.com/v2/auth/authorize/?" + urllib.parse.urlencode(params)


def handle_tiktok_callback(code: str) -> dict:
    if not CLIENT_KEY or not CLIENT_SECRET:
        raise RuntimeError("TikTok credentials missing")
    data = urllib.parse.urlencode({
        "client_key": CLIENT_KEY,
        "client_secret": CLIENT_SECRET,
        "code": code,
        "grant_type": "authorization_code",
        "redirect_uri": REDIRECT_URI,
    }).encode()
    req = urllib.request.Request(
        "https://open.tiktokapis.com/v2/oauth/token/",
        data=data,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        token = json.loads(resp.read())
    _save_token(token)
    return token


def is_tiktok_connected() -> bool:
    return _load_token() is not None and bool(CLIENT_KEY)


def disconnect_tiktok() -> dict:
    if TOKEN_PATH.exists():
        TOKEN_PATH.unlink()
    return {"disconnected": True}


def upload_to_tiktok(video_path: str, title: str, description: str, privacy: str = "SELF_ONLY", **kwargs) -> dict:
    """Upload an MP4 to TikTok via inbox endpoint (saves to drafts in sandbox)."""
    import time
    token = _load_token()
    if token is None:
        raise RuntimeError("TikTok not connected — call /oauth/tiktok/start first")
    access_token = token["access_token"]

    file_size = os.path.getsize(video_path)
    chunk_size = min(file_size, 64 * 1024 * 1024)  # max 64MB per chunk

    # Step 1: Init upload
    init_body = json.dumps({
        "post_info": {
            "title": (title + " " + description)[:2200],
            "privacy_level": privacy,  # SELF_ONLY for sandbox
            "disable_duet": False,
            "disable_comment": False,
            "disable_stitch": False,
        },
        "source_info": {
            "source": "FILE_UPLOAD",
            "video_size": file_size,
            "chunk_size": chunk_size,
            "total_chunk_count": 1,
        },
    }).encode()
    init_req = urllib.request.Request(
        "https://open.tiktokapis.com/v2/post/publish/video/init/",
        data=init_body,
        headers={
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(init_req, timeout=30) as resp:
        init_data = json.loads(resp.read())

    if init_data.get("error", {}).get("code") not in (None, "ok"):
        raise RuntimeError(f"TikTok init failed: {init_data['error']}")

    upload_url = init_data["data"]["upload_url"]
    publish_id = init_data["data"]["publish_id"]

    # Step 2: PUT video bytes
    with open(video_path, "rb") as f:
        video_bytes = f.read()
    upload_req = urllib.request.Request(
        upload_url,
        data=video_bytes,
        headers={
            "Content-Type": "video/mp4",
            "Content-Range": f"bytes 0-{file_size - 1}/{file_size}",
        },
        method="PUT",
    )
    urllib.request.urlopen(upload_req, timeout=600).read()

    # Step 3: Poll status (best-effort — return immediately with publish_id)
    for _ in range(10):
        time.sleep(3)
        status_req = urllib.request.Request(
            "https://open.tiktokapis.com/v2/post/publish/status/fetch/",
            data=json.dumps({"publish_id": publish_id}).encode(),
            headers={
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(status_req, timeout=15) as resp:
                status_data = json.loads(resp.read())
            status = status_data.get("data", {}).get("status", "")
            if status in ("PUBLISH_COMPLETE", "SEND_TO_USER_INBOX"):
                break
            if status == "FAILED":
                raise RuntimeError(f"TikTok publish failed: {status_data}")
        except Exception:
            continue

    return {
        "platform": "tiktok",
        "publish_id": publish_id,
        "url": f"https://www.tiktok.com/@me",  # TikTok doesn't return final URL via API
    }
