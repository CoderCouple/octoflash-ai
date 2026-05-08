"""
Instagram Reels publishing via Instagram Graph API.

REQUIREMENTS:
  - Instagram Business or Creator account, linked to a Facebook Page
  - Facebook Developer App with `instagram_content_publish` permission (requires app review for production)
  - The video URL passed to upload MUST be publicly accessible (Instagram fetches it from a URL)

Setup:
  1. Create app at https://developers.facebook.com/apps
  2. Add Instagram Graph API + Login products
  3. Add `IG_APP_ID`, `IG_APP_SECRET`, `IG_REDIRECT_URI`, `IG_BUSINESS_ACCOUNT_ID` to .env
  4. For local development you can host the rendered MP4 on Cloudinary, S3, or via a tunnel (ngrok/cloudflared).
"""

import json
import logging
import os
import time
import urllib.parse
import urllib.request
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

logger = logging.getLogger(__name__)

APP_ID = os.getenv("IG_APP_ID", "")
APP_SECRET = os.getenv("IG_APP_SECRET", "")
REDIRECT_URI = os.getenv("IG_REDIRECT_URI", "http://localhost:8000/oauth/instagram/callback")
BUSINESS_ACCOUNT_ID = os.getenv("IG_BUSINESS_ACCOUNT_ID", "")
SCOPES = "instagram_basic,instagram_content_publish,pages_show_list,pages_read_engagement"

TOKEN_DIR = Path(__file__).resolve().parent.parent.parent / "storage" / ".tokens"
TOKEN_PATH = TOKEN_DIR / "instagram.json"

GRAPH_API = "https://graph.facebook.com/v19.0"


def _save_token(data: dict):
    TOKEN_DIR.mkdir(parents=True, exist_ok=True)
    TOKEN_PATH.write_text(json.dumps(data))


def _load_token() -> dict | None:
    if not TOKEN_PATH.exists():
        return None
    return json.loads(TOKEN_PATH.read_text())


def get_instagram_auth_url() -> str:
    if not APP_ID:
        raise RuntimeError("IG_APP_ID not configured in .env")
    params = {
        "client_id": APP_ID,
        "redirect_uri": REDIRECT_URI,
        "scope": SCOPES,
        "response_type": "code",
        "state": "octoflash",
    }
    return "https://www.facebook.com/v19.0/dialog/oauth?" + urllib.parse.urlencode(params)


def handle_instagram_callback(code: str) -> dict:
    if not APP_ID or not APP_SECRET:
        raise RuntimeError("Instagram credentials missing")
    params = urllib.parse.urlencode({
        "client_id": APP_ID,
        "client_secret": APP_SECRET,
        "redirect_uri": REDIRECT_URI,
        "code": code,
    })
    with urllib.request.urlopen(f"{GRAPH_API}/oauth/access_token?{params}", timeout=15) as resp:
        token = json.loads(resp.read())

    # Exchange short-lived for long-lived (60-day) token
    long_params = urllib.parse.urlencode({
        "grant_type": "fb_exchange_token",
        "client_id": APP_ID,
        "client_secret": APP_SECRET,
        "fb_exchange_token": token["access_token"],
    })
    with urllib.request.urlopen(f"{GRAPH_API}/oauth/access_token?{long_params}", timeout=15) as resp:
        long_token = json.loads(resp.read())

    _save_token(long_token)
    return long_token


def is_instagram_connected() -> bool:
    return _load_token() is not None and bool(APP_ID) and bool(BUSINESS_ACCOUNT_ID)


def disconnect_instagram() -> dict:
    if TOKEN_PATH.exists():
        TOKEN_PATH.unlink()
    return {"disconnected": True}


def upload_to_instagram(video_path: str, title: str, description: str, public_video_url: str = "", **kwargs) -> dict:
    """Publish a Reel. Requires a publicly accessible video URL.

    `public_video_url` must be passed in — Instagram fetches the video from this URL,
    not the local file. Caller is responsible for hosting (S3/Cloudinary/tunnel).
    """
    token = _load_token()
    if token is None:
        raise RuntimeError("Instagram not connected — call /oauth/instagram/start first")
    if not BUSINESS_ACCOUNT_ID:
        raise RuntimeError("IG_BUSINESS_ACCOUNT_ID missing in .env")
    if not public_video_url:
        raise RuntimeError(
            "Instagram needs a public video URL. "
            "Host the MP4 on S3/Cloudinary/ngrok and pass public_video_url."
        )

    access_token = token["access_token"]
    caption = (title + "\n\n" + description).strip()[:2200]

    # Step 1: Create media container (REELS)
    create_params = urllib.parse.urlencode({
        "media_type": "REELS",
        "video_url": public_video_url,
        "caption": caption,
        "access_token": access_token,
    })
    req = urllib.request.Request(
        f"{GRAPH_API}/{BUSINESS_ACCOUNT_ID}/media?{create_params}",
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        create_data = json.loads(resp.read())
    container_id = create_data["id"]

    # Step 2: Wait until container is FINISHED (Instagram processes video)
    for _ in range(30):
        time.sleep(4)
        status_url = f"{GRAPH_API}/{container_id}?fields=status_code&access_token={access_token}"
        with urllib.request.urlopen(status_url, timeout=15) as resp:
            status = json.loads(resp.read()).get("status_code")
        if status == "FINISHED":
            break
        if status == "ERROR":
            raise RuntimeError(f"Instagram processing failed for container {container_id}")

    # Step 3: Publish container
    pub_params = urllib.parse.urlencode({
        "creation_id": container_id,
        "access_token": access_token,
    })
    pub_req = urllib.request.Request(
        f"{GRAPH_API}/{BUSINESS_ACCOUNT_ID}/media_publish?{pub_params}",
        method="POST",
    )
    with urllib.request.urlopen(pub_req, timeout=30) as resp:
        pub_data = json.loads(resp.read())

    media_id = pub_data["id"]
    return {
        "platform": "instagram",
        "media_id": media_id,
        "url": f"https://www.instagram.com/reel/{media_id}",
    }
