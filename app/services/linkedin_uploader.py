"""
LinkedIn OAuth2 + video upload.

Uses LinkedIn's Posts API (REST). Two-step upload:
  1. Register upload → get uploadUrl
  2. PUT the video bytes to uploadUrl
  3. Create a post referencing the uploaded asset

Setup:
  1. Create app at https://www.linkedin.com/developers/apps
  2. Add Sign In with LinkedIn + Share on LinkedIn products
  3. Add `LINKEDIN_CLIENT_ID`, `LINKEDIN_CLIENT_SECRET`, `LINKEDIN_REDIRECT_URI` to .env
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

CLIENT_ID = os.getenv("LINKEDIN_CLIENT_ID", "")
CLIENT_SECRET = os.getenv("LINKEDIN_CLIENT_SECRET", "")
REDIRECT_URI = os.getenv("LINKEDIN_REDIRECT_URI", "http://localhost:8000/oauth/linkedin/callback")
SCOPES = "w_member_social openid profile"

TOKEN_DIR = Path(__file__).resolve().parent.parent.parent / "storage" / ".tokens"
TOKEN_PATH = TOKEN_DIR / "linkedin.json"


def _save_token(data: dict):
    TOKEN_DIR.mkdir(parents=True, exist_ok=True)
    TOKEN_PATH.write_text(json.dumps(data))


def _load_token() -> dict | None:
    if not TOKEN_PATH.exists():
        return None
    return json.loads(TOKEN_PATH.read_text())


def get_linkedin_auth_url() -> str:
    if not CLIENT_ID:
        raise RuntimeError("LINKEDIN_CLIENT_ID not configured in .env")
    params = {
        "response_type": "code",
        "client_id": CLIENT_ID,
        "redirect_uri": REDIRECT_URI,
        "scope": SCOPES,
        "state": "octoflash",
    }
    return "https://www.linkedin.com/oauth/v2/authorization?" + urllib.parse.urlencode(params)


def handle_linkedin_callback(code: str) -> dict:
    if not CLIENT_ID or not CLIENT_SECRET:
        raise RuntimeError("LinkedIn credentials missing")
    data = urllib.parse.urlencode({
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": REDIRECT_URI,
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
    }).encode()
    req = urllib.request.Request(
        "https://www.linkedin.com/oauth/v2/accessToken",
        data=data,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        token = json.loads(resp.read())

    # Fetch member URN (sub from /v2/userinfo)
    userinfo_req = urllib.request.Request(
        "https://api.linkedin.com/v2/userinfo",
        headers={"Authorization": f"Bearer {token['access_token']}"},
    )
    with urllib.request.urlopen(userinfo_req, timeout=15) as resp:
        userinfo = json.loads(resp.read())

    token["member_urn"] = f"urn:li:person:{userinfo['sub']}"
    _save_token(token)
    return token


def is_linkedin_connected() -> bool:
    return _load_token() is not None and bool(CLIENT_ID)


def disconnect_linkedin() -> dict:
    if TOKEN_PATH.exists():
        TOKEN_PATH.unlink()
    return {"disconnected": True}


def upload_to_linkedin(video_path: str, title: str, description: str, **kwargs) -> dict:
    """Upload an MP4 to LinkedIn as a video post."""
    token = _load_token()
    if token is None:
        raise RuntimeError("LinkedIn not connected — call /oauth/linkedin/start first")

    access_token = token["access_token"]
    member_urn = token["member_urn"]
    headers_auth = {
        "Authorization": f"Bearer {access_token}",
        "X-Restli-Protocol-Version": "2.0.0",
    }

    # Step 1: Register upload
    register_body = json.dumps({
        "registerUploadRequest": {
            "recipes": ["urn:li:digitalmediaRecipe:feedshare-video"],
            "owner": member_urn,
            "serviceRelationships": [{
                "relationshipType": "OWNER",
                "identifier": "urn:li:userGeneratedContent",
            }],
        }
    }).encode()
    reg_req = urllib.request.Request(
        "https://api.linkedin.com/v2/assets?action=registerUpload",
        data=register_body,
        headers={**headers_auth, "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(reg_req, timeout=30) as resp:
        reg_data = json.loads(resp.read())

    upload_url = reg_data["value"]["uploadMechanism"][
        "com.linkedin.digitalmedia.uploading.MediaUploadHttpRequest"
    ]["uploadUrl"]
    asset_urn = reg_data["value"]["asset"]

    # Step 2: PUT the video bytes
    with open(video_path, "rb") as f:
        video_bytes = f.read()
    upload_req = urllib.request.Request(
        upload_url,
        data=video_bytes,
        headers={"Authorization": f"Bearer {access_token}"},
        method="PUT",
    )
    urllib.request.urlopen(upload_req, timeout=600).read()

    # Step 3: Create post referencing the asset
    post_body = json.dumps({
        "author": member_urn,
        "lifecycleState": "PUBLISHED",
        "specificContent": {
            "com.linkedin.ugc.ShareContent": {
                "shareCommentary": {"text": f"{title}\n\n{description}".strip()},
                "shareMediaCategory": "VIDEO",
                "media": [{
                    "status": "READY",
                    "media": asset_urn,
                    "title": {"text": title[:200]},
                }],
            }
        },
        "visibility": {"com.linkedin.ugc.MemberNetworkVisibility": "PUBLIC"},
    }).encode()
    post_req = urllib.request.Request(
        "https://api.linkedin.com/v2/ugcPosts",
        data=post_body,
        headers={**headers_auth, "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(post_req, timeout=30) as resp:
        post_id = resp.headers.get("X-RestLi-Id") or json.loads(resp.read()).get("id")

    return {
        "platform": "linkedin",
        "post_id": post_id,
        "url": f"https://www.linkedin.com/feed/update/{post_id}",
    }
