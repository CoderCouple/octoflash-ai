import re
import uuid
import subprocess
from pathlib import Path

from app.utils.files import get_video_path, get_video_dir

YOUTUBE_REGEX = re.compile(
    r"^(https?://)?(www\.)?(youtube\.com/(watch\?v=|shorts/)|youtu\.be/)[\w\-]+"
)


def validate_youtube_url(url: str) -> bool:
    return bool(YOUTUBE_REGEX.match(url))


def download_video(url: str) -> tuple[str, Path]:
    """Download a YouTube video using yt-dlp. Returns (video_id, video_path)."""
    if not validate_youtube_url(url):
        raise ValueError(f"Invalid YouTube URL: {url}")

    video_id = uuid.uuid4().hex[:12]
    output_path = get_video_path(video_id)

    cmd = [
        "yt-dlp",
        "-f", "mp4/best[ext=mp4]",
        "-o", str(output_path),
        "--no-playlist",
        url,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp failed: {result.stderr.strip()}")

    if not output_path.exists():
        # yt-dlp sometimes adds extensions — find the actual file
        candidates = list(get_video_dir(video_id).glob("video.*"))
        if candidates:
            output_path = candidates[0]
        else:
            raise FileNotFoundError("Downloaded video file not found")

    return video_id, output_path
