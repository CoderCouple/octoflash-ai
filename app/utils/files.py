import shutil
from pathlib import Path

STORAGE_DIR = Path(__file__).resolve().parent.parent.parent / "storage"


def get_video_dir(video_id: str) -> Path:
    """Return the directory for a specific video's assets."""
    d = STORAGE_DIR / video_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_frames_dir(video_id: str) -> Path:
    d = get_video_dir(video_id) / "frames"
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_video_path(video_id: str) -> Path:
    return get_video_dir(video_id) / "video.mp4"


def cleanup_video(video_id: str) -> None:
    d = STORAGE_DIR / video_id
    if d.exists():
        shutil.rmtree(d)
