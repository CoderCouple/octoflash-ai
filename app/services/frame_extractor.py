import json
import subprocess
from pathlib import Path

from app.utils.files import get_frames_dir


def get_video_duration(video_path: Path) -> float:
    """Get video duration in seconds using ffprobe."""
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr.strip()}")

    info = json.loads(result.stdout)
    return float(info["format"]["duration"])


def extract_frames(video_id: str, video_path: Path) -> list[str]:
    """Extract one frame per second from the video. Returns list of frame paths."""
    frames_dir = get_frames_dir(video_id)

    cmd = [
        "ffmpeg",
        "-i", str(video_path),
        "-vf", "fps=1",
        "-q:v", "2",
        str(frames_dir / "frame_%04d.jpg"),
        "-y",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg frame extraction failed: {result.stderr.strip()}")

    frames = sorted(frames_dir.glob("frame_*.jpg"))
    if not frames:
        raise FileNotFoundError("No frames were extracted")

    return [str(f.relative_to(frames_dir.parent.parent)) for f in frames]
