import logging
import uuid
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.models.schemas import (
    AnalyzeRequest, AnalyzeResponse,
    GenerateRequest, GenerateResponse, JobStatusResponse,
)
from app.services.downloader import download_video, validate_youtube_url
from app.services.frame_extractor import extract_frames, get_video_duration
from app.services.transcriber import extract_youtube_id, fetch_transcript, process_transcript
from app.services.describer import generate_description
from app.services.prompt_builder import build_manin_prompt
from app.services.job_manager import create_job, get_job
from app.manim_pipeline.renderer import render_job
from app.services.youtube_metadata import generate_youtube_metadata
from app.utils.files import cleanup_video

logger = logging.getLogger(__name__)

app = FastAPI(title="Octoflash", version="0.3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = Path(__file__).parent / "static"
STORAGE_DIR = Path(__file__).parent.parent / "storage"

# Ensure storage/scripts directory exists
(STORAGE_DIR / "scripts").mkdir(parents=True, exist_ok=True)

# Serve storage files (frames, renders) at /storage/...
app.mount("/storage", StaticFiles(directory=str(STORAGE_DIR)), name="storage")


@app.get("/")
def root():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


def _get_oembed_duration(yt_id: str) -> float | None:
    """Get video duration via YouTube oEmbed (no auth needed)."""
    import urllib.request
    import json
    try:
        url = f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={yt_id}&format=json"
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read())
            # oEmbed doesn't return duration directly, but we can estimate from title
            # For a more accurate approach, use youtube-transcript-api snippet times
            return None
    except Exception:
        return None


def _get_youtube_thumbnails(video_id: str, yt_id: str) -> list[str]:
    """Download YouTube thumbnail images as fallback frames."""
    import urllib.request
    from app.utils.files import get_frames_dir

    frames_dir = get_frames_dir(video_id)
    thumbnails = [
        f"https://img.youtube.com/vi/{yt_id}/maxresdefault.jpg",
        f"https://img.youtube.com/vi/{yt_id}/hqdefault.jpg",
        f"https://img.youtube.com/vi/{yt_id}/mqdefault.jpg",
    ]

    saved = []
    for i, thumb_url in enumerate(thumbnails):
        try:
            out_path = frames_dir / f"frame_{i+1:04d}.jpg"
            urllib.request.urlretrieve(thumb_url, str(out_path))
            if out_path.stat().st_size > 1000:  # skip placeholder images
                saved.append(str(out_path.relative_to(frames_dir.parent.parent)))
        except Exception:
            continue

    logger.info("Fetched %d YouTube thumbnails as fallback frames", len(saved))
    return saved


def _get_duration_from_transcript(url: str) -> float:
    """Estimate duration from transcript snippet timestamps."""
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        yt_id = extract_youtube_id(url)
        if not yt_id:
            return 60.0
        api = YouTubeTranscriptApi()
        result = api.fetch(yt_id, languages=["en", "en-US", "en-GB"])
        if result.snippets:
            last = result.snippets[-1]
            return last.start + last.duration
    except Exception:
        pass
    return 60.0


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    # Validate URL
    if not validate_youtube_url(req.url):
        raise HTTPException(status_code=400, detail="Invalid YouTube URL")

    video_id = None
    try:
        # Process transcript — auto-fetch if not provided
        if req.transcript.strip():
            transcript = process_transcript(req.transcript)
        else:
            try:
                transcript = fetch_transcript(req.url)
            except RuntimeError:
                logger.warning("No transcript available, continuing without it")
                transcript = "(No transcript available — generate based on video title and thumbnails)"

        # Try downloading video for frame extraction
        frames = []
        duration = 60.0  # default
        try:
            video_id, video_path = download_video(req.url)
            duration = get_video_duration(video_path)
            frames = extract_frames(video_id, video_path)
            logger.info("Video downloaded, %d frames extracted", len(frames))
        except Exception as dl_err:
            logger.warning("Video download failed, using YouTube thumbnails: %s", dl_err)
            # Fallback: use YouTube thumbnails + transcript-based duration
            video_id = uuid.uuid4().hex[:12]
            yt_id = extract_youtube_id(req.url)
            if yt_id:
                duration = _get_duration_from_transcript(req.url)
                frames = _get_youtube_thumbnails(video_id, yt_id)

        # Generate description
        description = generate_description(transcript, frames, duration)

        # Build Manin prompt
        manin_prompt = build_manin_prompt(transcript, frames, description, duration)

        return AnalyzeResponse(
            video_id=video_id,
            duration_seconds=round(duration, 2),
            frames=frames,
            transcript=transcript,
            description=description,
            manin_prompt=manin_prompt,
        )

    except ValueError as e:
        if video_id:
            cleanup_video(video_id)
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        if video_id:
            cleanup_video(video_id)
        raise HTTPException(status_code=500, detail=str(e))
    except RuntimeError as e:
        if video_id:
            cleanup_video(video_id)
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        if video_id:
            cleanup_video(video_id)
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")


# ── Generate Endpoints ──────────────────────────────────────────────────────


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest, background_tasks: BackgroundTasks):
    """Start a Manim render job. Returns a job_id to poll for status."""
    job_id = uuid.uuid4().hex[:12]
    create_job(job_id)

    background_tasks.add_task(
        render_job,
        job_id=job_id,
        transcript=req.transcript,
        description=req.description,
        duration=req.duration,
        title=req.title,
        orientation=req.orientation,
        quality=req.quality,
        voiceover=req.voiceover,
        source_video_id=req.video_id,
        manin_prompt=req.manin_prompt,
    )

    return GenerateResponse(job_id=job_id, status="pending")


@app.get("/generate/{job_id}/status", response_model=JobStatusResponse)
def generate_status(job_id: str):
    """Poll the status of a render job."""
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    return JobStatusResponse(**job)


@app.get("/generate/{job_id}/video")
def generate_video(job_id: str, orientation: str = Query("landscape")):
    """Download the rendered MP4 for a completed job."""
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Job is {job['status']}, not completed")

    video_key = "portrait_video" if orientation == "portrait" else "landscape_video"
    video_path = job.get(video_key)

    if not video_path or not Path(video_path).exists():
        raise HTTPException(status_code=404, detail=f"No {orientation} video found for job {job_id}")

    return FileResponse(
        video_path,
        media_type="video/mp4",
        filename=f"octoflash_{job_id}_{orientation}.mp4",
    )


@app.get("/scripts/{video_id}")
def get_script(video_id: str):
    """Return the generated Manim script as plain text."""
    script_file = STORAGE_DIR / "scripts" / video_id / "episode.py"
    if not script_file.exists():
        raise HTTPException(status_code=404, detail=f"No script found for video {video_id}")

    return PlainTextResponse(
        content=script_file.read_text(),
        media_type="text/plain",
    )


@app.get("/generate/{job_id}/output-frames")
def get_output_frames(job_id: str):
    """Return list of output frame paths for a rendered job."""
    frames_dir = STORAGE_DIR / "renders" / job_id / "output_frames"
    if not frames_dir.exists():
        return {"frames": []}

    frames = sorted(frames_dir.glob("frame_*.jpg"))
    # Return paths relative to storage dir so /storage/... serves them
    return {
        "frames": [str(f.relative_to(STORAGE_DIR)) for f in frames]
    }



# ── YouTube Metadata ──────────────────────────────────────────────────────

class YouTubeMetadataRequest(BaseModel):
    transcript: str
    title: str = ""
    duration: float = 60.0


@app.post("/youtube-metadata")
def youtube_metadata_endpoint(req: YouTubeMetadataRequest):
    """Generate YouTube title, description, and tags from transcript."""
    if not req.transcript:
        raise HTTPException(status_code=400, detail="Transcript is required")

    try:
        meta = generate_youtube_metadata(req.transcript, req.title, req.duration)
        return meta
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Metadata generation failed: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
