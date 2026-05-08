import asyncio
import logging
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.models.schemas import (
    AnalyzeRequest, AnalyzeResponse, AnalyzeMultiRequest,
    GenerateRequest, GenerateResponse, JobStatusResponse,
    PublishRequest, QueueVideoRequest, VideoUpdateRequest,
    ChannelRequest, ChannelUpdateRequest,
)
from app.services.downloader import download_video, validate_youtube_url
from app.services.frame_extractor import extract_frames, get_video_duration
from app.services.transcriber import extract_youtube_id, fetch_transcript, process_transcript
from app.services.describer import generate_description
from app.services.prompt_builder import build_manin_prompt
from app.services.script_generator import synthesize_concepts
from app.services.job_manager import create_job, get_job, update_job
from app.manim_pipeline.renderer import render_job
from app.services.youtube_metadata import generate_youtube_metadata
from app.services.youtube_uploader import (
    get_youtube_auth_url, handle_youtube_callback, is_youtube_connected,
    upload_to_youtube, disconnect_youtube, list_playlists,
)
from app.services.video_manager import (
    create_video, get_video, update_video, list_videos,
)
from app.services.channel_manager import (
    create_channel, list_channels, delete_channel, update_channel,
)
from app.utils.files import cleanup_video

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start auto-processor on startup, stop on shutdown."""
    from app.services.auto_processor import auto_process_loop, stop as stop_processor
    task = asyncio.create_task(auto_process_loop())
    logger.info("Lifespan: auto-processor task created")
    yield
    stop_processor()
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    logger.info("Lifespan: auto-processor stopped")


app = FastAPI(title="Octoflash", version="0.4.0", lifespan=lifespan)

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
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def root():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/favicon.ico")
def favicon():
    return FileResponse(STATIC_DIR / "favicon.ico")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/api/config")
def api_config():
    """Return Firebase config for the frontend client SDK."""
    return {
        "firebase_project_id": os.environ.get("FIREBASE_PROJECT_ID", ""),
        "firebase_api_key": os.environ.get("FIREBASE_WEB_API_KEY", ""),
    }


# ── Video CRUD (Firebase) ────────────────────────────────────────────────


@app.get("/api/videos")
def api_list_videos():
    """List all videos from Firebase."""
    return {"videos": list_videos(limit=50)}


@app.post("/api/videos")
def api_queue_video(req: QueueVideoRequest):
    """Queue a new video URL for processing. Optionally tag with channel_id for tracking."""
    if not validate_youtube_url(req.url):
        raise HTTPException(status_code=400, detail="Invalid YouTube URL")
    video = create_video(
        req.url,
        source=req.source,
        channel_id=req.channel_id,
        source_short_youtube_id=req.source_short_youtube_id,
    )
    if video is None:
        raise HTTPException(status_code=503, detail="Firebase not available")
    return video


@app.get("/api/videos/{video_id}")
def api_get_video(video_id: str):
    """Get a single video by ID."""
    video = get_video(video_id)
    if video is None:
        raise HTTPException(status_code=404, detail=f"Video {video_id} not found")
    return video


@app.patch("/api/videos/{video_id}")
def api_update_video(video_id: str, req: VideoUpdateRequest):
    """Update editable fields on a video."""
    fields = {k: v for k, v in req.model_dump().items() if v is not None}
    if not fields:
        raise HTTPException(status_code=400, detail="No fields to update")
    video = update_video(video_id, **fields)
    if video is None:
        raise HTTPException(status_code=404, detail=f"Video {video_id} not found")
    return video


@app.post("/api/videos/{video_id}/analyze")
def api_trigger_analyze(video_id: str, background_tasks: BackgroundTasks):
    """Manually trigger analysis for a video."""
    video = get_video(video_id)
    if video is None:
        raise HTTPException(status_code=404, detail=f"Video {video_id} not found")
    update_video(video_id, status="queued", error=None)
    return {"status": "queued", "video_id": video_id}


@app.get("/api/channels")
def api_list_channels():
    """List all followed channels."""
    return {"channels": list_channels(limit=100)}


@app.post("/api/channels")
def api_create_channel(req: ChannelRequest):
    """Add a new channel to follow."""
    if "youtube.com" not in req.url and "youtu.be" not in req.url:
        raise HTTPException(status_code=400, detail="URL must be a YouTube channel URL")
    ch = create_channel(req.url, name=req.name, notes=req.notes)
    if ch is None:
        raise HTTPException(status_code=503, detail="Firebase not available")
    return ch


@app.delete("/api/channels/{channel_id}")
def api_delete_channel(channel_id: str):
    """Unfollow a channel."""
    ok = delete_channel(channel_id)
    if not ok:
        raise HTTPException(status_code=503, detail="Firebase not available")
    return {"deleted": channel_id}


@app.patch("/api/channels/{channel_id}")
def api_update_channel(channel_id: str, req: ChannelUpdateRequest):
    fields = {k: v for k, v in req.model_dump().items() if v is not None}
    ch = update_channel(channel_id, **fields)
    if ch is None:
        raise HTTPException(status_code=404, detail=f"Channel {channel_id} not found")
    return ch


@app.get("/api/channels/{channel_id}/used-shorts")
def api_used_shorts(channel_id: str):
    """Map of source_short_youtube_id → video metadata for shorts queued from this channel."""
    from app.services.firebase_client import videos_collection
    from google.cloud.firestore_v1.base_query import FieldFilter
    col = videos_collection()
    if col is None:
        raise HTTPException(status_code=503, detail="Firebase not available")
    used = {}
    for doc in col.where(filter=FieldFilter("channel_id", "==", channel_id)).stream():
        d = doc.to_dict()
        yt_id = d.get("source_short_youtube_id")
        if not yt_id:
            continue
        publish = (d.get("publish") or {}).get("youtube") or {}
        used[yt_id] = {
            "video_id": d.get("video_id"),
            "status": d.get("status"),
            "title": d.get("title"),
            "published_url": publish.get("url"),
            "created_at": d.get("created_at").isoformat() if hasattr(d.get("created_at"), "isoformat") else None,
        }
    return {"channel_id": channel_id, "used_shorts": used}


@app.get("/api/channels/{channel_id}/shorts")
def api_channel_shorts(channel_id: str, limit: int = Query(5)):
    """Fetch recent shorts from a followed channel via yt-dlp."""
    from app.services.channel_manager import _channels_collection
    col = _channels_collection()
    if col is None:
        raise HTTPException(status_code=503, detail="Firebase not available")
    doc = col.document(channel_id).get()
    if not doc.exists:
        raise HTTPException(status_code=404, detail=f"Channel {channel_id} not found")

    channel = doc.to_dict()
    url = channel.get("url", "")

    # Try shorts URL variant first, fall back to channel URL
    shorts_urls = []
    if "/shorts" not in url:
        if url.endswith("/"):
            shorts_urls.append(url + "shorts")
        else:
            shorts_urls.append(url + "/shorts")
    shorts_urls.append(url)

    import json as _json
    import subprocess as _sp
    last_err = None
    for try_url in shorts_urls:
        try:
            result = _sp.run(
                [
                    "yt-dlp", "--flat-playlist", "--dump-json",
                    "--playlist-end", str(limit),
                    "--skip-download",
                    try_url,
                ],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode != 0:
                last_err = result.stderr[-300:]
                continue
            shorts = []
            for line in result.stdout.strip().splitlines():
                try:
                    item = _json.loads(line)
                    yt_id = item.get("id", "")
                    shorts.append({
                        "id": yt_id,
                        "title": item.get("title", ""),
                        "url": item.get("url") or f"https://www.youtube.com/shorts/{yt_id}",
                        "thumbnail": f"https://img.youtube.com/vi/{yt_id}/mqdefault.jpg" if yt_id else "",
                        "duration": item.get("duration"),
                    })
                except Exception:
                    continue
            return {"channel_id": channel_id, "shorts": shorts}
        except Exception as e:
            last_err = str(e)
            continue

    raise HTTPException(status_code=502, detail=f"Failed to fetch shorts: {last_err}")


@app.post("/api/videos/{video_id}/generate")
def api_trigger_generate(video_id: str, background_tasks: BackgroundTasks):
    """Manually trigger generation for an analyzed video."""
    video = get_video(video_id)
    if video is None:
        raise HTTPException(status_code=404, detail=f"Video {video_id} not found")
    if video.get("status") not in ("analyzed", "generated", "failed"):
        raise HTTPException(status_code=400, detail=f"Video must be analyzed first (current: {video.get('status')})")

    job_id = uuid.uuid4().hex[:12]
    update_video(video_id, status="generating", job_id=job_id, error=None)

    from app.services.auto_processor import _run_generation
    background_tasks.add_task(
        _run_generation,
        job_id=job_id,
        video_id=video_id,
        analysis={
            "transcript": video.get("transcript", ""),
            "description": video.get("description", ""),
            "duration_seconds": video.get("duration_seconds", 60.0),
            "manin_prompt": video.get("manin_prompt", ""),
            "frames": video.get("frame_paths", []),
        },
        orientation=video.get("orientation", "portrait"),
        quality=video.get("quality", "qm"),
        voiceover=video.get("voiceover", True),
        title=video.get("title", ""),
    )
    return {"status": "generating", "video_id": video_id, "job_id": job_id}


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


# ── Multi-URL Analyze ──────────────────────────────────────────────────────


@app.post("/analyze-multi", response_model=AnalyzeResponse)
def analyze_multi(req: AnalyzeMultiRequest):
    """Analyze multiple YouTube URLs and combine their content into one."""
    if not req.urls or len(req.urls) < 2:
        raise HTTPException(status_code=400, detail="Provide at least 2 URLs")
    if len(req.urls) > 5:
        raise HTTPException(status_code=400, detail="Maximum 5 URLs allowed")

    # Validate all URLs first
    for url in req.urls:
        if not validate_youtube_url(url):
            raise HTTPException(status_code=400, detail=f"Invalid YouTube URL: {url}")

    all_transcripts = []
    all_frames = []
    total_duration = 0.0
    combined_video_id = uuid.uuid4().hex[:12]

    for i, url in enumerate(req.urls):
        logger.info("Analyzing URL %d/%d: %s", i + 1, len(req.urls), url)

        # Fetch transcript
        try:
            transcript = fetch_transcript(url)
        except RuntimeError:
            logger.warning("No transcript for URL %d, using placeholder", i + 1)
            transcript = f"(No transcript available for video {i + 1})"
        all_transcripts.append(f"--- Video {i + 1} ---\n{transcript}")

        # Try video download for frames, fallback to thumbnails
        try:
            vid_id, video_path = download_video(url)
            duration = get_video_duration(video_path)
            frames = extract_frames(vid_id, video_path)
            total_duration += duration
            all_frames.extend(frames)
        except Exception as dl_err:
            logger.warning("Download failed for URL %d, using thumbnails: %s", i + 1, dl_err)
            yt_id = extract_youtube_id(url)
            if yt_id:
                total_duration += _get_duration_from_transcript(url)
                frames = _get_youtube_thumbnails(combined_video_id, yt_id)
                all_frames.extend(frames)
            else:
                total_duration += 60.0

    # Synthesize concepts across all videos
    transcript_dicts = [
        {"url": url, "transcript": t, "title": f"Video {i + 1}"}
        for i, (url, t) in enumerate(zip(req.urls, all_transcripts))
    ]

    try:
        synthesis = synthesize_concepts(
            transcripts=transcript_dicts,
            frames=all_frames[:12],
            total_duration=total_duration,
        )
        combined_transcript = synthesis["combined_transcript"]
        synthesized_title = synthesis.get("title")
        logger.info("Concept synthesis succeeded: title='%s'", synthesized_title)
    except Exception as synth_err:
        logger.warning("Concept synthesis failed, falling back to concatenation: %s", synth_err)
        combined_transcript = "\n\n".join(all_transcripts)
        synthesis = None
        synthesized_title = None

    # Generate combined description
    description = generate_description(combined_transcript, all_frames[:12], total_duration)

    # Build prompt (with synthesis data if available)
    manin_prompt = build_manin_prompt(
        combined_transcript, all_frames[:12], description, total_duration,
        synthesis=synthesis,
    )

    logger.info("Multi-URL analysis complete: %d URLs, %.1fs total, %d frames",
                len(req.urls), total_duration, len(all_frames))

    return AnalyzeResponse(
        video_id=combined_video_id,
        duration_seconds=round(total_duration, 2),
        frames=all_frames[:12],  # cap at 12 frames
        transcript=combined_transcript,
        description=description,
        manin_prompt=manin_prompt,
        synthesized_title=synthesized_title,
    )


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


# ── YouTube OAuth & Publish ─────────────────────────────────────────────


@app.get("/oauth/youtube/start")
def youtube_oauth_start():
    """Return Google OAuth2 consent URL for YouTube."""
    try:
        auth_url = get_youtube_auth_url()
        return {"auth_url": auth_url}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/oauth/youtube/callback")
def youtube_oauth_callback(code: str = Query(...)):
    """Handle OAuth callback — exchange code for tokens, close popup."""
    from fastapi.responses import HTMLResponse
    try:
        handle_youtube_callback(code)
        return HTMLResponse("""
        <html><body style="font-family:Inter,sans-serif;display:flex;align-items:center;justify-content:center;height:100vh;background:#f8f9fb;">
        <div style="text-align:center;">
            <h2 style="color:#16a34a;">Connected!</h2>
            <p style="color:#6b7280;">YouTube account linked. You can close this window.</p>
        </div>
        <script>
            if (window.opener) {
                window.opener.postMessage({type: 'youtube-connected'}, '*');
            }
            setTimeout(() => window.close(), 1500);
        </script>
        </body></html>
        """)
    except Exception as e:
        from fastapi.responses import HTMLResponse
        return HTMLResponse(f"""
        <html><body style="font-family:Inter,sans-serif;display:flex;align-items:center;justify-content:center;height:100vh;background:#f8f9fb;">
        <div style="text-align:center;">
            <h2 style="color:#dc2626;">Connection Failed</h2>
            <p style="color:#6b7280;">{e}</p>
        </div>
        </body></html>
        """, status_code=400)


@app.get("/oauth/youtube/status")
def youtube_oauth_status():
    """Check if YouTube is connected."""
    return {"connected": is_youtube_connected()}


@app.post("/oauth/youtube/disconnect")
def youtube_oauth_disconnect():
    """Remove stored YouTube tokens."""
    return disconnect_youtube()


@app.get("/oauth/youtube/playlists")
def youtube_playlists():
    """List the user's YouTube playlists."""
    return {"playlists": list_playlists()}


@app.post("/publish")
def publish_video(req: PublishRequest, background_tasks: BackgroundTasks):
    """Start publishing to selected platforms."""
    job = get_job(req.job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {req.job_id} not found")
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail="Job not completed yet")

    # Initialize publish status
    publish_status = {}
    for platform in req.platforms:
        publish_status[platform] = {"status": "pending"}

    update_job(req.job_id, publish=publish_status)

    background_tasks.add_task(
        _run_publish,
        job_id=req.job_id,
        job=job,
        title=req.title,
        description=req.description,
        tags=req.tags,
        privacy=req.privacy,
        platforms=req.platforms,
        video_type=req.video_type,
        playlist=req.playlist,
    )

    return {"job_id": req.job_id, "publish": publish_status}


@app.get("/publish/{job_id}/status")
def publish_status(job_id: str):
    """Get publish status for a job."""
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return {"job_id": job_id, "publish": job.get("publish")}


def _run_publish(
    job_id: str,
    job: dict,
    title: str,
    description: str,
    tags: str,
    privacy: str,
    platforms: list[str],
    video_type: str = "shorts",
    playlist: str = "",
):
    """Background task: upload to each selected platform."""
    for platform in platforms:
        try:
            update_job(job_id, publish={
                **get_job(job_id).get("publish", {}),
                platform: {"status": "uploading"},
            })

            if platform == "youtube":
                # Pick the best available video file
                video_path = job.get("portrait_video") or job.get("landscape_video")
                if not video_path:
                    raise FileNotFoundError("No rendered video found")

                result = upload_to_youtube(
                    video_path=video_path,
                    title=title,
                    description=description,
                    tags=tags,
                    privacy=privacy,
                    video_type=video_type,
                    playlist=playlist,
                )
                update_job(job_id, publish={
                    **get_job(job_id).get("publish", {}),
                    platform: {"status": "completed", **result},
                })
            else:
                update_job(job_id, publish={
                    **get_job(job_id).get("publish", {}),
                    platform: {"status": "unsupported", "error": f"{platform} coming soon"},
                })

        except Exception as e:
            logger.error("Publish to %s failed: %s", platform, e)
            update_job(job_id, publish={
                **get_job(job_id).get("publish", {}),
                platform: {"status": "failed", "error": str(e)},
            })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app", host="0.0.0.0", port=8000,
        reload=True,
        reload_dirs=["app"],
    )
