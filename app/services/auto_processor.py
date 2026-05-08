"""
Background auto-processing loop.

Polls Firestore for queued videos every 15 seconds,
then runs analyze → generate automatically.
"""

import asyncio
import logging
import uuid

from app.services.video_manager import get_queued_videos, update_video
from app.services.firebase_client import is_firebase_available

logger = logging.getLogger(__name__)

_running = False


async def auto_process_loop():
    """Main loop: poll for queued videos and process them."""
    global _running

    if not is_firebase_available():
        logger.warning("Firebase not available — auto-processor disabled")
        return

    _running = True
    logger.info("Auto-processor started (polling every 15s)")

    while _running:
        try:
            queued = get_queued_videos(limit=3)
            if queued:
                logger.info("Auto-processor found %d queued video(s)", len(queued))

            for video in queued:
                try:
                    await _process_video(video)
                except Exception as e:
                    video_id = video.get("video_id", "?")
                    logger.error("Auto-processor failed for %s: %s", video_id, e)
                    update_video(video_id, status="failed", error=str(e)[:500])

        except Exception as e:
            logger.error("Auto-processor poll error: %s", e)

        await asyncio.sleep(15)


async def _process_video(video: dict):
    """Process a single queued video: analyze → generate."""
    video_id = video["video_id"]
    source_url = video["source_url"]

    logger.info("Processing video %s: %s", video_id, source_url)

    # --- Phase 1: Analyze ---
    update_video(video_id, status="analyzing")

    analysis = await asyncio.to_thread(_run_analysis, source_url, video_id)

    # Derive a title from the first line of description
    auto_title = (analysis.get("description") or "").split("\n")[0][:60].strip() or "Auto-Generated Video"

    update_video(
        video_id,
        status="analyzed",
        duration_seconds=analysis["duration_seconds"],
        frame_paths=analysis["frames"],
        frame_count=len(analysis["frames"]),
        transcript=analysis["transcript"],
        description=analysis["description"],
        manin_prompt=analysis["manin_prompt"],
        title=auto_title,
    )
    logger.info("Video %s analyzed successfully (title='%s')", video_id, auto_title)

    # --- Phase 2: Generate ---
    update_video(video_id, status="generating")

    job_id = uuid.uuid4().hex[:12]
    update_video(video_id, job_id=job_id)

    await asyncio.to_thread(
        _run_generation,
        job_id=job_id,
        video_id=video_id,
        analysis=analysis,
        orientation=video.get("orientation", "portrait"),
        quality=video.get("quality", "qm"),
        voiceover=video.get("voiceover", True),
    )

    # Check job result
    from app.services.job_manager import get_job
    job = get_job(job_id)

    if job and job.get("status") == "completed":
        update_video(
            video_id,
            status="generated",
            landscape_video=job.get("landscape_video"),
            portrait_video=job.get("portrait_video"),
            scene_file=job.get("scene_file"),
            script_file=job.get("script_file"),
        )
        logger.info("Video %s generated successfully (job_id=%s)", video_id, job_id)
    else:
        error = job.get("error", "Unknown render error") if job else "Job not found"
        update_video(video_id, status="failed", error=error)
        logger.error("Video %s generation failed: %s", video_id, error)


def _run_analysis(source_url: str, video_id: str) -> dict:
    """Run the analysis pipeline (same logic as /analyze endpoint)."""
    import uuid as _uuid
    from app.services.downloader import download_video, validate_youtube_url
    from app.services.frame_extractor import extract_frames, get_video_duration
    from app.services.transcriber import extract_youtube_id, fetch_transcript
    from app.services.describer import generate_description
    from app.services.prompt_builder import build_manin_prompt

    if not validate_youtube_url(source_url):
        raise ValueError(f"Invalid YouTube URL: {source_url}")

    # Fetch transcript
    try:
        transcript = fetch_transcript(source_url)
    except RuntimeError:
        logger.warning("No transcript available for %s", source_url)
        transcript = "(No transcript available)"

    # Download and extract frames
    frames = []
    duration = 60.0
    try:
        dl_video_id, video_path = download_video(source_url)
        duration = get_video_duration(video_path)
        frames = extract_frames(dl_video_id, video_path)
    except Exception as dl_err:
        logger.warning("Download failed for %s, using thumbnails: %s", source_url, dl_err)
        yt_id = extract_youtube_id(source_url)
        if yt_id:
            from app.main import _get_duration_from_transcript, _get_youtube_thumbnails
            duration = _get_duration_from_transcript(source_url)
            frames = _get_youtube_thumbnails(video_id, yt_id)

    # Generate description and prompt
    description = generate_description(transcript, frames, duration)
    manin_prompt = build_manin_prompt(transcript, frames, description, duration)

    return {
        "video_id": video_id,
        "duration_seconds": round(duration, 2),
        "frames": frames,
        "transcript": transcript,
        "description": description,
        "manin_prompt": manin_prompt,
    }


def _run_generation(
    job_id: str,
    video_id: str,
    analysis: dict,
    orientation: str = "portrait",
    quality: str = "qm",
    voiceover: bool = True,
    title: str = "",
):
    """Run the render pipeline."""
    from app.services.job_manager import create_job
    from app.manim_pipeline.renderer import render_job

    create_job(job_id, video_id=video_id)

    render_title = title or (analysis.get("description") or "").split("\n")[0][:60].strip() or "Auto-Generated Video"

    render_job(
        job_id=job_id,
        transcript=analysis["transcript"],
        description=analysis["description"],
        duration=analysis["duration_seconds"],
        title=render_title,
        orientation=orientation,
        quality=quality,
        voiceover=voiceover,
        source_video_id=video_id,
        manin_prompt=analysis.get("manin_prompt", ""),
        _firebase_video_id=video_id,
    )


def stop():
    """Stop the auto-processor loop."""
    global _running
    _running = False
    logger.info("Auto-processor stopped")
