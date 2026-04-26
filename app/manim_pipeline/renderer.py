"""
Render a generated Manim scene to MP4.

Writes the scene .py file to disk, invokes `manim` as a subprocess,
and returns the path to the rendered video. Integrates with job_manager
for status updates throughout the render lifecycle.

Fallback chain: Claude+voice -> Claude-no-voice -> simple generator.
After successful render, optionally runs iterative improvement loop.
"""

import logging
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from app.services.job_manager import update_job
from app.manim_pipeline.scene_generator import validate_scene_code, generate_scene_code
from app.services.script_generator import (
    generate_episode_script,
    save_script,
    evaluate_output,
    extract_video_frames,
    strip_voiceover,
    sanitize_script,
)

logger = logging.getLogger(__name__)

STORAGE_DIR = Path(__file__).resolve().parent.parent.parent / "storage"

MAX_IMPROVEMENT_ITERATIONS = 3


def _detect_scene_class(scene_code: str) -> str:
    """Detect the scene class name from generated code."""
    match = re.search(
        r"class\s+(\w+)\s*\(\s*(?:OctoflashScene|Scene|VoiceoverScene|Octoflash3DScene)",
        scene_code,
    )
    return match.group(1) if match else "InspiredVideoScene"


def render_scene(
    video_id: str,
    scene_code: str,
    quality: str = "qh",
    portrait: bool = False,
) -> dict:
    """Write scene to disk and render it. Returns paths to scene file and video."""
    job_dir = STORAGE_DIR / "renders" / video_id
    job_dir.mkdir(parents=True, exist_ok=True)

    scene_file = job_dir / "scene.py"
    scene_file.write_text(scene_code)

    scene_class = _detect_scene_class(scene_code)

    orientation = "portrait" if portrait else "landscape"
    media_dir = job_dir / "media" / orientation

    resolution = "1080,1920" if portrait else "1920,1080"

    cmd = [
        "manim",
        f"-{quality}",
        str(scene_file),
        scene_class,
        "--media_dir", str(media_dir),
        "--resolution", resolution,
    ]

    logger.info("Running manim: %s", " ".join(cmd))

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=600,
        cwd=str(STORAGE_DIR.parent),
        env=_build_env(),
    )

    if result.returncode != 0:
        error_msg = _classify_error(result.stderr)
        # Log full stderr for debugging (truncated at 2000 chars)
        logger.error("Manim render failed (class=%s):\n--- STDOUT ---\n%s\n--- STDERR ---\n%s",
                     scene_class, result.stdout[:1000], result.stderr[:2000])
        raise RuntimeError(error_msg)

    video_file = _find_rendered_video(media_dir)

    return {
        "scene_file": str(scene_file),
        "video_file": str(video_file) if video_file else None,
        "media_dir": str(media_dir),
        "stderr": result.stderr,
    }


def render_job(
    job_id: str,
    transcript: str,
    description: str,
    duration: float,
    title: str,
    orientation: str = "landscape",
    quality: str = "qm",
    voiceover: bool = True,
    source_video_id: str = "",
    manin_prompt: str = "",
) -> None:
    """Full render pipeline with fallback chain and iterative improvement.

    Fallback order:
    1. Claude script with voiceover (OctoflashScene)
    2. Claude script WITHOUT voiceover (Scene) — keeps rich animations
    3. Simple generator without voiceover — last resort

    After successful render, evaluates output quality and may iterate.
    """
    try:
        update_job(job_id, status="rendering")
        portrait = orientation == "portrait"

        # Collect source frame paths for vision analysis (use source video_id)
        source_frame_paths = _find_source_frames(source_video_id or job_id)

        # === STEP 1: Generate script with Claude ===
        claude_code = None
        script_file_path = None

        try:
            claude_code = generate_episode_script(
                transcript=transcript,
                description=description,
                duration=duration,
                title=title,
                video_id=job_id,
                voiceover=voiceover,
                source_frames=source_frame_paths,
                manin_prompt=manin_prompt,
            )
            script_file_path = save_script(job_id, claude_code)
            logger.info("Claude script (voiceover=%s) saved to %s", voiceover, script_file_path)
        except Exception as e:
            logger.warning("Claude script generation failed: %s", e)

        # === STEP 2: Try rendering with fallback chain ===
        scene_code = None
        result = None
        render_method = None

        # Attempt 1: Claude script with voiceover
        if claude_code and voiceover:
            try:
                logger.info("Attempt 1: Rendering Claude script with voiceover (%d chars)", len(claude_code))
                validate_scene_code(claude_code)
                result = render_scene(job_id, claude_code, quality=quality, portrait=portrait)
                scene_code = claude_code
                render_method = "claude+voice"
                logger.info("Render succeeded: claude+voice")
            except Exception as e:
                logger.warning("Claude+voice render failed: %s", str(e)[:500])

        # Attempt 2: Strip voiceover from Claude script (keep rich animations)
        if result is None and claude_code:
            try:
                logger.info("Attempt 2: Stripping voiceover and retrying")
                claude_no_voice = strip_voiceover(claude_code)
                validate_scene_code(claude_no_voice)
                result = render_scene(job_id, claude_no_voice, quality=quality, portrait=portrait)
                scene_code = claude_no_voice
                render_method = "claude-no-voice"
                script_file_path = save_script(job_id, claude_no_voice)
                logger.info("Render succeeded: claude-no-voice (stripped from voiceover script)")
            except Exception as e:
                logger.warning("Claude-no-voice render failed: %s", str(e)[:500])

        # Attempt 3: Generate fresh no-voice script from Claude
        if result is None:
            try:
                logger.info("Attempt 3: Generating fresh no-voice script from Claude")
                fresh_no_voice = generate_episode_script(
                    transcript=transcript,
                    description=description,
                    duration=duration,
                    title=title,
                    video_id=job_id,
                    voiceover=False,
                    source_frames=source_frame_paths,
                    manin_prompt=manin_prompt,
                )
                validate_scene_code(fresh_no_voice)
                result = render_scene(job_id, fresh_no_voice, quality=quality, portrait=portrait)
                scene_code = fresh_no_voice
                render_method = "claude-fresh-no-voice"
                script_file_path = save_script(job_id, fresh_no_voice)
                logger.info("Render succeeded: claude-fresh-no-voice")
            except Exception as e:
                logger.warning("Claude fresh-no-voice render failed: %s", str(e)[:300])

        # Attempt 4: Simple generator (last resort)
        if result is None:
            logger.warning("All Claude attempts failed, falling back to simple generator")
            simple_code = generate_scene_code(
                video_id=job_id,
                transcript=transcript,
                description=description,
                duration=duration,
                title=title,
                voiceover=False,
            )
            validate_scene_code(simple_code)
            result = render_scene(job_id, simple_code, quality=quality, portrait=portrait)
            scene_code = simple_code
            render_method = "simple-fallback"

        # === STEP 3: Store paths ===
        scene_file = str(STORAGE_DIR / "renders" / job_id / "scene.py")
        update_fields = {"scene_file": scene_file}
        if script_file_path:
            update_fields["script_file"] = str(script_file_path)
        update_job(job_id, **update_fields)

        # === STEP 4: Iterative improvement (only for Claude scripts) ===
        if render_method and render_method.startswith("claude") and result.get("video_file"):
            result, scene_code = _improvement_loop(
                job_id=job_id,
                current_result=result,
                current_code=scene_code,
                transcript=transcript,
                description=description,
                duration=duration,
                title=title,
                quality=quality,
                portrait=portrait,
                voiceover=(render_method == "claude+voice"),
                source_frames=source_frame_paths,
                manin_prompt=manin_prompt,
            )

        # === STEP 5: Apply watermark (intro + outro branding) ===
        if result.get("video_file") and Path(result["video_file"]).exists():
            try:
                res = "1080x1920" if portrait else "1920x1080"
                _apply_watermark(Path(result["video_file"]), resolution=res)
            except Exception as e:
                logger.warning("Watermark application failed: %s", e)

        # === STEP 6: Extract output frames for UI display ===
        if result.get("video_file") and Path(result["video_file"]).exists():
            try:
                output_frames_dir = STORAGE_DIR / "renders" / job_id / "output_frames"
                output_frames_dir.mkdir(parents=True, exist_ok=True)
                _extract_output_frames(Path(result["video_file"]), output_frames_dir)
                logger.info("Output frames saved to %s", output_frames_dir)
            except Exception as e:
                logger.warning("Failed to extract output frames: %s", e)

        # === STEP 7: Mark complete ===
        video_key = "portrait_video" if portrait else "landscape_video"
        update_job(
            job_id,
            status="completed",
            completed_at=datetime.now(timezone.utc).isoformat(),
            **{video_key: result.get("video_file")},
        )

    except Exception as e:
        logger.exception("render_job failed for %s", job_id)
        update_job(
            job_id,
            status="failed",
            completed_at=datetime.now(timezone.utc).isoformat(),
            error=str(e)[:1000],
        )


def _improvement_loop(
    job_id: str,
    current_result: dict,
    current_code: str,
    transcript: str,
    description: str,
    duration: float,
    title: str,
    quality: str,
    portrait: bool,
    voiceover: bool,
    source_frames: list[Path] | None,
    manin_prompt: str = "",
) -> tuple[dict, str]:
    """Run evaluate-improve-rerender loop up to MAX_IMPROVEMENT_ITERATIONS times."""

    result = current_result
    code = current_code

    for iteration in range(MAX_IMPROVEMENT_ITERATIONS):
        video_path = result.get("video_file")
        if not video_path or not Path(video_path).exists():
            break

        try:
            update_job(job_id, status="rendering",
                       error=f"Self-correcting: evaluating output (iteration {iteration + 1}/{MAX_IMPROVEMENT_ITERATIONS})")

            # Extract frames from rendered output
            output_frames = extract_video_frames(Path(video_path), count=8)
            if not output_frames:
                logger.warning("No output frames extracted, stopping improvement")
                break

            # Evaluate quality by comparing output against source frames
            evaluation = evaluate_output(output_frames, transcript, code, source_frames)
            logger.info("Iteration %d evaluation: score=%d", iteration + 1, evaluation["score"])

            if evaluation["passed"]:
                logger.info("Quality passed (score=%d), stopping improvement", evaluation["score"])
                update_job(job_id, error=f"Quality check passed (score={evaluation['score']}/10)")
                break

            # Generate improved script
            feedback = evaluation["feedback"]
            logger.info("Improving script (iteration %d, score=%d): %s",
                        iteration + 1, evaluation["score"], feedback[:300])

            update_job(job_id, status="rendering",
                       error=f"Self-correcting: score {evaluation['score']}/10, regenerating (iteration {iteration + 1}/{MAX_IMPROVEMENT_ITERATIONS})")

            improved_code = generate_episode_script(
                transcript=transcript,
                description=description,
                duration=duration,
                title=title,
                video_id=job_id,
                voiceover=voiceover,
                source_frames=source_frames,
                feedback=feedback,
                manin_prompt=manin_prompt,
            )

            validate_scene_code(improved_code)

            update_job(job_id, status="rendering",
                       error=f"Self-correcting: re-rendering improved script (iteration {iteration + 1}/{MAX_IMPROVEMENT_ITERATIONS})")

            new_result = render_scene(job_id, improved_code, quality=quality, portrait=portrait)

            # Success — update
            code = improved_code
            result = new_result
            save_script(job_id, improved_code)
            logger.info("Improvement iteration %d rendered successfully", iteration + 1)

        except Exception as e:
            logger.warning("Improvement iteration %d failed: %s", iteration + 1, e)
            break  # Keep the last working version

    # Clear the status error field before returning
    update_job(job_id, error="")

    return result, code


def _find_source_frames(video_id: str) -> list[Path] | None:
    """Find source video frames for vision analysis, in sequential order.

    First tries storage/{video_id}/frames/, then falls back to any
    storage/*/frames/ directory. Returns frames sorted sequentially.
    """
    # Try exact video_id directory first
    exact_dir = STORAGE_DIR / video_id / "frames"
    if exact_dir.exists():
        frames = sorted(exact_dir.glob("frame_*.jpg"))
        if frames:
            logger.info("Found %d source frames in %s", len(frames), exact_dir)
            return frames

    # Fallback: search all frame directories, pick the one with most frames
    best_frames = []
    for frames_dir in sorted(STORAGE_DIR.glob("*/frames")):
        frames = sorted(frames_dir.glob("frame_*.jpg"))
        if len(frames) > len(best_frames):
            best_frames = frames

    if best_frames:
        logger.info("Found %d source frames (fallback search)", len(best_frames))
        return best_frames

    return None


def _classify_error(stderr: str) -> str:
    """Classify manim subprocess errors for clearer reporting.

    Covers all common Manim CE error categories for actionable diagnostics.
    """
    stderr_lower = stderr.lower()

    if "modulenotfounderror" in stderr_lower or "importerror" in stderr_lower:
        return f"Import error — missing dependency:\n{stderr.strip()}"
    if "elevenlabs" in stderr_lower or "api_key" in stderr_lower or "voiceover" in stderr_lower:
        return f"Voiceover/ElevenLabs error:\n{stderr.strip()}"
    if "syntaxerror" in stderr_lower:
        return f"Syntax error in generated scene:\n{stderr.strip()}"
    if "timeout" in stderr_lower:
        return f"Render timed out:\n{stderr.strip()}"
    if "cannot create a mobject from an empty string" in stderr_lower:
        return f"Empty string mobject error (Text/Tex/MathTex given empty string):\n{stderr.strip()}"
    if "latex error converting" in stderr_lower or "pdflatex error" in stderr_lower:
        return f"LaTeX compilation error (invalid TeX in MathTex/Tex):\n{stderr.strip()}"
    if "no tex installation" in stderr_lower or "no such file or directory: 'latex'" in stderr_lower:
        return f"LaTeX not installed:\n{stderr.strip()}"
    if "only works for vmobjects" in stderr_lower or "only works for vectorized" in stderr_lower:
        return f"Animation type error (Create/Write used on non-VMobject):\n{stderr.strip()}"
    if "truth value of an array" in stderr_lower:
        return f"Numpy array truth value error (use np.maximum/np.minimum instead of max/min):\n{stderr.strip()}"
    if "called scene.play with no animations" in stderr_lower:
        return f"Empty play() call (no animations passed):\n{stderr.strip()}"
    if "animation only works on mobjects" in stderr_lower:
        return f"Non-Mobject passed to animation:\n{stderr.strip()}"
    if "run_time" in stderr_lower and "cannot be negative" in stderr_lower:
        return f"Negative run_time error (guard with 'if remaining > 0'):\n{stderr.strip()}"
    if "nameerror" in stderr_lower:
        return f"NameError (likely missing imports):\n{stderr.strip()}"
    if "attributeerror" in stderr_lower:
        return f"Attribute error (possibly wrong API — check manimlib vs CE):\n{stderr.strip()}"
    if "typeerror" in stderr_lower:
        return f"Type error in generated scene:\n{stderr.strip()}"
    if "valueerror" in stderr_lower:
        return f"Value error in generated scene:\n{stderr.strip()}"
    if "memoryerror" in stderr_lower or "killed" in stderr_lower:
        return f"Memory/resource error (scene too complex):\n{stderr.strip()}"

    return f"Manim render failed:\n{stderr.strip()}"


def _is_voiceover_error(error_msg: str) -> bool:
    """Check if an error is related to voiceover/ElevenLabs."""
    keywords = ["elevenlabs", "voiceover", "api_key", "speech_service", "ElevenLabsService"]
    return any(kw.lower() in error_msg.lower() for kw in keywords)


def render_both_orientations(
    video_id: str,
    scene_code: str,
    quality: str = "qh",
) -> dict:
    """Render both landscape (16:9) and portrait (9:16)."""
    landscape = render_scene(video_id, scene_code, quality, portrait=False)
    portrait = render_scene(video_id, scene_code, quality, portrait=True)
    return {"landscape": landscape, "portrait": portrait}


def _find_rendered_video(media_dir: Path) -> Path | None:
    """Find the output .mp4 in manim's media directory structure."""
    videos_dir = media_dir / "videos"
    if not videos_dir.exists():
        return None

    for mp4 in sorted(videos_dir.rglob("*.mp4")):
        if "partial_movie_files" not in str(mp4):
            return mp4
    return None


def _extract_output_frames(video_path: Path, output_dir: Path, count: int = 12) -> list[Path]:
    """Extract evenly-spaced frames from rendered video for UI display."""
    # Get duration
    probe = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
        capture_output=True, text=True,
    )
    try:
        vid_duration = float(probe.stdout.strip())
    except ValueError:
        vid_duration = 30.0

    interval = max(0.5, vid_duration / count)

    subprocess.run(
        ["ffmpeg", "-y", "-i", str(video_path),
         "-vf", f"fps=1/{interval:.2f}",
         "-frames:v", str(count),
         "-q:v", "2",
         str(output_dir / "frame_%04d.jpg")],
        capture_output=True, text=True,
        timeout=60,
    )

    return sorted(output_dir.glob("frame_*.jpg"))


def _create_watermark_image(
    width: int,
    height: int,
    main_text: str,
    sub_text: str,
    output_path: Path,
) -> None:
    """Create a branded watermark image using Pillow."""
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (width, height), (0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Try to load a bold font, fall back to default
    main_font = None
    sub_font = None
    for font_path in [
        "/System/Library/Fonts/Supplemental/Verdana Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]:
        if Path(font_path).exists():
            try:
                main_font = ImageFont.truetype(font_path, size=int(height * 0.12))
                sub_font = ImageFont.truetype(font_path, size=int(height * 0.035))
                break
            except Exception:
                continue

    if main_font is None:
        main_font = ImageFont.load_default()
        sub_font = ImageFont.load_default()

    # Draw main text centered
    bbox = draw.textbbox((0, 0), main_text, font=main_font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (width - tw) // 2
    y = (height - th) // 2 - int(height * 0.05)
    draw.text((x, y), main_text, fill=(255, 255, 255), font=main_font)

    # Draw subtitle centered below
    bbox2 = draw.textbbox((0, 0), sub_text, font=sub_font)
    tw2 = bbox2[2] - bbox2[0]
    x2 = (width - tw2) // 2
    y2 = y + th + int(height * 0.03)
    draw.text((x2, y2), sub_text, fill=(79, 195, 247), font=sub_font)  # ACCENT_CYAN

    img.save(str(output_path))


def _apply_watermark(video_path: Path, resolution: str = "1920x1080") -> Path:
    """Prepend and append branded OCTOFLASH watermark clips to the video.

    Creates 3-second intro and 3-second outro clips with big centered
    "OCTOFLASH" text on black background, then concatenates them with
    the main video. Returns path to the watermarked video (replaces original).
    """
    width, height = [int(x) for x in resolution.split("x")]
    parent = video_path.parent
    intro_img = parent / "_wm_intro.png"
    outro_img = parent / "_wm_outro.png"
    intro_clip = parent / "_wm_intro.mp4"
    outro_clip = parent / "_wm_outro.mp4"
    concat_list = parent / "_wm_concat.txt"
    output_path = parent / f"wm_{video_path.name}"

    try:
        # Create watermark images with Pillow
        _create_watermark_image(width, height, "OCTOFLASH", "AI-Powered Visual Learning", intro_img)
        _create_watermark_image(width, height, "OCTOFLASH", "Like & Subscribe", outro_img)

        # Convert images to 3-second video clips with fade in/out
        for img_path, clip_path in [(intro_img, intro_clip), (outro_img, outro_clip)]:
            subprocess.run(
                ["ffmpeg", "-y",
                 "-loop", "1", "-i", str(img_path),
                 "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo",
                 "-vf", "fade=t=in:st=0:d=0.8,fade=t=out:st=2.2:d=0.8",
                 "-c:v", "libx264", "-pix_fmt", "yuv420p",
                 "-c:a", "aac",
                 "-t", "3", "-shortest",
                 str(clip_path)],
                capture_output=True, text=True, timeout=30,
            )

        if not intro_clip.exists() or not outro_clip.exists():
            logger.warning("Watermark clip creation failed, skipping watermark")
            return video_path

        # Ensure main video has audio track (manim videos usually don't)
        main_with_audio = parent / "_wm_main.mp4"
        probe = subprocess.run(
            ["ffprobe", "-v", "quiet", "-select_streams", "a",
             "-show_entries", "stream=codec_type", "-of", "csv=p=0",
             str(video_path)],
            capture_output=True, text=True, timeout=10,
        )
        if "audio" not in (probe.stdout or ""):
            # Add silent audio track so concat works
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(video_path),
                 "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo",
                 "-c:v", "copy", "-c:a", "aac", "-shortest",
                 str(main_with_audio)],
                capture_output=True, text=True, timeout=120,
            )
            if main_with_audio.exists():
                video_path.unlink()
                main_with_audio.rename(video_path)

        # Write concat list (use absolute paths)
        concat_list.write_text(
            f"file '{intro_clip.resolve()}'\n"
            f"file '{video_path.resolve()}'\n"
            f"file '{outro_clip.resolve()}'\n"
        )

        # Concatenate: intro + main + outro
        result = subprocess.run(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
             "-i", str(concat_list),
             "-c", "copy",
             str(output_path)],
            capture_output=True, text=True, timeout=120,
        )

        if result.returncode != 0:
            # Re-encode if stream copy fails (codec/resolution mismatch)
            logger.info("Concat copy failed, re-encoding...")
            subprocess.run(
                ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
                 "-i", str(concat_list),
                 "-c:v", "libx264", "-pix_fmt", "yuv420p",
                 "-c:a", "aac",
                 str(output_path)],
                capture_output=True, text=True, timeout=300,
            )

        if output_path.exists() and output_path.stat().st_size > 0:
            # Replace original with watermarked version
            video_path.unlink()
            output_path.rename(video_path)
            logger.info("Watermark applied: %s", video_path)
        else:
            logger.warning("Watermarked output missing, keeping original")

    except Exception as e:
        logger.warning("Watermark failed: %s — keeping original video", e)

    finally:
        # Clean up temp files
        main_with_audio = parent / "_wm_main.mp4"
        for f in [intro_img, outro_img, intro_clip, outro_clip, concat_list, output_path, main_with_audio]:
            if f.exists():
                f.unlink(missing_ok=True)

    return video_path


def _build_env() -> dict:
    """Build environment variables for the manim subprocess."""
    import os
    from dotenv import dotenv_values

    env = os.environ.copy()
    # Load .env values so the subprocess has API keys
    dotenv_path = STORAGE_DIR.parent / ".env"
    if dotenv_path.exists():
        for key, val in dotenv_values(dotenv_path).items():
            if val is not None:
                env[key] = val
    # Ensure the project root is on PYTHONPATH
    project_root = str(STORAGE_DIR.parent)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{project_root}:{existing}" if existing else project_root
    return env
