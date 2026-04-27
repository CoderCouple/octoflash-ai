"""
Transcript fetcher — auto-fetches YouTube captions via youtube-transcript-api.
Falls back to yt-dlp subtitle extraction if the API fails.
"""

import logging
import re
import subprocess
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

# Lazy-loaded Whisper model singleton
_whisper_model = None


def extract_youtube_id(url: str) -> str | None:
    """Extract the YouTube video ID from various URL formats."""
    patterns = [
        r"youtube\.com/watch\?v=([\w-]+)",
        r"youtube\.com/shorts/([\w-]+)",
        r"youtu\.be/([\w-]+)",
    ]
    for p in patterns:
        m = re.search(p, url)
        if m:
            return m.group(1)
    return None


def fetch_transcript(url: str) -> str:
    """Auto-fetch transcript from a YouTube video.

    Tries youtube-transcript-api first, then yt-dlp subtitle extraction.
    Raises RuntimeError if no transcript can be obtained.
    """
    yt_id = extract_youtube_id(url)
    if not yt_id:
        raise ValueError(f"Cannot extract YouTube ID from URL: {url}")

    # Attempt 1: youtube-transcript-api
    transcript = _fetch_via_api(yt_id)
    if transcript:
        logger.info("Transcript fetched via youtube-transcript-api (%d chars)", len(transcript))
        return transcript

    # Attempt 2: yt-dlp subtitle extraction
    transcript = _fetch_via_ytdlp(url)
    if transcript:
        logger.info("Transcript fetched via yt-dlp subtitles (%d chars)", len(transcript))
        return transcript

    # Attempt 3: faster-whisper local transcription
    transcript = _fetch_via_whisper(url)
    if transcript:
        logger.info("Transcript fetched via faster-whisper (%d chars)", len(transcript))
        return transcript

    raise RuntimeError(
        f"No transcript available for {url}. "
        "All methods failed (youtube-transcript-api, yt-dlp, faster-whisper)."
    )


def _fetch_via_api(yt_id: str) -> str | None:
    """Fetch transcript using youtube-transcript-api (v1.x API)."""
    try:
        from youtube_transcript_api import YouTubeTranscriptApi

        api = YouTubeTranscriptApi()
        result = api.fetch(yt_id, languages=["en", "en-US", "en-GB"])
        text = " ".join(snippet.text for snippet in result.snippets)
        return text.strip() if text.strip() else None
    except Exception as e:
        logger.warning("youtube-transcript-api failed for %s: %s", yt_id, e)
        return None


def _fetch_via_ytdlp(url: str) -> str | None:
    """Fetch subtitles using yt-dlp as fallback."""
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_template = str(Path(tmpdir) / "subs")
            cmd = [
                "yt-dlp",
                "--write-auto-sub",
                "--write-sub",
                "--sub-lang", "en",
                "--skip-download",
                "-o", out_template,
                url,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                logger.warning("yt-dlp subtitle download returned code %d: %s", result.returncode, result.stderr[:300])
                return None

            # Find the subtitle file (.vtt or .srt)
            tmppath = Path(tmpdir)
            for ext in ["*.vtt", "*.srt"]:
                subs = list(tmppath.glob(ext))
                if subs:
                    raw = subs[0].read_text()
                    return _clean_subtitle_text(raw)
        logger.warning("yt-dlp produced no subtitle files")
        return None
    except Exception as e:
        logger.warning("yt-dlp subtitle extraction failed: %s", e)
        return None


def _fetch_via_whisper(url: str) -> str | None:
    """Download audio and transcribe locally using faster-whisper."""
    global _whisper_model
    try:
        from faster_whisper import WhisperModel

        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = str(Path(tmpdir) / "audio.wav")
            cmd = [
                "yt-dlp",
                "-x", "--audio-format", "wav",
                "-o", audio_path,
                url,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if result.returncode != 0:
                logger.warning("yt-dlp audio download failed: %s", result.stderr[:500])
                return None

            # yt-dlp may append extension — find the actual file
            actual = Path(audio_path)
            if not actual.exists():
                candidates = list(Path(tmpdir).glob("audio.*"))
                if not candidates:
                    logger.warning("No audio file found after yt-dlp download")
                    return None
                actual = candidates[0]

            # Lazy-load model (singleton)
            if _whisper_model is None:
                logger.info("Loading faster-whisper 'base' model (first use)...")
                _whisper_model = WhisperModel("base", compute_type="int8")

            segments, _info = _whisper_model.transcribe(
                str(actual), beam_size=5, language="en"
            )
            text = " ".join(seg.text.strip() for seg in segments if seg.text.strip())
            return text.strip() if text.strip() else None

    except Exception as e:
        logger.warning("faster-whisper transcription failed: %s", e)
        return None


def _clean_subtitle_text(raw: str) -> str | None:
    """Clean VTT/SRT subtitle text into plain transcript."""
    # Remove VTT header
    raw = re.sub(r"WEBVTT.*?\n\n", "", raw, flags=re.DOTALL)
    # Remove timestamp lines (00:00:00.000 --> 00:00:01.000)
    raw = re.sub(r"\d{2}:\d{2}:\d{2}[\.,]\d{3}\s*-->\s*\d{2}:\d{2}:\d{2}[\.,]\d{3}.*\n", "", raw)
    # Remove sequence numbers
    raw = re.sub(r"^\d+\s*$", "", raw, flags=re.MULTILINE)
    # Remove HTML tags like <c> </c>
    raw = re.sub(r"<[^>]+>", "", raw)
    # Remove duplicate lines (auto-subs repeat)
    lines = []
    prev = ""
    for line in raw.strip().split("\n"):
        line = line.strip()
        if line and line != prev:
            lines.append(line)
            prev = line
    text = " ".join(lines)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text if text else None


def process_transcript(raw_transcript: str) -> str:
    """Clean and return the user-provided transcript."""
    return raw_transcript.strip()
