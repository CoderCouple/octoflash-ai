"""
Video description generation.

Uses Claude vision API to analyze extracted frames and produce a rich
description of the visual style, animations, and content — which is then
fed to the script generator for high-quality Manim reproduction.
"""

import logging

from app.services.script_generator import analyze_source_frames

logger = logging.getLogger(__name__)


def generate_description(
    transcript: str,
    frames: list[str],
    duration: float,
) -> str:
    """Generate a rich video description using vision analysis of frames.

    Sends sampled frames to Claude vision API for analysis.
    Falls back to basic text description if API is unavailable.
    """
    return analyze_source_frames(frames, transcript, duration)
