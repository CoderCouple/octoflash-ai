"""
Shared styles, colors, helpers, and base scene for Octoflash animations.
Adapted from context-zero-manin-demo/aiintuition/styles.py.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

from manim import (
    Scene, VGroup, Text, RoundedRectangle, Code,
    FadeIn, FadeOut, Write,
    UP, DOWN, LEFT, RIGHT, ORIGIN,
    config, ThreeDScene,
)
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.elevenlabs import ElevenLabsService

# Load .env from project root
load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

# ── Brand Colors ─────────────────────────────────────────────────────────────
BG_COLOR       = "#000000"   # pitch black background
CODE_BG        = "#0a0a0a"
ACCENT_BLUE    = "#4fc3f7"
ACCENT_ORANGE  = "#ff9800"
ACCENT_GREEN   = "#66bb6a"
ACCENT_RED     = "#ef5350"
ACCENT_PURPLE  = "#ab47bc"
ACCENT_YELLOW  = "#ffee58"
ACCENT_CYAN    = "#26c6da"
ACCENT_PINK    = "#ec407a"
TEXT_PRIMARY    = "#ffffff"   # bright white for all general text
TEXT_SECONDARY  = "#e0e0e0"  # near-white, readable on black
TEXT_DIM        = "#9e9e9e"  # muted but still visible on black

# ── Font Sizes ───────────────────────────────────────────────────────────────
TITLE_SIZE     = 52   # bold section titles at top
SUBTITLE_SIZE  = 30
BODY_SIZE      = 24
LABEL_SIZE     = 20
CODE_FONT_SIZE = 20


# ── ElevenLabs Voice Config ──────────────────────────────────────────────────
ELEVENLABS_VOICE_ID = "onwK4e9ZLuTAKqWW03F9"  # Daniel - Steady Broadcaster
ELEVENLABS_MODEL    = "eleven_multilingual_v2"


def get_speech_service():
    """Return configured ElevenLabs service.

    Patches the elevenlabs SDK's voices() call to avoid the /v1/voices endpoint,
    which requires the voices_read permission that restricted API keys lack.
    """
    from elevenlabs.api.voice import Voice
    import manim_voiceover.services.elevenlabs as el_service

    # Create a fake voice object — bypass the voices listing API call
    fake_voice = Voice(voice_id=ELEVENLABS_VOICE_ID, name="Daniel")
    original_voices = el_service.voices

    # Patch at the module level where it's actually called
    el_service.voices = lambda: [fake_voice]

    try:
        service = ElevenLabsService(
            voice_id=ELEVENLABS_VOICE_ID,
            model=ELEVENLABS_MODEL,
            transcription_model=None,
        )
    finally:
        el_service.voices = original_voices

    return service


# ── Base Scene (2D) ──────────────────────────────────────────────────────────
class OctoflashScene(VoiceoverScene):
    """Base 2D scene with dark background + ElevenLabs voiceover."""

    def setup(self):
        super().setup()
        self.camera.background_color = BG_COLOR
        self.set_speech_service(get_speech_service())


# ── Base Scene (3D) ──────────────────────────────────────────────────────────
class Octoflash3DScene(ThreeDScene, VoiceoverScene):
    """Base 3D scene with dark background + ElevenLabs voiceover."""

    def setup(self):
        ThreeDScene.setup(self)
        VoiceoverScene.setup(self)
        self.camera.background_color = BG_COLOR
        self.set_speech_service(get_speech_service())


# ── Helper: Title Card ───────────────────────────────────────────────────────
def make_title_card(title: str, subtitle: str = "") -> VGroup:
    """Branded intro card with title and optional subtitle."""
    title_text = Text(
        title,
        font_size=TITLE_SIZE,
        color=TEXT_PRIMARY,
        weight="BOLD",
    )

    if subtitle:
        sub_text = Text(
            subtitle,
            font_size=SUBTITLE_SIZE,
            color=ACCENT_CYAN,
        ).next_to(title_text, DOWN, buff=0.4)
        return VGroup(title_text, sub_text).move_to(ORIGIN)

    return VGroup(title_text).move_to(ORIGIN)


# ── Helper: Rounded Rect Cell ──────────────────────────────────────────────
def make_cell(
    label: str,
    width: float = 0.8,
    height: float = 0.6,
    color: str = ACCENT_BLUE,
    font_size: int = LABEL_SIZE,
) -> VGroup:
    """Single rounded-rect cell with centered label."""
    rect = RoundedRectangle(
        corner_radius=0.08,
        width=width,
        height=height,
        fill_color=color,
        fill_opacity=0.3,
        stroke_color=color,
        stroke_width=2,
    )
    txt = Text(str(label), font_size=font_size, color=TEXT_PRIMARY)
    txt.move_to(rect.get_center())
    return VGroup(rect, txt)


def make_cell_row(
    labels: list,
    color: str = ACCENT_BLUE,
    cell_width: float = 0.8,
    cell_height: float = 0.6,
    buff: float = 0.0,
    font_size: int = LABEL_SIZE,
) -> VGroup:
    """Horizontal row of rounded-rect cells."""
    cells = VGroup(*[
        make_cell(lbl, cell_width, cell_height, color, font_size)
        for lbl in labels
    ])
    cells.arrange(RIGHT, buff=buff)
    return cells


# ── Helper: Code Block ───────────────────────────────────────────────────────
def make_code_block(
    code_str: str,
    font_size: int = CODE_FONT_SIZE,
    language: str = "python",
) -> Code:
    """Styled code block with consistent look."""
    return Code(
        code_string=code_str,
        language=language,
        formatter_style="monokai",
        background="rectangle",
        add_line_numbers=False,
        background_config={
            "stroke_color": ACCENT_BLUE,
            "stroke_width": 1,
        },
        paragraph_config={
            "font_size": font_size,
        },
    )


# ── Helper: Intro / Outro Sequences ─────────────────────────────────────────
def intro_sequence(scene, title: str, subtitle: str = "", duration: float = 3.0):
    """Play standard branded intro. ~3 seconds."""
    card = make_title_card(title, subtitle)
    scene.play(FadeIn(card, shift=UP * 0.3), run_time=0.8)
    scene.wait(duration - 1.6)
    scene.play(FadeOut(card), run_time=0.8)


def make_mcq_card(
    question: str,
    options: list[str],
    correct_idx: int | None = None,
) -> VGroup:
    """Create a multiple-choice question card.

    Args:
        question: The question text.
        options: List of answer option strings (A, B, C, D labels added automatically).
        correct_idx: If provided, highlight this option in green as the correct answer.

    Returns:
        VGroup containing the question and all option rows.
    """
    labels = "ABCDEFGH"

    q_text = Text(question, font_size=BODY_SIZE, color=TEXT_PRIMARY, weight="BOLD")
    q_text.to_edge(UP, buff=1.0)

    option_vgroup = VGroup()
    for i, opt in enumerate(options):
        letter = labels[i] if i < len(labels) else str(i + 1)
        is_correct = correct_idx is not None and i == correct_idx
        text_color = ACCENT_GREEN if is_correct else TEXT_PRIMARY
        border_color = ACCENT_GREEN if is_correct else TEXT_DIM
        weight = "BOLD" if is_correct else "NORMAL"

        opt_rect = RoundedRectangle(
            corner_radius=0.15,
            width=10,
            height=0.7,
            fill_color=ACCENT_GREEN if is_correct else BG_COLOR,
            fill_opacity=0.15 if is_correct else 0.0,
            stroke_color=border_color,
            stroke_width=2 if is_correct else 1,
        )
        opt_text = Text(
            f"{letter})  {opt}",
            font_size=LABEL_SIZE,
            color=text_color,
            weight=weight,
        )
        opt_text.move_to(opt_rect.get_center())
        option_vgroup.add(VGroup(opt_rect, opt_text))

    option_vgroup.arrange(DOWN, buff=0.2)
    option_vgroup.next_to(q_text, DOWN, buff=0.6)

    return VGroup(q_text, option_vgroup).move_to(ORIGIN)


def outro_sequence(scene, text: str = "Octoflash", duration: float = 3.0):
    """Play standard outro. ~3 seconds."""
    outro = Text(
        text,
        font_size=SUBTITLE_SIZE,
        color=ACCENT_CYAN,
        weight="BOLD",
    )
    scene.play(FadeIn(outro, shift=UP * 0.3), run_time=0.8)
    scene.wait(duration - 1.6)
    scene.play(FadeOut(outro), run_time=0.8)
