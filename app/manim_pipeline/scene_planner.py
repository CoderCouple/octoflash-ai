"""
Scene planner — v1 simple wrapper around transcript splitting and visual picking.

Extensibility hook for future LLM-powered planning (vision model analysis,
intelligent visual selection, etc.).
"""

import re
import textwrap


def split_transcript_by_sentences(transcript: str, duration: float) -> list[dict]:
    """Split transcript into timed sections at sentence boundaries.

    Splits on sentence-ending punctuation (. ? !) then groups sentences
    to hit ~6 second target sections.
    """
    sentences = re.split(r'(?<=[.?!])\s+', transcript.strip())
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return [{"text": "Visual summary of the video.", "start": 0.0, "end": duration}]

    # Cap duration and limit sections for tight, fast videos
    capped_duration = min(duration, 90.0)
    target_section_duration = 10.0
    num_sections = max(3, min(6, round(capped_duration / target_section_duration)))

    # Distribute sentences evenly across sections
    sentences_per_section = max(1, len(sentences) // num_sections)
    section_duration = duration / num_sections

    sections = []
    for i in range(num_sections):
        start_sent = i * sentences_per_section
        end_sent = start_sent + sentences_per_section if i < num_sections - 1 else len(sentences)
        section_text = " ".join(sentences[start_sent:end_sent])
        if not section_text:
            continue
        sections.append({
            "text": section_text,
            "start": round(i * section_duration, 1),
            "end": round((i + 1) * section_duration, 1),
        })

    return sections


def pick_visual(section_idx: int, total_sections: int) -> str:
    """Pick a visual animation style for each section.

    v1: Deterministic cycling. Future: LLM-powered selection based on content.
    """
    visuals = [
        "title_card",
        "text_reveal",
        "bullet_points",
        "highlight_box",
        "split_screen_text",
        "closing_card",
    ]

    if section_idx == 0:
        return "title_card"
    if section_idx == total_sections - 1:
        return "closing_card"

    interior_idx = (section_idx - 1) % (len(visuals) - 2)
    return visuals[1 + interior_idx]


def plan_scenes(transcript: str, duration: float, title: str = "Inspired Video") -> list[dict]:
    """Plan the full scene sequence from transcript.

    Returns a list of dicts with: text, start, end, visual, title.
    """
    sections = split_transcript_by_sentences(transcript, duration)
    total = len(sections)

    for i, section in enumerate(sections):
        section["visual"] = pick_visual(i, total)
        section["title"] = title

    return sections
