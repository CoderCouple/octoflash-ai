"""
YouTube metadata generator — creates title, description, tags, and hashtags
from a transcript using Claude API. Follows the AI with Intuition template style.
"""

import logging
import os
import re

import anthropic
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

logger = logging.getLogger(__name__)

CLAUDE_MODEL = "claude-opus-4-7"

SYSTEM_PROMPT = """You are a YouTube metadata expert for educational AI/tech channels.
Given a video transcript, generate optimized YouTube metadata.

## Output Format (STRICT — follow exactly)

TITLE: <catchy title, max 70 chars, use — em dash, include topic keywords>

HOOK: <1-2 sentence hook for the first line of description, creates curiosity>

POINTS:
→ <key point 1 covered in the video>
→ <key point 2>
→ <key point 3>
→ <key point 4>

TOPIC_LINE: <short topic summary, e.g. "Activation Functions — ReLU, Sigmoid & Nonlinearity">

TAGS: <comma-separated tags, 15-20 tags, mix of specific and broad>

## Rules
- Title should be attention-grabbing but accurate
- Hook should make viewers want to watch
- Points should summarize what the viewer will learn (use → bullet prefix)
- Tags should include both specific topic terms and broader AI/ML terms
- Keep everything educational and professional
- Do NOT include hashtags in tags — those are added automatically
"""

SERIES_NAME = "Octoflash"
SERIES_HASHTAG = "#octoflash"

BASE_TAGS = [
    "ai", "machinelearning", "deeplearning", "neuralnetworks",
    "artificialintelligence", "programming", "shorts", "learnai",
    "computerscience", "manim", "animation", "datascience", "python",
]

DESCRIPTION_FOOTER = """
---
🎙️ Voice: AI-generated (ElevenLabs)
🎬 Animation: Manim (3Blue1Brown engine)
"""

HASHTAGS = "#ai #machinelearning #deeplearning #neuralnetworks #programming #shorts #learnai #computerscience"


def _get_client() -> anthropic.Anthropic:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")
    return anthropic.Anthropic(api_key=api_key)


def generate_youtube_metadata(
    transcript: str,
    title_hint: str = "",
    duration: float = 60.0,
) -> dict:
    """Generate YouTube metadata from a transcript using Claude API.

    Returns dict with: title, description, tags, hook, points, topic_line
    """
    client = _get_client()

    user_msg = (
        f"Video duration: {duration:.0f} seconds\n"
        f"{'Suggested title: ' + title_hint if title_hint else ''}\n\n"
        f"Transcript:\n{transcript[:3000]}\n\n"
        f"Generate YouTube metadata for this educational video."
    )

    try:
        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        )
        raw = response.content[0].text
        logger.info("YouTube metadata raw response (%d chars): %s", len(raw), raw[:200])
        return _parse_metadata(raw, title_hint)
    except Exception as e:
        logger.exception("YouTube metadata generation failed: %s", e)
        # Fallback: return minimal valid metadata so the publish flow can proceed
        return {
            "title": (title_hint or "Educational Video")[:70],
            "description": (transcript[:500] + "\n\n" + HASHTAGS + DESCRIPTION_FOOTER).strip(),
            "tags": ", ".join(BASE_TAGS),
            "hook": "",
            "points": [],
            "topic_line": "",
        }


def _parse_metadata(raw: str, title_hint: str = "") -> dict:
    """Parse Claude's structured response into a metadata dict."""
    title = ""
    hook = ""
    points = []
    topic_line = ""
    tags_str = ""

    title_match = re.search(r"TITLE:\s*(.+)", raw)
    if title_match:
        title = title_match.group(1).strip()

    hook_match = re.search(r"HOOK:\s*(.+?)(?=\nPOINTS:|\n\n)", raw, re.DOTALL)
    if hook_match:
        hook = hook_match.group(1).strip()

    points_match = re.search(r"POINTS:\s*\n((?:→.+\n?)+)", raw)
    if points_match:
        points = [
            line.strip().lstrip("→").strip()
            for line in points_match.group(1).strip().split("\n")
            if line.strip()
        ]

    topic_match = re.search(r"TOPIC_LINE:\s*(.+)", raw)
    if topic_match:
        topic_line = topic_match.group(1).strip()

    tags_match = re.search(r"TAGS:\s*(.+)", raw, re.DOTALL)
    if tags_match:
        tags_str = tags_match.group(1).strip().split("\n")[0]  # first line only

    # Build final title
    if not title:
        title = title_hint or "Educational Video"

    # Build description
    points_str = "\n".join(f"→ {p}" for p in points) if points else ""
    description = f"""{hook}

In this video, we visualize:
{points_str}

🧠 Series: {SERIES_NAME}
📐 {topic_line}

{HASHTAGS}
{DESCRIPTION_FOOTER}"""

    # Build tags list
    parsed_tags = [t.strip() for t in tags_str.split(",") if t.strip()]
    all_tags = parsed_tags + BASE_TAGS
    seen = set()
    unique_tags = []
    for t in all_tags:
        t_lower = t.lower()
        if t_lower not in seen:
            seen.add(t_lower)
            unique_tags.append(t)

    return {
        "title": title,
        "description": description,
        "tags": ", ".join(unique_tags),
        "hook": hook,
        "points": points,
        "topic_line": topic_line,
    }


def format_metadata(meta: dict) -> str:
    """Format metadata dict into a printable string."""
    lines = [
        "=" * 60,
        "",
        f"📌 Title:",
        f"{meta['title']}",
        "",
        f"📝 Description:",
        f"{meta['description']}",
        "",
        f"🏷️ Tags:",
        f"{meta['tags']}",
        "",
        "=" * 60,
    ]
    return "\n".join(lines)
