"""
Generate YouTube metadata (title, description, tags) for a rendered Octoflash video.
"""

BASE_TAGS = [
    "shorts", "animation", "manim", "ai", "octoflash",
]

DESCRIPTION_FOOTER = """
---
🎙️ Voice: AI-generated (ElevenLabs)
🎬 Animation: Manim (3Blue1Brown engine)
🤖 Generated with Octoflash
"""


def generate_metadata(
    title: str,
    description: str,
    tags: list[str] | None = None,
) -> dict:
    """Generate YouTube-ready metadata for a rendered video."""

    extra_tags = tags or []
    all_tags = extra_tags + BASE_TAGS

    # Deduplicate while preserving order
    seen = set()
    unique_tags = []
    for t in all_tags:
        if t.lower() not in seen:
            seen.add(t.lower())
            unique_tags.append(t)

    hashtags = " ".join(f"#{t}" for t in unique_tags[:15])

    full_description = f"""{description}

{hashtags}
{DESCRIPTION_FOOTER}"""

    return {
        "title": title,
        "description": full_description,
        "tags": ", ".join(unique_tags),
        "made_for_kids": False,
    }
