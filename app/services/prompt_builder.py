def build_manin_prompt(
    transcript: str,
    frames: list[str],
    description: str,
    duration: float,
) -> str:
    """Build the structured prompt for Manin to generate an inspired video."""
    frame_list = "\n".join(f"  - {f}" for f in frames)

    return f"""# Inspired Video Concept

## Source Analysis

### Transcript
{transcript if transcript else "(No transcript provided)"}

### Frame References ({len(frames)} frames, 1 per second)
{frame_list}

### Video Description
{description}

### Source Duration
{duration:.1f} seconds

---

## Instructions for Manin

Generate a **new short-form video concept** inspired by the source material above.

**Preserve** the general energy, pacing, topic, and emotional style.
**Do not copy** exact scenes, faces, branding, dialogue, or copyrighted creative expression.

## Required Output

1. **Video Concept** — One-paragraph creative brief for the new video.

2. **Shot-by-Shot Plan** — For each shot:
   - Shot number and duration
   - Scene description
   - Camera movement (static, pan, zoom, tracking, etc.)
   - Visual style notes (lighting, color grade, mood)

3. **Suggested Narration** — New voiceover script that captures the same tone without copying the original.

4. **On-Screen Text** — Any titles, captions, or text overlays with timing.

5. **Timing Breakdown** — Second-by-second plan matching the original duration (~{duration:.0f}s).

6. **Visual Style Guide** — Overall aesthetic direction (color palette, typography, transitions).
"""
