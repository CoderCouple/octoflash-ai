import json


def build_manin_prompt(
    transcript: str,
    frames: list[str],
    description: str,
    duration: float,
    synthesis: dict | None = None,
) -> str:
    """Build the structured prompt for Manin to generate an inspired video.

    Args:
        transcript: The video transcript (or synthesized combined transcript).
        frames: List of frame paths.
        description: Generated description of the video.
        duration: Target duration in seconds.
        synthesis: Optional synthesis data from synthesize_concepts() for multi-video.
    """
    frame_list = "\n".join(f"  - {f}" for f in frames)

    if synthesis and synthesis.get("section_outline"):
        return _build_multi_video_prompt(transcript, frame_list, frames, description, duration, synthesis)

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


def _build_multi_video_prompt(
    transcript: str,
    frame_list: str,
    frames: list[str],
    description: str,
    duration: float,
    synthesis: dict,
) -> str:
    """Build a prompt specialized for multi-video concept synthesis."""
    num_videos = max(
        len(set(v for c in synthesis.get("core_concepts", []) for v in c.get("from_videos", []))),
        2,
    )

    # Format core concepts
    concepts_text = ""
    for c in synthesis.get("core_concepts", []):
        sources = ", ".join(f"Video {v}" for v in c.get("from_videos", []))
        concepts_text += f"- **{c['name']}** (from {sources}): {c.get('explanation', '')}\n"

    # Format section outline
    outline_text = ""
    for i, s in enumerate(synthesis.get("section_outline", []), 1):
        concepts = ", ".join(s.get("concepts", []))
        outline_text += (
            f"  {i}. **{s['title']}** ({s.get('duration', 15)}s) — "
            f"Concepts: [{concepts}] — Visual: {s.get('visual', 'axes/graph')}\n"
        )

    # Format MCQ if present
    mcq_text = ""
    mcq = synthesis.get("mcq", {})
    if mcq.get("question"):
        options = ", ".join(f'"{o}"' for o in mcq.get("options", []))
        mcq_text = (
            f"\n### Quiz Question\n"
            f"- Question: {mcq['question']}\n"
            f"- Options: [{options}]\n"
            f"- Correct index: {mcq.get('correct_idx', 0)}\n"
        )

    return f"""# Unified Concept Explainer — {synthesis.get('title', 'Combined Topics')}

## Multi-Video Synthesis (from {num_videos} source videos)

### Core Concepts Extracted
{concepts_text}

### Narrative Arc
{synthesis.get('narrative_arc', 'Sequential presentation of topics.')}

### Section Outline (FOLLOW THIS STRUCTURE)
{outline_text}
{mcq_text}

---

## Synthesized Transcript
{transcript if transcript else "(No transcript provided)"}

### Frame References ({len(frames)} frames from {num_videos} videos)
{frame_list}

### Video Description
{description}

### Target Duration
{duration:.1f} seconds

---

## Instructions for Manin

Generate a **unified concept explainer** that combines {num_videos} source topics into ONE cohesive animation.

**CRITICAL**: Follow the section outline above exactly. Each section must:
- Cover the specified concepts with the specified visual type
- Flow naturally from the previous section using the narrative arc
- Build toward a unified understanding, NOT separate summaries

**Preserve** the educational depth from all sources while creating a single story.
**Do not** simply present topics sequentially — weave them together.

## Required Output

1. **Video Concept** — One-paragraph brief for the unified explainer.

2. **Section-by-Section Plan** — Follow the outline above:
   - Section title and duration
   - Concepts covered (from which source videos)
   - Visual type (axes/graph, diagram, formula, etc.)
   - How it connects to the next section

3. **Unified Narration** — Single flowing voiceover that ties all concepts together.

4. **Visual Style Guide** — Consistent aesthetic across all sections.
"""
