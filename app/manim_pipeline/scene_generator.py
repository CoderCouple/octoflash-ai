"""
Generate a Manim scene .py file programmatically from video analysis output.

Takes transcript, description, and frame data -> produces a complete Manim scene
that can be rendered directly with `manim -qh`.
"""

import textwrap

from app.manim_pipeline.scene_planner import plan_scenes


def _escape(text: str) -> str:
    """Escape text for safe embedding inside triple-quoted Python strings."""
    return text.replace("\\", "\\\\").replace('"""', '\\"\\"\\"').replace("\n", " ")


def generate_scene_code(
    video_id: str,
    transcript: str,
    description: str,
    duration: float,
    title: str = "Inspired Video",
    voiceover: bool = True,
) -> str:
    """Generate a complete Manim scene .py file as a string."""

    sections = plan_scenes(transcript, duration, title)
    safe_title = _escape(title)
    # Wrap title if too long
    if len(safe_title) > 35:
        words = safe_title.split()
        mid = len(words) // 2
        safe_title = " ".join(words[:mid]) + "\\n" + " ".join(words[mid:])

    lines = []

    # Module docstring
    lines.append(f'"""')
    lines.append(f"Auto-generated Manim scene for: {title}")
    lines.append(f"Video ID: {video_id}")
    lines.append(f"Duration: {duration:.1f}s | Sections: {len(sections)}")
    lines.append(f'"""')
    lines.append("")

    # Imports
    lines.append("from manim import (")
    lines.append("    VGroup, Text, RoundedRectangle,")
    lines.append("    FadeIn, FadeOut, Write, Create,")
    lines.append("    UP, DOWN, LEFT, RIGHT, ORIGIN,")
    lines.append("    Axes, MathTex,")
    lines.append(")")
    lines.append("import numpy as np")
    lines.append("")

    if voiceover:
        lines.append("from app.manim_pipeline.styles import (")
        lines.append("    OctoflashScene, make_title_card, intro_sequence, outro_sequence,")
    else:
        lines.append("from app.manim_pipeline.styles import OctoflashSceneNoVoice")
        lines.append("from app.manim_pipeline.styles import (")
        lines.append("    make_title_card, intro_sequence, outro_sequence,")

    lines.append("    BG_COLOR, ACCENT_BLUE, ACCENT_ORANGE, ACCENT_GREEN,")
    lines.append("    ACCENT_CYAN, ACCENT_PURPLE, ACCENT_PINK,")
    lines.append("    TEXT_PRIMARY, TEXT_SECONDARY, TEXT_DIM,")
    lines.append("    TITLE_SIZE, SUBTITLE_SIZE, BODY_SIZE, LABEL_SIZE,")
    lines.append(")")
    lines.append("")
    lines.append("")

    # Class
    base = "OctoflashScene" if voiceover else "OctoflashSceneNoVoice"
    lines.append(f"class InspiredVideoScene({base}):")
    lines.append("")
    lines.append("    def construct(self):")

    # Persistent title at TOP ZONE
    lines.append(f'        # ── Persistent title (TOP ZONE) ──')
    lines.append(f'        title = Text("{safe_title}", font_size=TITLE_SIZE, color=TEXT_PRIMARY, weight="BOLD")')
    lines.append(f'        title.to_edge(UP, buff=0.3)')
    lines.append(f'        self.play(FadeIn(title), run_time=0.5)')
    lines.append("")

    # Sections
    for i, sec in enumerate(sections):
        section_lines = _build_section(i, sec, len(sections), voiceover)
        for sl in section_lines:
            lines.append(("        " + sl) if sl.strip() else "")

        lines.append("")

    # Fade out title and outro
    lines.append("        self.play(FadeOut(title), run_time=0.3)")
    lines.append("        outro_sequence(self)")
    lines.append("")

    code = "\n".join(lines)

    # Pre-validate syntax
    validate_scene_code(code)

    return code


def validate_scene_code(code: str) -> None:
    """Check generated code for syntax errors before rendering."""
    try:
        compile(code, "<generated_scene>", "exec")
    except SyntaxError as e:
        raise ValueError(f"Generated scene has syntax error at line {e.lineno}: {e.msg}")


def _build_section(idx: int, section: dict, total: int, voiceover: bool) -> list[str]:
    """Build lines for a single section, at zero indentation."""

    text = _escape(section["text"])
    visual = section["visual"]
    title = _escape(section.get("title", ""))

    # Wrap for display
    display_lines = textwrap.wrap(text, width=38)
    if len(display_lines) > 3:
        display_lines = display_lines[:3]
        display_lines[-1] += "..."
    display_text = "\\n".join(display_lines)

    # Short caption for bottom zone
    caption_text = textwrap.shorten(text, width=50, placeholder="...")

    lines = []
    lines.append(f"# -- Section {idx + 1}/{total} ({section['start']}s - {section['end']}s) --")

    # Caption at BOTTOM ZONE (every section)
    cap_var = f"cap_{idx}"
    lines.append(f'{cap_var} = Text("{caption_text}", font_size=LABEL_SIZE, color=TEXT_SECONDARY)')
    lines.append(f'{cap_var}.to_edge(DOWN, buff=0.4)')
    if idx == 0:
        lines.append(f'self.play(FadeIn({cap_var}), run_time=0.3)')
    else:
        lines.append(f'self.play(FadeOut(cap_{idx - 1}), FadeIn({cap_var}), run_time=0.3)')
    lines.append("")

    if visual == "title_card":
        lines.append(f'card_{idx} = make_title_card("""{title}""")')
        lines.append("")
        lines += _vo_block(voiceover, text, section, [
            f"self.play(FadeIn(card_{idx}, shift=UP * 0.3), run_time=0.8)",
        ])
        lines.append(f"self.play(FadeOut(card_{idx}), run_time=0.5)")

    elif visual == "text_reveal":
        lines.append(f"text_{idx} = Text(")
        lines.append(f'    """{display_text}""",')
        lines.append(f"    font_size=BODY_SIZE,")
        lines.append(f"    color=TEXT_PRIMARY,")
        lines.append(f"    line_spacing=1.4,")
        lines.append(f")")
        lines.append(f"text_{idx}.move_to(ORIGIN)")
        lines.append("")
        lines += _vo_block(voiceover, text, section, [
            f"self.play(Write(text_{idx}), run_time=1.2)",
        ])
        lines.append(f"self.play(FadeOut(text_{idx}), run_time=0.4)")

    elif visual == "bullet_points":
        bullets = display_lines[:4]
        bullets_code = ", ".join(f'"""{b}"""' for b in bullets)
        lines.append(f"bullets_{idx} = VGroup(*[")
        lines.append(f'    Text(f"• {{b}}", font_size=LABEL_SIZE, color=TEXT_SECONDARY)')
        lines.append(f"    for b in [{bullets_code}]")
        lines.append(f"])")
        lines.append(f"bullets_{idx}.arrange(DOWN, aligned_edge=LEFT, buff=0.35)")
        lines.append(f"bullets_{idx}.move_to(ORIGIN)")
        lines.append("")
        lines += _vo_block(voiceover, text, section, [
            f"for bullet in bullets_{idx}:",
            f"    self.play(FadeIn(bullet, shift=RIGHT * 0.3), run_time=0.3)",
        ])
        lines.append(f"self.play(FadeOut(bullets_{idx}), run_time=0.4)")

    elif visual == "highlight_box":
        lines.append(f"box_{idx} = RoundedRectangle(")
        lines.append(f"    corner_radius=0.2, width=9, height=2.5,")
        lines.append(f"    fill_color=ACCENT_BLUE, fill_opacity=0.15,")
        lines.append(f"    stroke_color=ACCENT_BLUE, stroke_width=2,")
        lines.append(f")")
        lines.append(f"box_{idx}.move_to(ORIGIN)")
        lines.append(f"box_text_{idx} = Text(")
        lines.append(f'    """{display_text}""",')
        lines.append(f"    font_size=BODY_SIZE,")
        lines.append(f"    color=TEXT_PRIMARY,")
        lines.append(f"    line_spacing=1.4,")
        lines.append(f").move_to(box_{idx}.get_center())")
        lines.append(f"box_group_{idx} = VGroup(box_{idx}, box_text_{idx})")
        lines.append("")
        lines += _vo_block(voiceover, text, section, [
            f"self.play(FadeIn(box_group_{idx}, shift=UP * 0.2), run_time=0.8)",
        ])
        lines.append(f"self.play(FadeOut(box_group_{idx}), run_time=0.4)")

    elif visual == "split_screen_text":
        left_text = display_lines[0] if display_lines else ""
        right_text = display_lines[1] if len(display_lines) > 1 else ""
        lines.append(f'left_{idx} = Text("""{left_text}""", font_size=LABEL_SIZE, color=ACCENT_CYAN)')
        lines.append(f'right_{idx} = Text("""{right_text}""", font_size=LABEL_SIZE, color=ACCENT_ORANGE)')
        lines.append(f"left_{idx}.move_to(LEFT * 3 + UP * 0.5)")
        lines.append(f"right_{idx}.move_to(RIGHT * 3 + DOWN * 0.5)")
        lines.append("")
        lines += _vo_block(voiceover, text, section, [
            f"self.play(FadeIn(left_{idx}, shift=LEFT * 0.3), run_time=0.5)",
            f"self.play(FadeIn(right_{idx}, shift=RIGHT * 0.3), run_time=0.5)",
        ])
        lines.append(f"self.play(FadeOut(left_{idx}), FadeOut(right_{idx}), run_time=0.4)")

    elif visual == "closing_card":
        lines.append(f"closing_{idx} = Text(")
        lines.append(f'    """{display_text}""",')
        lines.append(f"    font_size=BODY_SIZE,")
        lines.append(f"    color=ACCENT_GREEN,")
        lines.append(f'    weight="BOLD",')
        lines.append(f"    line_spacing=1.4,")
        lines.append(f")")
        lines.append(f"closing_{idx}.move_to(ORIGIN)")
        lines.append("")
        lines += _vo_block(voiceover, text, section, [
            f"self.play(FadeIn(closing_{idx}, shift=UP * 0.3), run_time=0.8)",
        ])
        lines.append(f"self.play(FadeOut(closing_{idx}), run_time=0.5)")

    else:
        lines.append(f"# Unsupported visual: {visual}")

    return lines


def _vo_block(voiceover: bool, text: str, section: dict, body_lines: list[str]) -> list[str]:
    """Build the voiceover with-block (or plain wait) wrapping the animation lines."""
    lines = []

    if voiceover:
        lines.append(f'with self.voiceover(text="""{text}""") as tracker:')
        for bl in body_lines:
            lines.append(f"    {bl}")
        lines.append(f"    remaining = tracker.get_remaining_duration(buff=-0.3)")
        lines.append(f"    if remaining > 0:")
        lines.append(f"        self.wait(remaining)")
    else:
        for bl in body_lines:
            lines.append(bl)
        wait_time = min(2.0, max(0.5, section["end"] - section["start"] - 1.5))
        lines.append(f"self.wait({wait_time:.1f})")

    return lines
