"""
Visual effects, transitions, and polish patterns for Octoflash Manim animations.

Comprehensive library of production-quality effects for educational content.
Each function is self-contained and compatible with both OctoflashScene and Scene.

Usage in generated scripts:
    from app.manim_pipeline.visual_effects import (
        crossfade_transition, zoom_transition, wipe_transition,
        slide_transition, iris_transition,
        typewriter_reveal, word_by_word_reveal, highlight_then_transform,
        scanning_highlight, equation_step_through,
        make_progress_bar, make_step_counter, make_section_marker,
        make_speech_bubble, make_callout_box, make_labeled_arrow,
        make_thought_cloud,
        subtle_grid_background, gradient_background, dot_grid_background,
        glow_effect, pulse_effect, emphasis_box,
        sweep_in_group, cascade_fade_in,
    )
"""

from __future__ import annotations

import numpy as np
from manim import (
    VGroup, VMobject, Text, MathTex, Tex,
    Rectangle, RoundedRectangle, Square, Circle, Line, Arrow, Dot,
    DashedLine, Polygon, ArcBetweenPoints, Ellipse,
    SurroundingRectangle, BackgroundRectangle, Underline,
    FadeIn, FadeOut, FadeTransform, Write, Create, Uncreate,
    Transform, ReplacementTransform, TransformMatchingTex,
    GrowFromCenter, GrowFromEdge, GrowArrow, SpinInFromNothing,
    AnimationGroup, LaggedStart, LaggedStartMap, Succession,
    Indicate, Circumscribe, Flash, FocusOn, ShowPassingFlash, ApplyWave, Wiggle,
    AddTextLetterByLetter,
    ScaleInPlace, ShrinkToCenter,
    ShowIncreasingSubsets,
    UP, DOWN, LEFT, RIGHT, ORIGIN, UL, UR, DL, DR,
    PI, TAU, DEGREES,
    config,
    there_and_back, smooth, linear, rate_functions,
    ValueTracker, always_redraw, DecimalNumber,
    WHITE, GREY, GREY_A, GREY_B, GREY_C, GREY_D, GREY_E,
    BLUE, BLUE_A, BLUE_B, BLUE_C, BLUE_D, BLUE_E,
    GREEN, GREEN_A, GREEN_B, GREEN_C, GREEN_D,
    RED, RED_A, RED_B, RED_C, RED_D,
    YELLOW, YELLOW_A, YELLOW_B, YELLOW_C, YELLOW_D,
    ORANGE, PURE_YELLOW,
    SMALL_BUFF, MED_SMALL_BUFF, MED_LARGE_BUFF, LARGE_BUFF, DEFAULT_STROKE_WIDTH,
)

from app.manim_pipeline.styles import (
    BG_COLOR, ACCENT_BLUE, ACCENT_ORANGE, ACCENT_GREEN, ACCENT_RED,
    ACCENT_PURPLE, ACCENT_YELLOW, ACCENT_CYAN, ACCENT_PINK,
    TEXT_PRIMARY, TEXT_SECONDARY, TEXT_DIM,
    TITLE_SIZE, SUBTITLE_SIZE, BODY_SIZE, LABEL_SIZE,
)


# =============================================================================
# 1. SCENE TRANSITIONS
# =============================================================================

def crossfade_transition(scene, old_group: VGroup, new_group: VGroup,
                         run_time: float = 1.0):
    """Crossfade: simultaneously fade out old content and fade in new content.

    When to use: Between major sections when both old and new content are
    unrelated. Smooth, professional, and the most versatile transition.

    With voiceover:
        self.play(FadeOut(old_cap), run_time=0.3)
        crossfade_transition(self, old_content, new_content, run_time=0.8)
        # Then start new voiceover block

    Args:
        scene: The Scene instance (self in construct).
        old_group: VGroup of mobjects to fade out.
        new_group: VGroup of mobjects to fade in.
        run_time: Duration of the crossfade.
    """
    scene.play(
        FadeOut(old_group, shift=UP * 0.3),
        FadeIn(new_group, shift=UP * 0.3),
        run_time=run_time,
    )


def zoom_transition(scene, old_group: VGroup, new_group: VGroup,
                     zoom_in: bool = True, run_time: float = 1.0):
    """Zoom in or out: old content scales away, new content scales in.

    When to use: Zooming IN when drilling deeper into a detail (e.g., from
    overview to specific formula). Zooming OUT when pulling back to see
    the big picture after examining a detail.

    With voiceover: Place between voiceover blocks. Use zoom_in=True when the
    narration says "let's look closer at..." and zoom_in=False for "stepping
    back to see the full picture..."

    Args:
        scene: The Scene instance.
        old_group: VGroup to zoom away.
        new_group: VGroup to zoom in.
        zoom_in: True = zoom into new content, False = zoom out.
        run_time: Duration.
    """
    if zoom_in:
        scene.play(
            FadeOut(old_group, scale=1.5),
            FadeIn(new_group, scale=0.5),
            run_time=run_time,
        )
    else:
        scene.play(
            FadeOut(old_group, scale=0.5),
            FadeIn(new_group, scale=1.5),
            run_time=run_time,
        )


def wipe_transition(scene, old_group: VGroup, new_group: VGroup,
                     direction=LEFT, run_time: float = 1.0):
    """Wipe: old content slides out one direction, new slides in from opposite.

    When to use: Sequential content -- "now that we've seen X, let's move to Y."
    LEFT wipe (default) feels like turning a page forward.
    RIGHT wipe feels like going back to review.

    Args:
        scene: The Scene instance.
        old_group: VGroup to slide out.
        new_group: VGroup to slide in.
        direction: Which way old content exits (LEFT, RIGHT, UP, DOWN).
        run_time: Duration.
    """
    shift_out = direction * (config.frame_width + 1)
    shift_in = -direction * (config.frame_width + 1)
    scene.play(
        old_group.animate.shift(shift_out),
        new_group.animate.shift(shift_in).set_opacity(1),
        run_time=run_time,
    )


def slide_transition(scene, old_group: VGroup, new_group: VGroup,
                      direction=LEFT, run_time: float = 1.0):
    """Slide: old and new content move together like a conveyor belt.

    When to use: Showing a sequence of related items (step 1 -> step 2 -> step 3).
    Gives a sense of progression through a linear process.

    Args:
        scene: The Scene instance.
        old_group: VGroup to slide out.
        new_group: VGroup to slide in (should be pre-positioned off-screen).
        direction: Direction of the slide.
        run_time: Duration.
    """
    offset = -direction * config.frame_width
    new_group.shift(offset)
    scene.play(
        AnimationGroup(
            old_group.animate.shift(direction * config.frame_width),
            new_group.animate.shift(-offset),
            lag_ratio=0,
        ),
        run_time=run_time,
    )


def iris_transition(scene, old_group: VGroup, focus_point=ORIGIN,
                    run_time: float = 1.0):
    """Iris close: a circle shrinks to a point, hiding the scene.

    When to use: End of a major section or chapter. Classic "closing" feel.
    Often paired with iris_open for the next section.

    Args:
        scene: The Scene instance.
        old_group: Content that will be hidden behind the iris.
        focus_point: Where the iris closes to.
        run_time: Duration.
    """
    iris = Circle(
        radius=config.frame_width,
        fill_color=BG_COLOR,
        fill_opacity=1,
        stroke_width=0,
    ).move_to(focus_point)
    scene.play(
        iris.animate.scale(0.01),
        FadeOut(old_group),
        run_time=run_time,
    )
    scene.remove(iris)


def section_wipe(scene, color=ACCENT_BLUE, run_time: float = 0.6):
    """A colored bar sweeps across the screen as a section divider.

    When to use: Quick visual break between sections. Cleaner than a fade
    when you want to signal "new topic" without a title card.

    Args:
        scene: The Scene instance.
        color: Color of the sweeping bar.
        run_time: Duration.
    """
    bar = Rectangle(
        width=0.15, height=config.frame_height + 1,
        fill_color=color, fill_opacity=0.8, stroke_width=0,
    ).move_to(LEFT * (config.frame_width / 2 + 1))
    scene.add(bar)
    scene.play(
        bar.animate.move_to(RIGHT * (config.frame_width / 2 + 1)),
        run_time=run_time,
    )
    scene.remove(bar)


# =============================================================================
# 2. EMPHASIS EFFECTS
# =============================================================================

def glow_effect(mobject: VMobject, color=ACCENT_CYAN, num_layers: int = 8,
                max_stroke_width: float = 20, opacity_start: float = 0.15):
    """Add a soft glow behind a mobject by layering blurred copies.

    When to use: Highlighting the "answer" or key result. Makes a formula or
    number visually pop against the dark background.

    Returns a VGroup containing the glow layers (add BEHIND the mobject).

    Args:
        mobject: The mobject to glow around.
        color: Glow color.
        num_layers: More layers = smoother glow.
        max_stroke_width: Outer stroke width for the glow.
        opacity_start: Starting opacity of outermost layer.

    Example:
        eq = MathTex(r"E = mc^2")
        glow = glow_effect(eq, color=ACCENT_YELLOW)
        self.play(FadeIn(glow), Write(eq))
    """
    layers = VGroup()
    for i in range(num_layers):
        t = (i + 1) / num_layers
        layer = mobject.copy()
        layer.set_stroke(
            color=color,
            width=max_stroke_width * t,
            opacity=opacity_start * (1 - t),
        )
        layer.set_fill(opacity=0)
        layers.add(layer)
    return layers


def pulse_effect(scene, mobject: VMobject, scale_factor: float = 1.2,
                 color=PURE_YELLOW, n_pulses: int = 2, run_time: float = 1.5):
    """Pulse a mobject: briefly scale up and recolor, then return to normal.

    When to use: Drawing attention to a specific term, variable, or result
    mid-explanation. Less intrusive than Flash but more noticeable than Indicate.

    With voiceover: Place right when narrator emphasizes a word.
        "The KEY insight is..." -> pulse the key formula

    Args:
        scene: The Scene instance.
        mobject: Mobject to pulse.
        scale_factor: How much to scale up during pulse.
        color: Flash color during pulse.
        n_pulses: Number of pulse cycles.
        run_time: Total duration.
    """
    for _ in range(n_pulses):
        scene.play(
            mobject.animate.scale(scale_factor).set_color(color),
            run_time=run_time / (n_pulses * 2),
            rate_func=smooth,
        )
        scene.play(
            mobject.animate.scale(1 / scale_factor).set_color(TEXT_PRIMARY),
            run_time=run_time / (n_pulses * 2),
            rate_func=smooth,
        )


def emphasis_box(scene, mobject: VMobject, color=ACCENT_YELLOW,
                 buff: float = 0.15, run_time: float = 0.8,
                 fade_after: float = 0.0):
    """Draw a surrounding rectangle around a mobject for emphasis.

    When to use: Highlighting a specific part of an equation or diagram.
    The box draws itself around the target, lingers, then optionally fades.

    Args:
        scene: The Scene instance.
        mobject: What to emphasize.
        color: Box color.
        buff: Padding around the mobject.
        run_time: Draw time.
        fade_after: Seconds to wait then fade. 0 = leave it.

    Returns:
        The SurroundingRectangle mobject (for later removal).
    """
    box = SurroundingRectangle(
        mobject, color=color, buff=buff,
        stroke_width=2.5, corner_radius=0.1,
    )
    scene.play(Create(box), run_time=run_time)
    if fade_after > 0:
        scene.wait(fade_after)
        scene.play(FadeOut(box), run_time=0.3)
    return box


def underline_emphasis(scene, mobject: VMobject, color=ACCENT_CYAN,
                       run_time: float = 0.6):
    """Draw an underline beneath a mobject.

    When to use: Subtler than a box. Good for emphasizing a word within a
    sentence or a single variable in a formula.

    Args:
        scene: The Scene instance.
        mobject: What to underline.
        color: Underline color.
        run_time: Animation duration.

    Returns:
        The Underline mobject.
    """
    ul = Underline(mobject, color=color, stroke_width=3)
    scene.play(Create(ul), run_time=run_time)
    return ul


def flash_and_circumscribe(scene, mobject: VMobject, flash_color=PURE_YELLOW,
                           circ_color=ACCENT_CYAN, run_time: float = 1.5):
    """Combined Flash + Circumscribe for maximum emphasis.

    When to use: The "aha moment" -- when revealing a key result, answer to
    a quiz, or the final simplified form of an equation.

    Args:
        scene: The Scene instance.
        mobject: What to emphasize.
        flash_color: Color of the flash burst.
        circ_color: Color of the circumscription line.
        run_time: Total duration.
    """
    scene.play(
        Flash(mobject, color=flash_color, flash_radius=0.5,
              line_length=0.3, num_lines=16, run_time=run_time * 0.6),
        Circumscribe(mobject, color=circ_color, run_time=run_time),
    )


# =============================================================================
# 3. TEXT REVEAL PATTERNS
# =============================================================================

def typewriter_reveal(scene, text_mobject, run_time: float = None,
                      time_per_char: float = 0.05):
    """Typewriter effect: reveal text character by character.

    When to use: Code snippets, definitions, or any text where the sequential
    reveal adds to comprehension. Mimics someone typing.

    IMPORTANT: Only works with Text() mobjects, NOT MathTex/Tex.

    With voiceover: Set time_per_char so the text finishes roughly when
    the narrator finishes reading it.

    Args:
        scene: The Scene instance.
        text_mobject: A Text() mobject.
        run_time: Override total duration. If None, computed from time_per_char.
        time_per_char: Seconds per character.
    """
    scene.play(
        AddTextLetterByLetter(text_mobject, time_per_char=time_per_char,
                              run_time=run_time),
    )


def word_by_word_reveal(scene, words: list[str], position=ORIGIN,
                        font_size: int = BODY_SIZE, color=TEXT_PRIMARY,
                        word_delay: float = 0.3, fade_together: bool = True):
    """Reveal text word by word, building up a sentence.

    When to use: Definitions, key takeaways, or any sentence where each
    word adds meaning. Creates a "building" feeling.

    With voiceover: Time word_delay to match the narrator's pacing.

    Args:
        scene: The Scene instance.
        words: List of word strings.
        position: Where to place the text.
        font_size: Text size.
        color: Text color.
        word_delay: Delay between each word appearing.
        fade_together: If True, returns the full text. If False, each word
                       is a separate mobject.

    Returns:
        VGroup of word mobjects.
    """
    word_mobjects = VGroup()
    for word in words:
        t = Text(word, font_size=font_size, color=color)
        word_mobjects.add(t)
    word_mobjects.arrange(RIGHT, buff=0.2).move_to(position)

    for word_mob in word_mobjects:
        scene.play(FadeIn(word_mob, shift=UP * 0.15), run_time=word_delay)

    return word_mobjects


def highlight_then_transform(scene, tex_start: MathTex, tex_end: MathTex,
                              highlight_indices: list[int] = None,
                              highlight_color=ACCENT_YELLOW,
                              run_time: float = 2.0):
    """Highlight parts of an equation, then transform to the next step.

    When to use: Step-by-step algebraic manipulation. Highlight the part
    that changes, then morph the entire equation to its new form.

    IMPORTANT: Both MathTex must use matching {{ }} delimiters for
    TransformMatchingTex to work correctly.

    With voiceover: Highlight during "notice that..." then transform during
    "which simplifies to..."

    Args:
        scene: The Scene instance.
        tex_start: The starting MathTex (already on screen).
        tex_end: The target MathTex (will replace tex_start).
        highlight_indices: Which submobject indices to highlight first.
        highlight_color: Color for highlighting.
        run_time: Total transform time.
    """
    if highlight_indices:
        highlights = [tex_start[i] for i in highlight_indices
                      if i < len(tex_start)]
        scene.play(
            *[Indicate(h, color=highlight_color, scale_factor=1.15)
              for h in highlights],
            run_time=run_time * 0.3,
        )

    tex_end.move_to(tex_start)
    scene.play(
        TransformMatchingTex(tex_start, tex_end),
        run_time=run_time * 0.7,
    )


def scanning_highlight(scene, mobject: VMobject, color=ACCENT_CYAN,
                       direction=RIGHT, run_time: float = 1.5):
    """A highlight bar scans across the mobject left to right.

    When to use: Reading through a line of code or a formula left-to-right.
    Guides the viewer's eye across the content.

    Args:
        scene: The Scene instance.
        mobject: What to scan across.
        color: Highlight color.
        direction: Scan direction.
        run_time: Duration.
    """
    highlight = Rectangle(
        width=0.08, height=mobject.height + 0.3,
        fill_color=color, fill_opacity=0.4, stroke_width=0,
    )
    start = mobject.get_left() + LEFT * 0.2
    end = mobject.get_right() + RIGHT * 0.2
    highlight.move_to(start)
    scene.add(highlight)
    scene.play(highlight.animate.move_to(end), run_time=run_time)
    scene.remove(highlight)


def equation_step_through(scene, equations: list[MathTex],
                          position=ORIGIN, step_time: float = 1.5,
                          caption_texts: list[str] = None):
    """Show a sequence of equations, transforming one into the next.

    When to use: Multi-step derivations. Each equation morphs into the next,
    showing algebraic progression. The gold standard for math animations.

    With voiceover: One voiceover block per step, with the transform
    playing during the narration.

    Args:
        scene: The Scene instance.
        equations: List of MathTex objects (use {{ }} for matching parts).
        position: Where to center the equations.
        step_time: Time per transformation.
        caption_texts: Optional list of caption strings for each step.

    Returns:
        The final equation mobject (still on screen).
    """
    if not equations:
        return None

    equations[0].move_to(position)
    scene.play(Write(equations[0]), run_time=step_time * 0.8)

    cap = None
    for i in range(1, len(equations)):
        equations[i].move_to(position)

        if caption_texts and i < len(caption_texts):
            new_cap = Text(caption_texts[i], font_size=LABEL_SIZE,
                           color=TEXT_SECONDARY).to_edge(DOWN, buff=0.4)
            if cap:
                scene.play(
                    TransformMatchingTex(equations[i - 1], equations[i]),
                    FadeOut(cap), FadeIn(new_cap),
                    run_time=step_time,
                )
            else:
                scene.play(
                    TransformMatchingTex(equations[i - 1], equations[i]),
                    FadeIn(new_cap),
                    run_time=step_time,
                )
            cap = new_cap
        else:
            scene.play(
                TransformMatchingTex(equations[i - 1], equations[i]),
                run_time=step_time,
            )

    return equations[-1]


# =============================================================================
# 4. BACKGROUND EFFECTS
# =============================================================================

def subtle_grid_background(
    x_range: tuple = (-8, 8, 1),
    y_range: tuple = (-5, 5, 1),
    stroke_color=TEXT_DIM,
    stroke_width: float = 0.3,
    stroke_opacity: float = 0.15,
) -> VGroup:
    """Create a subtle grid background for visual depth.

    When to use: Any scene where you want more visual polish than a plain
    black background. Especially good for coordinate-based content.

    Add to scene FIRST (behind everything):
        grid = subtle_grid_background()
        self.add(grid)

    Args:
        x_range: (start, end, step) for vertical lines.
        y_range: (start, end, step) for horizontal lines.
        stroke_color: Grid line color.
        stroke_width: Grid line thickness.
        stroke_opacity: Grid line opacity.

    Returns:
        VGroup of grid lines.
    """
    grid = VGroup()
    x_start, x_end, x_step = x_range
    y_start, y_end, y_step = y_range

    # Vertical lines
    x = x_start
    while x <= x_end:
        line = Line(
            start=[x, y_start, 0], end=[x, y_end, 0],
            stroke_color=stroke_color, stroke_width=stroke_width,
            stroke_opacity=stroke_opacity,
        )
        grid.add(line)
        x += x_step

    # Horizontal lines
    y = y_start
    while y <= y_end:
        line = Line(
            start=[x_start, y, 0], end=[x_end, y, 0],
            stroke_color=stroke_color, stroke_width=stroke_width,
            stroke_opacity=stroke_opacity,
        )
        grid.add(line)
        y += y_step

    return grid


def gradient_background(
    top_color="#1a1a2e",
    bottom_color="#000000",
    num_strips: int = 30,
) -> VGroup:
    """Create a vertical gradient background using horizontal strips.

    When to use: When a pure black background feels too stark. Adds depth
    and a cinematic feel. Use dark colors only -- bright gradients distract.

    Add FIRST: self.add(gradient_background())

    Args:
        top_color: Color at the top of frame.
        bottom_color: Color at the bottom.
        num_strips: Number of color bands (more = smoother gradient).

    Returns:
        VGroup of rectangles forming the gradient.
    """
    strips = VGroup()
    strip_height = (config.frame_height + 0.5) / num_strips

    for i in range(num_strips):
        t = i / (num_strips - 1) if num_strips > 1 else 0
        # Interpolate manually via hex
        strip = Rectangle(
            width=config.frame_width + 0.5,
            height=strip_height + 0.02,  # slight overlap
            fill_opacity=1,
            stroke_width=0,
        )
        strip.set_fill(color=top_color if t < 0.5 else bottom_color)
        y_pos = config.frame_height / 2 - i * strip_height
        strip.move_to([0, y_pos, 0])
        strips.add(strip)

    return strips


def dot_grid_background(
    rows: int = 12,
    cols: int = 18,
    dot_radius: float = 0.02,
    color=TEXT_DIM,
    opacity: float = 0.1,
) -> VGroup:
    """Create a subtle dot grid background.

    When to use: Similar to grid lines but lighter, less technical feel.
    Good for non-mathematical topics where grid lines would feel out of place.

    Args:
        rows: Number of dot rows.
        cols: Number of dot columns.
        dot_radius: Size of each dot.
        color: Dot color.
        opacity: Dot opacity.

    Returns:
        VGroup of dots.
    """
    dots = VGroup()
    x_spacing = config.frame_width / (cols + 1)
    y_spacing = config.frame_height / (rows + 1)

    for r in range(rows):
        for c in range(cols):
            x = -config.frame_width / 2 + (c + 1) * x_spacing
            y = -config.frame_height / 2 + (r + 1) * y_spacing
            dot = Dot(
                point=[x, y, 0], radius=dot_radius,
                color=color, fill_opacity=opacity,
                stroke_width=0,
            )
            dots.add(dot)

    return dots


def vignette_overlay(opacity: float = 0.3) -> VGroup:
    """Create a vignette (darker edges) overlay for cinematic feel.

    When to use: Final polish layer. Subtly darkens the edges of the frame,
    drawing the viewer's eye to the center content.

    Add LAST (on top of everything, behind foreground mobjects):
        self.add(vignette_overlay())

    Args:
        opacity: Maximum darkness at edges.

    Returns:
        VGroup of gradient rectangles.
    """
    vignette = VGroup()
    for edge_dir, size in [(LEFT, (2, config.frame_height)),
                            (RIGHT, (2, config.frame_height)),
                            (UP, (config.frame_width, 2)),
                            (DOWN, (config.frame_width, 2))]:
        rect = Rectangle(
            width=size[0], height=size[1],
            fill_color=BG_COLOR, fill_opacity=opacity,
            stroke_width=0,
        )
        rect.next_to(ORIGIN, edge_dir, buff=config.frame_width / 2 - 0.5)
        vignette.add(rect)
    return vignette


# =============================================================================
# 5. ANNOTATION PATTERNS
# =============================================================================

def make_speech_bubble(
    text: str,
    width: float = 4.0,
    height: float = 1.5,
    font_size: int = LABEL_SIZE,
    text_color=TEXT_PRIMARY,
    bubble_color=ACCENT_BLUE,
    fill_opacity: float = 0.1,
    tail_direction=DL,
) -> VGroup:
    """Create a speech bubble with text and a tail pointer.

    When to use: Character dialogue, direct quotes, or when you want to
    present information as if someone is saying it.

    Args:
        text: The speech text.
        width: Bubble width.
        height: Bubble height.
        font_size: Text size.
        text_color: Text color.
        bubble_color: Bubble border/fill color.
        fill_opacity: Background fill opacity.
        tail_direction: Where the tail points (DL, DR, UL, UR).

    Returns:
        VGroup(bubble_rect, text, tail_triangle).

    Example:
        bubble = make_speech_bubble("Hello!", tail_direction=DL)
        bubble.next_to(character, UP)
        self.play(FadeIn(bubble, shift=UP*0.2))
    """
    rect = RoundedRectangle(
        corner_radius=0.2, width=width, height=height,
        fill_color=bubble_color, fill_opacity=fill_opacity,
        stroke_color=bubble_color, stroke_width=2,
    )
    txt = Text(text, font_size=font_size, color=text_color)
    txt.move_to(rect.get_center())

    # Tail
    tail_base = rect.get_corner(tail_direction)
    tail_tip = tail_base + tail_direction * 0.4
    tail = Polygon(
        tail_base + RIGHT * 0.15,
        tail_base + LEFT * 0.15,
        tail_tip,
        fill_color=bubble_color, fill_opacity=fill_opacity,
        stroke_color=bubble_color, stroke_width=2,
    )

    return VGroup(rect, txt, tail)


def make_thought_cloud(
    text: str,
    width: float = 4.0,
    height: float = 1.5,
    font_size: int = LABEL_SIZE,
    color=TEXT_DIM,
) -> VGroup:
    """Create a thought cloud with trailing dots.

    When to use: Internal monologue, hypothetical scenarios, or "what if"
    questions during the explanation.

    Args:
        text: Thought text.
        width: Cloud width.
        height: Cloud height.
        font_size: Text size.
        color: Cloud color.

    Returns:
        VGroup containing the cloud and trailing dots.
    """
    cloud = Ellipse(width=width, height=height,
                    fill_color=color, fill_opacity=0.08,
                    stroke_color=color, stroke_width=1.5)
    txt = Text(text, font_size=font_size, color=TEXT_PRIMARY)
    txt.move_to(cloud.get_center())

    # Trailing dots (like a thought bubble trail)
    d1 = Dot(radius=0.08, color=color, fill_opacity=0.5)
    d2 = Dot(radius=0.05, color=color, fill_opacity=0.4)
    d3 = Dot(radius=0.03, color=color, fill_opacity=0.3)
    d1.next_to(cloud, DL, buff=0.15)
    d2.next_to(d1, DL, buff=0.1)
    d3.next_to(d2, DL, buff=0.08)

    return VGroup(cloud, txt, d1, d2, d3)


def make_callout_box(
    text: str,
    title: str = "",
    width: float = 5.0,
    color=ACCENT_ORANGE,
    font_size: int = LABEL_SIZE,
    title_size: int = BODY_SIZE,
) -> VGroup:
    """Create a callout box with optional title bar.

    When to use: Important notes, warnings, tips, or key definitions.
    The colored title bar makes it stand out from regular content.

    Args:
        text: Body text.
        title: Optional title (shown in colored bar at top).
        width: Box width.
        color: Accent color.
        font_size: Body text size.
        title_size: Title text size.

    Returns:
        VGroup containing the callout box.

    Example:
        callout = make_callout_box("Time complexity: O(n log n)", title="Key Insight")
        callout.to_edge(RIGHT, buff=0.5)
        self.play(GrowFromEdge(callout, LEFT))
    """
    body_text = Text(text, font_size=font_size, color=TEXT_PRIMARY)
    body_width = max(body_text.width + 0.8, width)
    body_height = body_text.height + 0.6

    body_rect = RoundedRectangle(
        corner_radius=0.12, width=body_width, height=body_height,
        fill_color=BG_COLOR, fill_opacity=0.9,
        stroke_color=color, stroke_width=2,
    )

    parts = [body_rect, body_text]
    body_text.move_to(body_rect.get_center())

    if title:
        title_text = Text(title, font_size=title_size, color=TEXT_PRIMARY,
                          weight="BOLD")
        title_bar = Rectangle(
            width=body_width, height=0.5,
            fill_color=color, fill_opacity=0.3,
            stroke_width=0,
        )
        title_text.move_to(title_bar.get_center())
        title_group = VGroup(title_bar, title_text)
        title_group.next_to(body_rect, UP, buff=0)
        parts.extend([title_bar, title_text])

    return VGroup(*parts)


def make_labeled_arrow(
    start, end,
    label: str = "",
    color=ACCENT_CYAN,
    font_size: int = LABEL_SIZE,
    label_direction=UP,
    buff: float = 0.15,
) -> VGroup:
    """Create an arrow with a text label.

    When to use: Pointing from a label to a specific part of a diagram,
    connecting cause to effect, or annotating parts of a graph.

    Args:
        start: Arrow start point.
        end: Arrow end point.
        label: Text label.
        color: Arrow and label color.
        font_size: Label font size.
        label_direction: Which side of the arrow to place the label.
        buff: Space between arrow and label.

    Returns:
        VGroup(arrow, label_text).

    Example:
        arr = make_labeled_arrow(
            axes.c2p(2, 4), axes.c2p(2, 0),
            label="f(2)=4", label_direction=RIGHT,
        )
        self.play(GrowArrow(arr[0]), FadeIn(arr[1]))
    """
    arrow = Arrow(start, end, color=color, stroke_width=3,
                  max_tip_length_to_length_ratio=0.15)
    parts = [arrow]

    if label:
        lbl = Text(label, font_size=font_size, color=color)
        lbl.next_to(arrow, label_direction, buff=buff)
        parts.append(lbl)

    return VGroup(*parts)


def make_brace_annotation(
    mobject: VMobject,
    text: str,
    direction=DOWN,
    color=ACCENT_CYAN,
    font_size: int = LABEL_SIZE,
) -> VGroup:
    """Create a brace with label text under/over a mobject.

    When to use: Labeling a span of items (e.g., "these 3 elements" or
    "this entire term"). Classic mathematical annotation.

    Args:
        mobject: What to put the brace around.
        text: Label text.
        direction: Where the brace goes (DOWN, UP, LEFT, RIGHT).
        color: Brace and text color.
        font_size: Label font size.

    Returns:
        VGroup(brace, label).
    """
    from manim import Brace
    brace = Brace(mobject, direction, color=color)
    label = brace.get_text(text, font_size=font_size).set_color(color)
    return VGroup(brace, label)


# =============================================================================
# 6. PROGRESS INDICATORS
# =============================================================================

def make_progress_bar(
    total_steps: int,
    current_step: int = 0,
    width: float = 10.0,
    height: float = 0.12,
    bg_color=TEXT_DIM,
    fill_color=ACCENT_BLUE,
    position=None,
) -> VGroup:
    """Create a progress bar showing completion through the video.

    When to use: Multi-section educational videos. Place at the top or bottom
    edge and update it as each section completes.

    Args:
        total_steps: Total number of sections/steps.
        current_step: Current step (0-indexed).
        width: Bar width.
        height: Bar height.
        bg_color: Background track color.
        fill_color: Filled portion color.
        position: Where to place it. Default: bottom edge.

    Returns:
        VGroup(bg_bar, fill_bar, step_dots).

    Example:
        bar = make_progress_bar(5, current_step=0)
        self.add(bar)
        # Later, to advance:
        new_bar = make_progress_bar(5, current_step=2)
        self.play(Transform(bar, new_bar))
    """
    if position is None:
        position = DOWN * 3.7

    # Background track
    bg = RoundedRectangle(
        corner_radius=height / 2, width=width, height=height,
        fill_color=bg_color, fill_opacity=0.3,
        stroke_width=0,
    ).move_to(position)

    # Filled portion
    fill_width = max(0.01, width * (current_step / total_steps))
    fill = RoundedRectangle(
        corner_radius=height / 2, width=fill_width, height=height,
        fill_color=fill_color, fill_opacity=0.8,
        stroke_width=0,
    )
    fill.move_to(bg.get_left(), aligned_edge=LEFT)

    # Step dots
    dots = VGroup()
    for i in range(total_steps):
        x = bg.get_left()[0] + (i + 0.5) * width / total_steps
        dot = Dot(
            point=[x, position[1], 0], radius=0.04,
            color=TEXT_PRIMARY if i < current_step else TEXT_DIM,
            fill_opacity=1.0 if i < current_step else 0.3,
        )
        dots.add(dot)

    return VGroup(bg, fill, dots)


def make_step_counter(
    total_steps: int,
    current_step: int = 1,
    label_prefix: str = "Step",
    color=ACCENT_BLUE,
    font_size: int = LABEL_SIZE,
    position=None,
) -> VGroup:
    """Create a step counter like "Step 2/5".

    When to use: Procedural explanations (algorithms, proofs, recipes).
    Shows the viewer where they are in the sequence.

    Args:
        total_steps: Total steps.
        current_step: Current step (1-indexed).
        label_prefix: Text before the number.
        color: Accent color.
        font_size: Text size.
        position: Where to place. Default: top-right corner.

    Returns:
        VGroup(bg_rect, text).

    Example:
        counter = make_step_counter(5, current_step=1)
        self.add(counter)
        # Advance: Transform(counter, make_step_counter(5, 2))
    """
    if position is None:
        position = UR + LEFT * 1.5 + DOWN * 0.5

    text = Text(
        f"{label_prefix} {current_step}/{total_steps}",
        font_size=font_size, color=color, weight="BOLD",
    )
    bg = RoundedRectangle(
        corner_radius=0.1,
        width=text.width + 0.4,
        height=text.height + 0.25,
        fill_color=BG_COLOR, fill_opacity=0.8,
        stroke_color=color, stroke_width=1.5,
    )
    text.move_to(bg.get_center())
    group = VGroup(bg, text).move_to(position)
    return group


def make_section_marker(
    sections: list[str],
    current_idx: int = 0,
    active_color=ACCENT_BLUE,
    inactive_color=TEXT_DIM,
    font_size: int = 16,
    position=None,
) -> VGroup:
    """Create section markers showing the current position in a multi-part lesson.

    When to use: Videos with distinct sections (Intro, Theory, Example, Quiz).
    Shows at the top or side as a persistent navigation indicator.

    Args:
        sections: List of section names.
        current_idx: Index of the active section.
        active_color: Color of the active section.
        inactive_color: Color of inactive sections.
        font_size: Text size.
        position: Where to place. Default: top center.

    Returns:
        VGroup of section indicators.

    Example:
        sections = ["Intro", "Theory", "Example", "Quiz"]
        marker = make_section_marker(sections, current_idx=0)
        self.add(marker)
        # Later: Transform(marker, make_section_marker(sections, 2))
    """
    if position is None:
        position = UP * 3.6

    markers = VGroup()
    for i, name in enumerate(sections):
        is_active = i == current_idx
        is_done = i < current_idx

        color = active_color if is_active else (
            TEXT_SECONDARY if is_done else inactive_color
        )
        weight = "BOLD" if is_active else "NORMAL"

        txt = Text(name, font_size=font_size, color=color, weight=weight)

        # Active indicator dot
        dot = Dot(radius=0.04, color=color, fill_opacity=1.0 if is_active else 0.4)
        dot.next_to(txt, DOWN, buff=0.08)

        markers.add(VGroup(txt, dot))

        # Connecting line between sections
        if i < len(sections) - 1:
            spacer = Line(ORIGIN, RIGHT * 0.4, stroke_color=inactive_color,
                          stroke_width=1, stroke_opacity=0.3)
            markers.add(spacer)

    markers.arrange(RIGHT, buff=0.15).move_to(position)
    return markers


# =============================================================================
# 7. COMPOSITE ANIMATION HELPERS
# =============================================================================

def sweep_in_group(scene, group: VGroup, direction=RIGHT,
                   lag_ratio: float = 0.15, run_time: float = 1.5):
    """Sweep in a group of mobjects one by one from a direction.

    When to use: Introducing a list, series of bullet points, or multiple
    diagram elements. Creates a cascading "reveal" effect.

    Args:
        scene: The Scene instance.
        group: VGroup of mobjects to sweep in.
        direction: Direction items come from.
        lag_ratio: Overlap between successive items.
        run_time: Total animation time.
    """
    scene.play(
        LaggedStart(
            *[FadeIn(mob, shift=direction * 0.5) for mob in group],
            lag_ratio=lag_ratio,
        ),
        run_time=run_time,
    )


def cascade_fade_in(scene, group: VGroup, lag_ratio: float = 0.1,
                    run_time: float = 2.0, scale: float = 0.8):
    """Cascade fade in: each submobject fades in with a slight scale-up.

    When to use: Grid of items, matrix visualization, or any collection
    where you want elements appearing in sequence.

    Args:
        scene: The Scene instance.
        group: VGroup of mobjects.
        lag_ratio: Overlap between items.
        run_time: Total time.
        scale: Starting scale factor (< 1 means items grow in).
    """
    scene.play(
        LaggedStart(
            *[FadeIn(mob, scale=scale) for mob in group],
            lag_ratio=lag_ratio,
        ),
        run_time=run_time,
    )


def pop_in_sequence(scene, group: VGroup, lag_ratio: float = 0.1,
                    run_time: float = 1.5):
    """Pop in items one by one with GrowFromCenter.

    When to use: Introducing diagram nodes, data structure elements,
    or any discrete items. Playful but professional.

    Args:
        scene: The Scene instance.
        group: VGroup of mobjects.
        lag_ratio: Overlap.
        run_time: Total time.
    """
    scene.play(
        LaggedStart(
            *[GrowFromCenter(mob) for mob in group],
            lag_ratio=lag_ratio,
        ),
        run_time=run_time,
    )


def staggered_write(scene, group: VGroup, lag_ratio: float = 0.3,
                    run_time: float = 2.0):
    """Write (draw) multiple mobjects with staggered timing.

    When to use: Multiple formulas or text lines appearing in sequence.
    More polished than writing them one at a time with separate play() calls.

    Args:
        scene: The Scene instance.
        group: VGroup of VMobjects.
        lag_ratio: Overlap between writes.
        run_time: Total time.
    """
    scene.play(
        LaggedStart(
            *[Write(mob) for mob in group],
            lag_ratio=lag_ratio,
        ),
        run_time=run_time,
    )


def dynamic_counter(
    scene,
    start_value: float,
    end_value: float,
    position=ORIGIN,
    font_size: int = 48,
    color=ACCENT_CYAN,
    num_decimals: int = 0,
    prefix: str = "",
    suffix: str = "",
    run_time: float = 3.0,
):
    """Animate a number counting from start to end.

    When to use: Showing statistics, costs, or any numeric value that
    changes. Dramatic reveal of a final number.

    With voiceover: Start the counter as narrator says "the total comes
    out to..." and let it finish as they say the number.

    Args:
        scene: The Scene instance.
        start_value: Starting number.
        end_value: Ending number.
        position: Where to show the counter.
        font_size: Number size.
        color: Number color.
        num_decimals: Decimal places.
        prefix: Text before number (e.g., "$").
        suffix: Text after number (e.g., "%").
        run_time: How long the counting takes.

    Returns:
        The final DecimalNumber mobject (still on screen).
    """
    tracker = ValueTracker(start_value)

    number = always_redraw(lambda: DecimalNumber(
        tracker.get_value(),
        num_decimal_places=num_decimals,
        font_size=font_size,
        color=color,
    ).move_to(position))

    prefix_mob = None
    suffix_mob = None

    if prefix:
        prefix_mob = always_redraw(lambda: Text(
            prefix, font_size=font_size, color=color,
        ).next_to(number, LEFT, buff=0.1))

    if suffix:
        suffix_mob = always_redraw(lambda: Text(
            suffix, font_size=font_size, color=color,
        ).next_to(number, RIGHT, buff=0.1))

    to_add = [number]
    if prefix_mob:
        to_add.append(prefix_mob)
    if suffix_mob:
        to_add.append(suffix_mob)

    scene.add(*to_add)
    scene.play(
        tracker.animate.set_value(end_value),
        run_time=run_time,
        rate_func=smooth,
    )

    return VGroup(*to_add)


# =============================================================================
# 8. VOICEOVER INTEGRATION HELPERS
# =============================================================================

def timed_sequence(scene, animations_with_durations: list[tuple],
                   total_duration: float = None):
    """Play a sequence of animations, auto-filling remaining time with waits.

    When to use: Inside a voiceover block when you have multiple animations
    that need to fit within a specific time window.

    Args:
        scene: The Scene instance.
        animations_with_durations: List of (animation, duration) tuples.
            Use None for duration to use the animation's default.
        total_duration: If set, add a wait at the end to fill remaining time.

    Example:
        with self.voiceover(text="Here we see...") as tracker:
            timed_sequence(self, [
                (Write(eq), 1.5),
                (Create(axes), 1.0),
                (FadeIn(graph), 0.8),
            ], total_duration=tracker.duration)
    """
    elapsed = 0
    for anim, dur in animations_with_durations:
        if dur is None:
            dur = anim.run_time
        scene.play(anim, run_time=dur)
        elapsed += dur

    if total_duration and elapsed < total_duration:
        scene.wait(total_duration - elapsed - 0.3)


def cleanup_and_transition(scene, old_mobjects: list, new_title: str = None,
                            title_ref=None):
    """Clean up current section and optionally show a new section title.

    When to use: Between voiceover sections. Clears the old content and
    optionally updates the persistent title.

    Args:
        scene: The Scene instance.
        old_mobjects: List of mobjects to fade out.
        new_title: If provided, transform the title to this new text.
        title_ref: Reference to the existing title mobject (for transform).

    Returns:
        The new title mobject (or None).
    """
    fade_group = VGroup(*[m for m in old_mobjects if m is not title_ref])

    if new_title and title_ref:
        new_title_mob = Text(new_title, font_size=TITLE_SIZE,
                             color=TEXT_PRIMARY, weight="BOLD")
        new_title_mob.to_edge(UP, buff=0.3)
        scene.play(
            FadeOut(fade_group, run_time=0.5),
            Transform(title_ref, new_title_mob, run_time=0.8),
        )
        return title_ref
    else:
        scene.play(FadeOut(fade_group), run_time=0.5)
        return None
