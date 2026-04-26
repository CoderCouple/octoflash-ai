"""
Reusable mathematical animation helpers inspired by 3Blue1Brown patterns.

Adapted from 3b1b/videos (manimgl) to Manim Community Edition.
All functions return VGroup/Mobject objects or animate directly on a scene.

Includes:
    - Fourier epicycle visualization (rotating vectors summing to trace a shape)
    - Parametric curve tracing with moving dot
    - Phase space dual visualization
    - Probability distribution visualization (histogram, bell curve, area model)
    - Calculus animations (Riemann sum, shell integration, tangent line)
    - Equation derivation chains (step-by-step TransformMatchingTex)
    - Number line with animated tracker dot
    - Matrix element visualization and transformation
    - TracedPath motion history
    - Color-coded equation highlighting

Compatible with Manim Community Edition (tested 0.18+).

Usage:
    from app.manim_pipeline.math_animations import (
        make_epicycle_vectors, animate_epicycles,
        trace_curve_with_dot, animate_parametric_trace,
        make_number_line_tracker, animate_tracker_sweep,
        make_histogram, animate_histogram_build, animate_histogram_morph,
        make_bell_curve, animate_area_under_curve,
        animate_riemann_refinement,
        animate_tangent_line_sweep,
        animate_equation_chain,
        make_matrix_grid, animate_matrix_highlight,
        make_coordinate_pair, animate_coordinate_mapping,
    )
"""

from __future__ import annotations

import numpy as np
from manim import (
    VGroup, VMobject, Circle, Line, Arrow, Dot, DashedLine,
    Text, MathTex, Tex, Axes, NumberLine, BarChart,
    Rectangle, RoundedRectangle, Square, Polygon,
    SurroundingRectangle, Brace, DecimalNumber,
    FadeIn, FadeOut, Create, Write, Uncreate,
    Transform, ReplacementTransform, TransformMatchingTex,
    GrowFromCenter, GrowArrow,
    AnimationGroup, LaggedStart, LaggedStartMap, Succession,
    Indicate, Circumscribe, Flash,
    ShowPassingFlash, ShowIncreasingSubsets,
    UP, DOWN, LEFT, RIGHT, ORIGIN, UL, UR, DL, DR,
    PI, TAU, DEGREES,
    ValueTracker, always_redraw, TracedPath,
    ParametricFunction,
    config,
    linear, smooth, rate_functions,
    interpolate_color, color_gradient,
)

from app.manim_pipeline.styles import (
    ACCENT_BLUE, ACCENT_ORANGE, ACCENT_GREEN, ACCENT_RED,
    ACCENT_PURPLE, ACCENT_YELLOW, ACCENT_CYAN, ACCENT_PINK,
    TEXT_PRIMARY, TEXT_SECONDARY, TEXT_DIM,
    TITLE_SIZE, SUBTITLE_SIZE, BODY_SIZE, LABEL_SIZE,
    BG_COLOR,
)


# ════���══════════════════════════════════════════════════════════════════════════
# FOURIER EPICYCLE VISUALIZATION
# ════════════════════════���══════════════════════════════���═══════════════════════

def compute_fourier_coefficients(points: list[complex], n_vectors: int = 20) -> list[dict]:
    """Compute Fourier coefficients from a list of complex points.

    Args:
        points: List of complex numbers tracing a closed path.
        n_vectors: Number of rotating vectors (more = finer detail).

    Returns:
        List of dicts with 'freq', 'amplitude', 'phase' sorted by amplitude.
    """
    N = len(points)
    coefficients = []
    freqs = list(range(-n_vectors // 2, n_vectors // 2 + 1))

    for freq in freqs:
        coeff = sum(
            p * np.exp(-2j * PI * freq * k / N)
            for k, p in enumerate(points)
        ) / N
        coefficients.append({
            "freq": freq,
            "amplitude": abs(coeff),
            "phase": np.angle(coeff),
            "coeff": coeff,
        })

    # Sort by amplitude (largest first) for visual impact
    coefficients.sort(key=lambda c: -c["amplitude"])
    return coefficients


def make_epicycle_vectors(
    coefficients: list[dict],
    time_tracker: ValueTracker,
    center: np.ndarray = ORIGIN,
    circle_color: str = ACCENT_BLUE,
    vector_color: str = TEXT_PRIMARY,
) -> dict:
    """Create rotating vector chain for Fourier epicycle animation.

    Args:
        coefficients: From compute_fourier_coefficients().
        time_tracker: ValueTracker controlling animation time (0 to 1 = one cycle).
        center: Center point of the outermost circle.
        circle_color: Color for the circles.
        vector_color: Color for the vector arrows.

    Returns:
        dict with 'vectors' (VGroup), 'circles' (VGroup), 'tip_dot' (Dot),
        'all' (VGroup of everything).
    """
    def get_tip_position():
        t = time_tracker.get_value()
        pos = np.array([center[0], center[1], 0], dtype=float)
        for c in coefficients:
            angle = 2 * PI * c["freq"] * t + c["phase"]
            pos[0] += c["amplitude"] * np.cos(angle)
            pos[1] += c["amplitude"] * np.sin(angle)
        return pos

    circles = always_redraw(lambda: _build_epicycle_circles(
        coefficients, time_tracker.get_value(), center, circle_color
    ))

    vectors = always_redraw(lambda: _build_epicycle_vectors(
        coefficients, time_tracker.get_value(), center, vector_color
    ))

    tip_dot = always_redraw(lambda: Dot(
        get_tip_position(), color=ACCENT_ORANGE, radius=0.06
    ))

    all_group = VGroup(circles, vectors, tip_dot)
    return {
        "circles": circles,
        "vectors": vectors,
        "tip_dot": tip_dot,
        "all": all_group,
        "get_tip": get_tip_position,
    }


def _build_epicycle_circles(coefficients, t, center, color):
    """Internal: build circle chain at time t."""
    circles = VGroup()
    pos = np.array([center[0], center[1], 0], dtype=float)
    for c in coefficients:
        if c["amplitude"] < 0.01:
            continue
        circle = Circle(
            radius=c["amplitude"],
            stroke_color=color,
            stroke_width=1,
            stroke_opacity=0.3,
            fill_opacity=0,
        ).move_to(pos)
        circles.add(circle)
        angle = 2 * PI * c["freq"] * t + c["phase"]
        pos = pos.copy()
        pos[0] += c["amplitude"] * np.cos(angle)
        pos[1] += c["amplitude"] * np.sin(angle)
    return circles


def _build_epicycle_vectors(coefficients, t, center, color):
    """Internal: build vector chain at time t."""
    vectors = VGroup()
    pos = np.array([center[0], center[1], 0], dtype=float)
    for c in coefficients:
        if c["amplitude"] < 0.01:
            continue
        angle = 2 * PI * c["freq"] * t + c["phase"]
        new_pos = pos.copy()
        new_pos[0] += c["amplitude"] * np.cos(angle)
        new_pos[1] += c["amplitude"] * np.sin(angle)
        line = Line(pos, new_pos, stroke_color=color, stroke_width=1.5)
        vectors.add(line)
        pos = new_pos
    return vectors


def animate_epicycles(
    scene,
    coefficients: list[dict],
    n_cycles: float = 1.0,
    run_time: float = 8.0,
    center: np.ndarray = ORIGIN,
    trace_color: str = ACCENT_CYAN,
    fade_trace: bool = True,
) -> dict:
    """Full epicycle animation: rotating vectors trace a path.

    Args:
        scene: The Manim Scene to animate in.
        coefficients: From compute_fourier_coefficients().
        n_cycles: Number of full rotations.
        run_time: Total animation time.
        center: Center point.
        trace_color: Color of the traced path.
        fade_trace: Whether the trace fades over time.

    Returns:
        dict with all created mobjects.
    """
    t = ValueTracker(0)
    epicycle_data = make_epicycle_vectors(coefficients, t, center)

    # Traced path from the tip
    trace = TracedPath(
        epicycle_data["get_tip"],
        stroke_color=trace_color,
        stroke_width=3,
        dissipating_time=run_time * 0.8 if fade_trace else None,
    )

    scene.add(trace, epicycle_data["all"])
    scene.play(t.animate.set_value(n_cycles), run_time=run_time, rate_func=linear)

    return {"tracker": t, "epicycle": epicycle_data, "trace": trace}


# ════════════════���══════════════════════════════════════════════════════���═══════
# CURVE TRACING WITH DOT
# ═════════════��═════════════════════════════════════════════════════════════════

def trace_curve_with_dot(
    axes,
    func,
    x_range: tuple = (-4, 4),
    curve_color: str = ACCENT_CYAN,
    dot_color: str = ACCENT_ORANGE,
    dot_radius: float = 0.08,
) -> dict:
    """Create a curve that traces from left to right with a leading dot.

    Args:
        axes: Manim Axes object.
        func: Function to plot.
        x_range: (x_min, x_max).
        curve_color: Color of the growing curve.
        dot_color: Color of the leading dot.
        dot_radius: Radius of the dot.

    Returns:
        dict with 'tracker' (ValueTracker), 'curve', 'dot', 'all' (VGroup).
    """
    t = ValueTracker(x_range[0])

    curve = always_redraw(lambda: axes.plot(
        func, x_range=[x_range[0], t.get_value()],
        color=curve_color, stroke_width=3,
    ))

    dot = always_redraw(lambda: Dot(
        axes.c2p(t.get_value(), func(t.get_value())),
        color=dot_color, radius=dot_radius,
    ))

    return {
        "tracker": t,
        "curve": curve,
        "dot": dot,
        "all": VGroup(curve, dot),
    }


def animate_parametric_trace(
    scene,
    axes,
    func,
    x_range: tuple = (-4, 4),
    run_time: float = 4.0,
    curve_color: str = ACCENT_CYAN,
    dot_color: str = ACCENT_ORANGE,
    show_label: bool = True,
) -> dict:
    """Animate a function being traced with a moving dot and optional value label.

    Returns dict with all created mobjects.
    """
    data = trace_curve_with_dot(axes, func, x_range, curve_color, dot_color)
    t = data["tracker"]

    mobjects = [data["curve"], data["dot"]]

    if show_label:
        label = always_redraw(lambda: MathTex(
            f"x = {t.get_value():.1f}",
            font_size=20, color=dot_color,
        ).next_to(data["dot"], UR, buff=0.1))
        mobjects.append(label)
        data["label"] = label

    scene.add(*mobjects)
    scene.play(t.animate.set_value(x_range[1]), run_time=run_time, rate_func=linear)

    data["all"] = VGroup(*mobjects)
    return data


# ���═════════════════════���════════════════════════════════════════════════════════
# NUMBER LINE TRACKER
# ═══════════════════════════════════════════════════════════════════���═══════════

def make_number_line_tracker(
    x_range: tuple = (0, 10, 1),
    length: float = 10,
    initial_value: float = 0,
    dot_color: str = ACCENT_CYAN,
    label_fmt: str = "{:.1f}",
) -> dict:
    """Create a number line with an animated tracker dot and value label.

    Returns:
        dict with 'line' (NumberLine), 'tracker' (ValueTracker),
        'dot', 'label', 'all' (VGroup).
    """
    nl = NumberLine(
        x_range=list(x_range), length=length,
        include_numbers=True, include_tip=True,
        color=TEXT_DIM,
    )

    t = ValueTracker(initial_value)

    dot = always_redraw(lambda: Dot(
        nl.n2p(t.get_value()), color=dot_color, radius=0.1
    ))

    label = always_redraw(lambda: MathTex(
        label_fmt.format(t.get_value()),
        font_size=24, color=dot_color,
    ).next_to(dot, UP, buff=0.15))

    return {
        "line": nl,
        "tracker": t,
        "dot": dot,
        "label": label,
        "all": VGroup(nl, dot, label),
    }


def animate_tracker_sweep(
    scene, tracker_data: dict, target_value: float,
    run_time: float = 3.0, rate_func=linear,
):
    """Animate the tracker dot sweeping to a target value."""
    t = tracker_data["tracker"]
    scene.play(t.animate.set_value(target_value), run_time=run_time, rate_func=rate_func)


# ═══════════════════════════════════��══════════════════════════════��════════════
# PROBABILITY DISTRIBUTIONS
# ══════════════════════════��════════════════════════════════════════════════════

def make_histogram(
    values: list[float],
    bar_names: list[str] | None = None,
    y_range: tuple = (0, 1, 0.2),
    bar_colors: list[str] | None = None,
    width: float = 8,
    height: float = 3.5,
) -> dict:
    """Create a styled histogram/bar chart.

    Returns:
        dict with 'chart' (BarChart), 'all' (VGroup).
    """
    if bar_names is None:
        bar_names = [str(i) for i in range(len(values))]
    if bar_colors is None:
        bar_colors = [ACCENT_BLUE, ACCENT_GREEN, ACCENT_ORANGE, ACCENT_RED,
                      ACCENT_PURPLE, ACCENT_CYAN, ACCENT_PINK, ACCENT_YELLOW]

    chart = BarChart(
        values=values,
        bar_names=bar_names,
        y_range=list(y_range),
        x_length=width,
        y_length=height,
        bar_colors=bar_colors[:len(values)],
        bar_width=0.6,
        bar_fill_opacity=0.8,
        bar_stroke_width=1,
    )

    return {"chart": chart, "all": VGroup(chart)}


def animate_histogram_build(scene, hist_data: dict, run_time: float = 2.0):
    """Animate histogram bars growing from zero."""
    chart = hist_data["chart"]
    scene.play(Create(chart), run_time=run_time)


def animate_histogram_morph(scene, chart, new_values: list[float], run_time: float = 2.0):
    """Morph histogram to new values."""
    target = chart.copy()
    target.change_bar_values(new_values)
    scene.play(chart.animate.become(target), run_time=run_time)


def make_bell_curve(
    axes,
    mean: float = 0,
    std: float = 1,
    color: str = ACCENT_CYAN,
    fill_opacity: float = 0.3,
) -> dict:
    """Create a Gaussian bell curve on given axes.

    Returns:
        dict with 'curve', 'area', 'all' (VGroup).
    """
    def gaussian(x):
        return (1 / (std * np.sqrt(2 * PI))) * np.exp(-0.5 * ((x - mean) / std) ** 2)

    curve = axes.plot(gaussian, color=color, stroke_width=3)
    area = axes.get_area(curve, color=color, opacity=fill_opacity)

    return {"curve": curve, "area": area, "all": VGroup(area, curve)}


def animate_area_under_curve(
    scene, axes, func, x_range: tuple = (-3, 3),
    color: str = ACCENT_GREEN, run_time: float = 2.0,
) -> VMobject:
    """Animate the area under a curve filling in."""
    graph = axes.plot(func, x_range=list(x_range), color=color)
    area = axes.get_area(graph, x_range=list(x_range), color=color, opacity=0.4)
    scene.play(Create(graph), run_time=run_time * 0.4)
    scene.play(FadeIn(area), run_time=run_time * 0.6)
    return VGroup(graph, area)


# ══════════════════════════════════════════��═══════════════════════════���════════
# CALCULUS ANIMATIONS
# ═════════════════════════════════════════════════════��═════════════════════════

def animate_riemann_refinement(
    scene,
    axes,
    graph,
    x_range: tuple = (0, 4),
    dx_values: list[float] | None = None,
    color_range: tuple = (ACCENT_BLUE, ACCENT_GREEN),
    run_time_per_step: float = 1.5,
) -> VMobject:
    """Animate Riemann rectangles getting finer, then morph to exact area.

    Args:
        scene: Manim Scene.
        axes: Axes object.
        graph: The plotted function graph.
        x_range: Integration range.
        dx_values: List of dx widths (coarse to fine).
        color_range: Color gradient for rectangles.
        run_time_per_step: Time for each refinement step.

    Returns:
        The final area mobject.
    """
    if dx_values is None:
        dx_values = [1.0, 0.5, 0.25, 0.1]

    prev_rects = None
    for dx in dx_values:
        rects = axes.get_riemann_rectangles(
            graph, x_range=list(x_range), dx=dx,
            color=color_range, fill_opacity=0.5,
        )
        if prev_rects is None:
            scene.play(Create(rects), run_time=run_time_per_step)
        else:
            scene.play(Transform(prev_rects, rects), run_time=run_time_per_step)
        if prev_rects is None:
            prev_rects = rects

    # Morph to exact area
    area = axes.get_area(graph, x_range=list(x_range), color=color_range, opacity=0.5)
    scene.play(FadeOut(prev_rects), FadeIn(area), run_time=run_time_per_step)
    return area


def animate_tangent_line_sweep(
    scene,
    axes,
    func,
    deriv_func,
    x_range: tuple = (-3, 3),
    line_color: str = ACCENT_YELLOW,
    line_length: float = 3,
    run_time: float = 5.0,
) -> dict:
    """Animate a tangent line sweeping along a curve.

    Args:
        func: The function y = f(x).
        deriv_func: The derivative f'(x).
        x_range: Sweep range.
        line_color: Tangent line color.
        line_length: Total length of tangent line segment.

    Returns:
        dict with 'tracker', 'dot', 'tangent', 'all'.
    """
    t = ValueTracker(x_range[0])

    dot = always_redraw(lambda: Dot(
        axes.c2p(t.get_value(), func(t.get_value())),
        color=ACCENT_ORANGE, radius=0.06,
    ))

    def get_tangent():
        x = t.get_value()
        slope = deriv_func(x)
        p = axes.c2p(x, func(x))
        # Direction vector along tangent
        dx = line_length / 2
        angle = np.arctan(slope)
        offset = np.array([dx * np.cos(angle), dx * np.sin(angle), 0])
        return Line(
            p - offset, p + offset,
            stroke_color=line_color, stroke_width=2.5,
        )

    tangent = always_redraw(get_tangent)

    slope_label = always_redraw(lambda: MathTex(
        rf"f'({t.get_value():.1f}) = {deriv_func(t.get_value()):.2f}",
        font_size=20, color=line_color,
    ).next_to(dot, UL, buff=0.15))

    scene.add(dot, tangent, slope_label)
    scene.play(t.animate.set_value(x_range[1]), run_time=run_time, rate_func=linear)

    return {
        "tracker": t, "dot": dot, "tangent": tangent,
        "slope_label": slope_label,
        "all": VGroup(dot, tangent, slope_label),
    }


# ═════════��════════════════════════════════��════════════════════════════��═══════
# EQUATION DERIVATION CHAINS
# ════════════════════���═══════════════════��═══════════════════════��══════════════

def animate_equation_chain(
    scene,
    equations: list[MathTex],
    position=ORIGIN,
    wait_between: float = 1.0,
    transform_run_time: float = 1.5,
    use_matching_tex: bool = True,
) -> MathTex:
    """Animate a chain of equation derivation steps.

    Each equation morphs into the next using TransformMatchingTex.

    Args:
        scene: Manim Scene.
        equations: List of MathTex objects.
        position: Where to place the equations.
        wait_between: Wait time between steps.
        transform_run_time: Time for each morph.
        use_matching_tex: Use TransformMatchingTex (True) or Transform (False).

    Returns:
        The final equation mobject.
    """
    if not equations:
        return None

    current = equations[0].move_to(position)
    scene.play(Write(current), run_time=transform_run_time)

    for next_eq in equations[1:]:
        scene.wait(wait_between)
        next_eq.move_to(position)
        if use_matching_tex:
            scene.play(TransformMatchingTex(current, next_eq),
                       run_time=transform_run_time)
        else:
            scene.play(Transform(current, next_eq),
                       run_time=transform_run_time)
        current = next_eq

    return current


# ════════════════════════════════════════��══════════════════════════════════════
# MATRIX VISUALIZATION
# ═════════════════════════════════════════���══════════════════════════��══════════

def make_matrix_grid(
    rows: int,
    cols: int,
    values: list[list[float]] | None = None,
    cell_size: float = 0.6,
    positive_color: str = ACCENT_BLUE,
    negative_color: str = ACCENT_RED,
    zero_color: str = TEXT_DIM,
    show_values: bool = True,
    font_size: int = 18,
) -> dict:
    """Create a color-coded matrix grid (like weight matrices in neural nets).

    Positive values are blue, negative are red, zero is grey.

    Returns:
        dict with 'cells' (2D list), 'labels' (2D list), 'all' (VGroup).
    """
    cells = []
    labels = []
    all_group = VGroup()

    for r in range(rows):
        cell_row = []
        label_row = []
        for c in range(cols):
            val = values[r][c] if values else 0.0
            # Color based on sign and magnitude
            if val > 0:
                opacity = min(abs(val), 1.0)
                fill_color = positive_color
            elif val < 0:
                opacity = min(abs(val), 1.0)
                fill_color = negative_color
            else:
                opacity = 0.1
                fill_color = zero_color

            cell = Square(
                side_length=cell_size,
                fill_color=fill_color,
                fill_opacity=opacity * 0.6,
                stroke_color=TEXT_DIM,
                stroke_width=0.5,
            )
            cell.move_to(
                RIGHT * c * cell_size + DOWN * r * cell_size
            )
            cell_row.append(cell)
            all_group.add(cell)

            if show_values and values:
                lbl = MathTex(
                    f"{val:.1f}", font_size=font_size,
                    color=TEXT_PRIMARY,
                )
                lbl.move_to(cell.get_center())
                label_row.append(lbl)
                all_group.add(lbl)
            else:
                label_row.append(None)

        cells.append(cell_row)
        labels.append(label_row)

    # Center the grid
    all_group.move_to(ORIGIN)

    return {"cells": cells, "labels": labels, "all": all_group}


def animate_matrix_highlight(
    scene, matrix_data: dict, row: int = None, col: int = None,
    color: str = ACCENT_YELLOW, run_time: float = 1.0,
):
    """Highlight a row or column of the matrix grid."""
    cells = matrix_data["cells"]
    targets = []

    if row is not None:
        targets = cells[row]
    elif col is not None:
        targets = [cells[r][col] for r in range(len(cells))]

    if targets:
        rects = VGroup(*[
            SurroundingRectangle(cell, color=color, buff=0.02, stroke_width=2)
            for cell in targets
        ])
        scene.play(Create(rects), run_time=run_time)
        return rects
    return VGroup()


# ═══════════════��══════════════════════════════════════════════════════���════════
# COORDINATE MAPPING (PHASE SPACE / DUAL VIEWS)
# ════════��═════════════════���══════════════════════════��═════════════════════════

def make_coordinate_pair(
    left_axes_config: dict | None = None,
    right_axes_config: dict | None = None,
    left_title: str = "Physical Space",
    right_title: str = "Phase Space",
    spacing: float = 0.5,
) -> dict:
    """Create two side-by-side coordinate systems for dual visualization.

    Returns:
        dict with 'left_axes', 'right_axes', 'left_title', 'right_title',
        'all' (VGroup).
    """
    l_cfg = left_axes_config or {
        "x_range": [-4, 4, 1], "y_range": [-3, 3, 1],
        "x_length": 5, "y_length": 3.5,
    }
    r_cfg = right_axes_config or {
        "x_range": [-4, 4, 1], "y_range": [-3, 3, 1],
        "x_length": 5, "y_length": 3.5,
    }

    left_axes = Axes(
        **l_cfg,
        axis_config={"color": TEXT_DIM, "stroke_width": 1.5},
        tips=False,
    )
    right_axes = Axes(
        **r_cfg,
        axis_config={"color": TEXT_DIM, "stroke_width": 1.5},
        tips=False,
    )

    pair = VGroup(left_axes, right_axes).arrange(RIGHT, buff=spacing)
    pair.shift(DOWN * 0.3)

    lt = Text(left_title, font_size=LABEL_SIZE, color=TEXT_SECONDARY)
    lt.next_to(left_axes, UP, buff=0.2)
    rt = Text(right_title, font_size=LABEL_SIZE, color=TEXT_SECONDARY)
    rt.next_to(right_axes, UP, buff=0.2)

    return {
        "left_axes": left_axes,
        "right_axes": right_axes,
        "left_title": lt,
        "right_title": rt,
        "all": VGroup(left_axes, right_axes, lt, rt),
    }


def animate_coordinate_mapping(
    scene,
    left_axes,
    right_axes,
    mapping_func,
    x_range: tuple = (-3, 3),
    n_points: int = 50,
    dot_color: str = ACCENT_CYAN,
    run_time: float = 3.0,
):
    """Animate points being mapped from left axes to right axes.

    Args:
        mapping_func: Takes (x, y) and returns (new_x, new_y).
    """
    # Create points on left
    left_dots = VGroup()
    right_dots = VGroup()

    for i in range(n_points):
        x = x_range[0] + (x_range[1] - x_range[0]) * i / (n_points - 1)
        y = np.sin(x)  # example curve
        left_dot = Dot(left_axes.c2p(x, y), color=dot_color, radius=0.04)
        left_dots.add(left_dot)

        nx, ny = mapping_func(x, y)
        right_dot = Dot(right_axes.c2p(nx, ny), color=dot_color, radius=0.04)
        right_dots.add(right_dot)

    scene.play(LaggedStartMap(FadeIn, left_dots, lag_ratio=0.02), run_time=run_time * 0.4)

    # Animate mapping with connecting arrows for a few sample points
    scene.play(
        *[
            ReplacementTransform(left_dots[i].copy(), right_dots[i])
            for i in range(0, n_points, max(1, n_points // 10))
        ],
        run_time=run_time * 0.6,
    )
    scene.play(
        LaggedStartMap(FadeIn, right_dots, lag_ratio=0.02),
        run_time=run_time * 0.3,
    )

    return {"left_dots": left_dots, "right_dots": right_dots}


# ═══════════��════════════════════════���═══════════════════════════��══════════════
# COLOR-CODED EQUATION HIGHLIGHTING
# ═════════���═══════════════════════��═══════════════════════════════���═════════════

def highlight_equation_parts(
    equation: MathTex,
    color_map: dict[str, str],
) -> MathTex:
    """Apply color coding to specific parts of a MathTex equation.

    Args:
        equation: MathTex object.
        color_map: Dict mapping tex strings to colors, e.g. {"x": BLUE, "y": RED}.
    """
    for tex_str, color in color_map.items():
        equation.set_color_by_tex(tex_str, color)
    return equation


def animate_equation_highlight(
    scene, equation: MathTex, parts: list[str],
    highlight_color: str = ACCENT_YELLOW,
    run_time: float = 0.8,
):
    """Sequentially highlight parts of an equation with Indicate."""
    for part in parts:
        targets = equation.get_parts_by_tex(part)
        if targets:
            scene.play(Indicate(targets, color=highlight_color), run_time=run_time)


# ══════��══════════════════���═════════════════════════════════════════════════════
# TRACED PATH ANIMATION
# ═��═════════════════════════════════════════════════════════════════════════════

def make_traced_dot(
    initial_point: np.ndarray = ORIGIN,
    dot_color: str = ACCENT_ORANGE,
    trace_color: str = ACCENT_CYAN,
    dot_radius: float = 0.08,
    dissipating_time: float | None = None,
) -> dict:
    """Create a dot with a TracedPath that shows its motion history.

    Returns:
        dict with 'dot', 'trace', 'all' (VGroup).
    """
    dot = Dot(initial_point, color=dot_color, radius=dot_radius)

    trace_kwargs = {
        "stroke_color": trace_color,
        "stroke_width": 2,
    }
    if dissipating_time is not None:
        trace_kwargs["dissipating_time"] = dissipating_time

    trace = TracedPath(dot.get_center, **trace_kwargs)

    return {"dot": dot, "trace": trace, "all": VGroup(trace, dot)}


# ════════════════════════════════════════════════════════════════════���══════════
# SAVE STATE / RESTORE PATTERN HELPER
# ══════���═════════════════════��══════════════════════════════════════════════════

def animate_save_and_transform(
    scene, mobject, transform_func, restore_after: bool = True,
    transform_run_time: float = 2.0, hold_time: float = 1.0,
):
    """Save state, apply transformation, optionally restore.

    Args:
        scene: Manim Scene.
        mobject: The mobject to transform.
        transform_func: Callable that modifies the mobject (e.g. lambda m: m.scale(2).shift(UP)).
        restore_after: Whether to restore original state after hold_time.
        transform_run_time: Duration of the transformation.
        hold_time: How long to hold the transformed state.
    """
    from manim import Restore
    mobject.save_state()
    target = mobject.copy()
    transform_func(target)
    scene.play(Transform(mobject, target), run_time=transform_run_time)
    if hold_time > 0:
        scene.wait(hold_time)
    if restore_after:
        scene.play(Restore(mobject), run_time=transform_run_time * 0.7)
