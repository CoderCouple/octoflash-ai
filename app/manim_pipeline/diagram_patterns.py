"""
Reusable Manim CE diagram, flowchart, and architectural visualization helpers.

Provides factory functions and animation patterns for:
1. Flowcharts / process diagrams
2. Layer diagrams (neural-net style, pipeline stages)
3. Comparison layouts (side-by-side A vs B)
4. Timeline animations (sequential steps)
5. Data-flow animations (data moving through pipeline)
6. Table / grid animations (confusion matrices, data grids)

All helpers return VGroup objects that can be positioned freely.
Animation functions accept a Scene (or OctoflashScene) as the first arg.

Compatible with Manim Community Edition (tested 0.18+).
"""

from __future__ import annotations

import numpy as np
from manim import (
    # Mobjects
    VGroup, Group, VMobject, Mobject,
    Text, MathTex, Tex,
    RoundedRectangle, Rectangle, Square, Circle, Dot,
    Arrow, Line, DashedLine, CurvedArrow, DoubleArrow,
    SurroundingRectangle, Brace,
    Table, MathTable, MobjectTable, IntegerTable,
    # Animations
    FadeIn, FadeOut, Create, Write, Unwrite, Uncreate,
    GrowArrow, GrowFromCenter,
    Transform, ReplacementTransform,
    Indicate, Flash, Circumscribe, ShowPassingFlash,
    AnimationGroup, LaggedStart, LaggedStartMap, Succession,
    MoveAlongPath,
    # Constants
    UP, DOWN, LEFT, RIGHT, ORIGIN, UL, UR, DL, DR,
    PI, TAU,
    # Rate functions
    linear, smooth, rush_into, rush_from, there_and_back,
    # Utilities
    always_redraw, ValueTracker,
    config,
)

from app.manim_pipeline.styles import (
    ACCENT_BLUE, ACCENT_ORANGE, ACCENT_GREEN, ACCENT_RED,
    ACCENT_PURPLE, ACCENT_YELLOW, ACCENT_CYAN, ACCENT_PINK,
    TEXT_PRIMARY, TEXT_SECONDARY, TEXT_DIM,
    TITLE_SIZE, SUBTITLE_SIZE, BODY_SIZE, LABEL_SIZE,
    BG_COLOR,
)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. FLOWCHARTS & PROCESS DIAGRAMS
# ═══════════════════════════════════════════════════════════════════════════════

def make_flowchart_box(
    label: str,
    width: float = 2.5,
    height: float = 0.8,
    color: str = ACCENT_BLUE,
    font_size: int = LABEL_SIZE,
    corner_radius: float = 0.15,
    fill_opacity: float = 0.2,
) -> VGroup:
    """Create a single flowchart box (rounded rectangle + centered label).

    Returns VGroup with .rect and .label attributes for direct access.
    """
    rect = RoundedRectangle(
        corner_radius=corner_radius,
        width=width,
        height=height,
        fill_color=color,
        fill_opacity=fill_opacity,
        stroke_color=color,
        stroke_width=2,
    )
    txt = Text(label, font_size=font_size, color=TEXT_PRIMARY)
    txt.move_to(rect.get_center())
    group = VGroup(rect, txt)
    group.rect = rect
    group.label = txt
    return group


def make_diamond(
    label: str,
    size: float = 1.2,
    color: str = ACCENT_ORANGE,
    font_size: int = LABEL_SIZE,
) -> VGroup:
    """Create a diamond (decision node) for flowcharts."""
    diamond = Square(side_length=size, color=color, fill_opacity=0.2, stroke_width=2)
    diamond.rotate(PI / 4)
    txt = Text(label, font_size=font_size, color=TEXT_PRIMARY)
    txt.move_to(diamond.get_center())
    group = VGroup(diamond, txt)
    group.diamond = diamond
    group.label = txt
    return group


def connect_boxes(
    start: VGroup,
    end: VGroup,
    direction: str = "down",
    color: str = TEXT_DIM,
    label: str = "",
    label_font_size: int = 16,
    buff: float = 0.1,
    stroke_width: float = 2,
) -> VGroup:
    """Create an arrow connecting two flowchart boxes.

    Args:
        direction: "down", "right", "left", "up" - which edge to connect from/to
        label: optional text label placed alongside the arrow

    Returns VGroup containing arrow and optional label.
    """
    dir_map = {
        "down":  (start.get_bottom, end.get_top),
        "up":    (start.get_top,    end.get_bottom),
        "right": (start.get_right,  end.get_left),
        "left":  (start.get_left,   end.get_right),
    }
    get_start, get_end = dir_map.get(direction, dir_map["down"])

    arrow = Arrow(
        get_start(), get_end(),
        buff=buff,
        color=color,
        stroke_width=stroke_width,
        max_tip_length_to_length_ratio=0.15,
    )

    result = VGroup(arrow)
    if label:
        lbl = Text(label, font_size=label_font_size, color=TEXT_SECONDARY)
        lbl.next_to(arrow, RIGHT if direction in ("down", "up") else UP, buff=0.1)
        result.add(lbl)

    result.arrow = arrow
    return result


def make_flowchart(
    steps: list[str],
    direction: str = "down",
    box_color: str = ACCENT_BLUE,
    arrow_color: str = TEXT_DIM,
    spacing: float = 1.2,
    box_width: float = 2.8,
    box_height: float = 0.7,
) -> VGroup:
    """Build a complete linear flowchart (boxes + arrows).

    Args:
        steps: list of label strings for each box
        direction: "down" or "right" - layout direction

    Returns VGroup with .boxes (VGroup of box VGroups) and .arrows (VGroup of arrows).

    Example:
        flow = make_flowchart(["Input", "Process", "Validate", "Output"])
        flow.scale(0.8).move_to(ORIGIN)
        self.play(LaggedStartMap(FadeIn, flow.boxes, lag_ratio=0.3))
        self.play(LaggedStartMap(GrowArrow, flow.arrows, lag_ratio=0.3))
    """
    arrange_dir = DOWN if direction == "down" else RIGHT
    connect_dir = direction

    boxes = VGroup()
    for step_label in steps:
        box = make_flowchart_box(step_label, width=box_width, height=box_height, color=box_color)
        boxes.add(box)
    boxes.arrange(arrange_dir, buff=spacing)

    arrows = VGroup()
    for i in range(len(boxes) - 1):
        conn = connect_boxes(boxes[i], boxes[i + 1], direction=connect_dir, color=arrow_color)
        arrows.add(conn.arrow)

    chart = VGroup(boxes, arrows)
    chart.boxes = boxes
    chart.arrows = arrows
    return chart


def animate_flowchart_build(scene, flowchart: VGroup, run_time_per_step: float = 0.5):
    """Animate a flowchart appearing step by step (box then arrow).

    Example:
        flow = make_flowchart(["Start", "Process", "End"])
        flow.move_to(ORIGIN)
        animate_flowchart_build(self, flow)
    """
    for i, box in enumerate(flowchart.boxes):
        scene.play(FadeIn(box, shift=DOWN * 0.2), run_time=run_time_per_step)
        if i < len(flowchart.arrows):
            scene.play(GrowArrow(flowchart.arrows[i]), run_time=run_time_per_step * 0.6)


def animate_flow_pulse(
    scene,
    flowchart: VGroup,
    pulse_color: str = ACCENT_CYAN,
    run_time: float = 0.4,
):
    """Send a visual pulse through each box in a flowchart sequentially.

    Each box briefly highlights then returns to normal.

    Example:
        animate_flow_pulse(self, flow, pulse_color=ACCENT_GREEN)
    """
    for box in flowchart.boxes:
        rect = box[0]  # the RoundedRectangle
        scene.play(
            rect.animate.set_fill(pulse_color, opacity=0.5),
            run_time=run_time,
        )
        scene.play(
            rect.animate.set_fill(rect.get_fill_color(), opacity=0.2),
            run_time=run_time * 0.5,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 2. LAYER DIAGRAMS (Neural-net, pipeline stages, architecture)
# ═══════════════════════════════════════════════════════════════════════════════

def make_layer_block(
    label: str,
    width: float = 3.0,
    height: float = 0.7,
    color: str = ACCENT_BLUE,
    sublabel: str = "",
    font_size: int = LABEL_SIZE,
) -> VGroup:
    """Create a single pipeline/layer block with optional sublabel.

    Returns VGroup with .rect, .label, and optionally .sublabel.
    """
    rect = RoundedRectangle(
        corner_radius=0.1,
        width=width,
        height=height,
        fill_color=color,
        fill_opacity=0.25,
        stroke_color=color,
        stroke_width=2,
    )
    txt = Text(label, font_size=font_size, color=TEXT_PRIMARY, weight="BOLD")
    txt.move_to(rect.get_center())

    group = VGroup(rect, txt)
    group.rect = rect
    group.label = txt

    if sublabel:
        sub = Text(sublabel, font_size=font_size - 4, color=TEXT_SECONDARY)
        sub.next_to(rect, DOWN, buff=0.05)
        group.add(sub)
        group.sublabel = sub

    return group


def make_layer_stack(
    layers: list[dict],
    spacing: float = 0.8,
    direction: str = "vertical",
    layer_width: float = 3.5,
    layer_height: float = 0.7,
    connect: bool = True,
    arrow_color: str = TEXT_DIM,
) -> VGroup:
    """Build a stack of layer blocks connected by arrows.

    Args:
        layers: list of dicts with keys: "label", "color", optional "sublabel"
            Example: [
                {"label": "Input Layer", "color": ACCENT_GREEN},
                {"label": "Conv2D (64)", "color": ACCENT_BLUE, "sublabel": "3x3 kernel"},
                {"label": "Dense (10)", "color": ACCENT_ORANGE},
            ]
        direction: "vertical" (top-to-bottom) or "horizontal" (left-to-right)

    Returns VGroup with .layers and .arrows.

    Example:
        stack = make_layer_stack([
            {"label": "Input", "color": ACCENT_GREEN},
            {"label": "Hidden 1", "color": ACCENT_BLUE},
            {"label": "Hidden 2", "color": ACCENT_BLUE},
            {"label": "Output", "color": ACCENT_RED},
        ])
        stack.scale(0.8).move_to(ORIGIN)
        self.play(LaggedStartMap(FadeIn, stack.layers, lag_ratio=0.2))
        self.play(LaggedStartMap(GrowArrow, stack.arrows, lag_ratio=0.2))
    """
    arrange_dir = DOWN if direction == "vertical" else RIGHT

    layer_vg = VGroup()
    for layer_def in layers:
        block = make_layer_block(
            label=layer_def["label"],
            color=layer_def.get("color", ACCENT_BLUE),
            sublabel=layer_def.get("sublabel", ""),
            width=layer_width,
            height=layer_height,
        )
        layer_vg.add(block)
    layer_vg.arrange(arrange_dir, buff=spacing)

    arrows_vg = VGroup()
    if connect:
        for i in range(len(layer_vg) - 1):
            if direction == "vertical":
                arrow = Arrow(
                    layer_vg[i].get_bottom(), layer_vg[i + 1].get_top(),
                    buff=0.08, color=arrow_color, stroke_width=2,
                    max_tip_length_to_length_ratio=0.2,
                )
            else:
                arrow = Arrow(
                    layer_vg[i].get_right(), layer_vg[i + 1].get_left(),
                    buff=0.08, color=arrow_color, stroke_width=2,
                    max_tip_length_to_length_ratio=0.2,
                )
            arrows_vg.add(arrow)

    result = VGroup(layer_vg, arrows_vg)
    result.layers = layer_vg
    result.arrows = arrows_vg
    return result


def make_parallel_layers(
    left_layers: list[dict],
    right_layers: list[dict],
    left_title: str = "",
    right_title: str = "",
    merge_label: str = "",
    gap: float = 3.0,
    layer_width: float = 2.5,
) -> VGroup:
    """Create two parallel layer stacks with optional merge at bottom.

    Useful for encoder-decoder, two-branch architectures, etc.

    Returns VGroup with .left_stack, .right_stack, and optionally .merge_box.
    """
    left = make_layer_stack(left_layers, layer_width=layer_width)
    right = make_layer_stack(right_layers, layer_width=layer_width)

    left.move_to(LEFT * gap / 2)
    right.move_to(RIGHT * gap / 2)

    group = VGroup(left, right)
    group.left_stack = left
    group.right_stack = right

    if left_title:
        lt = Text(left_title, font_size=LABEL_SIZE, color=ACCENT_CYAN, weight="BOLD")
        lt.next_to(left, UP, buff=0.3)
        group.add(lt)

    if right_title:
        rt = Text(right_title, font_size=LABEL_SIZE, color=ACCENT_CYAN, weight="BOLD")
        rt.next_to(right, UP, buff=0.3)
        group.add(rt)

    if merge_label:
        merge_box = make_layer_block(merge_label, color=ACCENT_PURPLE, width=layer_width * 2 + gap - 1)
        merge_y = min(left.get_bottom()[1], right.get_bottom()[1]) - 0.8
        merge_box.move_to([0, merge_y, 0])
        group.add(merge_box)
        group.merge_box = merge_box

        # Arrows from each stack to merge
        arr_l = Arrow(left.get_bottom(), merge_box.get_top() + LEFT * 1, buff=0.1,
                       color=TEXT_DIM, stroke_width=2)
        arr_r = Arrow(right.get_bottom(), merge_box.get_top() + RIGHT * 1, buff=0.1,
                       color=TEXT_DIM, stroke_width=2)
        group.add(arr_l, arr_r)

    return group


def animate_data_through_layers(
    scene,
    layer_stack: VGroup,
    data_color: str = ACCENT_CYAN,
    run_time_per_layer: float = 0.4,
):
    """Animate a 'data dot' flowing through each layer of a stack.

    The dot grows at the top layer, moves through each arrow, and pulses each layer.

    Example:
        stack = make_layer_stack([...])
        animate_data_through_layers(self, stack)
    """
    if not hasattr(layer_stack, 'layers') or len(layer_stack.layers) == 0:
        return

    dot = Dot(
        layer_stack.layers[0].get_center(),
        color=data_color,
        radius=0.12,
    )
    scene.play(GrowFromCenter(dot), run_time=0.3)

    for i, layer in enumerate(layer_stack.layers):
        # Highlight current layer
        rect = layer[0]
        scene.play(
            dot.animate.move_to(layer.get_center()),
            rect.animate.set_fill(data_color, opacity=0.45),
            run_time=run_time_per_layer,
        )
        scene.play(
            rect.animate.set_fill(rect.get_fill_color(), opacity=0.25),
            run_time=run_time_per_layer * 0.4,
        )

    scene.play(FadeOut(dot), run_time=0.2)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. COMPARISON LAYOUTS (Side-by-side, A vs B)
# ═══════════════════════════════════════════════════════════════════════════════

def make_comparison_layout(
    left_title: str,
    right_title: str,
    left_items: list[str],
    right_items: list[str],
    left_color: str = ACCENT_BLUE,
    right_color: str = ACCENT_ORANGE,
    header_font_size: int = BODY_SIZE,
    item_font_size: int = LABEL_SIZE,
    column_width: float = 5.0,
    item_spacing: float = 0.55,
) -> VGroup:
    """Create a two-column comparison layout with headers and bullet items.

    Returns VGroup with .left_col, .right_col, .divider, .left_header, .right_header.

    Example:
        comp = make_comparison_layout(
            "Python", "JavaScript",
            ["Dynamic typing", "Indentation", "GIL"],
            ["Prototype-based", "Curly braces", "Event loop"],
        )
        comp.scale(0.9).move_to(ORIGIN)
        self.play(FadeIn(comp))
    """
    half_gap = 0.3

    # Headers
    lh = Text(left_title, font_size=header_font_size, color=left_color, weight="BOLD")
    rh = Text(right_title, font_size=header_font_size, color=right_color, weight="BOLD")
    lh.move_to(LEFT * (column_width / 2 + half_gap) + UP * 2)
    rh.move_to(RIGHT * (column_width / 2 + half_gap) + UP * 2)

    # Underlines
    l_line = Line(
        lh.get_left() + DOWN * 0.2, lh.get_right() + DOWN * 0.2,
        color=left_color, stroke_width=2,
    )
    r_line = Line(
        rh.get_left() + DOWN * 0.2, rh.get_right() + DOWN * 0.2,
        color=right_color, stroke_width=2,
    )

    # Divider
    divider = DashedLine(
        UP * 2.3, DOWN * 2.3,
        color=TEXT_DIM, stroke_width=1, dash_length=0.1,
    )

    # Items
    left_items_vg = VGroup()
    for i, item in enumerate(left_items):
        bullet = Text(f"  {item}", font_size=item_font_size, color=TEXT_PRIMARY)
        dot = Dot(radius=0.04, color=left_color)
        row = VGroup(dot, bullet).arrange(RIGHT, buff=0.15)
        row.move_to(
            LEFT * (column_width / 2 + half_gap)
            + UP * (1.2 - i * item_spacing)
        )
        row.align_to(lh, LEFT)
        left_items_vg.add(row)

    right_items_vg = VGroup()
    for i, item in enumerate(right_items):
        bullet = Text(f"  {item}", font_size=item_font_size, color=TEXT_PRIMARY)
        dot = Dot(radius=0.04, color=right_color)
        row = VGroup(dot, bullet).arrange(RIGHT, buff=0.15)
        row.move_to(
            RIGHT * (column_width / 2 + half_gap)
            + UP * (1.2 - i * item_spacing)
        )
        row.align_to(rh, LEFT)
        right_items_vg.add(row)

    result = VGroup(lh, rh, l_line, r_line, divider, left_items_vg, right_items_vg)
    result.left_header = lh
    result.right_header = rh
    result.left_col = left_items_vg
    result.right_col = right_items_vg
    result.divider = divider
    return result


def make_before_after(
    before_content: Mobject,
    after_content: Mobject,
    before_label: str = "Before",
    after_label: str = "After",
    before_color: str = ACCENT_RED,
    after_color: str = ACCENT_GREEN,
    gap: float = 5.5,
) -> VGroup:
    """Create a before/after side-by-side comparison with labeled panels.

    before_content and after_content can be any Mobject (code blocks, diagrams, etc).

    Returns VGroup with .before_panel, .after_panel, .arrow.

    Example:
        before = make_code_block("x = x + 1")
        after = make_code_block("x += 1")
        ba = make_before_after(before, after)
        self.play(FadeIn(ba))
    """
    # Before panel
    before_frame = SurroundingRectangle(
        before_content, color=before_color, buff=0.2, stroke_width=2
    )
    before_title = Text(before_label, font_size=LABEL_SIZE, color=before_color, weight="BOLD")

    before_panel = VGroup(before_content, before_frame)
    before_panel.move_to(LEFT * gap / 2)
    before_title.next_to(before_panel, UP, buff=0.2)
    before_panel.add(before_title)

    # After panel
    after_frame = SurroundingRectangle(
        after_content, color=after_color, buff=0.2, stroke_width=2
    )
    after_title = Text(after_label, font_size=LABEL_SIZE, color=after_color, weight="BOLD")

    after_panel = VGroup(after_content, after_frame)
    after_panel.move_to(RIGHT * gap / 2)
    after_title.next_to(after_panel, UP, buff=0.2)
    after_panel.add(after_title)

    # Arrow between
    arrow = Arrow(
        before_panel.get_right(), after_panel.get_left(),
        buff=0.2, color=TEXT_DIM, stroke_width=2,
    )

    result = VGroup(before_panel, arrow, after_panel)
    result.before_panel = before_panel
    result.after_panel = after_panel
    result.arrow = arrow
    return result


def animate_comparison_reveal(scene, comparison: VGroup, run_time: float = 0.5):
    """Animate a comparison layout: headers first, then items staggered left-right.

    Example:
        comp = make_comparison_layout("A", "B", [...], [...])
        animate_comparison_reveal(self, comp)
    """
    scene.play(
        FadeIn(comparison.left_header, shift=DOWN * 0.2),
        FadeIn(comparison.right_header, shift=DOWN * 0.2),
        Create(comparison.divider),
        run_time=run_time,
    )

    max_items = max(len(comparison.left_col), len(comparison.right_col))
    for i in range(max_items):
        anims = []
        if i < len(comparison.left_col):
            anims.append(FadeIn(comparison.left_col[i], shift=RIGHT * 0.2))
        if i < len(comparison.right_col):
            anims.append(FadeIn(comparison.right_col[i], shift=LEFT * 0.2))
        scene.play(*anims, run_time=run_time * 0.7)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. TIMELINE ANIMATIONS
# ═══════════════════════════════════════════════════════════════════════════════

def make_timeline(
    events: list[dict],
    total_width: float = 11.0,
    node_radius: float = 0.12,
    line_color: str = TEXT_DIM,
    default_color: str = ACCENT_BLUE,
    label_font_size: int = 16,
    sublabel_font_size: int = 14,
    label_direction: str = "alternate",
) -> VGroup:
    """Create a horizontal timeline with labeled event nodes.

    Args:
        events: list of dicts with "label" (required), "color" (optional), "sublabel" (optional)
            Example: [
                {"label": "Step 1", "sublabel": "Init", "color": ACCENT_GREEN},
                {"label": "Step 2", "sublabel": "Train"},
                {"label": "Step 3", "sublabel": "Evaluate"},
            ]
        label_direction: "up", "down", or "alternate" (alternates up/down)

    Returns VGroup with .line, .nodes, .labels, .sublabels.

    Example:
        tl = make_timeline([
            {"label": "2020", "sublabel": "GPT-3"},
            {"label": "2022", "sublabel": "ChatGPT"},
            {"label": "2024", "sublabel": "GPT-4o"},
        ])
        tl.move_to(ORIGIN)
        self.play(Create(tl.line))
        self.play(LaggedStartMap(GrowFromCenter, tl.nodes, lag_ratio=0.2))
        self.play(LaggedStartMap(FadeIn, tl.labels, lag_ratio=0.15))
    """
    n = len(events)
    half_w = total_width / 2

    # Main horizontal line
    main_line = Line(
        LEFT * half_w, RIGHT * half_w,
        color=line_color, stroke_width=2,
    )

    nodes = VGroup()
    labels_vg = VGroup()
    sublabels_vg = VGroup()

    for i, evt in enumerate(events):
        # Position evenly along the line
        t = i / (n - 1) if n > 1 else 0.5
        x_pos = -half_w + t * total_width
        color = evt.get("color", default_color)

        # Node dot
        node = Dot(
            point=[x_pos, 0, 0],
            radius=node_radius,
            color=color,
        )
        nodes.add(node)

        # Determine label direction
        if label_direction == "alternate":
            up = i % 2 == 0
        elif label_direction == "up":
            up = True
        else:
            up = False

        direction = UP if up else DOWN

        # Main label
        lbl = Text(evt["label"], font_size=label_font_size, color=TEXT_PRIMARY, weight="BOLD")
        lbl.next_to(node, direction, buff=0.25)
        labels_vg.add(lbl)

        # Sublabel
        if evt.get("sublabel"):
            sub = Text(evt["sublabel"], font_size=sublabel_font_size, color=TEXT_SECONDARY)
            sub.next_to(lbl, direction, buff=0.08)
            sublabels_vg.add(sub)

    result = VGroup(main_line, nodes, labels_vg, sublabels_vg)
    result.line = main_line
    result.nodes = nodes
    result.labels = labels_vg
    result.sublabels = sublabels_vg
    return result


def animate_timeline_progress(
    scene,
    timeline: VGroup,
    highlight_color: str = ACCENT_GREEN,
    run_time_per_step: float = 0.6,
):
    """Animate progress through a timeline, highlighting each node in sequence.

    Example:
        tl = make_timeline([...])
        # First build the timeline
        scene.play(Create(tl.line), LaggedStartMap(GrowFromCenter, tl.nodes))
        scene.play(LaggedStartMap(FadeIn, tl.labels))
        # Then animate progress
        animate_timeline_progress(scene, tl)
    """
    for i, node in enumerate(timeline.nodes):
        anims = [
            node.animate.set_color(highlight_color).scale(1.5),
        ]
        if i < len(timeline.labels):
            anims.append(timeline.labels[i].animate.set_color(highlight_color))
        scene.play(*anims, run_time=run_time_per_step)

    # Optionally draw a completion check
    scene.wait(0.3)


def make_vertical_timeline(
    events: list[dict],
    total_height: float = 5.0,
    node_radius: float = 0.1,
    line_color: str = TEXT_DIM,
    default_color: str = ACCENT_BLUE,
    label_font_size: int = LABEL_SIZE,
) -> VGroup:
    """Create a vertical timeline (top to bottom) with labels on the right.

    Args:
        events: list of dicts with "label", optional "color"

    Returns VGroup with .line, .nodes, .labels.
    """
    n = len(events)
    half_h = total_height / 2

    main_line = Line(
        UP * half_h, DOWN * half_h,
        color=line_color, stroke_width=2,
    )

    nodes = VGroup()
    labels_vg = VGroup()

    for i, evt in enumerate(events):
        t = i / (n - 1) if n > 1 else 0.5
        y_pos = half_h - t * total_height
        color = evt.get("color", default_color)

        node = Dot(point=[0, y_pos, 0], radius=node_radius, color=color)
        nodes.add(node)

        lbl = Text(evt["label"], font_size=label_font_size, color=TEXT_PRIMARY)
        lbl.next_to(node, RIGHT, buff=0.3)
        labels_vg.add(lbl)

    result = VGroup(main_line, nodes, labels_vg)
    result.line = main_line
    result.nodes = nodes
    result.labels = labels_vg
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 5. DATA FLOW ANIMATIONS
# ═══════════════════════════════════════════════════════════════════════════════

def make_pipeline(
    stages: list[dict],
    direction: str = "right",
    box_width: float = 2.2,
    box_height: float = 0.8,
    spacing: float = 0.8,
    arrow_color: str = TEXT_DIM,
) -> VGroup:
    """Build a horizontal or vertical data pipeline with labeled stages.

    Args:
        stages: list of dicts with "label", "color", optional "sublabel"
            Example: [
                {"label": "Raw Data", "color": ACCENT_GREEN},
                {"label": "Clean", "color": ACCENT_BLUE},
                {"label": "Transform", "color": ACCENT_ORANGE},
                {"label": "Model", "color": ACCENT_RED},
            ]

    Returns VGroup with .stages, .arrows.
    """
    arrange_dir = RIGHT if direction == "right" else DOWN

    stages_vg = VGroup()
    for stage_def in stages:
        box = make_flowchart_box(
            stage_def["label"],
            width=box_width,
            height=box_height,
            color=stage_def.get("color", ACCENT_BLUE),
        )
        if stage_def.get("sublabel"):
            sub = Text(stage_def["sublabel"], font_size=14, color=TEXT_SECONDARY)
            sub.next_to(box, DOWN, buff=0.1)
            box.add(sub)
        stages_vg.add(box)

    stages_vg.arrange(arrange_dir, buff=spacing)

    arrows_vg = VGroup()
    for i in range(len(stages_vg) - 1):
        if direction == "right":
            arrow = Arrow(
                stages_vg[i].get_right(), stages_vg[i + 1].get_left(),
                buff=0.08, color=arrow_color, stroke_width=2,
                max_tip_length_to_length_ratio=0.2,
            )
        else:
            arrow = Arrow(
                stages_vg[i].get_bottom(), stages_vg[i + 1].get_top(),
                buff=0.08, color=arrow_color, stroke_width=2,
                max_tip_length_to_length_ratio=0.2,
            )
        arrows_vg.add(arrow)

    result = VGroup(stages_vg, arrows_vg)
    result.stages = stages_vg
    result.arrows = arrows_vg
    return result


def animate_data_packet(
    scene,
    pipeline: VGroup,
    packet_label: str = "data",
    packet_color: str = ACCENT_CYAN,
    run_time_per_hop: float = 0.6,
    transform_labels: list[str] | None = None,
):
    """Animate a labeled 'data packet' moving through a pipeline, transforming at each stage.

    Args:
        transform_labels: if provided, the packet label changes at each stage.
            Length should match number of stages.

    Example:
        pipe = make_pipeline([
            {"label": "CSV", "color": ACCENT_GREEN},
            {"label": "Clean", "color": ACCENT_BLUE},
            {"label": "Features", "color": ACCENT_ORANGE},
            {"label": "Model", "color": ACCENT_RED},
        ])
        pipe.move_to(ORIGIN)
        self.play(FadeIn(pipe))
        animate_data_packet(
            self, pipe,
            packet_label="raw.csv",
            transform_labels=["raw.csv", "clean_df", "features", "predictions"],
        )
    """
    if len(pipeline.stages) == 0:
        return

    # Create initial packet
    first_stage = pipeline.stages[0]
    current_label = packet_label
    if transform_labels and len(transform_labels) > 0:
        current_label = transform_labels[0]

    packet_rect = RoundedRectangle(
        corner_radius=0.08, width=1.2, height=0.4,
        fill_color=packet_color, fill_opacity=0.4,
        stroke_color=packet_color, stroke_width=2,
    )
    packet_text = Text(current_label, font_size=14, color=TEXT_PRIMARY)
    packet_text.move_to(packet_rect.get_center())
    packet = VGroup(packet_rect, packet_text)
    packet.next_to(first_stage, UP, buff=0.3)

    scene.play(FadeIn(packet, shift=DOWN * 0.2), run_time=0.3)

    for i, stage in enumerate(pipeline.stages):
        # Move packet to stage
        scene.play(packet.animate.next_to(stage, UP, buff=0.3), run_time=run_time_per_hop)

        # Flash the stage
        stage_rect = stage[0]
        scene.play(
            stage_rect.animate.set_fill(packet_color, opacity=0.4),
            run_time=0.2,
        )

        # Transform label if specified
        if transform_labels and i < len(transform_labels):
            new_label = transform_labels[i]
            if new_label != current_label:
                new_text = Text(new_label, font_size=14, color=TEXT_PRIMARY)
                new_text.move_to(packet_text.get_center())
                scene.play(
                    Transform(packet_text, new_text),
                    run_time=0.3,
                )
                current_label = new_label

        # Reset stage color
        scene.play(
            stage_rect.animate.set_fill(stage_rect.get_fill_color(), opacity=0.2),
            run_time=0.15,
        )

    scene.play(FadeOut(packet, shift=UP * 0.3), run_time=0.3)


def make_branching_pipeline(
    input_stage: dict,
    branches: list[list[dict]],
    output_stage: dict | None = None,
    branch_gap: float = 2.0,
    box_width: float = 2.0,
) -> VGroup:
    """Create a pipeline that splits into parallel branches and optionally merges.

    Args:
        input_stage: dict with "label", "color" for the input node
        branches: list of lists of stage dicts — each inner list is one branch
        output_stage: optional merge node at the end

    Returns VGroup with .input_box, .branches (list of VGroups), .output_box.

    Example:
        pipe = make_branching_pipeline(
            {"label": "Input", "color": ACCENT_GREEN},
            [
                [{"label": "Path A1", "color": ACCENT_BLUE}, {"label": "Path A2", "color": ACCENT_BLUE}],
                [{"label": "Path B1", "color": ACCENT_ORANGE}, {"label": "Path B2", "color": ACCENT_ORANGE}],
            ],
            {"label": "Merge", "color": ACCENT_PURPLE},
        )
    """
    # Input box
    input_box = make_flowchart_box(
        input_stage["label"], width=box_width,
        color=input_stage.get("color", ACCENT_GREEN),
    )
    input_box.move_to(UP * 2.5)

    num_branches = len(branches)
    total_width = (num_branches - 1) * branch_gap

    branch_groups = []
    branch_arrows_from_input = VGroup()

    for b_idx, branch_stages in enumerate(branches):
        x_offset = -total_width / 2 + b_idx * branch_gap
        branch_vg = VGroup()
        for s_idx, stage_def in enumerate(branch_stages):
            box = make_flowchart_box(
                stage_def["label"], width=box_width, height=0.6,
                color=stage_def.get("color", ACCENT_BLUE),
            )
            box.move_to([x_offset, 1.2 - s_idx * 1.0, 0])
            branch_vg.add(box)

        # Arrows within branch
        for i in range(len(branch_vg) - 1):
            arr = Arrow(
                branch_vg[i].get_bottom(), branch_vg[i + 1].get_top(),
                buff=0.08, color=TEXT_DIM, stroke_width=2,
            )
            branch_vg.add(arr)

        # Arrow from input to first in branch
        arr_in = Arrow(
            input_box.get_bottom(), branch_vg[0].get_top(),
            buff=0.08, color=TEXT_DIM, stroke_width=2,
        )
        branch_arrows_from_input.add(arr_in)
        branch_groups.append(branch_vg)

    all_branches = VGroup(*branch_groups)

    result = VGroup(input_box, branch_arrows_from_input, all_branches)
    result.input_box = input_box
    result.branches = branch_groups

    if output_stage:
        output_box = make_flowchart_box(
            output_stage["label"], width=box_width,
            color=output_stage.get("color", ACCENT_PURPLE),
        )
        # Position below lowest branch element
        lowest_y = min(bg[-1].get_bottom()[1] for bg in branch_groups if len(bg) > 0)
        output_box.move_to([0, lowest_y - 1.0, 0])
        result.add(output_box)
        result.output_box = output_box

        # Arrows from each branch end to output
        for bg in branch_groups:
            # Find the last box (not arrow) in the branch
            last_box = bg[0]
            for mob in bg:
                if isinstance(mob, VGroup) and hasattr(mob, '__len__') and len(mob) >= 2:
                    last_box = mob
            arr_out = Arrow(
                last_box.get_bottom(), output_box.get_top(),
                buff=0.08, color=TEXT_DIM, stroke_width=2,
            )
            result.add(arr_out)

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 6. TABLE / GRID ANIMATIONS
# ═══════════════════════════════════════════════════════════════════════════════

def make_styled_table(
    data: list[list[str]],
    row_labels: list[str] | None = None,
    col_labels: list[str] | None = None,
    header_color: str = ACCENT_BLUE,
    cell_color: str = TEXT_PRIMARY,
    include_outer_lines: bool = True,
    font_size: int = LABEL_SIZE,
    h_buff: float = 1.0,
    v_buff: float = 0.6,
) -> Table:
    """Create a styled Manim Table with consistent theming.

    Args:
        data: 2D list of string values
        row_labels: optional list of row label strings
        col_labels: optional list of column header strings

    Returns a Manim Table object.

    Example:
        tbl = make_styled_table(
            [["95%", "5%"], ["10%", "90%"]],
            row_labels=["Actual Cat", "Actual Dog"],
            col_labels=["Pred Cat", "Pred Dog"],
        )
        tbl.scale(0.7).move_to(ORIGIN)
        self.play(tbl.create())
    """
    rl = [Text(r, font_size=font_size, color=header_color) for r in row_labels] if row_labels else None
    cl = [Text(c, font_size=font_size, color=header_color) for c in col_labels] if col_labels else None

    tbl = Table(
        data,
        row_labels=rl,
        col_labels=cl,
        include_outer_lines=include_outer_lines,
        h_buff=h_buff,
        v_buff=v_buff,
        line_config={"stroke_width": 1, "color": TEXT_DIM},
        element_to_mobject_config={"font_size": font_size, "color": cell_color},
    )

    return tbl


def make_confusion_matrix(
    values: list[list[int | float | str]],
    class_labels: list[str],
    title: str = "Confusion Matrix",
    high_color: str = ACCENT_GREEN,
    low_color: str = ACCENT_RED,
    font_size: int = LABEL_SIZE,
) -> VGroup:
    """Create a color-coded confusion matrix.

    Diagonal elements (correct predictions) are colored green,
    off-diagonal (errors) are colored red with opacity proportional to value.

    Args:
        values: 2D list of numeric values (will be displayed as strings)
        class_labels: labels for both rows and columns

    Returns VGroup with .table, .title.

    Example:
        cm = make_confusion_matrix(
            [[45, 5], [8, 42]],
            ["Cat", "Dog"],
        )
        cm.scale(0.8).move_to(ORIGIN)
        self.play(FadeIn(cm))
    """
    str_data = [[str(v) for v in row] for row in values]

    tbl = Table(
        str_data,
        row_labels=[Text(lbl, font_size=font_size, color=ACCENT_CYAN) for lbl in class_labels],
        col_labels=[Text(lbl, font_size=font_size, color=ACCENT_CYAN) for lbl in class_labels],
        include_outer_lines=True,
        h_buff=1.0,
        v_buff=0.7,
        line_config={"stroke_width": 1, "color": TEXT_DIM},
        element_to_mobject_config={"font_size": font_size + 2, "color": TEXT_PRIMARY},
    )

    # Color code cells: diagonal = green, off-diagonal = red
    # Table positions are 1-indexed; row_labels shift columns by 1, col_labels shift rows by 1
    n = len(class_labels)
    for r in range(n):
        for c in range(n):
            # get_cell is 1-indexed, +1 for header offset
            cell_pos = (r + 2, c + 2)
            if r == c:
                tbl.add_highlighted_cell(cell_pos, color=high_color)
            else:
                # Only highlight if value is non-trivial
                val = values[r][c]
                if isinstance(val, (int, float)) and val > 0:
                    tbl.add_highlighted_cell(cell_pos, color=low_color)

    title_text = Text(title, font_size=BODY_SIZE, color=TEXT_PRIMARY, weight="BOLD")
    title_text.next_to(tbl, UP, buff=0.4)

    # Axis labels
    pred_label = Text("Predicted", font_size=font_size - 2, color=TEXT_SECONDARY)
    pred_label.next_to(tbl, UP, buff=0.05)
    actual_label = Text("Actual", font_size=font_size - 2, color=TEXT_SECONDARY)
    actual_label.rotate(PI / 2)
    actual_label.next_to(tbl, LEFT, buff=0.15)

    result = VGroup(tbl, title_text, pred_label, actual_label)
    result.table = tbl
    result.title = title_text
    return result


def make_data_grid(
    rows: int,
    cols: int,
    cell_size: float = 0.6,
    values: list[list[str]] | None = None,
    colors: list[list[str]] | None = None,
    default_color: str = ACCENT_BLUE,
    font_size: int = 16,
    gap: float = 0.05,
) -> VGroup:
    """Create a colored grid of cells, optionally with values.

    Useful for visualizing matrices, heatmaps, feature grids, etc.

    Args:
        values: optional 2D list of display values
        colors: optional 2D list of hex color strings per cell

    Returns VGroup with .cells (2D list accessible as grid.cells[r][c]).

    Example:
        grid = make_data_grid(3, 3, values=[["1","2","3"],["4","5","6"],["7","8","9"]])
        grid.move_to(ORIGIN)
        self.play(LaggedStartMap(FadeIn, grid, lag_ratio=0.05))
    """
    all_cells = VGroup()
    cells_2d = []

    for r in range(rows):
        row_cells = []
        for c in range(cols):
            color = default_color
            if colors and r < len(colors) and c < len(colors[r]):
                color = colors[r][c]

            rect = Square(
                side_length=cell_size,
                fill_color=color,
                fill_opacity=0.3,
                stroke_color=color,
                stroke_width=1.5,
            )
            cell_group = VGroup(rect)

            if values and r < len(values) and c < len(values[r]):
                txt = Text(str(values[r][c]), font_size=font_size, color=TEXT_PRIMARY)
                txt.move_to(rect.get_center())
                cell_group.add(txt)

            cell_group.move_to([
                c * (cell_size + gap) - (cols - 1) * (cell_size + gap) / 2,
                -r * (cell_size + gap) + (rows - 1) * (cell_size + gap) / 2,
                0,
            ])

            all_cells.add(cell_group)
            row_cells.append(cell_group)
        cells_2d.append(row_cells)

    all_cells.cells = cells_2d
    return all_cells


def animate_grid_highlight_row(
    scene,
    grid: VGroup,
    row: int,
    color: str = ACCENT_YELLOW,
    run_time: float = 0.4,
):
    """Highlight all cells in a specific row of a data grid.

    Example:
        animate_grid_highlight_row(self, grid, 1, color=ACCENT_GREEN)
    """
    if not hasattr(grid, 'cells') or row >= len(grid.cells):
        return
    anims = []
    for cell in grid.cells[row]:
        rect = cell[0]
        anims.append(rect.animate.set_fill(color, opacity=0.5))
    scene.play(*anims, run_time=run_time)


def animate_grid_highlight_col(
    scene,
    grid: VGroup,
    col: int,
    color: str = ACCENT_YELLOW,
    run_time: float = 0.4,
):
    """Highlight all cells in a specific column of a data grid."""
    if not hasattr(grid, 'cells'):
        return
    anims = []
    for row_cells in grid.cells:
        if col < len(row_cells):
            rect = row_cells[col][0]
            anims.append(rect.animate.set_fill(color, opacity=0.5))
    scene.play(*anims, run_time=run_time)


def animate_grid_highlight_cell(
    scene,
    grid: VGroup,
    row: int,
    col: int,
    color: str = ACCENT_YELLOW,
    run_time: float = 0.3,
):
    """Highlight a single cell in a data grid with a surrounding glow."""
    if not hasattr(grid, 'cells') or row >= len(grid.cells) or col >= len(grid.cells[row]):
        return
    cell = grid.cells[row][col]
    scene.play(
        cell[0].animate.set_fill(color, opacity=0.6).set_stroke(color, width=3),
        run_time=run_time,
    )


def animate_table_row_by_row(scene, table: Table, run_time_per_row: float = 0.5):
    """Animate a Manim Table appearing row by row.

    Example:
        tbl = make_styled_table([["a","b"],["c","d"]], col_labels=["X","Y"])
        tbl.move_to(ORIGIN)
        # Show structure first
        scene.play(Create(VGroup(table.get_horizontal_lines(), table.get_vertical_lines())))
        # Then fill row by row
        animate_table_row_by_row(scene, tbl)
    """
    rows = table.get_rows()
    for row in rows:
        scene.play(FadeIn(row, shift=RIGHT * 0.2), run_time=run_time_per_row)


def animate_table_cell_by_cell(scene, table: Table, run_time_per_cell: float = 0.15):
    """Animate a Manim Table appearing cell by cell with a wave effect.

    Example:
        tbl = make_styled_table([...])
        animate_table_cell_by_cell(scene, tbl)
    """
    entries = table.get_entries()
    scene.play(
        LaggedStart(
            *[FadeIn(entry, shift=UP * 0.1) for entry in entries],
            lag_ratio=0.08,
        ),
        run_time=len(entries) * run_time_per_cell,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# HIGHLIGHT UTILITIES (work with any mobject)
# ═══════════════════════════════════════════════════════════════════════════════

def highlight_box(
    target: Mobject,
    color: str = ACCENT_YELLOW,
    buff: float = 0.15,
    stroke_width: float = 2.5,
) -> SurroundingRectangle:
    """Create a SurroundingRectangle highlight around any mobject.

    Example:
        box = highlight_box(some_text, color=ACCENT_RED)
        self.play(Create(box))
    """
    return SurroundingRectangle(
        target, color=color, buff=buff, stroke_width=stroke_width,
        corner_radius=0.1,
    )


def animate_highlight_sequence(
    scene,
    targets: list[Mobject],
    color: str = ACCENT_YELLOW,
    run_time: float = 0.4,
    pause: float = 0.3,
):
    """Sequentially highlight each target with a surrounding rectangle, then remove it.

    Example:
        animate_highlight_sequence(self, [box1, box2, box3], color=ACCENT_GREEN)
    """
    for target in targets:
        rect = highlight_box(target, color=color)
        scene.play(Create(rect), run_time=run_time)
        scene.wait(pause)
        scene.play(FadeOut(rect), run_time=run_time * 0.5)


# ═══════════════════════════════════════════════════════════════════════════════
# COMPLETE SCENE EXAMPLES (for reference in prompts)
# ═══════════════════════════════════════════════════════════════════════════════

FLOWCHART_EXAMPLE = '''
# ── Flowchart Example ──
from manim import *
import numpy as np
from app.manim_pipeline.styles import *
from app.manim_pipeline.diagram_patterns import *

class FlowchartDemo(Scene):
    def construct(self):
        self.camera.background_color = BG_COLOR

        title = Text("ML Training Pipeline", font_size=TITLE_SIZE,
                      color=TEXT_PRIMARY, weight="BOLD")
        title.to_edge(UP, buff=0.3)
        self.play(FadeIn(title))

        # Build flowchart
        flow = make_flowchart(
            ["Load Data", "Preprocess", "Split Train/Test", "Train Model", "Evaluate"],
            direction="down",
            box_color=ACCENT_BLUE,
            box_width=3.2,
            box_height=0.65,
            spacing=0.7,
        )
        flow.scale(0.85).move_to(DOWN * 0.3)

        # Animate step by step
        animate_flowchart_build(self, flow, run_time_per_step=0.4)
        self.wait(0.5)

        # Send pulse through
        animate_flow_pulse(self, flow, pulse_color=ACCENT_GREEN)
        self.wait(1)

        self.play(FadeOut(VGroup(title, flow)))
'''

LAYER_DIAGRAM_EXAMPLE = '''
# ── Neural Network Layer Diagram ──
from manim import *
import numpy as np
from app.manim_pipeline.styles import *
from app.manim_pipeline.diagram_patterns import *

class NeuralNetDemo(Scene):
    def construct(self):
        self.camera.background_color = BG_COLOR

        title = Text("CNN Architecture", font_size=TITLE_SIZE,
                      color=TEXT_PRIMARY, weight="BOLD")
        title.to_edge(UP, buff=0.3)
        self.play(FadeIn(title))

        stack = make_layer_stack([
            {"label": "Input (224x224x3)", "color": ACCENT_GREEN},
            {"label": "Conv2D (64)", "color": ACCENT_BLUE, "sublabel": "3x3 kernel, ReLU"},
            {"label": "MaxPool2D", "color": ACCENT_CYAN},
            {"label": "Conv2D (128)", "color": ACCENT_BLUE, "sublabel": "3x3 kernel, ReLU"},
            {"label": "Flatten", "color": ACCENT_ORANGE},
            {"label": "Dense (10)", "color": ACCENT_RED, "sublabel": "softmax"},
        ], spacing=0.55, layer_width=4.5, layer_height=0.55)
        stack.scale(0.8).move_to(DOWN * 0.2)

        # Build layers with stagger
        self.play(LaggedStartMap(FadeIn, stack.layers, shift=LEFT * 0.3, lag_ratio=0.15), run_time=2)
        self.play(LaggedStartMap(GrowArrow, stack.arrows, lag_ratio=0.15), run_time=1)
        self.wait(0.5)

        # Animate data flowing through
        animate_data_through_layers(self, stack, data_color=ACCENT_CYAN, run_time_per_layer=0.35)
        self.wait(1)

        self.play(FadeOut(VGroup(title, stack)))
'''

COMPARISON_EXAMPLE = '''
# ── Side-by-Side Comparison ──
from manim import *
import numpy as np
from app.manim_pipeline.styles import *
from app.manim_pipeline.diagram_patterns import *

class ComparisonDemo(Scene):
    def construct(self):
        self.camera.background_color = BG_COLOR

        title = Text("RNN vs Transformer", font_size=TITLE_SIZE,
                      color=TEXT_PRIMARY, weight="BOLD")
        title.to_edge(UP, buff=0.3)
        self.play(FadeIn(title))

        comp = make_comparison_layout(
            "RNN", "Transformer",
            [
                "Sequential processing",
                "O(n) parallelism",
                "Vanishing gradients",
                "Good for short sequences",
            ],
            [
                "Parallel processing",
                "O(1) with attention",
                "Stable gradients",
                "Handles long context",
            ],
            left_color=ACCENT_ORANGE,
            right_color=ACCENT_GREEN,
        )
        comp.scale(0.85).move_to(DOWN * 0.3)

        animate_comparison_reveal(self, comp)
        self.wait(1)

        # Highlight specific items
        winner_box = highlight_box(comp.right_col[1], color=ACCENT_GREEN)
        self.play(Create(winner_box))
        self.wait(0.5)

        self.play(FadeOut(VGroup(title, comp, winner_box)))
'''

TIMELINE_EXAMPLE = '''
# ── Timeline Animation ──
from manim import *
import numpy as np
from app.manim_pipeline.styles import *
from app.manim_pipeline.diagram_patterns import *

class TimelineDemo(Scene):
    def construct(self):
        self.camera.background_color = BG_COLOR

        title = Text("History of LLMs", font_size=TITLE_SIZE,
                      color=TEXT_PRIMARY, weight="BOLD")
        title.to_edge(UP, buff=0.3)
        self.play(FadeIn(title))

        tl = make_timeline([
            {"label": "2017", "sublabel": "Transformer", "color": ACCENT_BLUE},
            {"label": "2018", "sublabel": "BERT", "color": ACCENT_GREEN},
            {"label": "2020", "sublabel": "GPT-3", "color": ACCENT_ORANGE},
            {"label": "2022", "sublabel": "ChatGPT", "color": ACCENT_RED},
            {"label": "2023", "sublabel": "GPT-4", "color": ACCENT_PURPLE},
            {"label": "2024", "sublabel": "Claude 3.5", "color": ACCENT_CYAN},
        ], total_width=10.0)
        tl.move_to(DOWN * 0.5)

        # Build timeline
        self.play(Create(tl.line), run_time=1)
        self.play(LaggedStartMap(GrowFromCenter, tl.nodes, lag_ratio=0.15), run_time=1.5)
        self.play(
            LaggedStartMap(FadeIn, tl.labels, shift=DOWN * 0.1, lag_ratio=0.1),
            run_time=1,
        )
        self.play(
            LaggedStartMap(FadeIn, tl.sublabels, shift=DOWN * 0.1, lag_ratio=0.1),
            run_time=1,
        )
        self.wait(0.5)

        # Progress through
        animate_timeline_progress(self, tl, highlight_color=ACCENT_YELLOW)
        self.wait(1)

        self.play(FadeOut(VGroup(title, tl)))
'''

DATA_FLOW_EXAMPLE = '''
# ── Data Pipeline with Packet Animation ──
from manim import *
import numpy as np
from app.manim_pipeline.styles import *
from app.manim_pipeline.diagram_patterns import *

class DataFlowDemo(Scene):
    def construct(self):
        self.camera.background_color = BG_COLOR

        title = Text("ETL Pipeline", font_size=TITLE_SIZE,
                      color=TEXT_PRIMARY, weight="BOLD")
        title.to_edge(UP, buff=0.3)
        self.play(FadeIn(title))

        pipe = make_pipeline([
            {"label": "Source DB", "color": ACCENT_GREEN, "sublabel": "PostgreSQL"},
            {"label": "Extract", "color": ACCENT_BLUE},
            {"label": "Transform", "color": ACCENT_ORANGE, "sublabel": "clean + join"},
            {"label": "Load", "color": ACCENT_RED},
            {"label": "Data Lake", "color": ACCENT_PURPLE, "sublabel": "S3"},
        ], direction="right", box_width=2.0, spacing=0.6)
        pipe.scale(0.85).move_to(DOWN * 0.3)

        # Build pipeline
        self.play(LaggedStartMap(FadeIn, pipe.stages, shift=UP * 0.2, lag_ratio=0.15), run_time=1.5)
        self.play(LaggedStartMap(GrowArrow, pipe.arrows, lag_ratio=0.2), run_time=1)
        self.wait(0.5)

        # Animate data packet flowing through
        animate_data_packet(
            self, pipe,
            packet_label="records",
            packet_color=ACCENT_CYAN,
            transform_labels=["raw_rows", "raw_rows", "clean_df", "clean_df", "parquet"],
        )
        self.wait(1)

        self.play(FadeOut(VGroup(title, pipe)))
'''

TABLE_GRID_EXAMPLE = '''
# ── Confusion Matrix + Grid Animation ──
from manim import *
import numpy as np
from app.manim_pipeline.styles import *
from app.manim_pipeline.diagram_patterns import *

class TableGridDemo(Scene):
    def construct(self):
        self.camera.background_color = BG_COLOR

        # ── Part 1: Confusion Matrix ──
        title = Text("Model Evaluation", font_size=TITLE_SIZE,
                      color=TEXT_PRIMARY, weight="BOLD")
        title.to_edge(UP, buff=0.3)
        self.play(FadeIn(title))

        cm = make_confusion_matrix(
            [[45, 3, 2], [5, 40, 5], [1, 4, 45]],
            ["Cat", "Dog", "Bird"],
            title="Confusion Matrix",
        )
        cm.scale(0.6).move_to(DOWN * 0.3)
        self.play(FadeIn(cm), run_time=1)
        self.wait(2)

        self.play(FadeOut(cm), run_time=0.5)

        # ── Part 2: Data Grid with highlights ──
        cap = Text("Feature Importance Heatmap", font_size=BODY_SIZE,
                     color=TEXT_PRIMARY, weight="BOLD")
        cap.move_to(UP * 1.5)
        self.play(FadeIn(cap))

        grid = make_data_grid(
            4, 5,
            cell_size=0.8,
            values=[
                ["0.9", "0.3", "0.1", "0.7", "0.5"],
                ["0.2", "0.8", "0.6", "0.1", "0.4"],
                ["0.5", "0.1", "0.9", "0.3", "0.8"],
                ["0.7", "0.6", "0.2", "0.8", "0.1"],
            ],
            colors=[
                [ACCENT_GREEN, ACCENT_BLUE, ACCENT_RED, ACCENT_GREEN, ACCENT_ORANGE],
                [ACCENT_RED, ACCENT_GREEN, ACCENT_ORANGE, ACCENT_RED, ACCENT_BLUE],
                [ACCENT_ORANGE, ACCENT_RED, ACCENT_GREEN, ACCENT_BLUE, ACCENT_GREEN],
                [ACCENT_GREEN, ACCENT_ORANGE, ACCENT_RED, ACCENT_GREEN, ACCENT_RED],
            ],
            font_size=16,
        )
        grid.move_to(DOWN * 0.5)

        # Animate grid appearing
        self.play(LaggedStartMap(FadeIn, grid, lag_ratio=0.03), run_time=1.5)
        self.wait(0.5)

        # Highlight best row
        animate_grid_highlight_row(self, grid, 0, color=ACCENT_YELLOW)
        self.wait(0.5)

        # Highlight specific cell
        animate_grid_highlight_cell(self, grid, 2, 2, color=ACCENT_GREEN)
        self.wait(1)

        self.play(FadeOut(VGroup(title, cap, grid)))

        # ── Part 3: Styled Table ──
        tbl = make_styled_table(
            [["Adam", "0.001", "93.2%"],
             ["SGD", "0.01", "91.5%"],
             ["RMSprop", "0.001", "92.8%"]],
            col_labels=["Optimizer", "LR", "Accuracy"],
            header_color=ACCENT_CYAN,
        )
        tbl.scale(0.7).move_to(ORIGIN)

        tbl_title = Text("Hyperparameter Comparison", font_size=BODY_SIZE,
                          color=TEXT_PRIMARY, weight="BOLD")
        tbl_title.next_to(tbl, UP, buff=0.4)

        self.play(FadeIn(tbl_title))
        animate_table_row_by_row(self, tbl)
        self.wait(1)

        # Highlight best result
        best_cell = tbl.get_entries((2, 3))  # row 2, col 3
        hl = highlight_box(best_cell, color=ACCENT_GREEN)
        self.play(Create(hl))
        self.wait(1)

        self.play(FadeOut(VGroup(tbl, tbl_title, hl)))
'''


# ═══════════════════════════════════════════════════════════════════════════════
# 7. TREE DIAGRAMS (Binary trees, Decision trees, Org charts)
# ═══════════════════════════════════════════════════════════════════════════════

def make_tree_node(
    label: str,
    shape: str = "circle",
    size: float = 0.5,
    color: str = ACCENT_BLUE,
    font_size: int = 16,
    fill_opacity: float = 0.3,
) -> VGroup:
    """Create a single tree node (circle or rectangle) with a label.

    Args:
        shape: "circle" or "rect"

    Returns VGroup with .shape and .label attributes.
    """
    if shape == "circle":
        node_shape = Circle(
            radius=size,
            fill_color=color, fill_opacity=fill_opacity,
            stroke_color=color, stroke_width=2,
        )
    else:
        node_shape = RoundedRectangle(
            corner_radius=0.08,
            width=size * 2.5, height=size * 1.4,
            fill_color=color, fill_opacity=fill_opacity,
            stroke_color=color, stroke_width=2,
        )
    txt = Text(label, font_size=font_size, color=TEXT_PRIMARY)
    txt.move_to(node_shape.get_center())
    group = VGroup(node_shape, txt)
    group.shape = node_shape
    group.label = txt
    return group


def make_binary_tree(
    values: list,
    node_color: str = ACCENT_BLUE,
    leaf_color: str = ACCENT_GREEN,
    edge_color: str = TEXT_DIM,
    h_spacing: float = 1.5,
    v_spacing: float = 1.2,
    node_size: float = 0.35,
    font_size: int = 16,
) -> VGroup:
    """Build a binary tree from a list (level-order, None for missing nodes).

    Uses standard binary heap indexing: children of index i are 2i+1 and 2i+2.

    Args:
        values: list like [1, 2, 3, None, 4, 5, None].
                None entries are skipped (no node rendered).

    Returns VGroup with .nodes (dict: index->VGroup), .edges (VGroup of Lines).

    Example:
        tree = make_binary_tree(["A", "B", "C", "D", "E", "F", "G"])
        tree.scale(0.8).move_to(ORIGIN)
        self.play(FadeIn(tree))
    """
    import math
    n = len(values)
    if n == 0:
        return VGroup()

    depth = math.floor(math.log2(n)) + 1 if n > 0 else 0
    nodes = {}
    positions = {}

    for idx, val in enumerate(values):
        if val is None:
            continue
        level = math.floor(math.log2(idx + 1))
        pos_in_level = idx - (2**level - 1)
        nodes_at_level = 2**level

        # Horizontal spread decreases with depth
        spread = h_spacing * (2 ** (depth - level - 1))
        x = (pos_in_level - (nodes_at_level - 1) / 2) * spread
        y = -level * v_spacing

        is_leaf = (2 * idx + 1 >= n) or (
            (2 * idx + 1 < n and values[2 * idx + 1] is None)
            and (2 * idx + 2 >= n or values[2 * idx + 2] is None)
        )
        color = leaf_color if is_leaf else node_color

        node = make_tree_node(str(val), shape="circle", size=node_size,
                              color=color, font_size=font_size)
        node.move_to([x, y, 0])
        nodes[idx] = node
        positions[idx] = np.array([x, y, 0])

    # Create edges
    edges = VGroup()
    for idx in nodes:
        left_child = 2 * idx + 1
        right_child = 2 * idx + 2
        for child_idx in [left_child, right_child]:
            if child_idx in nodes:
                edge = Line(
                    positions[idx], positions[child_idx],
                    color=edge_color, stroke_width=2, buff=node_size + 0.05,
                )
                edges.add(edge)

    all_nodes = VGroup(*nodes.values())
    result = VGroup(edges, all_nodes)
    result.nodes = nodes
    result.edges = edges
    return result


def make_decision_tree(
    tree_data: dict,
    node_color: str = ACCENT_ORANGE,
    leaf_color: str = ACCENT_GREEN,
    edge_color: str = TEXT_DIM,
    h_spacing: float = 2.5,
    v_spacing: float = 1.5,
    node_width: float = 2.0,
    font_size: int = 14,
) -> VGroup:
    """Build a decision tree from a nested dictionary.

    Args:
        tree_data: nested dict like:
            {
                "label": "Age > 30?",
                "yes": {"label": "Income > 50k?",
                        "yes": {"label": "Approve"},
                        "no": {"label": "Deny"}},
                "no": {"label": "Deny"},
            }
            Leaf nodes have only "label". Decision nodes also have "yes" and "no".

    Returns VGroup with .nodes_list (VGroup of all nodes), .edges (VGroup),
           .edge_labels (VGroup).

    Example:
        dt = make_decision_tree({
            "label": "Is raining?",
            "yes": {"label": "Take umbrella"},
            "no": {"label": "Is sunny?",
                    "yes": {"label": "Sunglasses"},
                    "no": {"label": "Just go"}},
        })
        dt.scale(0.7).move_to(ORIGIN)
        self.play(FadeIn(dt))
    """
    nodes_list = VGroup()
    edges = VGroup()
    edge_labels_vg = VGroup()

    def _build(data, x, y, spread):
        is_leaf = "yes" not in data and "no" not in data
        color = leaf_color if is_leaf else node_color
        node = make_tree_node(
            data["label"], shape="rect",
            size=node_width / 2.5, color=color, font_size=font_size,
        )
        node.move_to([x, y, 0])
        nodes_list.add(node)

        if "yes" in data:
            child_y = y - v_spacing
            # Yes branch (left)
            yes_x = x - spread
            yes_node = _build(data["yes"], yes_x, child_y, spread * 0.5)
            edge = Line(
                node.get_bottom(), yes_node.get_top(),
                color=edge_color, stroke_width=2, buff=0.05,
            )
            edges.add(edge)
            lbl = Text("Yes", font_size=12, color=ACCENT_GREEN)
            lbl.move_to(edge.get_center()).shift(LEFT * 0.3)
            edge_labels_vg.add(lbl)

        if "no" in data:
            child_y = y - v_spacing
            # No branch (right)
            no_x = x + spread
            no_node = _build(data["no"], no_x, child_y, spread * 0.5)
            edge = Line(
                node.get_bottom(), no_node.get_top(),
                color=edge_color, stroke_width=2, buff=0.05,
            )
            edges.add(edge)
            lbl = Text("No", font_size=12, color=ACCENT_RED)
            lbl.move_to(edge.get_center()).shift(RIGHT * 0.3)
            edge_labels_vg.add(lbl)

        return node

    _build(tree_data, 0, 0, h_spacing)

    result = VGroup(edges, edge_labels_vg, nodes_list)
    result.nodes_list = nodes_list
    result.edges = edges
    result.edge_labels = edge_labels_vg
    return result


def make_org_chart(
    org_data: dict,
    boss_color: str = ACCENT_PURPLE,
    manager_color: str = ACCENT_BLUE,
    employee_color: str = ACCENT_GREEN,
    edge_color: str = TEXT_DIM,
    h_spacing: float = 2.5,
    v_spacing: float = 1.5,
    box_width: float = 2.0,
    font_size: int = 14,
) -> VGroup:
    """Build an org chart from a nested dictionary.

    Args:
        org_data: nested dict like:
            {
                "label": "CEO",
                "children": [
                    {"label": "CTO", "children": [
                        {"label": "Dev Lead"},
                        {"label": "QA Lead"},
                    ]},
                    {"label": "CFO"},
                ]
            }

    Returns VGroup with .all_nodes (VGroup), .edges (VGroup).

    Example:
        org = make_org_chart({
            "label": "CEO",
            "children": [
                {"label": "VP Eng", "children": [{"label": "Dev"}, {"label": "QA"}]},
                {"label": "VP Sales"},
            ],
        })
        org.scale(0.7).move_to(ORIGIN)
        self.play(FadeIn(org))
    """
    all_nodes = VGroup()
    edges = VGroup()

    def _get_depth(data):
        if "children" not in data or not data["children"]:
            return 0
        return 1 + max(_get_depth(c) for c in data["children"])

    def _count_leaves(data):
        if "children" not in data or not data["children"]:
            return 1
        return sum(_count_leaves(c) for c in data["children"])

    def _build(data, x, y, available_width, depth=0):
        # Pick color by depth
        if depth == 0:
            color = boss_color
        elif "children" in data and data["children"]:
            color = manager_color
        else:
            color = employee_color

        node = make_tree_node(
            data["label"], shape="rect",
            size=box_width / 2.5, color=color, font_size=font_size,
        )
        node.move_to([x, y, 0])
        all_nodes.add(node)

        if "children" in data and data["children"]:
            children = data["children"]
            n_children = len(children)
            total_leaves = sum(_count_leaves(c) for c in children)

            # Distribute width proportionally to leaf count
            child_x_start = x - available_width / 2
            for ch in children:
                ch_leaves = _count_leaves(ch)
                ch_width = (ch_leaves / total_leaves) * available_width
                ch_x = child_x_start + ch_width / 2
                child_node = _build(ch, ch_x, y - v_spacing, ch_width, depth + 1)
                edge = Line(
                    node.get_bottom(), child_node.get_top(),
                    color=edge_color, stroke_width=2, buff=0.05,
                )
                edges.add(edge)
                child_x_start += ch_width

        return node

    total_leaves = _count_leaves(org_data)
    total_width = total_leaves * h_spacing
    _build(org_data, 0, 0, total_width)

    result = VGroup(edges, all_nodes)
    result.all_nodes = all_nodes
    result.edges = edges
    return result


def animate_tree_level_by_level(scene, tree: VGroup, run_time_per_level: float = 0.5):
    """Animate a tree appearing level by level (top to bottom).

    Works with binary trees, decision trees, and org charts.
    Sorts nodes by y-coordinate descending (top first).

    Example:
        tree = make_binary_tree([1, 2, 3, 4, 5, 6, 7])
        animate_tree_level_by_level(self, tree)
    """
    # Collect all nodes and edges
    all_mobs = list(tree.submobjects)
    # Flatten
    flat_nodes = []
    flat_edges = []
    for mob in all_mobs:
        if isinstance(mob, VGroup):
            for sub in mob:
                if isinstance(sub, Line) or isinstance(sub, Arrow):
                    flat_edges.append(sub)
                elif isinstance(sub, VGroup):
                    flat_nodes.append(sub)
                else:
                    flat_nodes.append(sub)
        elif isinstance(mob, Line):
            flat_edges.append(mob)

    if not flat_nodes:
        scene.play(FadeIn(tree))
        return

    # Sort nodes by y coordinate (highest first)
    flat_nodes.sort(key=lambda m: -m.get_center()[1])

    # Group by approximate y level
    levels = []
    current_level = [flat_nodes[0]]
    for node in flat_nodes[1:]:
        if abs(node.get_center()[1] - current_level[0].get_center()[1]) < 0.3:
            current_level.append(node)
        else:
            levels.append(current_level)
            current_level = [node]
    levels.append(current_level)

    for level_nodes in levels:
        scene.play(
            *[FadeIn(n, shift=DOWN * 0.2) for n in level_nodes],
            run_time=run_time_per_level,
        )

    if flat_edges:
        scene.play(
            *[Create(e) for e in flat_edges],
            run_time=run_time_per_level,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 8. GRAPH / NETWORK LAYOUTS (using Manim's built-in Graph/DiGraph)
# ═══════════════════════════════════════════════════════════════════════════════

# These use Manim CE's built-in Graph and DiGraph classes which
# wrap NetworkX for automatic layout. Import them directly:
#   from manim import Graph, DiGraph

def make_graph_network(
    vertices: list,
    edges: list[tuple],
    labels: bool = True,
    layout: str = "spring",
    layout_scale: float = 2.5,
    layout_config: dict | None = None,
    vertex_config: dict | None = None,
    edge_config: dict | None = None,
    directed: bool = False,
) -> "Graph | DiGraph":
    """Create a Manim Graph or DiGraph with automatic layout.

    Available layouts: "spring", "circular", "kamada_kawai", "planar",
                       "random", "shell", "spectral", "spiral", "tree", "partite"

    For tree layout: layout_config={"root_vertex": root_id}
    For partite layout: layout_config={"partitions": [[v1,v2],[v3,v4]]}

    Args:
        vertices: list of hashable vertex identifiers (ints or strings)
        edges: list of (u, v) tuples
        directed: if True, uses DiGraph with arrow tips

    Returns a Manim Graph or DiGraph mobject.

    Example (undirected):
        from manim import Graph
        g = make_graph_network(
            [1,2,3,4,5],
            [(1,2),(2,3),(3,4),(4,5),(5,1),(1,3)],
            layout="circular",
        )
        self.play(Create(g))

    Example (directed):
        from manim import DiGraph
        g = make_graph_network(
            ["A","B","C","D"],
            [("A","B"),("B","C"),("C","D"),("D","A")],
            directed=True, layout="circular",
        )
        self.play(Create(g))

    Example (tree):
        g = make_graph_network(
            [1,2,3,4,5,6,7],
            [(1,2),(1,3),(2,4),(2,5),(3,6),(3,7)],
            layout="tree",
            layout_config={"root_vertex": 1},
        )
        self.play(Create(g))

    Example (neural network / partite):
        # 3-layer network: [1,2] -> [3,4,5] -> [6,7]
        verts = list(range(1, 8))
        edges = [(i, j) for i in [1,2] for j in [3,4,5]] + [(i,j) for i in [3,4,5] for j in [6,7]]
        g = make_graph_network(
            verts, edges,
            layout="partite",
            layout_config={"partitions": [[1,2],[3,4,5],[6,7]]},
            layout_scale=3,
        )
        self.play(Create(g))
    """
    from manim import Graph as ManimGraph, DiGraph as ManimDiGraph

    if layout_config is None:
        layout_config = {}

    kwargs = dict(
        labels=labels,
        layout=layout,
        layout_scale=layout_scale,
        layout_config=layout_config,
    )
    if vertex_config:
        kwargs["vertex_config"] = vertex_config
    if edge_config:
        kwargs["edge_config"] = edge_config

    if directed:
        return ManimDiGraph(vertices, edges, **kwargs)
    else:
        return ManimGraph(vertices, edges, **kwargs)


def animate_graph_build(scene, graph, run_time: float = 2.0):
    """Animate a Graph appearing: Create animation.

    Example:
        g = make_graph_network([1,2,3], [(1,2),(2,3)])
        animate_graph_build(self, g)
    """
    scene.play(Create(graph), run_time=run_time)


def animate_graph_layout_change(scene, graph, new_layout: str, run_time: float = 1.5):
    """Animate a smooth layout transition on an existing Graph.

    Example:
        g = make_graph_network([...], [...], layout="spring")
        self.play(Create(g))
        animate_graph_layout_change(self, g, "circular")
    """
    scene.play(graph.animate.change_layout(new_layout), run_time=run_time)


def make_neural_network_graph(
    layer_sizes: list[int],
    layout_scale: float = 3.0,
    vertex_config: dict | None = None,
) -> "Graph":
    """Create a neural network diagram using Graph with partite layout.

    Args:
        layer_sizes: list of neuron counts per layer, e.g. [3, 5, 5, 2]

    Returns a Manim Graph.

    Example:
        nn = make_neural_network_graph([2, 4, 4, 1])
        nn.scale(0.8).move_to(ORIGIN)
        self.play(Create(nn))
    """
    from manim import Graph as ManimGraph

    vertices = []
    edges = []
    partitions = []
    counter = 0

    for layer_idx, size in enumerate(layer_sizes):
        layer_verts = list(range(counter, counter + size))
        partitions.append(layer_verts)
        vertices.extend(layer_verts)

        # Connect to previous layer (fully connected)
        if layer_idx > 0:
            prev_layer = partitions[layer_idx - 1]
            for prev_v in prev_layer:
                for curr_v in layer_verts:
                    edges.append((prev_v, curr_v))

        counter += size

    v_config = {"radius": 0.15, "fill_opacity": 0.8}
    if vertex_config:
        v_config.update(vertex_config)

    return ManimGraph(
        vertices, edges,
        layout="partite",
        partitions=partitions,
        layout_scale=layout_scale,
        vertex_config=v_config,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 9. STATE MACHINES & AUTOMATA
# ═══════════════════════════════════════════════════════════════════════════════

def make_state_machine(
    states: list[dict],
    transitions: list[dict],
    layout: str = "circular",
    state_radius: float = 0.5,
    state_color: str = ACCENT_BLUE,
    accept_color: str = ACCENT_GREEN,
    start_color: str = ACCENT_ORANGE,
    edge_color: str = TEXT_DIM,
    font_size: int = 16,
    layout_scale: float = 2.5,
) -> VGroup:
    """Create a finite state machine / automaton diagram.

    Args:
        states: list of dicts with keys:
            "name" (str, required), "accept" (bool, optional), "start" (bool, optional)
            Example: [
                {"name": "q0", "start": True},
                {"name": "q1"},
                {"name": "q2", "accept": True},
            ]
        transitions: list of dicts with keys:
            "from" (str), "to" (str), "label" (str)
            Example: [
                {"from": "q0", "to": "q1", "label": "a"},
                {"from": "q1", "to": "q2", "label": "b"},
                {"from": "q2", "to": "q0", "label": "c"},
            ]

    Returns VGroup with .state_nodes (dict name->VGroup), .arrows (VGroup),
           .labels (VGroup), .start_arrow (optional).

    Example:
        sm = make_state_machine(
            [{"name": "q0", "start": True}, {"name": "q1"}, {"name": "q2", "accept": True}],
            [{"from": "q0", "to": "q1", "label": "0"},
             {"from": "q1", "to": "q2", "label": "1"},
             {"from": "q2", "to": "q0", "label": "0,1"}],
        )
        sm.move_to(ORIGIN)
        self.play(FadeIn(sm))
    """
    n = len(states)
    state_names = [s["name"] for s in states]

    # Compute positions based on layout
    if layout == "circular":
        angles = [i * TAU / n for i in range(n)]
        pos = {
            s["name"]: np.array([
                layout_scale * np.cos(a),
                layout_scale * np.sin(a),
                0,
            ])
            for s, a in zip(states, angles)
        }
    elif layout == "linear":
        total_w = (n - 1) * layout_scale
        pos = {
            s["name"]: np.array([
                -total_w / 2 + i * layout_scale, 0, 0
            ])
            for i, s in enumerate(states)
        }
    else:
        # Default to horizontal
        total_w = (n - 1) * layout_scale
        pos = {
            s["name"]: np.array([
                -total_w / 2 + i * layout_scale, 0, 0
            ])
            for i, s in enumerate(states)
        }

    # Build state nodes
    state_nodes = {}
    all_state_mobs = VGroup()

    for s_def in states:
        name = s_def["name"]
        is_accept = s_def.get("accept", False)
        is_start = s_def.get("start", False)

        if is_start:
            color = start_color
        elif is_accept:
            color = accept_color
        else:
            color = state_color

        outer = Circle(
            radius=state_radius,
            fill_color=color, fill_opacity=0.25,
            stroke_color=color, stroke_width=2,
        )
        outer.move_to(pos[name])

        txt = Text(name, font_size=font_size, color=TEXT_PRIMARY)
        txt.move_to(pos[name])

        group = VGroup(outer, txt)

        # Double circle for accept states
        if is_accept:
            inner = Circle(
                radius=state_radius * 0.8,
                stroke_color=color, stroke_width=2,
                fill_opacity=0,
            )
            inner.move_to(pos[name])
            group.add(inner)

        state_nodes[name] = group
        all_state_mobs.add(group)

    # Build transitions
    arrows_vg = VGroup()
    labels_vg = VGroup()

    for t in transitions:
        src = t["from"]
        dst = t["to"]
        label = t.get("label", "")

        if src == dst:
            # Self-loop: draw a small arc above the state
            center = pos[src]
            loop = CurvedArrow(
                center + UP * state_radius + LEFT * 0.3,
                center + UP * state_radius + RIGHT * 0.3,
                radius=-0.5,
                color=edge_color,
            )
            arrows_vg.add(loop)
            if label:
                lbl = Text(label, font_size=12, color=TEXT_SECONDARY)
                lbl.next_to(loop, UP, buff=0.1)
                labels_vg.add(lbl)
        else:
            # Check if reverse transition exists (need curved arrows to avoid overlap)
            has_reverse = any(
                tr["from"] == dst and tr["to"] == src for tr in transitions
            )
            if has_reverse:
                arrow = CurvedArrow(
                    pos[src], pos[dst],
                    radius=3.0,
                    color=edge_color,
                )
            else:
                arrow = Arrow(
                    pos[src], pos[dst],
                    buff=state_radius + 0.1,
                    color=edge_color, stroke_width=2,
                    max_tip_length_to_length_ratio=0.15,
                )
            arrows_vg.add(arrow)

            if label:
                lbl = Text(label, font_size=12, color=TEXT_SECONDARY)
                mid = arrow.point_from_proportion(0.5)
                # Offset label perpendicular to arrow direction
                direction = pos[dst] - pos[src]
                perp = np.array([-direction[1], direction[0], 0])
                perp = perp / (np.linalg.norm(perp) + 1e-8)
                lbl.move_to(mid + perp * 0.25)
                labels_vg.add(lbl)

    # Start arrow (pointing into start state)
    start_arrow_mob = VGroup()
    for s_def in states:
        if s_def.get("start", False):
            p = pos[s_def["name"]]
            start_pt = p + LEFT * (state_radius + 0.8)
            arr = Arrow(
                start_pt, p + LEFT * state_radius,
                buff=0.05, color=start_color, stroke_width=2,
            )
            start_arrow_mob.add(arr)
            break

    result = VGroup(arrows_vg, labels_vg, all_state_mobs, start_arrow_mob)
    result.state_nodes = state_nodes
    result.arrows = arrows_vg
    result.labels = labels_vg
    result.start_arrow = start_arrow_mob
    return result


def animate_state_transition(
    scene,
    state_machine: VGroup,
    path: list[str],
    highlight_color: str = ACCENT_YELLOW,
    run_time_per_step: float = 0.5,
):
    """Animate traversal through a state machine by highlighting states in sequence.

    Args:
        path: list of state names in the order visited, e.g. ["q0", "q1", "q2"]

    Example:
        sm = make_state_machine([...], [...])
        scene.play(FadeIn(sm))
        animate_state_transition(scene, sm, ["q0", "q1", "q2", "q1"])
    """
    for name in path:
        if name in state_machine.state_nodes:
            node = state_machine.state_nodes[name]
            outer = node[0]
            scene.play(
                outer.animate.set_fill(highlight_color, opacity=0.5),
                run_time=run_time_per_step,
            )
            scene.play(
                outer.animate.set_fill(outer.get_fill_color(), opacity=0.25),
                run_time=run_time_per_step * 0.4,
            )


# ═══════════════════════════════════════════════════════════════════════════════
# 10. VENN DIAGRAMS
# ═══════════════════════════════════════════════════════════════════════════════

def make_venn_2(
    label_a: str = "A",
    label_b: str = "B",
    label_ab: str = "",
    color_a: str = ACCENT_BLUE,
    color_b: str = ACCENT_ORANGE,
    radius: float = 1.5,
    overlap: float = 0.8,
    opacity: float = 0.3,
    font_size: int = 20,
) -> VGroup:
    """Create a 2-circle Venn diagram.

    Args:
        overlap: distance between circle centers (smaller = more overlap)

    Returns VGroup with .circle_a, .circle_b, .label_a_mob, .label_b_mob,
           .label_ab_mob (only if label_ab is non-empty).

    Example:
        venn = make_venn_2("Dogs", "Pets", "Pet Dogs")
        venn.move_to(ORIGIN)
        self.play(FadeIn(venn))
    """
    ca = Circle(radius=radius, fill_color=color_a, fill_opacity=opacity,
                stroke_color=color_a, stroke_width=2)
    cb = Circle(radius=radius, fill_color=color_b, fill_opacity=opacity,
                stroke_color=color_b, stroke_width=2)

    ca.shift(LEFT * overlap / 2)
    cb.shift(RIGHT * overlap / 2)

    la = Text(label_a, font_size=font_size, color=color_a, weight="BOLD")
    la.move_to(ca.get_center() + LEFT * (radius * 0.4))
    lb = Text(label_b, font_size=font_size, color=color_b, weight="BOLD")
    lb.move_to(cb.get_center() + RIGHT * (radius * 0.4))

    result = VGroup(ca, cb, la, lb)
    result.circle_a = ca
    result.circle_b = cb
    result.label_a_mob = la
    result.label_b_mob = lb

    if label_ab:
        lab = Text(label_ab, font_size=font_size - 2, color=TEXT_PRIMARY)
        lab.move_to((ca.get_center() + cb.get_center()) / 2)
        result.add(lab)
        result.label_ab_mob = lab

    return result


def make_venn_3(
    labels: list[str] = None,
    colors: list[str] = None,
    radius: float = 1.3,
    spread: float = 0.7,
    opacity: float = 0.25,
    font_size: int = 18,
) -> VGroup:
    """Create a 3-circle Venn diagram.

    Args:
        labels: list of 3 strings for each circle
        colors: list of 3 color strings

    Returns VGroup with .circles (list), .label_mobs (VGroup).

    Example:
        venn3 = make_venn_3(["ML", "Stats", "CS"],
                            [ACCENT_BLUE, ACCENT_GREEN, ACCENT_ORANGE])
        venn3.move_to(ORIGIN)
        self.play(FadeIn(venn3))
    """
    if labels is None:
        labels = ["A", "B", "C"]
    if colors is None:
        colors = [ACCENT_BLUE, ACCENT_GREEN, ACCENT_ORANGE]

    # Three circles arranged at 120-degree intervals
    angles = [PI / 2, PI / 2 + 2 * PI / 3, PI / 2 + 4 * PI / 3]
    circles = []
    label_mobs = VGroup()

    for i, (angle, label, color) in enumerate(zip(angles, labels, colors)):
        center = np.array([spread * np.cos(angle), spread * np.sin(angle), 0])
        c = Circle(
            radius=radius, fill_color=color, fill_opacity=opacity,
            stroke_color=color, stroke_width=2,
        )
        c.move_to(center)
        circles.append(c)

        # Label outside the circle
        label_pos = center + np.array([
            (radius + 0.3) * np.cos(angle),
            (radius + 0.3) * np.sin(angle),
            0,
        ])
        lbl = Text(label, font_size=font_size, color=color, weight="BOLD")
        lbl.move_to(label_pos)
        label_mobs.add(lbl)

    result = VGroup(*circles, label_mobs)
    result.circles = circles
    result.label_mobs = label_mobs
    return result


def animate_venn_highlight_intersection(
    scene,
    venn: VGroup,
    color: str = ACCENT_YELLOW,
    run_time: float = 0.5,
):
    """Flash-highlight the intersection area of a 2-circle Venn diagram.

    Uses Indicate on the overlap region.

    Example:
        venn = make_venn_2("A", "B", "A & B")
        self.play(FadeIn(venn))
        animate_venn_highlight_intersection(self, venn)
    """
    if hasattr(venn, 'label_ab_mob'):
        scene.play(
            Indicate(venn.label_ab_mob, color=color, scale_factor=1.3),
            venn.circle_a.animate.set_fill(opacity=0.5),
            venn.circle_b.animate.set_fill(opacity=0.5),
            run_time=run_time,
        )
        scene.play(
            venn.circle_a.animate.set_fill(opacity=0.3),
            venn.circle_b.animate.set_fill(opacity=0.3),
            run_time=run_time * 0.5,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 11. BAR CHARTS, LINE CHARTS, PIE CHARTS
# ═══════════════════════════════════════════════════════════════════════════════

# Manim CE has a built-in BarChart class (inherits from Axes).
# Import: from manim import BarChart

def make_bar_chart(
    values: list[float],
    bar_names: list[str] = None,
    y_range: list[float] = None,
    bar_colors: list[str] = None,
    x_length: float = 10,
    y_length: float = 5,
    bar_width: float = 0.6,
    bar_fill_opacity: float = 0.7,
    bar_stroke_width: float = 3,
    **kwargs,
) -> "BarChart":
    """Create a Manim BarChart with Octoflash styling.

    This wraps Manim CE's built-in BarChart class.

    Args:
        values: list of numeric bar heights
        bar_names: optional list of x-axis labels
        y_range: [y_min, y_max, y_step]
        bar_colors: list of colors for bars (cycles through if shorter)

    Returns a Manim BarChart object.

    Key methods:
        chart.get_bar_labels(font_size=24)  -- adds value labels above bars
        chart.change_bar_values([new_vals])  -- animatable bar value change

    Example (static):
        from manim import BarChart
        chart = make_bar_chart(
            [3, 7, 2, 9, 5],
            bar_names=["A", "B", "C", "D", "E"],
            y_range=[0, 10, 2],
        )
        chart.scale(0.8).move_to(ORIGIN)
        labels = chart.get_bar_labels(font_size=24)
        self.play(FadeIn(chart), FadeIn(labels))

    Example (animated value change):
        chart = make_bar_chart([3, 7, 2], bar_names=["X","Y","Z"], y_range=[0,10,2])
        self.play(FadeIn(chart))
        self.wait(1)
        chart.change_bar_values([8, 4, 6])
        new_labels = chart.get_bar_labels(font_size=24)
        self.play(FadeIn(new_labels))
    """
    from manim import BarChart as ManimBarChart

    if bar_colors is None:
        bar_colors = [ACCENT_BLUE, ACCENT_GREEN, ACCENT_ORANGE, ACCENT_RED, ACCENT_PURPLE]

    return ManimBarChart(
        values=values,
        bar_names=bar_names,
        y_range=y_range,
        x_length=x_length,
        y_length=y_length,
        bar_colors=bar_colors,
        bar_width=bar_width,
        bar_fill_opacity=bar_fill_opacity,
        bar_stroke_width=bar_stroke_width,
        **kwargs,
    )


def animate_bar_chart_build(scene, chart, run_time: float = 1.5):
    """Animate a BarChart growing from zero.

    Example:
        chart = make_bar_chart([3, 7, 2], bar_names=["A","B","C"])
        animate_bar_chart_build(self, chart)
    """
    # Store original values
    original = list(chart.values)
    # Set to zero
    chart.change_bar_values([0] * len(original))
    scene.add(chart)
    scene.wait(0.1)
    # Animate back to original
    chart.change_bar_values(original)
    labels = chart.get_bar_labels(font_size=24)
    scene.play(FadeIn(labels), run_time=run_time * 0.5)


def make_line_chart(
    x_values: list[float],
    y_values_list: list[list[float]],
    line_colors: list[str] = None,
    line_labels: list[str] = None,
    x_label: str = "x",
    y_label: str = "y",
    x_range: list[float] = None,
    y_range: list[float] = None,
    x_length: float = 8,
    y_length: float = 5,
    dot_radius: float = 0.05,
) -> VGroup:
    """Create a multi-line chart using Manim Axes.

    Args:
        x_values: shared x-axis values
        y_values_list: list of y-value lists (one per line)
        line_colors: colors for each line
        line_labels: legend labels for each line

    Returns VGroup with .axes, .lines (VGroup of curves), .dots (VGroup),
           .legend (VGroup, if line_labels provided).

    Example:
        lc = make_line_chart(
            [1, 2, 3, 4, 5],
            [[2, 4, 3, 7, 5], [1, 3, 5, 4, 6]],
            line_colors=[ACCENT_BLUE, ACCENT_ORANGE],
            line_labels=["Model A", "Model B"],
            x_label="Epoch", y_label="Loss",
        )
        lc.scale(0.8).move_to(ORIGIN)
        self.play(FadeIn(lc.axes))
        for line in lc.lines:
            self.play(Create(line), run_time=1)
    """
    from manim import Axes as ManimAxes

    if line_colors is None:
        line_colors = [ACCENT_BLUE, ACCENT_GREEN, ACCENT_ORANGE, ACCENT_RED, ACCENT_PURPLE]

    if x_range is None:
        x_range = [min(x_values), max(x_values), (max(x_values) - min(x_values)) / 5]
    if y_range is None:
        all_y = [y for ys in y_values_list for y in ys]
        y_min, y_max = min(all_y), max(all_y)
        y_range = [y_min, y_max, (y_max - y_min) / 5]

    axes = ManimAxes(
        x_range=x_range, y_range=y_range,
        x_length=x_length, y_length=y_length,
        axis_config={"color": TEXT_DIM, "include_numbers": True, "font_size": 16},
        tips=False,
    )

    x_lab = axes.get_x_axis_label(x_label, direction=DOWN)
    y_lab = axes.get_y_axis_label(y_label, direction=LEFT)

    lines_vg = VGroup()
    dots_vg = VGroup()

    for idx, y_values in enumerate(y_values_list):
        color = line_colors[idx % len(line_colors)]

        # Create line segments
        points = [axes.c2p(x, y) for x, y in zip(x_values, y_values)]
        line = VMobject(color=color, stroke_width=2.5)
        line.set_points_as_corners(points)
        lines_vg.add(line)

        # Create dots at data points
        for x, y in zip(x_values, y_values):
            dot = Dot(axes.c2p(x, y), radius=dot_radius, color=color)
            dots_vg.add(dot)

    result = VGroup(axes, x_lab, y_lab, lines_vg, dots_vg)
    result.axes = axes
    result.lines = lines_vg
    result.dots = dots_vg

    # Legend
    if line_labels:
        legend = VGroup()
        for idx, label in enumerate(line_labels):
            color = line_colors[idx % len(line_colors)]
            line_sample = Line(LEFT * 0.3, RIGHT * 0.3, color=color, stroke_width=3)
            label_text = Text(label, font_size=14, color=TEXT_PRIMARY)
            row = VGroup(line_sample, label_text).arrange(RIGHT, buff=0.15)
            legend.add(row)
        legend.arrange(DOWN, buff=0.15, aligned_edge=LEFT)
        legend.next_to(axes, RIGHT, buff=0.3).align_to(axes, UP)
        result.add(legend)
        result.legend = legend

    return result


def make_pie_chart(
    values: list[float],
    labels: list[str] = None,
    colors: list[str] = None,
    radius: float = 2.0,
    font_size: int = 16,
    show_percentages: bool = True,
    start_angle: float = PI / 2,
) -> VGroup:
    """Create a pie chart from values.

    Manim CE does not have a built-in PieChart, so this builds one from Sectors.

    Args:
        values: numeric values (will be normalized to percentages)
        labels: names for each slice
        colors: colors for each slice

    Returns VGroup with .slices (VGroup of Sectors), .label_mobs (VGroup).

    Example:
        pie = make_pie_chart(
            [40, 30, 20, 10],
            labels=["Python", "JS", "Go", "Rust"],
            colors=[ACCENT_BLUE, ACCENT_YELLOW, ACCENT_CYAN, ACCENT_ORANGE],
        )
        pie.move_to(ORIGIN)
        self.play(LaggedStartMap(FadeIn, pie.slices, lag_ratio=0.15))
        self.play(FadeIn(pie.label_mobs))
    """
    from manim import Sector

    if colors is None:
        colors = [ACCENT_BLUE, ACCENT_GREEN, ACCENT_ORANGE, ACCENT_RED,
                  ACCENT_PURPLE, ACCENT_CYAN, ACCENT_YELLOW, ACCENT_PINK]

    total = sum(values)
    if total == 0:
        return VGroup()

    slices = VGroup()
    label_mobs = VGroup()
    current_angle = start_angle

    for i, val in enumerate(values):
        pct = val / total
        angle = pct * TAU
        color = colors[i % len(colors)]

        sector = Sector(
            outer_radius=radius,
            inner_radius=0,
            angle=angle,
            start_angle=current_angle,
            fill_color=color,
            fill_opacity=0.7,
            stroke_color=color,
            stroke_width=2,
        )
        slices.add(sector)

        # Label position: midpoint of the arc
        mid_angle = current_angle + angle / 2
        label_r = radius + 0.4
        label_pos = np.array([
            label_r * np.cos(mid_angle),
            label_r * np.sin(mid_angle),
            0,
        ])

        label_str = ""
        if labels and i < len(labels):
            label_str = labels[i]
        if show_percentages:
            pct_str = f"{pct * 100:.0f}%"
            label_str = f"{label_str}\n{pct_str}" if label_str else pct_str

        if label_str:
            lbl = Text(label_str, font_size=font_size, color=TEXT_PRIMARY)
            lbl.move_to(label_pos)
            label_mobs.add(lbl)

        current_angle += angle

    result = VGroup(slices, label_mobs)
    result.slices = slices
    result.label_mobs = label_mobs
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 12. HEATMAPS & COLORED GRIDS
# ═══════════════════════════════════════════════════════════════════════════════

def make_heatmap(
    data: list[list[float]],
    low_color: str = ACCENT_BLUE,
    high_color: str = ACCENT_RED,
    mid_color: str = ACCENT_YELLOW,
    cell_size: float = 0.7,
    font_size: int = 14,
    show_values: bool = True,
    row_labels: list[str] = None,
    col_labels: list[str] = None,
    value_format: str = "{:.1f}",
) -> VGroup:
    """Create a color-coded heatmap from a 2D numeric array.

    Colors interpolate from low_color through mid_color to high_color
    based on value range.

    Args:
        data: 2D list of floats
        low_color, high_color, mid_color: color endpoints for gradient
        show_values: whether to display numeric values in cells
        row_labels, col_labels: optional axis labels

    Returns VGroup with .cells (VGroup), .row_label_mobs, .col_label_mobs.

    Example:
        hm = make_heatmap(
            [[0.1, 0.5, 0.9],
             [0.3, 0.7, 0.4],
             [0.8, 0.2, 0.6]],
            row_labels=["X", "Y", "Z"],
            col_labels=["A", "B", "C"],
        )
        hm.move_to(ORIGIN)
        self.play(FadeIn(hm))
    """
    from manim import interpolate_color as manim_interpolate

    rows = len(data)
    cols = len(data[0]) if rows > 0 else 0

    # Compute value range
    all_vals = [v for row in data for v in row]
    v_min, v_max = min(all_vals), max(all_vals)
    v_range = v_max - v_min if v_max != v_min else 1.0

    cells = VGroup()

    for r in range(rows):
        for c in range(cols):
            val = data[r][c]
            # Normalize to [0, 1]
            t = (val - v_min) / v_range

            # Two-segment interpolation: low->mid->high
            from manim import ManimColor
            if t < 0.5:
                color = manim_interpolate(ManimColor(low_color), ManimColor(mid_color), t * 2)
            else:
                color = manim_interpolate(ManimColor(mid_color), ManimColor(high_color), (t - 0.5) * 2)

            rect = Square(
                side_length=cell_size,
                fill_color=color,
                fill_opacity=0.8,
                stroke_color=TEXT_DIM,
                stroke_width=0.5,
            )
            rect.move_to([
                c * cell_size - (cols - 1) * cell_size / 2,
                -r * cell_size + (rows - 1) * cell_size / 2,
                0,
            ])

            cell_group = VGroup(rect)

            if show_values:
                txt = Text(
                    value_format.format(val),
                    font_size=font_size, color=TEXT_PRIMARY,
                )
                txt.move_to(rect.get_center())
                cell_group.add(txt)

            cells.add(cell_group)

    result = VGroup(cells)
    result.cells = cells

    # Row labels (left side)
    if row_labels:
        rl_mobs = VGroup()
        for r, label in enumerate(row_labels):
            lbl = Text(label, font_size=font_size, color=TEXT_SECONDARY)
            lbl.next_to(cells[r * cols], LEFT, buff=0.2)
            rl_mobs.add(lbl)
        result.add(rl_mobs)
        result.row_label_mobs = rl_mobs

    # Column labels (top)
    if col_labels:
        cl_mobs = VGroup()
        for c, label in enumerate(col_labels):
            lbl = Text(label, font_size=font_size, color=TEXT_SECONDARY)
            lbl.next_to(cells[c], UP, buff=0.2)
            cl_mobs.add(lbl)
        result.add(cl_mobs)
        result.col_label_mobs = cl_mobs

    return result


def animate_heatmap_reveal(scene, heatmap: VGroup, run_time: float = 2.0):
    """Animate a heatmap appearing cell by cell in a wave pattern.

    Example:
        hm = make_heatmap([[0.1, 0.5], [0.3, 0.7]])
        animate_heatmap_reveal(self, hm)
    """
    cells = heatmap.cells
    scene.play(
        LaggedStart(
            *[FadeIn(cell, shift=UP * 0.1) for cell in cells],
            lag_ratio=0.05,
        ),
        run_time=run_time,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 13. GANTT CHARTS / TIMELINE BARS
# ═══════════════════════════════════════════════════════════════════════════════

def make_gantt_chart(
    tasks: list[dict],
    time_range: tuple = (0, 10),
    chart_width: float = 10.0,
    row_height: float = 0.5,
    row_spacing: float = 0.15,
    font_size: int = 14,
    default_color: str = ACCENT_BLUE,
) -> VGroup:
    """Create a Gantt chart (horizontal bar timeline).

    Args:
        tasks: list of dicts with keys:
            "label" (str), "start" (float), "end" (float),
            "color" (str, optional)
            Example: [
                {"label": "Design", "start": 0, "end": 3, "color": ACCENT_BLUE},
                {"label": "Develop", "start": 2, "end": 7, "color": ACCENT_GREEN},
                {"label": "Test", "start": 6, "end": 9, "color": ACCENT_ORANGE},
                {"label": "Deploy", "start": 8, "end": 10, "color": ACCENT_RED},
            ]
        time_range: (min_time, max_time) for the x-axis

    Returns VGroup with .bars (VGroup), .labels (VGroup), .axis_line, .tick_labels.

    Example:
        gantt = make_gantt_chart([
            {"label": "Phase 1", "start": 0, "end": 3},
            {"label": "Phase 2", "start": 2, "end": 6},
            {"label": "Phase 3", "start": 5, "end": 10},
        ])
        gantt.move_to(ORIGIN)
        self.play(FadeIn(gantt))
    """
    t_min, t_max = time_range
    t_range = t_max - t_min
    n = len(tasks)
    total_height = n * (row_height + row_spacing)

    # Time axis
    axis_line = Line(
        LEFT * chart_width / 2, RIGHT * chart_width / 2,
        color=TEXT_DIM, stroke_width=2,
    )
    axis_line.shift(DOWN * total_height / 2 + DOWN * 0.3)

    # Tick marks and labels
    tick_labels = VGroup()
    n_ticks = min(int(t_range) + 1, 11)
    for i in range(n_ticks):
        t = t_min + i * (t_range / (n_ticks - 1)) if n_ticks > 1 else t_min
        x = -chart_width / 2 + (t - t_min) / t_range * chart_width
        tick = Line(
            [x, axis_line.get_center()[1] - 0.08, 0],
            [x, axis_line.get_center()[1] + 0.08, 0],
            color=TEXT_DIM, stroke_width=1,
        )
        lbl = Text(f"{t:.0f}", font_size=12, color=TEXT_SECONDARY)
        lbl.next_to(tick, DOWN, buff=0.1)
        tick_labels.add(VGroup(tick, lbl))

    # Bars
    bars = VGroup()
    labels_vg = VGroup()

    for i, task in enumerate(tasks):
        y = total_height / 2 - i * (row_height + row_spacing) - row_height / 2
        start_x = -chart_width / 2 + (task["start"] - t_min) / t_range * chart_width
        end_x = -chart_width / 2 + (task["end"] - t_min) / t_range * chart_width
        bar_w = end_x - start_x
        color = task.get("color", default_color)

        bar = Rectangle(
            width=bar_w, height=row_height,
            fill_color=color, fill_opacity=0.6,
            stroke_color=color, stroke_width=2,
        )
        bar.move_to([(start_x + end_x) / 2, y, 0])
        bars.add(bar)

        lbl = Text(task["label"], font_size=font_size, color=TEXT_PRIMARY)
        lbl.next_to(bar, LEFT, buff=0.2)
        labels_vg.add(lbl)

    result = VGroup(axis_line, tick_labels, bars, labels_vg)
    result.bars = bars
    result.labels = labels_vg
    result.axis_line = axis_line
    result.tick_labels = tick_labels
    return result


def animate_gantt_fill(scene, gantt: VGroup, run_time_per_bar: float = 0.5):
    """Animate Gantt chart bars growing left to right.

    Example:
        gantt = make_gantt_chart([...])
        scene.play(FadeIn(VGroup(gantt.axis_line, gantt.tick_labels, gantt.labels)))
        animate_gantt_fill(scene, gantt)
    """
    for bar in gantt.bars:
        scene.play(
            GrowFromCenter(bar),
            run_time=run_time_per_bar,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 14. ARCHITECTURE / SYSTEM DESIGN DIAGRAMS
# ═══════════════════════════════════════════════════════════════════════════════

def make_system_block(
    label: str,
    sublabel: str = "",
    width: float = 2.5,
    height: float = 1.0,
    color: str = ACCENT_BLUE,
    icon: str = "",
    font_size: int = LABEL_SIZE,
    corner_radius: float = 0.15,
) -> VGroup:
    """Create a system architecture block (service, database, queue, etc).

    Args:
        icon: optional emoji or short icon text placed above the label
        sublabel: smaller text below the main label

    Returns VGroup with .rect, .label, optionally .sublabel, .icon.

    Example:
        api = make_system_block("API Gateway", sublabel="nginx", color=ACCENT_GREEN)
        db = make_system_block("PostgreSQL", sublabel="Primary DB", color=ACCENT_ORANGE)
    """
    rect = RoundedRectangle(
        corner_radius=corner_radius,
        width=width, height=height,
        fill_color=color, fill_opacity=0.2,
        stroke_color=color, stroke_width=2,
    )

    elements = [rect]
    main_label = Text(label, font_size=font_size, color=TEXT_PRIMARY, weight="BOLD")

    if icon:
        icon_text = Text(icon, font_size=font_size + 6)
        icon_text.move_to(rect.get_center() + UP * 0.15)
        main_label.next_to(icon_text, DOWN, buff=0.05)
        elements.append(icon_text)
    else:
        main_label.move_to(rect.get_center())

    elements.append(main_label)

    group = VGroup(*elements)
    group.rect = rect
    group.label = main_label

    if sublabel:
        sub = Text(sublabel, font_size=font_size - 4, color=TEXT_SECONDARY)
        sub.next_to(main_label, DOWN, buff=0.05)
        group.add(sub)
        group.sublabel = sub

    return group


def make_database_shape(
    label: str,
    width: float = 2.0,
    height: float = 1.2,
    color: str = ACCENT_ORANGE,
    font_size: int = LABEL_SIZE,
) -> VGroup:
    """Create a cylinder-like database shape (rectangle with ellipse top/bottom).

    Uses stacked ellipse + rectangle to simulate a 3D cylinder.

    Returns VGroup with .body, .label.

    Example:
        db = make_database_shape("Users DB", color=ACCENT_GREEN)
        db.move_to(ORIGIN)
        self.play(FadeIn(db))
    """
    from manim import Ellipse

    body = Rectangle(
        width=width, height=height * 0.6,
        fill_color=color, fill_opacity=0.2,
        stroke_color=color, stroke_width=2,
    )

    top_ellipse = Ellipse(
        width=width, height=height * 0.35,
        fill_color=color, fill_opacity=0.3,
        stroke_color=color, stroke_width=2,
    )
    top_ellipse.next_to(body, UP, buff=0)

    bottom_ellipse = Ellipse(
        width=width, height=height * 0.35,
        fill_color=color, fill_opacity=0.1,
        stroke_color=color, stroke_width=2,
    )
    bottom_ellipse.next_to(body, DOWN, buff=0)

    txt = Text(label, font_size=font_size, color=TEXT_PRIMARY)
    txt.move_to(body.get_center())

    group = VGroup(bottom_ellipse, body, top_ellipse, txt)
    group.body = body
    group.label = txt
    return group


def make_architecture_diagram(
    components: list[dict],
    connections: list[dict],
    title: str = "",
) -> VGroup:
    """Create a system architecture diagram with positioned components and arrows.

    Args:
        components: list of dicts with keys:
            "id" (str), "label" (str), "pos" ([x, y]),
            "color" (str, optional), "sublabel" (str, optional),
            "shape" (str: "box" or "db", optional, default "box")
        connections: list of dicts with keys:
            "from" (str, component id), "to" (str, component id),
            "label" (str, optional), "style" (str: "arrow", "dashed", "double", optional)

    Returns VGroup with .components (dict: id->VGroup), .connections (VGroup).

    Example:
        arch = make_architecture_diagram(
            components=[
                {"id": "client", "label": "Client", "pos": [0, 2], "color": ACCENT_GREEN},
                {"id": "api", "label": "API Server", "pos": [0, 0], "color": ACCENT_BLUE},
                {"id": "db", "label": "Database", "pos": [-3, -2], "color": ACCENT_ORANGE, "shape": "db"},
                {"id": "cache", "label": "Redis Cache", "pos": [3, -2], "color": ACCENT_RED},
            ],
            connections=[
                {"from": "client", "to": "api", "label": "HTTPS"},
                {"from": "api", "to": "db", "label": "SQL"},
                {"from": "api", "to": "cache", "label": "GET/SET", "style": "dashed"},
            ],
            title="Microservice Architecture",
        )
        arch.scale(0.8).move_to(ORIGIN)
        self.play(FadeIn(arch))
    """
    comp_dict = {}
    comp_mobs = VGroup()

    for comp in components:
        cid = comp["id"]
        pos = np.array(comp["pos"] + [0]) if len(comp["pos"]) == 2 else np.array(comp["pos"])
        color = comp.get("color", ACCENT_BLUE)
        shape = comp.get("shape", "box")

        if shape == "db":
            mob = make_database_shape(
                comp["label"], color=color,
            )
        else:
            mob = make_system_block(
                comp["label"],
                sublabel=comp.get("sublabel", ""),
                color=color,
            )

        mob.move_to(pos)
        comp_dict[cid] = mob
        comp_mobs.add(mob)

    conn_mobs = VGroup()
    for conn in connections:
        src = comp_dict[conn["from"]]
        dst = comp_dict[conn["to"]]
        style = conn.get("style", "arrow")

        if style == "dashed":
            line = DashedLine(
                src.get_center(), dst.get_center(),
                color=TEXT_DIM, stroke_width=2,
                buff=0.6,
            )
            conn_mobs.add(line)
        elif style == "double":
            line = DoubleArrow(
                src.get_center(), dst.get_center(),
                buff=0.6, color=TEXT_DIM, stroke_width=2,
            )
            conn_mobs.add(line)
        else:
            arrow = Arrow(
                src.get_center(), dst.get_center(),
                buff=0.6, color=TEXT_DIM, stroke_width=2,
                max_tip_length_to_length_ratio=0.12,
            )
            conn_mobs.add(arrow)

        if conn.get("label"):
            line_mob = conn_mobs[-1]
            lbl = Text(conn["label"], font_size=12, color=TEXT_SECONDARY)
            lbl.move_to(line_mob.get_center())
            lbl.shift(UP * 0.2)
            conn_mobs.add(lbl)

    result = VGroup(conn_mobs, comp_mobs)
    result.components = comp_dict
    result.connections = conn_mobs

    if title:
        title_mob = Text(title, font_size=BODY_SIZE, color=TEXT_PRIMARY, weight="BOLD")
        title_mob.to_edge(UP, buff=0.3)
        result.add(title_mob)

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 15. ARROW PATTERNS (CurvedArrow, DoubleArrow, always_redraw, labeled)
# ═══════════════════════════════════════════════════════════════════════════════

def make_curved_connection(
    start: "Mobject | np.ndarray",
    end: "Mobject | np.ndarray",
    radius: float = 2.0,
    color: str = TEXT_DIM,
    label: str = "",
    label_font_size: int = 14,
    double: bool = False,
) -> VGroup:
    """Create a curved arrow between two points/mobjects with optional label.

    Args:
        start, end: points or Mobjects
        radius: curvature (positive = arc above, negative = arc below)
        double: if True, uses CurvedDoubleArrow

    Returns VGroup with .arrow and optionally .label.

    Example:
        box1 = make_flowchart_box("A").shift(LEFT * 3)
        box2 = make_flowchart_box("B").shift(RIGHT * 3)
        conn = make_curved_connection(box1, box2, radius=3, label="data flow")
        self.play(Create(conn))
    """
    from manim import CurvedDoubleArrow as CDA

    start_pt = start.get_center() if hasattr(start, 'get_center') else np.array(start)
    end_pt = end.get_center() if hasattr(end, 'get_center') else np.array(end)

    if double:
        arrow = CDA(start_pt, end_pt, radius=radius, color=color)
    else:
        arrow = CurvedArrow(start_pt, end_pt, radius=radius, color=color)

    result = VGroup(arrow)
    result.arrow = arrow

    if label:
        lbl = Text(label, font_size=label_font_size, color=TEXT_SECONDARY)
        lbl.move_to(arrow.point_from_proportion(0.5) + UP * 0.25)
        result.add(lbl)
        result.label = lbl

    return result


def make_double_arrow_connection(
    start_pt: "np.ndarray",
    end_pt: "np.ndarray",
    color: str = TEXT_DIM,
    label: str = "",
    label_font_size: int = 14,
    buff: float = 0.1,
) -> VGroup:
    """Create a double-headed arrow (bidirectional) with optional label.

    Example:
        darr = make_double_arrow_connection(LEFT * 3, RIGHT * 3, label="sync")
        self.play(GrowArrow(darr.arrow))
    """
    arrow = DoubleArrow(start_pt, end_pt, buff=buff, color=color, stroke_width=2)
    result = VGroup(arrow)
    result.arrow = arrow

    if label:
        lbl = Text(label, font_size=label_font_size, color=TEXT_SECONDARY)
        lbl.move_to(arrow.get_center() + UP * 0.25)
        result.add(lbl)
        result.label = lbl

    return result


def make_labeled_arrow(
    start_pt: "np.ndarray",
    end_pt: "np.ndarray",
    label: str,
    color: str = TEXT_DIM,
    font_size: int = 14,
    label_position: float = 0.5,
    buff: float = 0.1,
) -> VGroup:
    """Create an arrow with a label placed at a specified position along it.

    Uses Manim CE's LabeledArrow if available, otherwise manual placement.

    Args:
        label_position: 0.0 = at start, 1.0 = at end, 0.5 = middle

    Example:
        la = make_labeled_arrow(LEFT * 3, RIGHT * 3, "request", color=ACCENT_GREEN)
        self.play(GrowArrow(la.arrow))
        self.play(FadeIn(la.label))
    """
    arrow = Arrow(
        start_pt, end_pt, buff=buff,
        color=color, stroke_width=2,
        max_tip_length_to_length_ratio=0.15,
    )
    lbl = Text(label, font_size=font_size, color=TEXT_SECONDARY)
    point = arrow.point_from_proportion(label_position)
    lbl.move_to(point + UP * 0.2)

    result = VGroup(arrow, lbl)
    result.arrow = arrow
    result.label = lbl
    return result


def make_always_redraw_arrow(
    start_mob: "Mobject",
    end_mob: "Mobject",
    color: str = TEXT_DIM,
    buff: float = 0.1,
    stroke_width: float = 2,
) -> Arrow:
    """Create an arrow that automatically follows two mobjects using always_redraw.

    This is the key pattern for arrows that stay connected when mobjects move.

    Example:
        box1 = make_flowchart_box("A").shift(LEFT * 3)
        box2 = make_flowchart_box("B").shift(RIGHT * 3)
        arrow = make_always_redraw_arrow(box1, box2, color=ACCENT_BLUE)
        self.add(box1, box2, arrow)
        self.play(box1.animate.shift(UP * 2))  # arrow follows!
        self.play(box2.animate.shift(DOWN * 2))  # arrow follows!
    """
    return always_redraw(
        lambda: Arrow(
            start_mob.get_right(), end_mob.get_left(),
            buff=buff, color=color, stroke_width=stroke_width,
            max_tip_length_to_length_ratio=0.15,
        )
    )


def make_always_redraw_line(
    start_mob: "Mobject",
    end_mob: "Mobject",
    color: str = TEXT_DIM,
    stroke_width: float = 2,
) -> Line:
    """Create a line that automatically follows two mobjects using always_redraw.

    Example:
        dot1 = Dot(LEFT * 2)
        dot2 = Dot(RIGHT * 2)
        line = make_always_redraw_line(dot1, dot2, color=ACCENT_BLUE)
        self.add(dot1, dot2, line)
        self.play(dot1.animate.shift(UP * 2))  # line follows!
    """
    return always_redraw(
        lambda: Line(
            start_mob.get_center(), end_mob.get_center(),
            color=color, stroke_width=stroke_width,
        )
    )


def make_always_redraw_curved_arrow(
    start_mob: "Mobject",
    end_mob: "Mobject",
    radius: float = 2.0,
    color: str = TEXT_DIM,
) -> "CurvedArrow":
    """Create a CurvedArrow that auto-follows two mobjects.

    Example:
        a = Dot(LEFT * 2, color=RED)
        b = Dot(RIGHT * 2, color=BLUE)
        ca = make_always_redraw_curved_arrow(a, b, radius=3)
        self.add(a, b, ca)
        self.play(a.animate.shift(UP * 2))
    """
    return always_redraw(
        lambda: CurvedArrow(
            start_mob.get_center(), end_mob.get_center(),
            radius=radius, color=color,
        )
    )


def make_arrow_chain(
    points: list,
    color: str = TEXT_DIM,
    stroke_width: float = 2,
    curved: bool = False,
    radius: float = 2.0,
) -> VGroup:
    """Create a chain of arrows connecting a sequence of points.

    Args:
        points: list of [x, y, z] or [x, y] coordinates
        curved: if True, uses CurvedArrow between consecutive points

    Returns VGroup of arrows.

    Example:
        chain = make_arrow_chain(
            [LEFT * 3, UP * 2, RIGHT * 3, DOWN * 2],
            curved=True, radius=3,
        )
        self.play(LaggedStartMap(Create, chain, lag_ratio=0.2))
    """
    pts = [np.array(p) if len(p) == 3 else np.array(list(p) + [0]) for p in points]
    arrows = VGroup()

    for i in range(len(pts) - 1):
        if curved:
            arr = CurvedArrow(pts[i], pts[i + 1], radius=radius, color=color)
        else:
            arr = Arrow(
                pts[i], pts[i + 1],
                buff=0.1, color=color, stroke_width=stroke_width,
                max_tip_length_to_length_ratio=0.15,
            )
        arrows.add(arr)

    return arrows


# ═══════════════════════════════════════════════════════════════════════════════
# 16. MANIM TABLE METHODS - COMPREHENSIVE REFERENCE
# ═══════════════════════════════════════════════════════════════════════════════
#
# Manim CE provides 5 Table classes:
#   Table        - base class, uses Paragraph for entries
#   MathTable    - entries rendered with MathTex (LaTeX math mode)
#   MobjectTable - entries are arbitrary Mobjects (identity function)
#   IntegerTable - entries rendered with Integer
#   DecimalTable - entries rendered with DecimalNumber
#
# CONSTRUCTOR PARAMETERS (all tables):
#   table: 2D list of values
#   row_labels: list[VMobject] -- labels for each row
#   col_labels: list[VMobject] -- labels for each column
#   top_left_entry: VMobject -- corner entry (only if both row/col labels set)
#   v_buff: float = 0.8 -- vertical cell padding
#   h_buff: float = 1.3 -- horizontal cell padding
#   include_outer_lines: bool = False -- draw border lines
#   add_background_rectangles_to_entries: bool = False
#   entries_background_color: color = BLACK
#   include_background_rectangle: bool = False
#   background_rectangle_color: color = BLACK
#   element_to_mobject: Callable -- converter function for entries
#   element_to_mobject_config: dict -- kwargs for converter
#   arrange_in_grid_config: dict -- kwargs for grid layout
#   line_config: dict -- kwargs for Line objects (stroke_width, color)
#
# METHODS:
#   table.get_horizontal_lines() -> VGroup  -- all horizontal lines
#   table.get_vertical_lines() -> VGroup    -- all vertical lines
#   table.get_columns() -> VGroup of VGroup -- columns
#   table.get_rows() -> VGroup of VGroup    -- rows
#   table.get_entries(pos=None) -> VGroup or VMobject
#       pos=(row, col) 1-indexed. Returns single entry or all entries.
#   table.get_entries_without_labels(pos=None)
#   table.get_row_labels() -> VGroup
#   table.get_col_labels() -> VGroup
#   table.get_labels() -> VGroup
#   table.set_column_colors(*colors) -- color each column
#   table.set_row_colors(*colors) -- color each row
#   table.get_cell(pos=(1,1), **kwargs) -> Polygon -- cell rectangle
#   table.get_highlighted_cell(pos, color) -> BackgroundRectangle
#   table.add_highlighted_cell(pos, color) -- add highlight to cell
#   table.add_background_to_entries(color) -- add bg rect to all entries
#   table.create(lag_ratio=1, line_animation=Create, label_animation=Write,
#                element_animation=Create, entry_animation=FadeIn)
#       -> AnimationGroup  -- customizable creation animation
#   table.scale(factor) -- scales h_buff/v_buff proportionally
#
# DecimalTable EXTRAS:
#   element_to_mobject_config={"num_decimal_places": 2}
#
# IntegerTable EXTRAS:
#   element_to_mobject_config={"unit": r"^{\circ}"}  -- suffix
#
# COMPLETE TABLE USAGE EXAMPLES:
#
# Example 1: Basic Table with creation animation
#   t = Table(
#       [["95%", "5%"], ["10%", "90%"]],
#       row_labels=[Text("Cat"), Text("Dog")],
#       col_labels=[Text("Pred Cat"), Text("Pred Dog")],
#       include_outer_lines=True,
#       line_config={"stroke_width": 1, "color": GREY},
#   )
#   t.scale(0.8).move_to(ORIGIN)
#   self.play(t.create())
#
# Example 2: MathTable with colored lines
#   t = MathTable(
#       [["+", 0, 5, 10],
#        [0, 0, 5, 10],
#        [2, 2, 7, 12]],
#       include_outer_lines=True,
#   )
#   t.get_horizontal_lines()[:2].set_color(BLUE)
#   t.get_vertical_lines()[:2].set_color(BLUE)
#   self.add(t)
#
# Example 3: MobjectTable with arbitrary shapes
#   circle = Circle(color=RED, radius=0.3)
#   square = Square(side_length=0.5, color=BLUE)
#   t = MobjectTable(
#       [[circle.copy(), square.copy()],
#        [square.copy(), circle.copy()]],
#   )
#   self.play(t.create())
#
# Example 4: DecimalTable with scientific data
#   import numpy as np
#   x = np.linspace(-2, 2, 5)
#   y = np.exp(x)
#   t = DecimalTable(
#       [x, y],
#       row_labels=[MathTex("x"), MathTex("e^x")],
#       include_outer_lines=True,
#       element_to_mobject_config={"num_decimal_places": 2},
#   )
#   self.play(t.create())
#
# Example 5: IntegerTable with degree symbols
#   t = IntegerTable(
#       [[0, 30, 45, 60, 90],
#        [90, 60, 45, 30, 0]],
#       row_labels=[MathTex(r"\sin"), MathTex(r"\cos")],
#       element_to_mobject_config={"unit": r"^{\circ}"},
#   )
#   self.play(t.create())
#
# Example 6: Highlighting cells and rows
#   t = Table([["A","B"],["C","D"]], include_outer_lines=True)
#   t.add_highlighted_cell((1,1), color=GREEN)  # highlight top-left
#   t.add_highlighted_cell((2,2), color=RED)    # highlight bottom-right
#   t.set_row_colors(BLUE, YELLOW)
#   self.add(t)
#
# Example 7: Custom creation animations
#   t = Table([["X","Y"],["Z","W"]], include_outer_lines=True)
#   self.play(t.create(
#       lag_ratio=0.5,
#       line_animation=Create,
#       label_animation=Write,
#       element_animation=FadeIn,
#   ))


# ═══════════════════════════════════════════════════════════════════════════════
# COMPLETE SCENE EXAMPLES - ADVANCED PATTERNS
# ═══════════════════════════════════════════════════════════════════════════════

TREE_DIAGRAM_EXAMPLE = '''
# ── Tree Diagrams: Binary Tree, Decision Tree, Org Chart ──
from manim import *
import numpy as np
from app.manim_pipeline.styles import *
from app.manim_pipeline.diagram_patterns import *

class TreeDiagramDemo(Scene):
    def construct(self):
        self.camera.background_color = BG_COLOR

        # ── Part 1: Binary Tree ──
        title = Text("Binary Search Tree", font_size=TITLE_SIZE,
                      color=TEXT_PRIMARY, weight="BOLD")
        title.to_edge(UP, buff=0.3)
        self.play(FadeIn(title))

        tree = make_binary_tree(
            [8, 4, 12, 2, 6, 10, 14, 1, 3, 5, 7, 9, 11, 13, 15],
            node_color=ACCENT_BLUE, leaf_color=ACCENT_GREEN,
            h_spacing=1.0, v_spacing=1.0, node_size=0.3,
        )
        tree.scale(0.8).move_to(DOWN * 0.5)
        animate_tree_level_by_level(self, tree, run_time_per_level=0.4)
        self.wait(1)
        self.play(FadeOut(VGroup(title, tree)))

        # ── Part 2: Decision Tree ──
        title2 = Text("Decision Tree Classifier", font_size=TITLE_SIZE,
                       color=TEXT_PRIMARY, weight="BOLD")
        title2.to_edge(UP, buff=0.3)
        self.play(FadeIn(title2))

        dt = make_decision_tree({
            "label": "Age > 30?",
            "yes": {
                "label": "Income > 50k?",
                "yes": {"label": "Approve"},
                "no": {"label": "Review"},
            },
            "no": {
                "label": "Student?",
                "yes": {"label": "Approve"},
                "no": {"label": "Deny"},
            },
        })
        dt.scale(0.7).move_to(DOWN * 0.3)
        self.play(FadeIn(dt), run_time=1.5)
        self.wait(1)
        self.play(FadeOut(VGroup(title2, dt)))

        # ── Part 3: Org Chart ──
        title3 = Text("Organization Structure", font_size=TITLE_SIZE,
                       color=TEXT_PRIMARY, weight="BOLD")
        title3.to_edge(UP, buff=0.3)
        self.play(FadeIn(title3))

        org = make_org_chart({
            "label": "CEO",
            "children": [
                {"label": "CTO", "children": [
                    {"label": "Frontend"},
                    {"label": "Backend"},
                    {"label": "DevOps"},
                ]},
                {"label": "CFO", "children": [
                    {"label": "Accounting"},
                ]},
                {"label": "CMO", "children": [
                    {"label": "Marketing"},
                    {"label": "Sales"},
                ]},
            ],
        })
        org.scale(0.6).move_to(DOWN * 0.3)
        self.play(FadeIn(org), run_time=1.5)
        self.wait(1)
        self.play(FadeOut(VGroup(title3, org)))
'''

GRAPH_NETWORK_EXAMPLE = '''
# ── Graph/Network Layouts with Manim CE built-in Graph ──
from manim import *
import numpy as np
from app.manim_pipeline.styles import *
from app.manim_pipeline.diagram_patterns import *

class GraphNetworkDemo(Scene):
    def construct(self):
        self.camera.background_color = BG_COLOR

        # ── Undirected Graph with spring layout ──
        title = Text("Graph Layouts", font_size=TITLE_SIZE,
                      color=TEXT_PRIMARY, weight="BOLD")
        title.to_edge(UP, buff=0.3)
        self.play(FadeIn(title))

        g = Graph(
            [1, 2, 3, 4, 5, 6],
            [(1,2),(2,3),(3,4),(4,5),(5,6),(6,1),(1,3),(3,5)],
            layout="spring", labels=True, layout_scale=2,
        )
        g.move_to(DOWN * 0.5)
        self.play(Create(g), run_time=2)
        self.wait(0.5)

        # Animate layout change
        self.play(g.animate.change_layout("circular"), run_time=1.5)
        self.wait(0.5)

        # Move individual vertices
        self.play(
            g[1].animate.move_to(UP * 2),
            g[4].animate.move_to(DOWN * 2),
            run_time=1,
        )
        self.wait(0.5)
        self.play(FadeOut(VGroup(title, g)))

        # ── Directed Graph (DiGraph) ──
        title2 = Text("Directed Graph", font_size=TITLE_SIZE,
                       color=TEXT_PRIMARY, weight="BOLD")
        title2.to_edge(UP, buff=0.3)
        self.play(FadeIn(title2))

        dg = DiGraph(
            ["A", "B", "C", "D"],
            [("A","B"), ("B","C"), ("C","D"), ("D","A"), ("A","C")],
            layout="circular", labels=True, layout_scale=2,
            edge_config={"stroke_width": 2, "tip_config": {"tip_length": 0.2}},
        )
        dg.move_to(DOWN * 0.5)
        self.play(Create(dg), run_time=2)
        self.wait(1)
        self.play(FadeOut(VGroup(title2, dg)))

        # ── Neural Network (partite layout) ──
        title3 = Text("Neural Network", font_size=TITLE_SIZE,
                       color=TEXT_PRIMARY, weight="BOLD")
        title3.to_edge(UP, buff=0.3)
        self.play(FadeIn(title3))

        nn = make_neural_network_graph([3, 5, 5, 2])
        nn.scale(0.8).move_to(DOWN * 0.5)
        self.play(Create(nn), run_time=2)
        self.wait(1)
        self.play(FadeOut(VGroup(title3, nn)))

        # ── Tree Layout ──
        title4 = Text("Tree Graph", font_size=TITLE_SIZE,
                       color=TEXT_PRIMARY, weight="BOLD")
        title4.to_edge(UP, buff=0.3)
        self.play(FadeIn(title4))

        tg = Graph(
            [1,2,3,4,5,6,7],
            [(1,2),(1,3),(2,4),(2,5),(3,6),(3,7)],
            layout="tree",
            layout_config={"root_vertex": 1},
            labels=True,
            layout_scale=2.5,
        )
        tg.move_to(DOWN * 0.5)
        self.play(Create(tg), run_time=2)
        self.wait(1)
        self.play(FadeOut(VGroup(title4, tg)))
'''

STATE_MACHINE_EXAMPLE = '''
# ── State Machine / Automaton ──
from manim import *
import numpy as np
from app.manim_pipeline.styles import *
from app.manim_pipeline.diagram_patterns import *

class StateMachineDemo(Scene):
    def construct(self):
        self.camera.background_color = BG_COLOR

        title = Text("Finite State Machine", font_size=TITLE_SIZE,
                      color=TEXT_PRIMARY, weight="BOLD")
        title.to_edge(UP, buff=0.3)
        self.play(FadeIn(title))

        sm = make_state_machine(
            states=[
                {"name": "q0", "start": True},
                {"name": "q1"},
                {"name": "q2", "accept": True},
            ],
            transitions=[
                {"from": "q0", "to": "q1", "label": "a"},
                {"from": "q1", "to": "q2", "label": "b"},
                {"from": "q2", "to": "q0", "label": "c"},
                {"from": "q1", "to": "q1", "label": "a"},
            ],
            layout="circular",
            layout_scale=2.0,
        )
        sm.move_to(DOWN * 0.5)
        self.play(FadeIn(sm), run_time=1.5)
        self.wait(0.5)

        # Animate traversal: input "aab"
        animate_state_transition(self, sm, ["q0", "q1", "q1", "q2"])
        self.wait(1)
        self.play(FadeOut(VGroup(title, sm)))
'''

VENN_DIAGRAM_EXAMPLE = '''
# ── Venn Diagrams ──
from manim import *
import numpy as np
from app.manim_pipeline.styles import *
from app.manim_pipeline.diagram_patterns import *

class VennDiagramDemo(Scene):
    def construct(self):
        self.camera.background_color = BG_COLOR

        # ── 2-circle Venn ──
        title = Text("Venn Diagrams", font_size=TITLE_SIZE,
                      color=TEXT_PRIMARY, weight="BOLD")
        title.to_edge(UP, buff=0.3)
        self.play(FadeIn(title))

        venn = make_venn_2("ML", "Stats", "Data Science",
                           color_a=ACCENT_BLUE, color_b=ACCENT_GREEN)
        venn.move_to(DOWN * 0.5)
        self.play(FadeIn(venn.circle_a), FadeIn(venn.circle_b), run_time=1)
        self.play(FadeIn(venn.label_a_mob), FadeIn(venn.label_b_mob), run_time=0.5)
        if hasattr(venn, 'label_ab_mob'):
            self.play(Write(venn.label_ab_mob), run_time=0.5)
        self.wait(0.5)

        animate_venn_highlight_intersection(self, venn)
        self.wait(0.5)
        self.play(FadeOut(venn))

        # ── 3-circle Venn ──
        venn3 = make_venn_3(
            ["AI", "Statistics", "Engineering"],
            [ACCENT_BLUE, ACCENT_GREEN, ACCENT_ORANGE],
        )
        venn3.move_to(DOWN * 0.5)
        self.play(FadeIn(venn3), run_time=1.5)
        self.wait(1)
        self.play(FadeOut(VGroup(title, venn3)))

        # ── Boolean Venn using Manim CE Intersection/Union/Difference ──
        # These are built-in Manim CE classes for actual shape operations:
        #   from manim import Intersection, Union, Difference, Exclusion
        title2 = Text("Set Operations", font_size=TITLE_SIZE,
                       color=TEXT_PRIMARY, weight="BOLD")
        title2.to_edge(UP, buff=0.3)
        self.play(FadeIn(title2))

        a = Circle(radius=1.5, color=BLUE, fill_opacity=0.5).shift(LEFT * 0.7)
        b = Circle(radius=1.5, color=RED, fill_opacity=0.5).shift(RIGHT * 0.7)
        inter = Intersection(a, b, color=GREEN, fill_opacity=0.8)
        inter.move_to(DOWN * 0.5)
        self.play(FadeIn(a.copy().move_to(DOWN*0.5 + LEFT*0.7)),
                  FadeIn(b.copy().move_to(DOWN*0.5 + RIGHT*0.7)),
                  run_time=0.5)
        self.play(FadeIn(inter), run_time=0.5)
        self.wait(1)
        self.play(FadeOut(VGroup(title2), *self.mobjects))
'''

CHART_EXAMPLE = '''
# ── Bar Chart, Line Chart, Pie Chart ──
from manim import *
import numpy as np
from app.manim_pipeline.styles import *
from app.manim_pipeline.diagram_patterns import *

class ChartDemo(Scene):
    def construct(self):
        self.camera.background_color = BG_COLOR

        # ── Part 1: Bar Chart ──
        title = Text("Performance Metrics", font_size=TITLE_SIZE,
                      color=TEXT_PRIMARY, weight="BOLD")
        title.to_edge(UP, buff=0.3)
        self.play(FadeIn(title))

        chart = make_bar_chart(
            [85, 92, 78, 95, 88],
            bar_names=["Model A", "Model B", "Model C", "Model D", "Model E"],
            y_range=[0, 100, 20],
        )
        chart.scale(0.7).move_to(DOWN * 0.3)
        self.play(FadeIn(chart), run_time=1.5)
        labels = chart.get_bar_labels(font_size=20)
        self.play(FadeIn(labels), run_time=0.5)
        self.wait(1)

        # Animate value change
        chart.change_bar_values([90, 85, 95, 80, 92])
        new_labels = chart.get_bar_labels(font_size=20)
        self.play(Transform(labels, new_labels), run_time=1)
        self.wait(0.5)
        self.play(FadeOut(VGroup(title, chart, labels)))

        # ── Part 2: Line Chart ──
        title2 = Text("Training Progress", font_size=TITLE_SIZE,
                       color=TEXT_PRIMARY, weight="BOLD")
        title2.to_edge(UP, buff=0.3)
        self.play(FadeIn(title2))

        lc = make_line_chart(
            [1, 2, 3, 4, 5, 6, 7, 8],
            [[0.9, 0.7, 0.5, 0.35, 0.25, 0.18, 0.12, 0.08],
             [0.95, 0.85, 0.7, 0.55, 0.42, 0.35, 0.28, 0.22]],
            line_colors=[ACCENT_BLUE, ACCENT_ORANGE],
            line_labels=["Train Loss", "Val Loss"],
            x_label="Epoch", y_label="Loss",
            x_range=[1, 8, 1], y_range=[0, 1, 0.2],
        )
        lc.scale(0.7).move_to(DOWN * 0.3)
        self.play(FadeIn(lc.axes), run_time=0.5)
        for line in lc.lines:
            self.play(Create(line), run_time=1)
        self.play(FadeIn(lc.dots), run_time=0.3)
        if hasattr(lc, 'legend'):
            self.play(FadeIn(lc.legend), run_time=0.3)
        self.wait(1)
        self.play(FadeOut(VGroup(title2, lc)))

        # ── Part 3: Pie Chart ──
        title3 = Text("Market Share", font_size=TITLE_SIZE,
                       color=TEXT_PRIMARY, weight="BOLD")
        title3.to_edge(UP, buff=0.3)
        self.play(FadeIn(title3))

        pie = make_pie_chart(
            [45, 25, 18, 12],
            labels=["Product A", "Product B", "Product C", "Other"],
            colors=[ACCENT_BLUE, ACCENT_GREEN, ACCENT_ORANGE, ACCENT_RED],
        )
        pie.scale(0.8).move_to(DOWN * 0.3)
        self.play(LaggedStartMap(FadeIn, pie.slices, lag_ratio=0.2), run_time=1.5)
        self.play(FadeIn(pie.label_mobs), run_time=0.5)
        self.wait(1)
        self.play(FadeOut(VGroup(title3, pie)))
'''

HEATMAP_EXAMPLE = '''
# ── Heatmap Visualization ──
from manim import *
import numpy as np
from app.manim_pipeline.styles import *
from app.manim_pipeline.diagram_patterns import *

class HeatmapDemo(Scene):
    def construct(self):
        self.camera.background_color = BG_COLOR

        title = Text("Attention Heatmap", font_size=TITLE_SIZE,
                      color=TEXT_PRIMARY, weight="BOLD")
        title.to_edge(UP, buff=0.3)
        self.play(FadeIn(title))

        hm = make_heatmap(
            [[0.9, 0.1, 0.2, 0.05],
             [0.1, 0.8, 0.3, 0.1],
             [0.2, 0.3, 0.7, 0.2],
             [0.05, 0.1, 0.2, 0.85]],
            row_labels=["The", "cat", "sat", "down"],
            col_labels=["The", "cat", "sat", "down"],
        )
        hm.scale(0.9).move_to(DOWN * 0.3)
        animate_heatmap_reveal(self, hm, run_time=2)
        self.wait(1)
        self.play(FadeOut(VGroup(title, hm)))
'''

GANTT_CHART_EXAMPLE = '''
# ── Gantt Chart ──
from manim import *
import numpy as np
from app.manim_pipeline.styles import *
from app.manim_pipeline.diagram_patterns import *

class GanttChartDemo(Scene):
    def construct(self):
        self.camera.background_color = BG_COLOR

        title = Text("Project Timeline", font_size=TITLE_SIZE,
                      color=TEXT_PRIMARY, weight="BOLD")
        title.to_edge(UP, buff=0.3)
        self.play(FadeIn(title))

        gantt = make_gantt_chart([
            {"label": "Research", "start": 0, "end": 3, "color": ACCENT_BLUE},
            {"label": "Design", "start": 2, "end": 5, "color": ACCENT_GREEN},
            {"label": "Develop", "start": 4, "end": 8, "color": ACCENT_ORANGE},
            {"label": "Testing", "start": 7, "end": 9, "color": ACCENT_RED},
            {"label": "Deploy", "start": 9, "end": 10, "color": ACCENT_PURPLE},
        ], time_range=(0, 10))
        gantt.scale(0.8).move_to(DOWN * 0.3)

        self.play(FadeIn(VGroup(gantt.axis_line, gantt.tick_labels, gantt.labels)), run_time=0.5)
        animate_gantt_fill(self, gantt, run_time_per_bar=0.4)
        self.wait(1)
        self.play(FadeOut(VGroup(title, gantt)))
'''

ARCHITECTURE_EXAMPLE = '''
# ── System Architecture Diagram ──
from manim import *
import numpy as np
from app.manim_pipeline.styles import *
from app.manim_pipeline.diagram_patterns import *

class ArchitectureDemo(Scene):
    def construct(self):
        self.camera.background_color = BG_COLOR

        arch = make_architecture_diagram(
            components=[
                {"id": "client", "label": "Mobile App", "pos": [0, 3], "color": ACCENT_GREEN},
                {"id": "lb", "label": "Load Balancer", "pos": [0, 1.5], "color": ACCENT_CYAN},
                {"id": "api1", "label": "API Server 1", "pos": [-3, 0], "color": ACCENT_BLUE},
                {"id": "api2", "label": "API Server 2", "pos": [3, 0], "color": ACCENT_BLUE},
                {"id": "cache", "label": "Redis", "pos": [0, -1.5], "color": ACCENT_RED, "sublabel": "Cache"},
                {"id": "db", "label": "PostgreSQL", "pos": [-3, -3], "color": ACCENT_ORANGE, "shape": "db"},
                {"id": "queue", "label": "RabbitMQ", "pos": [3, -3], "color": ACCENT_PURPLE, "sublabel": "Message Queue"},
            ],
            connections=[
                {"from": "client", "to": "lb", "label": "HTTPS"},
                {"from": "lb", "to": "api1", "label": ""},
                {"from": "lb", "to": "api2", "label": ""},
                {"from": "api1", "to": "cache", "label": "GET/SET"},
                {"from": "api2", "to": "cache", "label": "GET/SET"},
                {"from": "api1", "to": "db", "label": "SQL"},
                {"from": "api2", "to": "db", "label": "SQL"},
                {"from": "api1", "to": "queue", "label": "publish", "style": "dashed"},
                {"from": "api2", "to": "queue", "label": "publish", "style": "dashed"},
            ],
            title="Microservice Architecture",
        )
        arch.scale(0.7).move_to(DOWN * 0.2)
        self.play(FadeIn(arch), run_time=2)
        self.wait(2)
        self.play(FadeOut(arch))
'''

ARROW_PATTERNS_EXAMPLE = '''
# ── Arrow Patterns: Every Variation ──
from manim import *
import numpy as np
from app.manim_pipeline.styles import *
from app.manim_pipeline.diagram_patterns import *

class ArrowPatternsDemo(Scene):
    def construct(self):
        self.camera.background_color = BG_COLOR

        title = Text("Arrow Patterns Reference", font_size=TITLE_SIZE,
                      color=TEXT_PRIMARY, weight="BOLD")
        title.to_edge(UP, buff=0.3)
        self.play(FadeIn(title))

        # ── 1. Basic Arrow ──
        arr1 = Arrow(LEFT * 3, RIGHT * 0, color=ACCENT_BLUE, stroke_width=3)
        lbl1 = Text("Arrow", font_size=14, color=TEXT_SECONDARY)
        lbl1.next_to(arr1, UP, buff=0.1)

        # ── 2. DoubleArrow (bidirectional) ──
        arr2 = DoubleArrow(LEFT * 3, RIGHT * 0, color=ACCENT_GREEN, stroke_width=3)
        lbl2 = Text("DoubleArrow", font_size=14, color=TEXT_SECONDARY)
        lbl2.next_to(arr2, UP, buff=0.1)

        # ── 3. CurvedArrow ──
        arr3 = CurvedArrow(LEFT * 3, RIGHT * 0, radius=3, color=ACCENT_ORANGE)
        lbl3 = Text("CurvedArrow", font_size=14, color=TEXT_SECONDARY)
        lbl3.next_to(arr3, UP, buff=0.1)

        # ── 4. CurvedDoubleArrow ──
        arr4 = CurvedDoubleArrow(LEFT * 3, RIGHT * 0, color=ACCENT_RED)
        lbl4 = Text("CurvedDoubleArrow", font_size=14, color=TEXT_SECONDARY)
        lbl4.next_to(arr4, UP, buff=0.1)

        # ── 5. DashedLine ──
        arr5 = DashedLine(LEFT * 3, RIGHT * 0, color=ACCENT_PURPLE, stroke_width=3)
        lbl5 = Text("DashedLine", font_size=14, color=TEXT_SECONDARY)
        lbl5.next_to(arr5, UP, buff=0.1)

        all_arrows = VGroup(
            VGroup(arr1, lbl1),
            VGroup(arr2, lbl2),
            VGroup(arr3, lbl3),
            VGroup(arr4, lbl4),
            VGroup(arr5, lbl5),
        )
        all_arrows.arrange(DOWN, buff=0.6).move_to(DOWN * 0.3)
        for grp in all_arrows:
            grp.shift(RIGHT * 1.5)

        self.play(
            LaggedStart(*[FadeIn(g) for g in all_arrows], lag_ratio=0.2),
            run_time=2,
        )
        self.wait(1)
        self.play(FadeOut(VGroup(title, all_arrows)))

        # ── 6. always_redraw Arrow (follows moving mobjects) ──
        title2 = Text("always_redraw Arrow", font_size=BODY_SIZE,
                       color=TEXT_PRIMARY, weight="BOLD")
        title2.to_edge(UP, buff=0.3)
        self.play(FadeIn(title2))

        box_a = make_flowchart_box("Source", color=ACCENT_BLUE).shift(LEFT * 3)
        box_b = make_flowchart_box("Target", color=ACCENT_GREEN).shift(RIGHT * 3)
        auto_arrow = make_always_redraw_arrow(box_a, box_b, color=ACCENT_CYAN)
        self.add(box_a, box_b, auto_arrow)
        self.play(FadeIn(VGroup(box_a, box_b)), run_time=0.5)
        self.wait(0.3)

        # Move boxes -- arrow follows automatically!
        self.play(box_a.animate.shift(UP * 2), run_time=0.8)
        self.play(box_b.animate.shift(DOWN * 2), run_time=0.8)
        self.play(
            box_a.animate.move_to(LEFT * 2 + DOWN),
            box_b.animate.move_to(RIGHT * 2 + UP),
            run_time=1,
        )
        self.wait(0.5)
        self.play(FadeOut(VGroup(title2, box_a, box_b, auto_arrow)))

        # ── 7. Arrow tip shapes ──
        title3 = Text("Arrow Tip Shapes", font_size=BODY_SIZE,
                       color=TEXT_PRIMARY, weight="BOLD")
        title3.to_edge(UP, buff=0.3)
        self.play(FadeIn(title3))

        from manim.mobject.geometry.tips import (
            ArrowTriangleTip, ArrowTriangleFilledTip,
            ArrowCircleTip, ArrowCircleFilledTip,
            ArrowSquareTip, ArrowSquareFilledTip,
            StealthTip,
        )

        tip_arrows = VGroup(
            Arrow(LEFT*3, RIGHT*0, tip_shape=ArrowTriangleFilledTip, color=BLUE),
            Arrow(LEFT*3, RIGHT*0, tip_shape=ArrowTriangleTip, color=GREEN),
            Arrow(LEFT*3, RIGHT*0, tip_shape=ArrowCircleFilledTip, color=YELLOW),
            Arrow(LEFT*3, RIGHT*0, tip_shape=ArrowCircleTip, color=ORANGE),
            Arrow(LEFT*3, RIGHT*0, tip_shape=ArrowSquareFilledTip, color=RED),
            Arrow(LEFT*3, RIGHT*0, tip_shape=ArrowSquareTip, color=PURPLE),
            Arrow(LEFT*3, RIGHT*0, tip_shape=StealthTip, color=TEAL),
        )
        tip_labels = VGroup(
            Text("TriangleFilled (default)", font_size=12, color=TEXT_SECONDARY),
            Text("Triangle (outline)", font_size=12, color=TEXT_SECONDARY),
            Text("CircleFilled", font_size=12, color=TEXT_SECONDARY),
            Text("Circle (outline)", font_size=12, color=TEXT_SECONDARY),
            Text("SquareFilled", font_size=12, color=TEXT_SECONDARY),
            Text("Square (outline)", font_size=12, color=TEXT_SECONDARY),
            Text("StealthTip", font_size=12, color=TEXT_SECONDARY),
        )

        for arr, lbl in zip(tip_arrows, tip_labels):
            lbl.next_to(arr, LEFT, buff=0.2)

        tip_group = VGroup(*[VGroup(a, l) for a, l in zip(tip_arrows, tip_labels)])
        tip_group.arrange(DOWN, buff=0.3).move_to(DOWN * 0.3)

        self.play(LaggedStartMap(FadeIn, tip_group, lag_ratio=0.1), run_time=2)
        self.wait(1)
        self.play(FadeOut(VGroup(title3, tip_group)))

        # ── 8. LabeledArrow and LabeledLine ──
        title4 = Text("Labeled Arrows", font_size=BODY_SIZE,
                       color=TEXT_PRIMARY, weight="BOLD")
        title4.to_edge(UP, buff=0.3)
        self.play(FadeIn(title4))

        from manim import LabeledArrow, LabeledLine
        la = LabeledArrow("0.5", start=LEFT*4, end=RIGHT*4, label_position=0.5,
                           label_config={"font_size": 20})
        la.shift(UP * 0.5)
        ll = LabeledLine("midpoint", start=LEFT*4+DOWN*1.5, end=RIGHT*4+DOWN*1.5,
                          label_position=0.5, label_config={"font_size": 20})

        self.play(Create(la), run_time=1)
        self.play(Create(ll), run_time=1)
        self.wait(1)
        self.play(FadeOut(VGroup(title4, la, ll)))
'''
