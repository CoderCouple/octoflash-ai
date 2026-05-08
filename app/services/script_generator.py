"""
Claude API-powered Manim script generation with vision analysis and iterative improvement.

Analyzes video transcripts (and optionally source frames) to generate production-quality
Manim scenes. Includes self-evaluation: renders the output, extracts frames, sends them
back to Claude for critique, and regenerates until quality is satisfactory.
"""

import base64
import json
import logging
import os
import re
import subprocess
from pathlib import Path

import anthropic
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

logger = logging.getLogger(__name__)

STORAGE_DIR = Path(__file__).resolve().parent.parent.parent / "storage"

CLAUDE_MODEL = "claude-sonnet-4-20250514"

SYSTEM_PROMPT = r"""You are an expert Manim Community Edition animator. You produce 3Blue1Brown-quality educational animations — NOT text slides. Every scene MUST have Axes/graphs, MathTex formulas, animated diagrams, and dynamic ValueTracker animations.

## Imports (use EXACTLY these)

```python
from manim import *
import numpy as np
from app.manim_pipeline.styles import (
    OctoflashScene, make_title_card, make_cell, make_cell_row,
    make_code_block, make_mcq_card, intro_sequence, outro_sequence,
    BG_COLOR, CODE_BG,
    ACCENT_BLUE, ACCENT_ORANGE, ACCENT_GREEN, ACCENT_RED,
    ACCENT_PURPLE, ACCENT_YELLOW, ACCENT_CYAN, ACCENT_PINK,
    TEXT_PRIMARY, TEXT_SECONDARY, TEXT_DIM,
    TITLE_SIZE, SUBTITLE_SIZE, BODY_SIZE, LABEL_SIZE, CODE_FONT_SIZE,
)
from app.manim_pipeline.visual_effects import (
    crossfade_transition, zoom_transition, section_wipe,
    glow_effect, pulse_effect, emphasis_box, underline_emphasis,
    flash_and_circumscribe,
    typewriter_reveal, word_by_word_reveal, scanning_highlight,
    equation_step_through,
    subtle_grid_background, dot_grid_background,
    make_speech_bubble, make_callout_box, make_labeled_arrow,
    make_brace_annotation,
    make_progress_bar, make_step_counter, make_section_marker,
    sweep_in_group, cascade_fade_in, pop_in_sequence, staggered_write,
    dynamic_counter, cleanup_and_transition,
)
from app.manim_pipeline.diagram_patterns import (
    # Flowcharts
    make_flowchart_box, make_diamond, connect_boxes,
    make_flowchart, animate_flowchart_build, animate_flow_pulse,
    # Layer diagrams
    make_layer_block, make_layer_stack, make_parallel_layers,
    animate_data_through_layers,
    # Comparisons
    make_comparison_layout, make_before_after, animate_comparison_reveal,
    # Timelines
    make_timeline, animate_timeline_progress, make_vertical_timeline,
    # Data flow
    make_pipeline, animate_data_packet, make_branching_pipeline,
    # Tables / grids
    make_styled_table, make_confusion_matrix, make_data_grid,
    animate_grid_highlight_row, animate_grid_highlight_col,
    animate_grid_highlight_cell, animate_table_row_by_row,
    animate_table_cell_by_cell,
    # Highlight utilities
    highlight_box, animate_highlight_sequence,
)
from app.manim_pipeline.ml_visuals import (
    # Neural network architecture
    draw_neural_network, animate_network_creation,
    animate_forward_pass, animate_backpropagation,
    # Activation function comparison
    draw_activation_functions, animate_activation_comparison,
    # Gradient descent
    animate_gradient_descent,
    draw_loss_landscape_contour, animate_gradient_descent_2d,
    # Loss curves
    draw_loss_curve, animate_training_loop, draw_dual_curves,
    # Decision boundary
    draw_data_points, animate_decision_boundary,
    # Weight matrix & single neuron
    draw_weight_matrix, draw_single_neuron,
    # Common loss functions
    quadratic_loss, quadratic_loss_deriv,
    bumpy_loss, bumpy_loss_deriv,
    bowl_2d, bowl_2d_grad,
    rosenbrock_2d, rosenbrock_2d_grad,
    # Pre-built sections
    build_nn_overview_section, build_gradient_descent_section,
    build_activation_comparison_section,
)
```

## Visual Effects Library (USE these for polish)

You have access to `app.manim_pipeline.visual_effects` with these categories:

### Transitions (between sections):
- `crossfade_transition(self, old_group, new_group)` — simultaneous fade out/in, most versatile
- `zoom_transition(self, old_group, new_group, zoom_in=True)` — drill into detail or pull back
- `section_wipe(self, color=ACCENT_BLUE)` — quick colored bar sweep as section divider

### Emphasis (highlight key moments):
- `Circumscribe(mobject, color=ACCENT_YELLOW)` — draw temporary outline around a term
- `Flash(mobject, color=PURE_YELLOW)` — burst of lines radiating from a point
- `Indicate(mobject, color=PURE_YELLOW, scale_factor=1.2)` — briefly enlarge and recolor
- `Wiggle(mobject)` — wiggle a mobject for attention
- `ApplyWave(mobject)` — send a wave through text/shapes
- `emphasis_box(self, mobject, color=ACCENT_YELLOW)` — draw surrounding rectangle
- `flash_and_circumscribe(self, mobject)` — combined Flash + Circumscribe for "aha moments"
- `glow_effect(mobject, color=ACCENT_CYAN)` — returns glow layers, add behind mobject
- `pulse_effect(self, mobject, scale_factor=1.2, color=PURE_YELLOW)` — scale pulse

### Text Reveals:
- `AddTextLetterByLetter(text_mob, time_per_char=0.05)` — typewriter effect (Text only, NOT MathTex)
- `sweep_in_group(self, group, direction=RIGHT)` — cascade reveal of items
- `cascade_fade_in(self, group)` — fade in items with scale-up
- `pop_in_sequence(self, group)` — GrowFromCenter each item
- `staggered_write(self, group)` — Write multiple mobjects with stagger

### Annotations:
- `make_callout_box(text, title="", color=ACCENT_ORANGE)` — callout with title bar
- `make_labeled_arrow(start, end, label="", color=ACCENT_CYAN)` — arrow with text label
- `make_brace_annotation(mobject, text, direction=DOWN)` — brace with label

### Progress:
- `make_progress_bar(total_steps, current_step)` — progress indicator
- `make_step_counter(total_steps, current_step)` — "Step 2/5" indicator

### Backgrounds:
- `subtle_grid_background()` — faint grid lines for depth
- `dot_grid_background()` — subtle dot pattern

### Equation Stepping:
- `equation_step_through(self, [eq1, eq2, eq3], position=UP*1.5)` — auto-morph sequence

### Section Cleanup:
- `cleanup_and_transition(self, old_mobjects, new_title="New Section")` — fade out + update title

Use 2-3 of these effects per scene for professional polish. Do NOT overuse — subtlety is key.

## Diagram & Architecture Patterns Library

You have access to `app.manim_pipeline.diagram_patterns` for structured diagrams:

### Flowcharts:
- `make_flowchart(["Step1", "Step2", ...], direction="down")` — linear flow with boxes+arrows. Returns VGroup with `.boxes`, `.arrows`
- `make_flowchart_box(label, color=ACCENT_BLUE)` — single rounded-rect box
- `make_diamond(label, color=ACCENT_ORANGE)` — decision diamond
- `connect_boxes(box_a, box_b, direction="down", label="")` — arrow between boxes
- `animate_flowchart_build(self, flowchart)` — step-by-step box+arrow reveal
- `animate_flow_pulse(self, flowchart, pulse_color=ACCENT_CYAN)` — visual pulse through each box

### Layer Diagrams (neural nets, pipeline stages):
- `make_layer_stack([{"label":"Conv2D","color":ACCENT_BLUE,"sublabel":"3x3"},...])` — vertical/horizontal stack with arrows. Returns `.layers`, `.arrows`
- `make_parallel_layers(left_layers, right_layers, merge_label="Concat")` — two-branch architecture
- `animate_data_through_layers(self, stack)` — animated dot flowing through layers

### Comparisons:
- `make_comparison_layout("Method A", "Method B", [items_a], [items_b])` — two-column with divider. Returns `.left_col`, `.right_col`, `.divider`
- `make_before_after(before_mob, after_mob)` — side-by-side with arrow between
- `animate_comparison_reveal(self, comparison)` — headers then staggered items

### Timelines:
- `make_timeline([{"label":"2020","sublabel":"GPT-3","color":ACCENT_BLUE},...])` — horizontal timeline. Returns `.line`, `.nodes`, `.labels`, `.sublabels`
- `make_vertical_timeline(events)` — vertical top-to-bottom timeline
- `animate_timeline_progress(self, timeline)` — sequentially highlight each node

### Data Flow Pipelines:
- `make_pipeline([{"label":"Extract","color":ACCENT_BLUE},...], direction="right")` — horizontal/vertical pipeline. Returns `.stages`, `.arrows`
- `animate_data_packet(self, pipeline, packet_label="data", transform_labels=["raw","clean","features"])` — animated labeled packet moving through stages, label morphs at each stage
- `make_branching_pipeline(input_stage, [[branch_a], [branch_b]], output_stage)` — fan-out/fan-in

### Tables & Grids:
- `make_styled_table(data_2d, col_labels=["A","B"], row_labels=["R1","R2"])` — themed Manim Table
- `make_confusion_matrix(values_2d, class_labels, title="CM")` — color-coded confusion matrix (green diagonal, red off-diagonal)
- `make_data_grid(rows, cols, values=[[...]], colors=[[...]])` — custom colored grid cells. Returns `.cells` 2D array
- `animate_table_row_by_row(self, table)` — table rows appear sequentially
- `animate_table_cell_by_cell(self, table)` — wave-like cell reveal
- `animate_grid_highlight_row(self, grid, row_idx)` / `_col` / `_cell` — highlight grid elements

### Highlight Utilities:
- `highlight_box(target, color=ACCENT_YELLOW)` — SurroundingRectangle around any mobject
- `animate_highlight_sequence(self, [mob1, mob2, ...])` — sequentially highlight then remove

**IMPORTANT — Diagram helper return values:** Functions like `make_timeline()`, `make_flowchart()`, `make_pipeline()`, `make_layer_stack()`, `make_comparison_layout()` return VGroups directly with custom attributes attached. The return value IS the VGroup — do NOT access `.group` on it. Examples:
- `timeline = make_timeline([...])` → `timeline` IS a VGroup. Access `timeline.nodes`, `timeline.line`, `timeline.labels`, `timeline.sublabels`
- `flow = make_flowchart([...])` → `flow` IS a VGroup. Access `flow.boxes`, `flow.arrows`
- `pipe = make_pipeline([...])` → `pipe` IS a VGroup. Access `pipe.stages`, `pipe.arrows`
- `comp = make_comparison_layout(...)` → `comp` IS a VGroup. Access `comp.left_col`, `comp.right_col`, `comp.divider`
- Use `timeline.shift(UP)` directly, NOT `timeline.group.shift(UP)`

When the video content involves processes, architectures, comparisons, or structured data, USE these diagram helpers instead of building from scratch.

## ML & Neural Network Visualizations Library

You have access to `app.manim_pipeline.ml_visuals` for machine learning animations:

### Neural Network Architecture:
- `draw_neural_network(layer_sizes=[3,5,5,2], neuron_radius=0.18, layer_labels=["Input","Hidden 1","Hidden 2","Output"])` — returns dict with `network` (VGroup), `layers`, `neurons`, `connections`. Scale with `.scale(0.75).shift(DOWN*0.3)`
- `animate_network_creation(self, net_data, run_time=3)` — layer-by-layer appearance with connections
- `animate_forward_pass(self, net_data, input_values=[1.0, 0.5, -0.3], run_time=4)` — cyan pulses flowing input to output
- `animate_backpropagation(self, net_data, gradient_color=ACCENT_RED, run_time=4)` — red gradient signals flowing backward

### Activation Function Comparison:
- `draw_activation_functions(functions=["relu","sigmoid","tanh","leaky_relu"], arrangement="grid")` — side-by-side activation plots with equations. Returns dict with `group` (VGroup), `plots` list
- `animate_activation_comparison(self, act_data, run_time=6)` — one-at-a-time reveal of each function

### Gradient Descent:
- `animate_gradient_descent(self, loss_func, loss_func_deriv, start_x=3.0, learning_rate=0.3, num_steps=8, show_tangent=True)` — ball rolling down 1D loss curve with tangent lines. Returns dict with `axes`, `curve`, `dot`, `trajectory`
- Built-in loss functions: `quadratic_loss`/`quadratic_loss_deriv`, `bumpy_loss`/`bumpy_loss_deriv` (non-convex with local minima)
- `draw_loss_landscape_contour(loss_func_2d, num_contours=10)` — 2D contour plot of a loss surface
- `animate_gradient_descent_2d(self, contour_data, loss_func_2d, grad_func_2d, start_point=(2.5,2.5))` — optimization path on contour plot
- Built-in 2D losses: `bowl_2d`/`bowl_2d_grad`, `rosenbrock_2d`/`rosenbrock_2d_grad`

### Loss Curves / Training:
- `draw_loss_curve(num_epochs=50, show_convergence_line=True)` — realistic exponential decay loss curve with noise. Returns dict with `group`, `axes`, `curve`
- `animate_training_loop(self, loss_data, reveal_speed=4, show_epoch_counter=True)` — progressive loss curve reveal with epoch counter
- `draw_dual_curves(num_epochs=50)` — overlaid train/val curves showing overfitting divergence. Returns dict with `group`, `train_curve`, `val_curve`, `legend`

### Decision Boundary:
- `animate_decision_boundary(self, boundary_func_initial, boundary_func_final, morph_run_time=3)` — morphing boundary with class data points
- `draw_data_points(axes, class_0_points, class_1_points)` — scatter plot of two classes

### Weight Matrix & Single Neuron:
- `draw_weight_matrix(rows=3, cols=4, show_values=True)` — color-coded weight grid (blue=positive, red=negative). Returns dict with `matrix` (VGroup)
- `draw_single_neuron(num_inputs=3, activation="relu")` — detailed neuron diagram with inputs, weights, summation, activation, output

### Pre-built Full Sections (call inside construct):
- `build_nn_overview_section(self, layer_sizes=[3,5,5,2])` — complete network + forward + backward animation
- `build_gradient_descent_section(self)` — complete GD demo on quadratic loss
- `build_activation_comparison_section(self)` — complete 2x2 activation comparison

When the video covers neural networks, machine learning, deep learning, optimization, or training — USE these ML helpers instead of building visualizations from scratch. They produce professional, animated results.

## REFERENCE SCENE (follow this pattern EXACTLY)

This shows the correct layout, zone management, subtitle captions, cleanup, and visual richness. Study it carefully:

```python
class ReLUExplainedScene(OctoflashScene):
    def construct(self):
        intro_sequence(self, "ReLU Activation Function")

        # ── Persistent title (TOP ZONE: y=3.2 to 4.0) ──
        title = Text("ReLU Activation", font_size=TITLE_SIZE,
                      color=TEXT_PRIMARY, weight="BOLD")
        title.to_edge(UP, buff=0.3)
        self.play(FadeIn(title))

        # ── Section 1: Show the function ──
        with self.voiceover(text="ReLU simply outputs zero for negatives and x for positives.") as tracker:
            # Caption (BOTTOM ZONE: y=-3.2 to -4.0)
            cap = Text("ReLU: zero for negatives, x for positives",
                        font_size=LABEL_SIZE, color=TEXT_SECONDARY)
            cap.to_edge(DOWN, buff=0.4)
            self.play(FadeIn(cap), run_time=0.5)

            # Formula (MIDDLE ZONE: y=-2.5 to 3.0)
            eq = MathTex(r"\text{ReLU}(x)", "=", r"\max(0,\,x)",
                         font_size=40, color=TEXT_PRIMARY)
            eq.shift(UP * 1.8)
            self.play(Write(eq), run_time=1.5)

            # Axes + plot (MIDDLE ZONE center)
            axes = Axes(x_range=[-4, 4, 1], y_range=[-1, 4, 1],
                        x_length=7, y_length=3.2,
                        axis_config={"color": TEXT_DIM, "stroke_width": 2})
            axes.shift(DOWN * 0.4)
            labels = axes.get_axis_labels(
                x_label=MathTex("x", font_size=24),
                y_label=MathTex("y", font_size=24))

            relu = axes.plot(lambda x: np.maximum(0, x),
                             color=ACCENT_GREEN, stroke_width=4)

            self.play(Create(axes), Write(labels), run_time=1)
            self.play(Create(relu), run_time=2)

            remaining = tracker.get_remaining_duration(buff=-0.3)
            if remaining > 0:
                self.wait(remaining)

        # ── Section 2: Dynamic sweep ──
        with self.voiceover(text="Watch how the output changes as x moves across the domain.") as tracker:
            # Update caption
            new_cap = Text("Sweeping x across the domain",
                            font_size=LABEL_SIZE, color=TEXT_SECONDARY)
            new_cap.to_edge(DOWN, buff=0.4)
            self.play(FadeOut(cap), FadeIn(new_cap), run_time=0.4)

            x_val = ValueTracker(-4)
            dot = always_redraw(lambda: Dot(
                axes.c2p(x_val.get_value(),
                         np.maximum(0, x_val.get_value())),
                color=ACCENT_CYAN, radius=0.1))
            x_label = always_redraw(lambda: MathTex(
                f"x={x_val.get_value():.1f}",
                font_size=24, color=ACCENT_CYAN
            ).next_to(dot, UR, buff=0.1))

            self.play(FadeIn(dot), FadeIn(x_label), run_time=0.5)
            self.play(x_val.animate.set_value(4),
                      run_time=4, rate_func=linear)

            remaining = tracker.get_remaining_duration(buff=-0.3)
            if remaining > 0:
                self.wait(remaining)

        # ── Cleanup before MCQ ──
        self.play(FadeOut(VGroup(eq, axes, labels, relu, dot,
                                 x_label, new_cap)), run_time=0.6)

        # ── MCQ ──
        with self.voiceover(text="Quick quiz: what is ReLU of negative five?") as tracker:
            mcq = make_mcq_card("What is ReLU(-5)?",
                                ["5", "0", "-5", "Undefined"])
            self.play(FadeIn(mcq), run_time=0.8)
            remaining = tracker.get_remaining_duration(buff=-0.3)
            if remaining > 0:
                self.wait(remaining)

        with self.voiceover(text="The answer is zero, since ReLU clamps all negatives to zero.") as tracker:
            mcq_ans = make_mcq_card("What is ReLU(-5)?",
                                    ["5", "0", "-5", "Undefined"],
                                    correct_idx=1)
            self.play(ReplacementTransform(mcq, mcq_ans), run_time=0.8)
            remaining = tracker.get_remaining_duration(buff=-0.3)
            if remaining > 0:
                self.wait(remaining)

        self.play(FadeOut(mcq_ans, title), run_time=0.5)
        outro_sequence(self)
```

## Screen Zones (NEVER overlap)

```
y=4.0  ┌──────── BRAND WATERMARK (RESERVED) ────────┐  auto-added by base scene — DO NOT touch
y=3.6  └───────────────────────────────────────────────┘
       ┌──────────────── TOP ZONE ─────────────────┐  title.to_edge(UP, buff=0.7)
y=3.0  └───────────────────────────────────────────────┘
       ┌──────────────── MIDDLE ZONE ───────────────┐  content: axes, formulas, diagrams
y=0.0  │              ORIGIN                        │  axes.shift(DOWN*0.4)
       │                                            │  equations.shift(UP*1.5)
y=-2.5 └───────────────────────────────────────────────┘
       ┌──────────────── BOTTOM ZONE ───────────────┐  caption.to_edge(DOWN, buff=0.4)
y=-4.0 └───────────────────────────────────────────────┘
```

- **Brand watermark**: a "ContextZeroAI" mark is auto-added at the very top by the base scene. NEVER place anything in the top 0.5 units. Use `buff=0.7` minimum on titles.
- **Axes**: `x_length=7, y_length=3.0-3.5`, position with `.shift(DOWN*0.4)` — NEVER `.move_to(ORIGIN)` without shifting down
- **Equations/formulas**: `.shift(UP*1.3)` to `.shift(UP*1.8)` — between title and axes
- **Title**: ALWAYS `.to_edge(UP, buff=0.7)` — leaves room for the brand watermark above
- **Caption**: ALWAYS `.to_edge(DOWN, buff=0.4)` — update each voiceover block

## MANDATORY Content Ratio

Every generated scene MUST have:
- At least **2 Axes+plot sections** with different visualizations (graphs, curves, animated sweeps)
- At least **2 MathTex formulas** with step-through animations (TransformMatchingTex)
- At least **1 ValueTracker dynamic animation** (sweeping parameter, moving dot, morphing graph)
- At least **1 MCQ** with answer reveal
- **Subtitle captions** updated every voiceover block
- **Zero text-only slides** — every section must have a visual element (graph, diagram, formula)

## Section Pattern (repeat for each concept)

```python
# 1. Update caption
cap = Text("Short phrase here", font_size=LABEL_SIZE, color=TEXT_SECONDARY)
cap.to_edge(DOWN, buff=0.4)
self.play(FadeOut(old_cap), FadeIn(cap), run_time=0.4)

# 2. Build visuals in MIDDLE ZONE
axes = Axes(..., x_length=7, y_length=3.2).shift(DOWN*0.4)
eq = MathTex(...).shift(UP*1.8)

# 3. Animate
self.play(Create(axes), Write(eq), run_time=1.5)

# 4. Dynamic element (ValueTracker, sweep, transform)
k = ValueTracker(1)
graph = always_redraw(lambda: axes.plot(...))
self.play(k.animate.set_value(5), run_time=3)

# 5. Cleanup ALL before next section
self.play(FadeOut(VGroup(axes, eq, graph, cap)), run_time=0.5)
```

## 3b1b-Style Animation Recipes (USE these patterns)

### Recipe 1: ValueTracker + always_redraw (Dynamic Graphs)
```python
k = ValueTracker(1.0)
graph = always_redraw(lambda: axes.plot(
    lambda x: np.sin(k.get_value() * x), color=ACCENT_CYAN, stroke_width=3
))
label = always_redraw(lambda: MathTex(
    rf"k = {k.get_value():.1f}", font_size=24, color=ACCENT_CYAN
).shift(UP * 1.8 + RIGHT * 3))
self.add(graph, label)
self.play(k.animate.set_value(5), run_time=4, rate_func=linear)
```

### Recipe 2: TransformMatchingTex (Equation Derivations)
```python
step1 = MathTex("{{a}}{{x}}^2", "+", "{{b}}{{x}}", "+", "{{c}}", "=", "0")
step1.set_color_by_tex("a", ACCENT_BLUE)
step2 = MathTex("{{x}}^2", "+", r"\frac{{{b}}}{{{a}}}", "{{x}}", "=", r"-\frac{{{c}}}{{{a}}}")
self.play(TransformMatchingTex(step1, step2), run_time=2)
```

### Recipe 3: LaggedStartMap (Staggered Reveals)
```python
dots = VGroup(*[Dot(point, color=ACCENT_BLUE) for point in points])
self.play(LaggedStartMap(FadeIn, dots, lag_ratio=0.05, shift=UP*0.3), run_time=2)
```

### Recipe 4: Animated Curve Tracing with Dot
```python
t = ValueTracker(x_min)
traced = always_redraw(lambda: axes.plot(
    func, x_range=[x_min, t.get_value()], color=ACCENT_CYAN, stroke_width=4
))
dot = always_redraw(lambda: Dot(
    axes.c2p(t.get_value(), func(t.get_value())), color=ACCENT_ORANGE, radius=0.08
))
self.add(traced, dot)
self.play(t.animate.set_value(x_max), run_time=5, rate_func=linear)
```

### Recipe 5: BarChart with Animated Value Morphing
```python
chart = BarChart(values=[72,85,91], bar_names=["A","B","C"],
    y_range=[0,100,20], x_length=8, y_length=3.5, bar_colors=[BLUE,GREEN,ORANGE])
chart.shift(DOWN * 0.3)
self.play(Create(chart), run_time=1.5)
target = chart.copy()
target.change_bar_values([90, 82, 96])
self.play(chart.animate.become(target), run_time=2)
```

### Recipe 6: NumberPlane Linear Transform (3b1b style)
```python
plane = NumberPlane(x_range=[-5,5,1], y_range=[-4,4,1],
    background_line_style={"stroke_color": ACCENT_BLUE, "stroke_opacity": 0.3})
ghost = plane.copy().set_stroke(opacity=0.15)
self.add(ghost)
matrix = [[2, 1], [0, 1]]
self.play(plane.animate.apply_matrix(matrix), run_time=3)
```

### Recipe 7: Riemann Sum → Integral
```python
for dx in [1.0, 0.5, 0.25, 0.1]:
    rects = axes.get_riemann_rectangles(graph, x_range=[0,4], dx=dx,
        color=(BLUE, GREEN), fill_opacity=0.5)
    self.play(Transform(prev_rects, rects), run_time=1.5)
area = axes.get_area(graph, x_range=[0,4], color=[BLUE,GREEN], opacity=0.5)
self.play(FadeOut(prev_rects), FadeIn(area), run_time=1.5)
```

## Rules

1. **Single class** inheriting `OctoflashScene`. Do NOT use `intro_sequence` — jump straight into content to hook viewers in the first 3 seconds. End with `outro_sequence`.
2. **Voiceover pattern**: wrap every group in `with self.voiceover(text="...") as tracker:` → animate → `remaining = tracker.get_remaining_duration(buff=-0.3)` → wait.
3. **Cleanup every section**: `self.play(FadeOut(VGroup(...)))` before building next section. NEVER leave stale objects.
4. **White text only**: all Text/MathTex is WHITE. Color ONLY for graph lines, highlights, MCQ correct answer.
5. **Axes sizing**: `x_length=7, y_length=3.0-3.5`. Position `.shift(DOWN*0.4)`. NEVER let axes touch the title or caption zone.
6. **Duration**: The rendered video MUST be close to the target duration. Use SHORT run_times: `run_time=0.5` for FadeIn/FadeOut, `run_time=1` for Create/Write, `run_time=2` max for sweeps. Keep `self.wait()` calls to 0.5-1s. Do NOT add long pauses. Limit to 4-6 voiceover sections MAX. Each section should be 8-15 seconds, not 30+.
7. **Title text**: Long titles MUST be split into 2 lines or use `font_size=36`. Never exceed 40 characters per line. Use `\n` to wrap.
8. **Valid Python**: No `ShowCreation` (use `Create`), no `get_graph()` (use `axes.plot()`), no `max()` in lambdas (use `np.maximum()`), no `math.sin` (use `np.sin`). `axes.get_coordinate_labels()` takes NO keyword arguments (no font_size, no color). `LaggedStartMap` takes (AnimClass, group, **kwargs) — do NOT pass direction constants like LEFT as positional args.
9. **2D only**: No ThreeDAxes, Surface, ThreeDScene, move_camera, set_camera_orientation.
10. **No `np.random`**: all plots must be deterministic.
11. **Imports**: only import from `manim`, `numpy`, `app.manim_pipeline.styles`, `app.manim_pipeline.visual_effects`, `app.manim_pipeline.diagram_patterns`, `app.manim_pipeline.ml_visuals`.
12. **No dangerous imports**: never `import os`, `import sys`, `import subprocess`, `import socket`.
13. **Lambda closures**: in loops, capture loop var with default arg: `lambda i=i: ...` not `lambda: ... i ...`.
14. **MathTex**: never use `$` inside MathTex (already in math mode). Use raw strings `r"..."` for LaTeX.
15. **`always_redraw`**: the callable MUST return a Mobject (e.g. `Arrow`, `Line`, `Dot`), NOT a numpy array or point. Wrong: `always_redraw(lambda: center + np.array([...]))`. Right: `always_redraw(lambda: Arrow(center, center + np.array([...]), color=ACCENT_BLUE))`.

## Output

Return ONLY a ```python``` code block. No explanations.
"""

SYSTEM_PROMPT_NO_VOICE = SYSTEM_PROMPT.replace(
    "OctoflashScene",
    "OctoflashSceneNoVoice",
).replace(
    "1. **Single class** inheriting `OctoflashSceneNoVoice`.",
    "1. **Single class** inheriting `OctoflashSceneNoVoice` (NOT bare `Scene` — the OctoflashSceneNoVoice base sets the background AND adds the brand watermark at the top).",
).replace(
    "2. **Voiceover pattern**: wrap every group in `with self.voiceover(text=\"...\") as tracker:` → animate → `remaining = tracker.get_remaining_duration(buff=-0.3)` → wait.",
    "2. **No voiceover**: do NOT use self.voiceover(). Use `self.wait(2)` after animations.",
).replace(
    "with self.voiceover(text=",
    "# Narration: ",
).replace(
    ") as tracker:",
    "",
).replace(
    "remaining = tracker.get_remaining_duration(buff=-0.3)",
    "self.wait(2)",
).replace(
    "if remaining > 0:",
    "",
).replace(
    "self.wait(remaining)",
    "",
)


def _get_client() -> anthropic.Anthropic:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set in environment")
    return anthropic.Anthropic(api_key=api_key)


SYNTHESIS_PROMPT = r"""You are an expert educational content synthesizer. Given transcripts from multiple YouTube videos on related topics, you must:

1. Extract the key concepts from each video
2. Find relationships and connections between concepts across videos
3. Create a unified narrative arc that weaves all concepts together
4. Produce a structured section outline for a single cohesive Manim animation

Respond with ONLY a valid JSON object (no markdown, no code fences) in this exact format:
{
  "title": "Combined topic title (concise, max 50 chars)",
  "core_concepts": [
    {"name": "Concept A", "from_videos": [1, 2], "explanation": "Brief explanation"},
    {"name": "Concept B", "from_videos": [3], "explanation": "Brief explanation"}
  ],
  "narrative_arc": "How concepts connect: A leads to B which enables C...",
  "section_outline": [
    {"title": "Section name", "concepts": ["A"], "visual": "axes/graph", "duration": 15},
    {"title": "Section name", "concepts": ["B", "C"], "visual": "diagram/flowchart", "duration": 20}
  ],
  "mcq": {"question": "Quiz question combining concepts", "options": ["A", "B", "C", "D"], "correct_idx": 0},
  "combined_transcript": "A single synthesized narrative that merges all source material into one coherent explainer script. This should read as a unified voiceover script, not a concatenation."
}

Rules:
- section_outline should have 4-6 sections, each 10-20 seconds
- visual types: "axes/graph", "diagram/flowchart", "formula/equation", "comparison/table", "timeline", "neural_net", "value_tracker"
- combined_transcript should be a SINGLE flowing narrative, NOT separate summaries per video
- The narrative_arc must explicitly state how concepts from different videos connect
- The MCQ should test understanding of the unified concept, not just one video
"""


def synthesize_concepts(
    transcripts: list[dict],
    frames: list[str],
    total_duration: float,
) -> dict:
    """Extract and synthesize concepts from multiple video transcripts using Claude.

    Args:
        transcripts: List of dicts with keys: url, transcript, title
        frames: List of frame paths (sampled across all videos)
        total_duration: Combined duration of all source videos

    Returns:
        Dict with: title, core_concepts, narrative_arc, section_outline, mcq, combined_transcript
    """
    client = _get_client()

    content = []

    # Include sample frames if available
    if frames:
        full_paths = [STORAGE_DIR / f for f in frames]
        existing = [p for p in full_paths if p.exists()]
        sampled = _sample_items(existing, 6)
        for frame_path in sampled:
            b64 = _image_to_base64(frame_path)
            if b64:
                content.append({
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/jpeg", "data": b64},
                })
        if sampled:
            content.append({
                "type": "text",
                "text": f"Above: sample frames from {len(transcripts)} source videos.\n\n",
            })

    # Build the transcript block
    transcript_block = ""
    for i, t in enumerate(transcripts):
        transcript_block += (
            f"### Video {i + 1}: {t.get('title', f'Video {i + 1}')}\n"
            f"URL: {t.get('url', 'N/A')}\n"
            f"Transcript:\n{t['transcript'][:3000]}\n\n"
        )

    target_duration = min(total_duration, 120)

    content.append({
        "type": "text",
        "text": (
            f"## Source Videos ({len(transcripts)} videos, ~{total_duration:.0f}s total)\n\n"
            f"{transcript_block}\n"
            f"## Target\n"
            f"Synthesize these {len(transcripts)} videos into ONE unified animation "
            f"(target ~{target_duration:.0f}s). Find the connecting threads between all videos "
            f"and create a single cohesive narrative.\n"
        ),
    })

    logger.info("Calling Claude API for concept synthesis (%d videos)", len(transcripts))

    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=4096,
        system=SYNTHESIS_PROMPT,
        messages=[{"role": "user", "content": content}],
    )

    raw_text = response.content[0].text.strip()

    # Strip markdown code fences if Claude added them
    if raw_text.startswith("```"):
        raw_text = re.sub(r"^```(?:json)?\s*\n?", "", raw_text)
        raw_text = re.sub(r"\n?```\s*$", "", raw_text)

    try:
        synthesis = json.loads(raw_text)
    except json.JSONDecodeError as e:
        logger.error("Failed to parse synthesis JSON: %s\nRaw: %s", e, raw_text[:500])
        # Fallback: return a basic structure with concatenated transcripts
        return _fallback_synthesis(transcripts, total_duration)

    # Validate required keys
    required = ["title", "core_concepts", "narrative_arc", "section_outline", "combined_transcript"]
    for key in required:
        if key not in synthesis:
            logger.warning("Synthesis missing key '%s', using fallback", key)
            return _fallback_synthesis(transcripts, total_duration)

    logger.info(
        "Concept synthesis complete: title='%s', %d concepts, %d sections",
        synthesis["title"],
        len(synthesis.get("core_concepts", [])),
        len(synthesis.get("section_outline", [])),
    )
    return synthesis


def _fallback_synthesis(transcripts: list[dict], total_duration: float) -> dict:
    """Fallback synthesis when Claude fails to return valid JSON."""
    combined = "\n\n".join(
        f"[From {t.get('title', f'Video {i+1}')}]: {t['transcript']}"
        for i, t in enumerate(transcripts)
    )
    return {
        "title": f"Combined: {len(transcripts)} Topics",
        "core_concepts": [],
        "narrative_arc": "Sequential presentation of topics from multiple sources.",
        "section_outline": [],
        "mcq": {"question": "What was the main topic?", "options": ["A", "B", "C", "D"], "correct_idx": 0},
        "combined_transcript": combined,
    }


def generate_episode_script(
    transcript: str,
    description: str,
    duration: float,
    title: str = "Inspired Video",
    video_id: str = "",
    voiceover: bool = True,
    source_frames: list[Path] | None = None,
    feedback: str | None = None,
    manin_prompt: str = "",
) -> str:
    """Generate a rich Manim scene script using Claude API.

    Args:
        transcript: The video transcript text.
        description: Generated description/analysis of the video.
        duration: Target video duration in seconds.
        title: Episode title.
        video_id: Identifier for logging.
        voiceover: Whether to use OctoflashScene with voiceover or plain Scene.
        source_frames: Optional list of source frame paths to send as vision input.
        feedback: Optional feedback from a previous iteration to improve the script.
    """
    client = _get_client()
    system = SYSTEM_PROMPT if voiceover else SYSTEM_PROMPT_NO_VOICE

    # Build user message content (text + optional images)
    content = []

    # Include source frames if available (sample up to 6)
    if source_frames:
        sampled = _sample_items(source_frames, 6)
        for frame_path in sampled:
            b64 = _image_to_base64(frame_path)
            if b64:
                content.append({
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/jpeg", "data": b64},
                })
        content.append({
            "type": "text",
            "text": "Above are sample frames from the SOURCE video. Match this visual style: dark background, clean typography, mathematical formulas, animated graphs/plots, persistent title at top, subtitle text at bottom.\n\n",
        })

    task = (
        f"## Video Info\n"
        f"- **Title**: {title}\n"
        f"- **Duration**: {duration:.1f} seconds\n"
        f"- **Video ID**: {video_id}\n\n"
        f"## Transcript\n{transcript}\n\n"
        f"## Visual Description\n{description}\n\n"
    )

    # Detect multi-video synthesis from manin_prompt
    is_multi_video = manin_prompt and "## Multi-Video Synthesis" in manin_prompt

    if manin_prompt:
        task += f"## Creative Direction (User-Edited Prompt)\n{manin_prompt}\n\n"

    if feedback:
        task += f"## IMPROVEMENT FEEDBACK (from previous iteration)\n{feedback}\n\n"

    # Cap target duration — keep videos tight and fast-paced
    target_secs = min(duration, 120)  # never exceed 2 minutes
    num_sections = max(3, min(6, int(target_secs / 15)))  # 3-6 sections

    if is_multi_video:
        task += (
            f"## Task\n"
            f"Write a complete Manim scene script for this **unified multi-video concept explainer**.\n"
            f"CRITICAL: The Creative Direction above contains a section outline from concept synthesis. "
            f"Follow that outline EXACTLY — each section must cover the specified concepts using the specified visual type.\n"
            f"The animation must feel like ONE cohesive explainer, NOT separate summaries stitched together.\n"
            f"Include: mathematical formulas (MathTex), Axes plots/graphs where relevant, "
            f"animated diagrams, and one MCQ with answer reveal.\n"
            f"{'Use OctoflashScene with voiceover.' if voiceover else 'Use Scene (no voiceover). Add self.wait() calls to fill duration.'}\n"
            f"CRITICAL: Target duration is ~{target_secs:.0f} seconds. Use ONLY {num_sections} sections. "
            f"Keep animations FAST: run_time=0.5 for transitions, run_time=1-2 for main animations. "
            f"NO long waits. The video must be TIGHT and FAST-PACED, not slow and boring.\n"
            f"Make it visually engaging — match the quality of 3Blue1Brown style animations."
        )
    else:
        task += (
            f"## Task\n"
            f"Write a complete Manim scene script for this educational content.\n"
            f"Include: mathematical formulas (MathTex), Axes plots/graphs where relevant, "
            f"animated diagrams, and one MCQ with answer reveal.\n"
            f"{'Use OctoflashScene with voiceover.' if voiceover else 'Use Scene (no voiceover). Add self.wait() calls to fill duration.'}\n"
            f"CRITICAL: Target duration is ~{target_secs:.0f} seconds. Use ONLY {num_sections} sections. "
            f"Keep animations FAST: run_time=0.5 for transitions, run_time=1-2 for main animations. "
            f"NO long waits. The video must be TIGHT and FAST-PACED, not slow and boring.\n"
            f"Make it visually engaging — match the quality of 3Blue1Brown style animations."
        )

    content.append({"type": "text", "text": task})

    logger.info("Calling Claude API for script generation (video_id=%s, voiceover=%s)", video_id, voiceover)

    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=8192,
        system=system,
        messages=[{"role": "user", "content": content}],
    )

    raw_text = response.content[0].text
    code = _extract_python_code(raw_text)

    if not code:
        raise RuntimeError("Claude response did not contain a valid Python code block")

    # Auto-fix common Claude mistakes
    code = sanitize_script(code)

    # Validate syntax
    try:
        compile(code, "<claude_generated_scene>", "exec")
    except SyntaxError as e:
        raise RuntimeError(f"Claude generated code with syntax error at line {e.lineno}: {e.msg}")

    logger.info("Claude script generated successfully (%d chars, voiceover=%s)", len(code), voiceover)
    return code


def save_script(video_id: str, script_code: str) -> Path:
    """Save generated script to storage/scripts/{video_id}/episode.py."""
    script_dir = STORAGE_DIR / "scripts" / video_id
    script_dir.mkdir(parents=True, exist_ok=True)
    script_file = script_dir / "episode.py"
    script_file.write_text(script_code)
    return script_file


def evaluate_output(
    output_frame_paths: list[Path],
    transcript: str,
    script_code: str,
    source_frame_paths: list[Path] | None = None,
) -> dict:
    """Evaluate rendered output by comparing output frames against source frames.

    Sends both source (input) and output frames to Claude vision for a
    side-by-side comparison. Claude scores the output and provides specific,
    actionable feedback for improvement.

    Returns:
        dict with 'score' (1-10), 'passed' (bool), and 'feedback' (str).
    """
    client = _get_client()

    content = []

    # --- SOURCE FRAMES (what the output should look like) ---
    if source_frame_paths:
        source_sampled = _sample_items(source_frame_paths, 4)
        for frame_path in source_sampled:
            b64 = _image_to_base64(frame_path)
            if b64:
                content.append({
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/jpeg", "data": b64},
                })
        content.append({
            "type": "text",
            "text": "^^^ ABOVE: SOURCE/INPUT frames — this is the REFERENCE style the output should match.\n\n",
        })

    # --- OUTPUT FRAMES (what was actually rendered) ---
    output_sampled = _sample_items(output_frame_paths, 6)
    for frame_path in output_sampled:
        b64 = _image_to_base64(frame_path)
        if b64:
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": "image/jpeg", "data": b64},
            })
    content.append({
        "type": "text",
        "text": "^^^ ABOVE: OUTPUT/RENDERED frames — this is what the Manim script produced.\n\n",
    })

    # --- EVALUATION PROMPT ---
    content.append({
        "type": "text",
        "text": (
            f"## Transcript\n{transcript[:1500]}\n\n"
            f"## Current Manim Script\n```python\n{script_code[:3000]}\n```\n\n"
            f"## Task\n"
            f"Compare the OUTPUT frames against the SOURCE frames and transcript.\n\n"
            f"Rate the output on a scale of 1-10, considering:\n"
            f"1. **Visual richness** — Does it have graphs, plots, diagrams, math formulas? Or just text slides?\n"
            f"2. **Style match** — Does it match the source's visual style (dark bg, colors, layout)?\n"
            f"3. **Content accuracy** — Does it cover the same concepts as the transcript?\n"
            f"4. **Animation quality** — Varied animations, not repetitive? No empty/black frames?\n"
            f"5. **Readability** — Text legible? Good contrast? Not too small?\n\n"
            f"IMPORTANT: A score of 5 or below means the output is mostly text slides with no real visualizations.\n"
            f"A score of 7+ means it has mathematical plots, diagrams, and dynamic animations.\n\n"
            f"Respond in EXACTLY this format:\n"
            f"SCORE: <number 1-10>\n"
            f"ISSUES: <bullet list of specific problems found>\n"
            f"FEEDBACK: <specific code-level fixes — e.g. 'Add an Axes plot showing the ReLU function', "
            f"'Replace the static text in section 3 with a ValueTracker animation', "
            f"'Add MathTex formulas for the equation steps'. Be extremely specific about what Manim objects to use.>\n"
        ),
    })

    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=2048,
        messages=[{"role": "user", "content": content}],
    )

    result_text = response.content[0].text
    score = 5  # default
    feedback = ""

    score_match = re.search(r"SCORE:\s*(\d+)", result_text)
    if score_match:
        score = int(score_match.group(1))

    # Combine ISSUES and FEEDBACK into one feedback string
    issues_match = re.search(r"ISSUES:\s*(.+?)(?=FEEDBACK:)", result_text, re.DOTALL)
    feedback_match = re.search(r"FEEDBACK:\s*(.+)", result_text, re.DOTALL)

    parts = []
    if issues_match:
        parts.append(f"Issues found:\n{issues_match.group(1).strip()}")
    if feedback_match:
        parts.append(f"Suggested fixes:\n{feedback_match.group(1).strip()}")
    feedback = "\n\n".join(parts) if parts else result_text

    passed = score >= 7
    logger.info("Output evaluation: score=%d, passed=%s, feedback=%s", score, passed, feedback[:200])

    return {"score": score, "passed": passed, "feedback": feedback}


def extract_video_frames(video_path: Path, count: int = 8) -> list[Path]:
    """Extract evenly-spaced frames from a rendered video using ffmpeg.

    Returns list of frame file paths.
    """
    output_dir = video_path.parent / "eval_frames"
    output_dir.mkdir(exist_ok=True)

    # Get video duration
    probe = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
        capture_output=True, text=True,
    )
    try:
        vid_duration = float(probe.stdout.strip())
    except ValueError:
        vid_duration = 30.0

    interval = max(1, vid_duration / count)

    # Extract frames
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(video_path),
         "-vf", f"fps=1/{interval:.2f}",
         "-frames:v", str(count),
         str(output_dir / "eval_%04d.jpg")],
        capture_output=True, text=True,
    )

    frames = sorted(output_dir.glob("eval_*.jpg"))
    return frames


def analyze_source_frames(
    frame_paths: list[str],
    transcript: str,
    duration: float,
) -> str:
    """Analyze source video frames with Claude vision to generate a rich description.

    Args:
        frame_paths: List of relative frame paths (e.g., 'video_id/frames/frame_0001.jpg').
        transcript: The video transcript.
        duration: Video duration in seconds.

    Returns:
        Rich visual description of the source video.
    """
    try:
        client = _get_client()
    except RuntimeError:
        logger.warning("No ANTHROPIC_API_KEY — returning basic description")
        return _basic_description(transcript, frame_paths, duration)

    # Sample ~8 frames evenly
    full_paths = [STORAGE_DIR / f for f in frame_paths]
    existing = [p for p in full_paths if p.exists()]
    if not existing:
        return _basic_description(transcript, frame_paths, duration)

    sampled = _sample_items(existing, 8)
    content = []

    for frame_path in sampled:
        b64 = _image_to_base64(frame_path)
        if b64:
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": "image/jpeg", "data": b64},
            })

    content.append({
        "type": "text",
        "text": (
            f"These are {len(sampled)} evenly-sampled frames from a {duration:.0f}-second educational video.\n\n"
            f"Transcript: {transcript}\n\n"
            f"Analyze the visual style and content. Describe:\n"
            f"1. Visual style (colors, background, typography)\n"
            f"2. Types of visualizations used (graphs, diagrams, formulas, 3D plots)\n"
            f"3. Animation techniques visible (transitions, transforms)\n"
            f"4. Layout pattern (title position, content area, subtitle/caption position)\n"
            f"5. Mathematical formulas or equations shown\n"
            f"6. Key visual scenes and what they depict\n\n"
            f"Be specific and detailed — this description will be used to generate a similar Manim animation."
        ),
    })

    try:
        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=2048,
            messages=[{"role": "user", "content": content}],
        )
        description = response.content[0].text
        logger.info("Vision-based frame analysis complete (%d chars)", len(description))
        return description
    except Exception as e:
        logger.warning("Vision analysis failed: %s — returning basic description", e)
        return _basic_description(transcript, frame_paths, duration)


def _basic_description(transcript: str, frames: list[str], duration: float) -> str:
    """Fallback description without vision analysis."""
    return (
        f"Educational video ({duration:.0f}s, {len(frames)} frames). "
        f"Content: {transcript[:500]}"
    )


def sanitize_script(code: str) -> str:
    """Auto-fix common Claude mistakes that crash Manim CE rendering.

    Comprehensive fixes for manimgl→CE API differences, vectorization issues,
    dangerous imports, 3D/camera removal, and LaTeX pitfalls.
    """
    original = code
    fixes_applied = []

    def _track(label, old_code, new_code):
        if old_code != new_code:
            fixes_applied.append(label)
        return new_code

    # ── CRITICAL: Inject missing import header ──
    # Claude sometimes omits imports entirely, causing NameError at runtime.
    if 'from manim import' not in code and 'from manim ' not in code:
        logger.warning("sanitize_script: Missing 'from manim import' — injecting full import header")
        has_voiceover = 'OctoflashScene' in code
        import_header = 'from manim import *\nimport numpy as np\n'
        if has_voiceover:
            import_header += (
                'from app.manim_pipeline.styles import (\n'
                '    OctoflashScene, make_title_card, make_cell, make_cell_row,\n'
                '    make_code_block, make_mcq_card, intro_sequence, outro_sequence,\n'
                '    BG_COLOR, CODE_BG,\n'
                '    ACCENT_BLUE, ACCENT_ORANGE, ACCENT_GREEN, ACCENT_RED,\n'
                '    ACCENT_PURPLE, ACCENT_YELLOW, ACCENT_CYAN, ACCENT_PINK,\n'
                '    TEXT_PRIMARY, TEXT_SECONDARY, TEXT_DIM,\n'
                '    TITLE_SIZE, SUBTITLE_SIZE, BODY_SIZE, LABEL_SIZE, CODE_FONT_SIZE,\n'
                ')\n'
            )
        else:
            import_header += (
                'from app.manim_pipeline.styles import (\n'
                '    BG_COLOR, ACCENT_BLUE, ACCENT_ORANGE, ACCENT_GREEN, ACCENT_RED,\n'
                '    ACCENT_PURPLE, ACCENT_YELLOW, ACCENT_CYAN, ACCENT_PINK,\n'
                '    TEXT_PRIMARY, TEXT_SECONDARY, TEXT_DIM,\n'
                '    TITLE_SIZE, SUBTITLE_SIZE, BODY_SIZE, LABEL_SIZE,\n'
                ')\n'
            )
        code = import_header + '\n' + code

    # ── CRITICAL: Inject missing styles helper imports ──
    # Even non-OctoflashScene scripts may use make_mcq_card, make_title_card, etc.
    _styles_helpers = [
        'make_title_card', 'make_cell', 'make_cell_row',
        'make_code_block', 'make_mcq_card', 'intro_sequence', 'outro_sequence',
    ]
    if 'from app.manim_pipeline.styles' in code:
        # Check if any helpers are used but not imported
        existing_styles_import = re.search(r'from app\.manim_pipeline\.styles import \([^)]*\)', code, re.DOTALL)
        if existing_styles_import:
            import_block = existing_styles_import.group(0)
            missing = [f for f in _styles_helpers if f in code and f not in import_block]
            if missing:
                # Add missing helpers to existing import (replace closing paren)
                new_block = import_block[:-1] + '    ' + ', '.join(missing) + ',\n)'
                code = code.replace(import_block, new_block)
    else:
        used_styles = [f for f in _styles_helpers if f in code]
        if used_styles:
            code = (
                'from app.manim_pipeline.styles import (\n'
                '    BG_COLOR, ACCENT_BLUE, ACCENT_ORANGE, ACCENT_GREEN, ACCENT_RED,\n'
                '    ACCENT_PURPLE, ACCENT_YELLOW, ACCENT_CYAN, ACCENT_PINK,\n'
                '    TEXT_PRIMARY, TEXT_SECONDARY, TEXT_DIM,\n'
                '    TITLE_SIZE, SUBTITLE_SIZE, BODY_SIZE, LABEL_SIZE,\n'
                '    ' + ', '.join(used_styles) + ',\n'
                ')\n'
            ) + code

    # ── CRITICAL: Inject missing visual_effects imports ──
    _ve_funcs = [
        'crossfade_transition', 'zoom_transition', 'section_wipe',
        'glow_effect', 'pulse_effect', 'emphasis_box', 'underline_emphasis',
        'flash_and_circumscribe',
        'typewriter_reveal', 'word_by_word_reveal', 'scanning_highlight',
        'equation_step_through',
        'subtle_grid_background', 'dot_grid_background',
        'make_speech_bubble', 'make_callout_box', 'make_labeled_arrow',
        'make_brace_annotation',
        'make_progress_bar', 'make_step_counter', 'make_section_marker',
        'sweep_in_group', 'cascade_fade_in', 'pop_in_sequence', 'staggered_write',
        'dynamic_counter', 'cleanup_and_transition',
    ]
    if 'visual_effects' not in code:
        used_ve = [f for f in _ve_funcs if f in code]
        if used_ve:
            code = (
                'from app.manim_pipeline.visual_effects import (\n'
                '    ' + ', '.join(used_ve) + ',\n'
                ')\n'
            ) + code

    # ── CRITICAL: Inject missing diagram_patterns imports ──
    _dp_funcs = [
        'make_flowchart_box', 'make_diamond', 'connect_boxes',
        'make_flowchart', 'animate_flowchart_build', 'animate_flow_pulse',
        'make_layer_block', 'make_layer_stack', 'make_parallel_layers',
        'animate_data_through_layers',
        'make_comparison_layout', 'make_before_after', 'animate_comparison_reveal',
        'make_timeline', 'animate_timeline_progress', 'make_vertical_timeline',
        'make_pipeline', 'animate_data_packet', 'make_branching_pipeline',
        'make_styled_table', 'make_confusion_matrix', 'make_data_grid',
        'animate_grid_highlight_row', 'animate_grid_highlight_col',
        'animate_grid_highlight_cell', 'animate_table_row_by_row',
        'animate_table_cell_by_cell',
        'highlight_box', 'animate_highlight_sequence',
    ]
    if 'diagram_patterns' not in code:
        used_dp = [f for f in _dp_funcs if f in code]
        if used_dp:
            code = (
                'from app.manim_pipeline.diagram_patterns import (\n'
                '    ' + ', '.join(used_dp) + ',\n'
                ')\n'
            ) + code

    # ── CRITICAL: Inject missing ml_visuals imports ──
    _ml_funcs = [
        'draw_neural_network', 'animate_network_creation',
        'animate_forward_pass', 'animate_backpropagation',
        'draw_activation_functions', 'animate_activation_comparison',
        'animate_gradient_descent',
        'draw_loss_landscape_contour', 'animate_gradient_descent_2d',
    ]
    if 'ml_visuals' not in code:
        used_ml = [f for f in _ml_funcs if f in code]
        if used_ml:
            code = (
                'from app.manim_pipeline.ml_visuals import (\n'
                '    ' + ', '.join(used_ml) + ',\n'
                ')\n'
            ) + code

    # ── CRITICAL: Push titles below the brand watermark zone ──
    # Any to_edge(UP, buff<0.6) collides with the watermark. Force buff=0.7 minimum.
    _prev = code
    code = re.sub(
        r'\.to_edge\(\s*UP\s*,\s*buff\s*=\s*(0?\.[0-5]\d*|0?\.6)\s*\)',
        '.to_edge(UP, buff=0.7)',
        code,
    )
    code = _track("push titles below watermark zone", _prev, code)

    # ── CRITICAL: Convert bare `Scene` to OctoflashSceneNoVoice so watermark always shows ──
    # If the script doesn't already inherit OctoflashScene/3DScene/NoVoice, force the no-voice base.
    if not re.search(r'class\s+\w+\s*\(\s*Octoflash(Scene|3DScene|SceneNoVoice)\s*\)', code):
        prev = code
        code = re.sub(
            r'class\s+(\w+)\s*\(\s*Scene\s*\)',
            r'class \1(OctoflashSceneNoVoice)',
            code,
        )
        if code != prev:
            # Ensure the import is present
            if 'OctoflashSceneNoVoice' not in (re.search(r'from app\.manim_pipeline\.styles import[^\)]*', code) or [None])[0] if re.search(r'from app\.manim_pipeline\.styles import[^\)]*', code) else False:
                pass
            if 'OctoflashSceneNoVoice' not in code.split('class ')[0]:
                # Inject import at the top of the styles import block
                if re.search(r'from app\.manim_pipeline\.styles import \(', code):
                    code = re.sub(
                        r'(from app\.manim_pipeline\.styles import \()',
                        r'\1\n    OctoflashSceneNoVoice,',
                        code,
                        count=1,
                    )
                else:
                    code = "from app.manim_pipeline.styles import OctoflashSceneNoVoice\n" + code
            fixes_applied.append("Scene→OctoflashSceneNoVoice (for watermark)")

    # ── CRITICAL: manimgl → Manim CE name fixes ──
    _prev = code
    code = re.sub(r'\bShowCreation\b', 'Create', code)
    code = re.sub(r'\bTexMobject\b', 'MathTex', code)
    code = re.sub(r'\bTextMobject\b', 'Text', code)
    code = re.sub(r'\.get_graph\(', '.plot(', code)
    code = _track("manimgl→CE names", _prev, code)

    # ── CRITICAL: Unwrap self.play(helper(self, ...)) for helpers that play internally ──
    # These functions already call self.play() and return None. Wrapping in self.play() crashes.
    _self_playing_helpers = [
        # visual_effects
        'staggered_write', 'sweep_in_group', 'cascade_fade_in', 'pop_in_sequence',
        'flash_and_circumscribe', 'crossfade_transition', 'zoom_transition',
        'section_wipe', 'equation_step_through', 'cleanup_and_transition',
        'pulse_effect', 'emphasis_box', 'underline_emphasis',
        'intro_sequence', 'outro_sequence',
        # diagram_patterns (all animate_* functions call scene.play() internally)
        'animate_flowchart_build', 'animate_flow_pulse',
        'animate_data_through_layers', 'animate_comparison_reveal',
        'animate_timeline_progress', 'animate_data_packet',
        'animate_table_row_by_row', 'animate_table_cell_by_cell',
        'animate_grid_highlight_row', 'animate_grid_highlight_col',
        'animate_grid_highlight_cell', 'animate_highlight_sequence',
        # ml_visuals
        'animate_network_creation', 'animate_forward_pass',
        'animate_backpropagation', 'animate_activation_comparison',
        'animate_gradient_descent', 'animate_gradient_descent_2d',
        'animate_training_loop', 'animate_decision_boundary',
    ]
    for helper in _self_playing_helpers:
        # self.play(helper(self, ...), run_time=X) → helper(self, ...)
        code = re.sub(
            rf'self\.play\(\s*{helper}\(self,\s*(.*?)\)\s*(?:,\s*run_time\s*=\s*[\d.]+)?\s*\)',
            rf'{helper}(self, \1)',
            code,
        )
        # self.play(helper(self)) → helper(self)
        code = re.sub(
            rf'self\.play\(\s*{helper}\(self\)\s*(?:,\s*run_time\s*=\s*[\d.]+)?\s*\)',
            rf'{helper}(self)',
            code,
        )

    # ── CRITICAL: Inject missing `self` into scene-helper calls ──
    # Claude often calls emphasis_box(mobject, ...) instead of emphasis_box(self, mobject, ...).
    # All _self_playing_helpers require `scene` (i.e. `self`) as the first argument.
    _prev = code
    for helper in _self_playing_helpers:
        # Match helper(something_not_self, ...) inside a method body (indented)
        # emphasis_box(bars[0], color=X) → emphasis_box(self, bars[0], color=X)
        code = re.sub(
            rf'(\s){helper}\((?!self[\s,)])',
            rf'\1{helper}(self, ',
            code,
        )
    # Clean up double-self: helper(self, self, ...) → helper(self, ...)
    for helper in _self_playing_helpers:
        code = re.sub(rf'{helper}\(self,\s*self,', rf'{helper}(self,', code)
    code = _track("inject missing self into scene-helper calls", _prev, code)

    # ── CRITICAL: Strip .group from diagram helper return accesses ──
    # make_flowchart/make_timeline/make_pipeline/make_comparison_layout etc.
    # return VGroups directly — they don't have a .group attribute.
    # Claude writes `pipeline.group.shift(...)` but it should be `pipeline.shift(...)`
    _prev = code
    # pipeline.group.scale(0.8) → pipeline.scale(0.8)
    code = re.sub(r'\.group\.', '.', code)
    # FadeIn(pipeline.group) or pipeline.group, → FadeIn(pipeline) or pipeline,
    code = re.sub(r'\.group(?=[\s),\]\n])', '', code)
    code = _track("strip .group from diagram helpers", _prev, code)

    # ── CRITICAL: Remove None from self.play() calls ──
    # Claude sometimes passes helper return values (None) directly to self.play().
    # self.play(None) → remove line; self.play(FadeOut(x), None) → self.play(FadeOut(x))
    _prev = code
    # Remove standalone self.play(None)
    code = re.sub(r'^\s*self\.play\(\s*None\s*\)\s*$', '', code, flags=re.MULTILINE)
    # Strip trailing None: self.play(FadeOut(x), None) → self.play(FadeOut(x))
    code = re.sub(r',\s*None\s*\)', ')', code)
    # Strip leading None: self.play(None, FadeIn(x)) → self.play(FadeIn(x))
    code = re.sub(r'(self\.play\(\s*)None\s*,\s*', r'\1', code)
    code = _track("strip None from self.play()", _prev, code)

    # ── CRITICAL: LaggedStartMap with direction as 3rd positional arg ──
    _prev = code
    _directions = r'(?:LEFT|RIGHT|UP|DOWN|UL|UR|DL|DR|ORIGIN)'
    code = re.sub(
        rf'(LaggedStartMap\(\s*\w+,\s*\w+),\s*{_directions}\s*,',
        r'\1,',
        code,
    )

    # ── CRITICAL: LaggedStartMap with lambda returning a bare direction ──
    # Claude writes LaggedStartMap(GrowFromEdge, bars, lambda m: DOWN, ...)
    # The lambda returns np.array([0,-1,0]) which gets *-unpacked into 3 scalar
    # args, crashing with "'numpy.float64' has no attribute 'get_critical_point'".
    # Fix: strip the lambda, pass direction as edge= kwarg instead.
    code = re.sub(
        rf'(LaggedStartMap\(\s*GrowFromEdge,\s*\w+),\s*lambda\s+\w+\s*:\s*({_directions})\s*,',
        r'\1, edge=\2,',
        code,
    )
    # Generic: strip any lambda returning bare direction for other animations
    code = re.sub(
        rf'(LaggedStartMap\(\s*(?!GrowFromEdge)\w+,\s*\w+),\s*lambda\s+\w+\s*:\s*{_directions}\s*,',
        r'\1,',
        code,
    )
    code = _track("LaggedStartMap direction fix", _prev, code)

    # ── CRITICAL: Python builtins not vectorized for numpy arrays ──
    _prev = code
    # max(0, x) → np.maximum(0, x) in any lambda context
    code = re.sub(r'lambda\s+(\w+)\s*:\s*max\((\d+),\s*\1\)', r'lambda \1: np.maximum(\2, \1)', code)
    code = re.sub(r'lambda\s+(\w+)\s*:\s*max\(\1,\s*(\d+)\)', r'lambda \1: np.maximum(\1, \2)', code)
    # min(1, x) → np.minimum(1, x)
    code = re.sub(r'lambda\s+(\w+)\s*:\s*min\((\d+),\s*\1\)', r'lambda \1: np.minimum(\2, \1)', code)
    code = re.sub(r'lambda\s+(\w+)\s*:\s*min\(\1,\s*(\d+)\)', r'lambda \1: np.minimum(\1, \2)', code)
    # abs(x) → np.abs(x) in lambdas
    code = re.sub(r'lambda\s+(\w+)\s*:\s*abs\(', r'lambda \1: np.abs(', code)
    code = _track("vectorize builtins (max/min/abs→np)", _prev, code)

    # ── HIGH: math.* → np.* (math module not vectorized) ──
    _prev = code
    code = re.sub(r'\bmath\.sin\b', 'np.sin', code)
    code = re.sub(r'\bmath\.cos\b', 'np.cos', code)
    code = re.sub(r'\bmath\.tan\b', 'np.tan', code)
    code = re.sub(r'\bmath\.exp\b', 'np.exp', code)
    code = re.sub(r'\bmath\.log\b', 'np.log', code)
    code = re.sub(r'\bmath\.sqrt\b', 'np.sqrt', code)
    code = re.sub(r'\bmath\.pi\b', 'np.pi', code)
    code = re.sub(r'\bmath\.e\b', 'np.e', code)
    code = re.sub(r'\bmath\.floor\b', 'np.floor', code)
    code = re.sub(r'\bmath\.ceil\b', 'np.ceil', code)
    code = re.sub(r'\bmath\.pow\b', 'np.power', code)
    code = re.sub(r'\bmath\.fabs\b', 'np.abs', code)
    # Remove import math (now unnecessary)
    code = re.sub(r'^\s*import math\s*$', '', code, flags=re.MULTILINE)
    code = _track("math→np conversion", _prev, code)

    # ── HIGH: 3D/Camera removal (2D scenes only) ──
    _prev = code
    code = re.sub(r'^\s*self\.move_camera\(.*?\)\s*$', '', code, flags=re.MULTILINE)
    code = re.sub(r'^\s*self\.set_camera_orientation\(.*?\)\s*$', '', code, flags=re.MULTILINE)
    code = re.sub(r'\bThreeDAxes\b', 'Axes', code)
    code = re.sub(r'^\s*self\.add_fixed_in_frame_mobjects\(.*?\)\s*$', '', code, flags=re.MULTILINE)

    # ── HIGH: Octoflash3DScene → OctoflashScene (no 3D support) ──
    code = re.sub(r'\bOctoflash3DScene\b', 'OctoflashScene', code)

    # ── HIGH: Arrow3D → Arrow (2D only) ──
    # Claude uses Arrow3D with 3D coords; convert to 2D Arrow, strip z-coordinate
    code = re.sub(r'\bArrow3D\b', 'Arrow', code)

    # ── HIGH: Surface / ThreeDScene removal ──
    code = re.sub(r'\bThreeDScene\b', 'Scene', code)
    code = re.sub(r'^\s*.*\bSurface\s*\(.*$', '', code, flags=re.MULTILINE)

    # ── HIGH: Strip 3D kwargs from Axes (z_range, z_length) ──
    code = re.sub(r',\s*z_range\s*=\s*\[[^\]]*\]', '', code)
    code = re.sub(r',\s*z_length\s*=\s*[\d.]+', '', code)
    code = _track("3D removal (ThreeDAxes/Arrow3D/camera/z_range)", _prev, code)

    # ── HIGH: Old animation names (manimlib removals) ──
    _prev = code
    code = re.sub(r'\bFadeInFromDown\b', 'FadeIn', code)
    code = re.sub(r'\bFadeInFromLarge\b', 'FadeIn', code)
    code = re.sub(r'\bFadeOutAndShiftDown\b', 'FadeOut', code)
    code = re.sub(r'\bFadeOutAndShift\b', 'FadeOut', code)
    code = re.sub(r'\bFadeInFrom\b', 'FadeIn', code)
    code = re.sub(r'\bFadeInFromPoint\b', 'FadeIn', code)
    code = re.sub(r'\bShowPassingFlashAround\b', 'Circumscribe', code)
    code = re.sub(r'\bParametricSurface\b', 'Surface', code)
    code = re.sub(r'\bplay_all\b', 'play', code)
    code = re.sub(r'self\.dither\(', 'self.wait(', code)
    code = re.sub(r'^\s*self\.embed\(\)\s*$', '', code, flags=re.MULTILINE)
    code = _track("old manimlib animation names", _prev, code)

    # ── HIGH: GraphScene removal ──
    code = re.sub(r'\(GraphScene\)', '(Scene)', code)
    code = re.sub(r'^\s*self\.setup_axes\(\)\s*$', '', code, flags=re.MULTILINE)

    # ── HIGH: Empty string mobjects (crash Manim CE) ──
    code = re.sub(r'Text\(\s*""\s*\)', 'Text(" ")', code)
    code = re.sub(r"Text\(\s*''\s*\)", "Text(' ')", code)
    code = re.sub(r'MathTex\(\s*""\s*\)', r'MathTex(r"\\quad")', code)
    code = re.sub(r"MathTex\(\s*''\s*\)", r"MathTex(r'\\quad')", code)
    code = re.sub(r'Tex\(\s*""\s*\)', r'Tex(r"\\quad")', code)
    code = re.sub(r"Tex\(\s*''\s*\)", r"Tex(r'\\quad')", code)

    # ── HIGH: get_coordinate_labels() doesn't accept font_size/color kwargs ──
    # Replace with get_coordinate_labels() (no kwargs) or just remove the call
    code = re.sub(
        r'\.get_coordinate_labels\([^)]*\)',
        '.get_coordinate_labels()',
        code,
    )

    # ── HIGH: self.play() with no args (ValueError) ──
    code = re.sub(r'^\s*self\.play\(\s*\)\s*$', '', code, flags=re.MULTILINE)

    # ── HIGH: manimlib CONFIG dict pattern ──
    code = re.sub(r'^\s*CONFIG\s*=\s*\{[^}]*\}\s*$', '', code, flags=re.MULTILINE)

    # ── HIGH: manimlib import path ──
    code = re.sub(r'from\s+manimlib\.imports\s+import\s+\*', 'from manim import *', code)
    code = re.sub(r'from\s+manimlib\s+import\s+\*', 'from manim import *', code)

    # ── MEDIUM: MathTex $ sign removal (already in math mode) ──
    code = re.sub(r'MathTex\(\s*r?\"\$', 'MathTex(r"', code)
    code = re.sub(r'\$\"\s*\)', '")', code)

    # ��─ MEDIUM: Ensure numpy is imported if np. is used ──
    if 'np.' in code and 'import numpy' not in code:
        # Add import after the first import line
        code = re.sub(
            r'(from manim import \*)',
            r'\1\nimport numpy as np',
            code,
            count=1,
        )

    # ── MEDIUM: Remove dangerous imports ──
    for dangerous in ['import os', 'import sys', 'import subprocess',
                      'import socket', 'import shutil', 'import random']:
        code = re.sub(rf'^\s*{re.escape(dangerous)}\b.*$', '', code, flags=re.MULTILINE)

    # ── LOW: Clean up empty lines from removals ──
    code = re.sub(r'\n{3,}', '\n\n', code)

    if fixes_applied:
        logger.info("sanitize_script applied %d fixes: %s", len(fixes_applied), ", ".join(fixes_applied))
    else:
        logger.info("sanitize_script: no fixes needed")

    return code


def strip_voiceover(code: str) -> str:
    """Convert a voiceover-based script to a plain Scene script.

    Replaces OctoflashScene with Scene, removes voiceover context managers,
    adds self.wait() calls, and removes OctoflashScene imports.
    """
    # Replace class inheritance — use OctoflashSceneNoVoice so the brand watermark is preserved
    code = re.sub(
        r'class\s+(\w+)\s*\(\s*OctoflashScene\s*\)',
        r'class \1(OctoflashSceneNoVoice)',
        code,
    )

    # Replace OctoflashScene with OctoflashSceneNoVoice in imports
    code = re.sub(r'\bOctoflashScene\b(?!NoVoice)', 'OctoflashSceneNoVoice', code)
    # Clean up double commas or trailing commas before )
    code = re.sub(r',\s*,', ',', code)
    code = re.sub(r',\s*\)', ')', code)
    code = re.sub(r'\(\s*,', '(', code)

    # Add background color setting after def construct
    if 'self.camera.background_color' not in code:
        code = re.sub(
            r'(def construct\(self\):)',
            r'\1\n        self.camera.background_color = BG_COLOR',
            code,
        )

    # Replace voiceover blocks: convert `with self.voiceover(text="...") as tracker:` to just the body
    # Remove the with statement line
    code = re.sub(
        r'^(\s*)with self\.voiceover\(text=["\'].*?["\']\) as tracker:\s*$',
        r'\1# --- voiceover section ---',
        code,
        flags=re.MULTILINE,
    )

    # Remove tracker.get_remaining_duration lines and associated if/wait
    code = re.sub(r'^\s*remaining\s*=\s*tracker\.get_remaining_duration.*$', '', code, flags=re.MULTILINE)
    code = re.sub(r'^\s*if\s+remaining\s*>\s*0\s*:\s*$', '', code, flags=re.MULTILINE)
    code = re.sub(r'^\s*self\.wait\(remaining\)\s*$', '        self.wait(2)', code, flags=re.MULTILINE)

    # Dedent the body that was inside the with block (remove one level of indentation)
    lines = code.split('\n')
    result = []
    in_voiceover_section = False
    for line in lines:
        if '# --- voiceover section ---' in line:
            in_voiceover_section = True
            result.append(line)
            continue
        if in_voiceover_section:
            # Check if this line is deeper indented (was inside the with block)
            stripped = line.lstrip()
            if stripped and not line.startswith('        '):
                in_voiceover_section = False
            elif stripped:
                # Remove one level of indentation (4 spaces)
                if line.startswith('            '):
                    line = '        ' + line[12:]
        result.append(line)
    code = '\n'.join(result)

    return code


def _extract_python_code(text: str) -> str | None:
    """Extract Python code from a markdown code block in Claude's response."""
    match = re.search(r"```python\s*\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    match = re.search(r"```\s*\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    stripped = text.strip()
    if stripped.startswith(("import ", "from ", '"""', "# ")):
        return stripped

    return None


def _sample_items(items: list, count: int) -> list:
    """Evenly sample `count` items from a list."""
    if len(items) <= count:
        return list(items)
    step = len(items) / count
    return [items[int(i * step)] for i in range(count)]


def _image_to_base64(path: Path) -> str | None:
    """Read an image file and return base64-encoded string."""
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except Exception:
        return None
