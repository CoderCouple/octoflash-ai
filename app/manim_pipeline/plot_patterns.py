"""
Drop-in Manim CE plot and graph animation patterns.

Each pattern is a standalone Scene that renders correctly with Manim CE.
All patterns follow the Octoflash screen zone layout:
  - TOP ZONE   (y=3.0 to 4.0): title.to_edge(UP, buff=0.3)
  - MIDDLE ZONE (y=-2.5 to 3.0): axes, graphs, diagrams
  - BOTTOM ZONE (y=-3.2 to -4.0): caption.to_edge(DOWN, buff=0.4)

Usage: Copy the relevant construct() body into a generated scene.
       All patterns use Scene (no voiceover) for testing portability.
"""

from manim import *
import numpy as np


# ── Brand colors (same as styles.py, duplicated for standalone rendering) ─────
BG       = "#000000"
BLUE     = "#4fc3f7"
ORANGE   = "#ff9800"
GREEN    = "#66bb6a"
RED      = "#ef5350"
PURPLE   = "#ab47bc"
YELLOW   = "#ffee58"
CYAN     = "#26c6da"
PINK     = "#ec407a"
WHITE    = "#ffffff"
DIM      = "#9e9e9e"


# ═════════════════════════════════════════════════════════════════════════════════
# 1. ANIMATED FUNCTION PLOTTING
#    Draw curves being traced out with a moving dot, not just appearing.
# ═════════════════════════════════════════════════════════════════════════════════

class AnimatedFunctionPlotting(Scene):
    """Traces a curve in real time with a dot + growing trail."""

    def construct(self):
        self.camera.background_color = BG

        title = Text("Function Tracing", font_size=52, color=WHITE, weight="BOLD")
        title.to_edge(UP, buff=0.3)
        self.play(FadeIn(title))

        caption = Text("Watching sin(x) being drawn", font_size=20, color=DIM)
        caption.to_edge(DOWN, buff=0.4)
        self.play(FadeIn(caption), run_time=0.5)

        # ── Axes ──
        axes = Axes(
            x_range=[-1, 2 * np.pi + 0.5, np.pi / 2],
            y_range=[-1.5, 1.5, 0.5],
            x_length=10,
            y_length=3.5,
            axis_config={"color": DIM, "stroke_width": 2},
        ).shift(DOWN * 0.3)

        x_labels = axes.get_axis_labels(
            x_label=MathTex("x", font_size=24),
            y_label=MathTex("y", font_size=24),
        )
        self.play(Create(axes), Write(x_labels), run_time=1)

        # ── Formula ──
        formula = MathTex(r"f(x) = \sin(x)", font_size=36, color=WHITE)
        formula.shift(UP * 1.8)
        self.play(Write(formula), run_time=0.8)

        # ── Animated trace with ValueTracker ──
        t = ValueTracker(-1)

        # The curve grows as t advances
        traced_curve = always_redraw(
            lambda: axes.plot(
                lambda x: np.sin(x),
                x_range=[-1, t.get_value()],
                color=CYAN,
                stroke_width=4,
            )
        )

        # Moving dot at the tip
        dot = always_redraw(
            lambda: Dot(
                axes.c2p(t.get_value(), np.sin(t.get_value())),
                color=ORANGE,
                radius=0.08,
            )
        )

        # Coordinate readout
        coord_label = always_redraw(
            lambda: MathTex(
                rf"({t.get_value():.1f},\;{np.sin(t.get_value()):.2f})",
                font_size=20,
                color=ORANGE,
            ).next_to(
                axes.c2p(t.get_value(), np.sin(t.get_value())),
                UR, buff=0.15,
            )
        )

        self.play(FadeIn(dot), FadeIn(coord_label), run_time=0.3)
        self.add(traced_curve)

        # Animate the trace
        self.play(
            t.animate.set_value(2 * np.pi + 0.3),
            run_time=5,
            rate_func=linear,
        )

        self.wait(1)
        self.play(FadeOut(VGroup(
            axes, x_labels, formula, traced_curve, dot, coord_label, caption,
        )), run_time=0.6)


# ═════════════════════════════════════════════════════════════════════════════════
# 2. MULTIPLE FUNCTION COMPARISON
#    Overlay 2-3 functions with color-coded legend, then morph between them.
# ═════════════════════════════════════════════════════════════════════════════════

class MultipleFunctionComparison(Scene):
    """Overlays sin, cos, and tan, then morphs sin into cos."""

    def construct(self):
        self.camera.background_color = BG

        title = Text("Function Comparison", font_size=52, color=WHITE, weight="BOLD")
        title.to_edge(UP, buff=0.3)
        self.play(FadeIn(title))

        caption = Text("Comparing trigonometric functions", font_size=20, color=DIM)
        caption.to_edge(DOWN, buff=0.4)
        self.play(FadeIn(caption), run_time=0.5)

        # ── Axes ──
        axes = Axes(
            x_range=[-2 * np.pi, 2 * np.pi, np.pi / 2],
            y_range=[-2, 2, 0.5],
            x_length=10,
            y_length=3.2,
            axis_config={"color": DIM, "stroke_width": 2},
        ).shift(DOWN * 0.3)

        labels = axes.get_axis_labels(
            x_label=MathTex("x", font_size=24),
            y_label=MathTex("y", font_size=24),
        )
        self.play(Create(axes), Write(labels), run_time=1)

        # ── Plot functions one by one ──
        sin_graph = axes.plot(lambda x: np.sin(x), color=CYAN, stroke_width=3)
        cos_graph = axes.plot(lambda x: np.cos(x), color=ORANGE, stroke_width=3)
        # Clamp tan to avoid huge spikes
        tan_graph = axes.plot(
            lambda x: np.clip(np.tan(x), -2, 2),
            color=GREEN,
            stroke_width=2,
            discontinuities=[-(3 * np.pi / 2), -np.pi / 2, np.pi / 2, 3 * np.pi / 2],
            dt=0.05,
        )

        # ── Color-coded legend ──
        legend_items = VGroup(
            VGroup(Line(LEFT * 0.3, RIGHT * 0.3, color=CYAN, stroke_width=4),
                   MathTex(r"\sin(x)", font_size=20, color=CYAN)).arrange(RIGHT, buff=0.15),
            VGroup(Line(LEFT * 0.3, RIGHT * 0.3, color=ORANGE, stroke_width=4),
                   MathTex(r"\cos(x)", font_size=20, color=ORANGE)).arrange(RIGHT, buff=0.15),
            VGroup(Line(LEFT * 0.3, RIGHT * 0.3, color=GREEN, stroke_width=4),
                   MathTex(r"\tan(x)", font_size=20, color=GREEN)).arrange(RIGHT, buff=0.15),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.15)
        legend_bg = SurroundingRectangle(
            legend_items, color=DIM, fill_color=BG,
            fill_opacity=0.8, corner_radius=0.1, buff=0.15,
        )
        legend = VGroup(legend_bg, legend_items)
        legend.move_to(axes.c2p(2 * np.pi - 1, 1.6))

        # Staggered creation
        self.play(Create(sin_graph), run_time=1.5)
        self.play(Create(cos_graph), run_time=1.5)
        self.play(Create(tan_graph), FadeIn(legend), run_time=1.5)
        self.wait(1)

        # ── Morph sin into cos ──
        morph_cap = Text("Morphing sin(x) into cos(x)", font_size=20, color=DIM)
        morph_cap.to_edge(DOWN, buff=0.4)
        self.play(FadeOut(caption), FadeIn(morph_cap), run_time=0.4)

        self.play(
            FadeOut(tan_graph),
            run_time=0.5,
        )
        sin_to_cos = axes.plot(lambda x: np.cos(x), color=CYAN, stroke_width=3)
        self.play(
            Transform(sin_graph, sin_to_cos),
            run_time=2,
            rate_func=smooth,
        )
        self.wait(1)

        # ── Dynamic amplitude with ValueTracker ──
        amp_cap = Text("Varying amplitude A*cos(x)", font_size=20, color=DIM)
        amp_cap.to_edge(DOWN, buff=0.4)
        self.play(FadeOut(morph_cap), FadeIn(amp_cap), run_time=0.4)

        A = ValueTracker(1.0)
        dynamic_graph = always_redraw(
            lambda: axes.plot(
                lambda x: A.get_value() * np.cos(x),
                color=PINK,
                stroke_width=4,
            )
        )
        amp_label = always_redraw(
            lambda: MathTex(
                rf"A = {A.get_value():.2f}",
                font_size=28, color=PINK,
            ).shift(UP * 1.8 + RIGHT * 3)
        )

        self.play(
            FadeOut(sin_graph), FadeOut(cos_graph), FadeOut(legend),
            FadeIn(dynamic_graph), FadeIn(amp_label),
            run_time=0.8,
        )
        self.play(A.animate.set_value(0.2), run_time=2)
        self.play(A.animate.set_value(1.8), run_time=2)
        self.play(A.animate.set_value(1.0), run_time=1)

        self.wait(0.5)
        self.play(FadeOut(VGroup(
            axes, labels, dynamic_graph, amp_label, amp_cap, title,
        )), run_time=0.6)


# ═════════════════════════════════════════════════════════════════════════════════
# 3. PARAMETRIC CURVES
#    Spirals, Lissajous, heart curve with animated parameter t.
# ═════════════════════════════════════════════════════════════════════════════════

class ParametricCurves(Scene):
    """Traces parametric curves: spiral, Lissajous, heart."""

    def construct(self):
        self.camera.background_color = BG

        title = Text("Parametric Curves", font_size=52, color=WHITE, weight="BOLD")
        title.to_edge(UP, buff=0.3)
        self.play(FadeIn(title))

        # ── Part A: Archimedean Spiral ──
        cap_a = Text("Archimedean Spiral: r = a + b*t", font_size=20, color=DIM)
        cap_a.to_edge(DOWN, buff=0.4)
        self.play(FadeIn(cap_a), run_time=0.5)

        axes_a = Axes(
            x_range=[-4, 4, 1], y_range=[-3, 3, 1],
            x_length=7, y_length=3.5,
            axis_config={"color": DIM, "stroke_width": 1},
        ).shift(DOWN * 0.3)
        self.play(Create(axes_a), run_time=0.5)

        t_spiral = ValueTracker(0.01)
        spiral = always_redraw(
            lambda: ParametricFunction(
                lambda t: axes_a.c2p(
                    (0.15 * t) * np.cos(t),
                    (0.15 * t) * np.sin(t),
                ),
                t_range=[0, t_spiral.get_value()],
                color=CYAN,
                stroke_width=3,
            )
        )
        dot_a = always_redraw(
            lambda: Dot(
                axes_a.c2p(
                    (0.15 * t_spiral.get_value()) * np.cos(t_spiral.get_value()),
                    (0.15 * t_spiral.get_value()) * np.sin(t_spiral.get_value()),
                ),
                color=ORANGE, radius=0.06,
            )
        )

        formula_a = MathTex(
            r"x = bt\cos t,\quad y = bt\sin t",
            font_size=30, color=WHITE,
        ).shift(UP * 1.8)
        self.play(Write(formula_a), run_time=0.8)
        self.add(spiral, dot_a)
        self.play(t_spiral.animate.set_value(6 * np.pi), run_time=4, rate_func=linear)
        self.wait(0.5)

        self.play(FadeOut(VGroup(axes_a, spiral, dot_a, formula_a, cap_a)), run_time=0.5)

        # ── Part B: Lissajous Curve ──
        cap_b = Text("Lissajous: x=sin(at), y=sin(bt+d)", font_size=20, color=DIM)
        cap_b.to_edge(DOWN, buff=0.4)
        self.play(FadeIn(cap_b), run_time=0.4)

        axes_b = Axes(
            x_range=[-1.5, 1.5, 0.5], y_range=[-1.5, 1.5, 0.5],
            x_length=6, y_length=3.5,
            axis_config={"color": DIM, "stroke_width": 1},
        ).shift(DOWN * 0.3)
        self.play(Create(axes_b), run_time=0.5)

        # Animate the frequency ratio
        freq_a, freq_b = 3, 2
        delta = ValueTracker(0)

        lissajous = always_redraw(
            lambda: ParametricFunction(
                lambda t: axes_b.c2p(
                    np.sin(freq_a * t + delta.get_value()),
                    np.sin(freq_b * t),
                ),
                t_range=[0, 2 * np.pi],
                color=PURPLE,
                stroke_width=3,
            )
        )
        formula_b = MathTex(
            r"x = \sin(3t + \delta),\quad y = \sin(2t)",
            font_size=30, color=WHITE,
        ).shift(UP * 1.8)
        delta_label = always_redraw(
            lambda: MathTex(
                rf"\delta = {delta.get_value():.2f}",
                font_size=24, color=PURPLE,
            ).shift(UP * 1.8 + RIGHT * 4)
        )

        self.play(Write(formula_b), run_time=0.8)
        self.add(lissajous, delta_label)
        self.play(delta.animate.set_value(2 * np.pi), run_time=4, rate_func=linear)
        self.wait(0.5)

        self.play(FadeOut(VGroup(axes_b, lissajous, formula_b, delta_label, cap_b)), run_time=0.5)

        # ── Part C: Heart Curve ──
        cap_c = Text("Heart Curve", font_size=20, color=DIM)
        cap_c.to_edge(DOWN, buff=0.4)
        self.play(FadeIn(cap_c), run_time=0.4)

        axes_c = Axes(
            x_range=[-2.5, 2.5, 1], y_range=[-2, 2.5, 1],
            x_length=6, y_length=3.5,
            axis_config={"color": DIM, "stroke_width": 1},
        ).shift(DOWN * 0.3)
        self.play(Create(axes_c), run_time=0.5)

        t_heart = ValueTracker(0.01)
        heart = always_redraw(
            lambda: ParametricFunction(
                lambda t: axes_c.c2p(
                    16 * np.sin(t) ** 3 / 10,
                    (13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t)) / 10,
                ),
                t_range=[0, t_heart.get_value()],
                color=RED,
                stroke_width=4,
            )
        )
        formula_c = MathTex(
            r"x = 16\sin^3 t", font_size=28, color=WHITE,
        ).shift(UP * 1.8 + LEFT * 2)
        formula_c2 = MathTex(
            r"y = 13\cos t - 5\cos 2t - \ldots",
            font_size=28, color=WHITE,
        ).next_to(formula_c, RIGHT, buff=0.3)

        self.play(Write(formula_c), Write(formula_c2), run_time=0.8)
        self.add(heart)
        self.play(t_heart.animate.set_value(2 * np.pi), run_time=4, rate_func=linear)
        self.wait(1)

        self.play(FadeOut(VGroup(
            axes_c, heart, formula_c, formula_c2, cap_c, title,
        )), run_time=0.6)


# ═════════════════════════════════════════════════════════════════════════════════
# 4. HEATMAPS AND CONTOUR PLOTS
#    Loss landscape / function surface as a 2D pixel heatmap.
#    Manim CE has no built-in Heatmap, so we build one from colored squares.
# ═════════════════════════════════════════════════════════════════════════════════

class HeatmapContourPlot(Scene):
    """2D heatmap of f(x,y) = sin(x)*cos(y) with contour overlay."""

    def construct(self):
        self.camera.background_color = BG

        title = Text("Loss Landscape Heatmap", font_size=52, color=WHITE, weight="BOLD")
        title.to_edge(UP, buff=0.3)
        self.play(FadeIn(title))

        caption = Text("f(x,y) = sin(x)*cos(y)", font_size=20, color=DIM)
        caption.to_edge(DOWN, buff=0.4)
        self.play(FadeIn(caption), run_time=0.5)

        formula = MathTex(
            r"f(x,y) = \sin(x)\cos(y)", font_size=36, color=WHITE,
        ).shift(UP * 1.8)
        self.play(Write(formula), run_time=0.8)

        # ── Build heatmap from colored squares ──
        resolution = 30  # 30x30 grid
        cell_size = 6.0 / resolution  # total width = 6 manim units
        x_vals = np.linspace(-np.pi, np.pi, resolution)
        y_vals = np.linspace(-np.pi, np.pi, resolution)

        heatmap_group = VGroup()
        for i, xv in enumerate(x_vals):
            for j, yv in enumerate(y_vals):
                val = np.sin(xv) * np.cos(yv)  # range [-1, 1]
                # Map to color: blue (low) -> white (mid) -> red (high)
                normalized = (val + 1) / 2  # [0, 1]
                if normalized < 0.5:
                    r_c = normalized * 2
                    g_c = normalized * 2
                    b_c = 1.0
                else:
                    r_c = 1.0
                    g_c = 2 * (1 - normalized)
                    b_c = 2 * (1 - normalized)

                sq = Square(
                    side_length=cell_size,
                    fill_opacity=0.9,
                    stroke_width=0,
                )
                sq.set_fill(rgb_to_color([r_c, g_c, b_c]))
                # Position: center at origin, shifted down
                sq.move_to(
                    np.array([
                        -3.0 + (i + 0.5) * cell_size,
                        -3.0 + (j + 0.5) * cell_size + 0.8,  # shift up from bottom
                        0,
                    ])
                )
                heatmap_group.add(sq)

        # Shift entire heatmap into MIDDLE ZONE
        heatmap_group.move_to(DOWN * 0.3)

        # Fade in row by row
        self.play(
            LaggedStart(
                *[FadeIn(sq) for sq in heatmap_group],
                lag_ratio=0.002,
                run_time=3,
            )
        )
        self.wait(0.5)

        # ── Add contour lines using ImplicitFunction ──
        contour_cap = Text("Adding contour lines", font_size=20, color=DIM)
        contour_cap.to_edge(DOWN, buff=0.4)
        self.play(FadeOut(caption), FadeIn(contour_cap), run_time=0.4)

        # Scale mapping: heatmap spans [-3,3] in screen coords for x in [-pi,pi]
        scale_x = 6.0 / (2 * np.pi)
        center_y_offset = heatmap_group.get_center()[1]

        contours = VGroup()
        for level in [-0.7, -0.3, 0.0, 0.3, 0.7]:
            contour = ImplicitFunction(
                lambda x, y, lv=level: np.sin(x / scale_x * np.pi) * np.cos(
                    (y - center_y_offset) / scale_x * np.pi
                ) - lv,
                x_range=[-3, 3],
                y_range=[center_y_offset - 3, center_y_offset + 3],
                color=WHITE,
                stroke_width=1.5,
                stroke_opacity=0.7,
            )
            contours.add(contour)

        self.play(Create(contours), run_time=2)
        self.wait(1)

        # ── Gradient descent dot animation ──
        gd_cap = Text("Gradient descent path", font_size=20, color=DIM)
        gd_cap.to_edge(DOWN, buff=0.4)
        self.play(FadeOut(contour_cap), FadeIn(gd_cap), run_time=0.4)

        # Start at a point and follow gradient descent
        gd_dot = Dot(heatmap_group.get_center() + UP * 1.5 + LEFT * 1.5, color=YELLOW, radius=0.1)
        gd_trail = TracedPath(gd_dot.get_center, stroke_color=YELLOW, stroke_width=2)
        self.add(gd_trail)
        self.play(FadeIn(gd_dot), run_time=0.3)

        # Animate descent path (predefined steps toward minimum at pi/2, 0)
        target = heatmap_group.get_center() + RIGHT * 1.5
        self.play(gd_dot.animate.move_to(target), run_time=3, rate_func=smooth)
        self.wait(1)

        self.play(FadeOut(VGroup(
            heatmap_group, contours, gd_dot, gd_trail, formula, gd_cap, title,
        )), run_time=0.6)


# ═════════════════════════════════════════════════════════════════════════════════
# 5. BAR CHART ANIMATIONS
#    Animated bar charts with morphing between datasets.
# ═════════════════════════════════════════════════════════════════════════════════

class BarChartAnimations(Scene):
    """BarChart creation, label animation, and value morphing."""

    def construct(self):
        self.camera.background_color = BG

        title = Text("Bar Chart Animation", font_size=52, color=WHITE, weight="BOLD")
        title.to_edge(UP, buff=0.3)
        self.play(FadeIn(title))

        caption = Text("Model accuracy comparison", font_size=20, color=DIM)
        caption.to_edge(DOWN, buff=0.4)
        self.play(FadeIn(caption), run_time=0.5)

        # ── Chart helper for consistent styling ──
        def make_chart(values):
            return BarChart(
                values=values,
                bar_names=["CNN", "RNN", "BERT", "SVM", "GPT"],
                y_range=[0, 100, 20],
                x_length=9,
                y_length=3.5,
                bar_colors=[BLUE, CYAN, GREEN, ORANGE, PINK],
                bar_fill_opacity=0.8,
                bar_stroke_width=1,
                bar_width=0.6,
            ).shift(DOWN * 0.3)

        # Start with near-zero bars, animate to real values
        chart_zero = make_chart([1, 1, 1, 1, 1])
        chart = make_chart([72, 85, 91, 68, 95])

        self.play(FadeIn(chart_zero), run_time=0.5)
        self.play(
            ReplacementTransform(chart_zero, chart),
            run_time=2,
            rate_func=smooth,
        )

        # ── Add value labels ──
        bar_labels = chart.get_bar_labels(font_size=20, color=WHITE)
        self.play(
            LaggedStart(
                *[FadeIn(lbl, shift=UP * 0.2) for lbl in bar_labels],
                lag_ratio=0.15,
                run_time=1.5,
            )
        )
        self.wait(1)

        # ── Morph to new values ──
        morph_cap = Text("Updated results after fine-tuning", font_size=20, color=DIM)
        morph_cap.to_edge(DOWN, buff=0.4)
        self.play(FadeOut(caption), FadeIn(morph_cap), FadeOut(bar_labels), run_time=0.4)

        chart_v2 = make_chart([88, 82, 96, 75, 98])
        self.play(
            ReplacementTransform(chart, chart_v2),
            run_time=2,
            rate_func=smooth,
        )
        chart = chart_v2  # update reference

        new_labels = chart.get_bar_labels(font_size=20, color=WHITE)
        self.play(FadeIn(new_labels), run_time=0.5)
        self.wait(1)

        # ── Highlight the winner ──
        highlight_cap = Text("GPT achieves highest accuracy", font_size=20, color=DIM)
        highlight_cap.to_edge(DOWN, buff=0.4)
        self.play(FadeOut(morph_cap), FadeIn(highlight_cap), run_time=0.4)

        winner_bar = chart.bars[4]
        highlight_rect = SurroundingRectangle(
            winner_bar, color=YELLOW, stroke_width=3, buff=0.05,
        )
        self.play(Create(highlight_rect), run_time=0.8)
        self.play(
            Indicate(winner_bar, color=YELLOW, scale_factor=1.05),
            run_time=1,
        )
        self.wait(1)

        self.play(FadeOut(VGroup(
            chart, new_labels, highlight_rect, highlight_cap, title,
        )), run_time=0.6)


# ═════════════════════════════════════════════════════════════════════════════════
# 6. COORDINATE SYSTEM TRANSFORMS
#    Showing how linear transforms distort space (3b1b Linear Algebra style).
#    NOTE: Uses NumberPlane + ApplyPointwiseFunction for standalone use
#    (LinearTransformationScene has specific inheritance requirements).
# ═════════════════════════════════════════════════════════════════════════════════

class CoordinateSystemTransforms(Scene):
    """Linear transform of the plane with basis vectors, 3b1b style."""

    def construct(self):
        self.camera.background_color = BG

        title = Text("Linear Transformations", font_size=52, color=WHITE, weight="BOLD")
        title.to_edge(UP, buff=0.3)
        self.play(FadeIn(title))

        caption = Text("How matrices transform space", font_size=20, color=DIM)
        caption.to_edge(DOWN, buff=0.4)
        self.play(FadeIn(caption), run_time=0.5)

        # ── Grid plane ──
        plane = NumberPlane(
            x_range=[-5, 5, 1],
            y_range=[-4, 4, 1],
            x_length=10,
            y_length=5,
            background_line_style={
                "stroke_color": BLUE,
                "stroke_width": 1,
                "stroke_opacity": 0.4,
            },
            axis_config={"color": DIM, "stroke_width": 2},
        ).shift(DOWN * 0.2)
        self.play(Create(plane), run_time=1.5)

        # ── Basis vectors ──
        i_hat = Arrow(
            plane.c2p(0, 0), plane.c2p(1, 0),
            buff=0, color=GREEN, stroke_width=5,
        )
        j_hat = Arrow(
            plane.c2p(0, 0), plane.c2p(0, 1),
            buff=0, color=RED, stroke_width=5,
        )
        i_label = MathTex(r"\hat{\imath}", font_size=28, color=GREEN)
        i_label.next_to(i_hat.get_end(), DOWN, buff=0.1)
        j_label = MathTex(r"\hat{\jmath}", font_size=28, color=RED)
        j_label.next_to(j_hat.get_end(), LEFT, buff=0.1)

        self.play(
            GrowArrow(i_hat), GrowArrow(j_hat),
            FadeIn(i_label), FadeIn(j_label),
            run_time=1,
        )

        # ── Matrix display ──
        matrix_tex = MathTex(
            r"A = \begin{bmatrix} 2 & 1 \\ 0 & 1 \end{bmatrix}",
            font_size=30, color=WHITE,
        ).shift(UP * 1.8 + RIGHT * 3.5)
        self.play(Write(matrix_tex), run_time=0.8)
        self.wait(0.5)

        # ── Apply shear transform ──
        transform_cap = Text("Applying the shear transform", font_size=20, color=DIM)
        transform_cap.to_edge(DOWN, buff=0.4)
        self.play(FadeOut(caption), FadeIn(transform_cap), run_time=0.4)

        # The 2x2 matrix: [[2, 1], [0, 1]]
        matrix = np.array([[2, 1], [0, 1]])

        def linear_transform(point):
            x, y, z = point
            new = matrix @ np.array([x, y])
            return np.array([new[0], new[1], z])

        # Save ghost of original
        ghost_plane = plane.copy().set_stroke(opacity=0.15)
        ghost_i = i_hat.copy().set_stroke(opacity=0.3)
        ghost_j = j_hat.copy().set_stroke(opacity=0.3)
        self.add(ghost_plane, ghost_i, ghost_j)

        # Animate the transform
        self.play(
            plane.animate.apply_function(linear_transform),
            i_hat.animate.apply_function(linear_transform),
            j_hat.animate.apply_function(linear_transform),
            i_label.animate.move_to(plane.c2p(2, 0) + DOWN * 0.2),
            j_label.animate.move_to(plane.c2p(1, 1) + LEFT * 0.2),
            run_time=3,
            rate_func=smooth,
        )

        # ── Show where basis vectors land ──
        new_i = MathTex(
            r"\hat{\imath} \to \begin{bmatrix} 2 \\ 0 \end{bmatrix}",
            font_size=24, color=GREEN,
        ).shift(UP * 1.8 + LEFT * 3.5)
        new_j = MathTex(
            r"\hat{\jmath} \to \begin{bmatrix} 1 \\ 1 \end{bmatrix}",
            font_size=24, color=RED,
        ).next_to(new_i, DOWN, buff=0.2)

        self.play(Write(new_i), Write(new_j), run_time=1)
        self.wait(1.5)

        # ── Show determinant ──
        det_cap = Text("Determinant = area scale factor", font_size=20, color=DIM)
        det_cap.to_edge(DOWN, buff=0.4)
        self.play(FadeOut(transform_cap), FadeIn(det_cap), run_time=0.4)

        det_tex = MathTex(
            r"\det(A) = 2 \cdot 1 - 1 \cdot 0 = 2",
            font_size=28, color=YELLOW,
        ).shift(DOWN * 2.2)
        self.play(Write(det_tex), run_time=1)
        self.wait(1)

        self.play(FadeOut(VGroup(
            plane, i_hat, j_hat, i_label, j_label,
            ghost_plane, ghost_i, ghost_j,
            matrix_tex, new_i, new_j, det_tex, det_cap, title,
        )), run_time=0.6)


# ═════════════════════════════════════════════════════════════════════════════════
# 7. VECTOR FIELD VISUALIZATIONS
#    Arrow fields showing gradient, plus StreamLines for flow.
# ═════════════════════════════════════════════════════════════════════════════════

class VectorFieldVisualizations(Scene):
    """ArrowVectorField + StreamLines with animated transitions."""

    def construct(self):
        self.camera.background_color = BG

        title = Text("Vector Fields", font_size=52, color=WHITE, weight="BOLD")
        title.to_edge(UP, buff=0.3)
        self.play(FadeIn(title))

        # ── Part A: Arrow Vector Field (gradient field) ──
        caption_a = Text("Gradient field of f(x,y)=x^2+y^2", font_size=20, color=DIM)
        caption_a.to_edge(DOWN, buff=0.4)
        self.play(FadeIn(caption_a), run_time=0.5)

        formula = MathTex(
            r"\nabla f = (2x,\; 2y)", font_size=36, color=WHITE,
        ).shift(UP * 1.6)
        self.play(Write(formula), run_time=0.8)

        # Gradient of x^2 + y^2 is (2x, 2y) — points radially outward
        arrow_field = ArrowVectorField(
            lambda pos: np.array([2 * pos[0], 2 * pos[1], 0]),
            x_range=[-5, 5, 0.8],
            y_range=[-2.5, 2.5, 0.8],
            length_func=lambda x: 0.4 * sigmoid(x),
            colors=[BLUE, CYAN, GREEN, YELLOW, RED],
        ).shift(DOWN * 0.3)

        self.play(
            LaggedStart(
                *[GrowArrow(arrow) for arrow in arrow_field],
                lag_ratio=0.01,
                run_time=3,
            )
        )
        self.wait(1)

        # ── Morph to a rotational field ──
        rot_cap = Text("Rotational field: (-y, x)", font_size=20, color=DIM)
        rot_cap.to_edge(DOWN, buff=0.4)
        self.play(FadeOut(caption_a), FadeIn(rot_cap), run_time=0.4)

        rot_formula = MathTex(
            r"\mathbf{F} = (-y,\; x)", font_size=36, color=WHITE,
        ).shift(UP * 1.6)
        self.play(TransformMatchingTex(formula, rot_formula), run_time=1)

        rot_field = ArrowVectorField(
            lambda pos: np.array([-pos[1], pos[0], 0]),
            x_range=[-5, 5, 0.8],
            y_range=[-2.5, 2.5, 0.8],
            length_func=lambda x: 0.4 * sigmoid(x),
            colors=[PURPLE, PINK, ORANGE, YELLOW],
        ).shift(DOWN * 0.3)

        self.play(arrow_field.animate.become(rot_field), run_time=2)
        self.wait(1)

        self.play(FadeOut(VGroup(arrow_field, rot_formula, rot_cap)), run_time=0.5)

        # ── Part B: StreamLines ──
        stream_cap = Text("Stream lines show flow trajectories", font_size=20, color=DIM)
        stream_cap.to_edge(DOWN, buff=0.4)
        self.play(FadeIn(stream_cap), run_time=0.4)

        stream_formula = MathTex(
            r"\mathbf{F} = (\sin y,\; \cos x)", font_size=36, color=WHITE,
        ).shift(UP * 1.6)
        self.play(Write(stream_formula), run_time=0.8)

        stream_lines = StreamLines(
            lambda pos: np.array([np.sin(pos[1]), np.cos(pos[0]), 0]),
            x_range=[-5, 5, 0.3],
            y_range=[-2.5, 2.5, 0.3],
            stroke_width=2,
            colors=[BLUE, CYAN, GREEN, YELLOW],
            padding=1,
        ).shift(DOWN * 0.3)

        self.play(stream_lines.create(), run_time=4)
        self.wait(1)

        # Animate flow
        flow_cap = Text("Watching the flow evolve", font_size=20, color=DIM)
        flow_cap.to_edge(DOWN, buff=0.4)
        self.play(FadeOut(stream_cap), FadeIn(flow_cap), run_time=0.4)

        stream_lines.start_animation(warm_up=False, flow_speed=1.5)
        self.wait(4)
        stream_lines.end_animation()

        self.wait(0.5)
        self.play(FadeOut(VGroup(
            stream_lines, stream_formula, flow_cap, title,
        )), run_time=0.6)


# ═════════════════════════════════════════════════════════════════════════════════
# 8. AREA UNDER CURVE ANIMATIONS
#    Riemann sums with decreasing dx, then smooth area fill.
# ═════════════════════════════════════════════════════════════════════════════════

class AreaUnderCurve(Scene):
    """Riemann rectangles converging to integral, then smooth fill."""

    def construct(self):
        self.camera.background_color = BG

        title = Text("Area Under a Curve", font_size=52, color=WHITE, weight="BOLD")
        title.to_edge(UP, buff=0.3)
        self.play(FadeIn(title))

        caption = Text("Approximating with Riemann sums", font_size=20, color=DIM)
        caption.to_edge(DOWN, buff=0.4)
        self.play(FadeIn(caption), run_time=0.5)

        # ── Axes ──
        axes = Axes(
            x_range=[-0.5, 4.5, 1],
            y_range=[-0.5, 5, 1],
            x_length=9,
            y_length=3.5,
            axis_config={"color": DIM, "stroke_width": 2},
        ).shift(DOWN * 0.3)

        labels = axes.get_axis_labels(
            x_label=MathTex("x", font_size=24),
            y_label=MathTex("y", font_size=24),
        )
        self.play(Create(axes), Write(labels), run_time=1)

        # ── Plot function ──
        func = lambda x: 0.25 * x ** 2 + 0.5
        graph = axes.plot(func, x_range=[0, 4], color=CYAN, stroke_width=4)

        formula = MathTex(
            r"f(x) = \frac{1}{4}x^2 + \frac{1}{2}",
            font_size=32, color=WHITE,
        ).shift(UP * 1.8)
        self.play(Write(formula), Create(graph), run_time=1.5)
        self.wait(0.5)

        # ── Riemann sums with decreasing dx ──
        dx_values = [1.0, 0.5, 0.25, 0.1]
        prev_rects = None

        for i, dx in enumerate(dx_values):
            dx_label = MathTex(
                rf"\Delta x = {dx}", font_size=24, color=ORANGE,
            ).shift(UP * 1.8 + RIGHT * 4)

            rects = axes.get_riemann_rectangles(
                graph,
                x_range=[0, 4],
                dx=dx,
                color=(BLUE, GREEN),
                fill_opacity=0.5,
                stroke_width=0.5,
                stroke_color=WHITE,
            )

            if prev_rects is None:
                self.play(
                    Create(rects),
                    FadeIn(dx_label),
                    run_time=1.5,
                )
            else:
                self.play(
                    Transform(prev_rects, rects),
                    Transform(prev_dx_label, dx_label),
                    run_time=1.5,
                )
                # Keep reference to the original for subsequent transforms
                rects = prev_rects

            prev_rects = rects
            prev_dx_label = dx_label
            self.wait(0.5)

        # ── Converge to smooth area ──
        converge_cap = Text("As dx approaches 0, we get the integral", font_size=20, color=DIM)
        converge_cap.to_edge(DOWN, buff=0.4)
        self.play(FadeOut(caption), FadeIn(converge_cap), run_time=0.4)

        area = axes.get_area(
            graph,
            x_range=[0, 4],
            color=[BLUE, GREEN],
            opacity=0.5,
        )

        self.play(
            FadeOut(prev_rects),
            FadeOut(prev_dx_label),
            FadeIn(area),
            run_time=1.5,
        )

        # ── Integral notation ──
        integral_formula = MathTex(
            r"\int_0^4 \left(\frac{x^2}{4} + \frac{1}{2}\right) dx",
            font_size=32, color=WHITE,
        ).shift(UP * 1.8)
        self.play(TransformMatchingTex(formula, integral_formula), run_time=1)

        # ── Compute result ──
        result = MathTex(
            r"= \left[\frac{x^3}{12} + \frac{x}{2}\right]_0^4 = \frac{64}{12} + 2 = \frac{22}{3}",
            font_size=28, color=YELLOW,
        ).next_to(integral_formula, DOWN, buff=0.3)
        self.play(Write(result), run_time=1.5)
        self.wait(1)

        # ── Animated sweep showing area growing ──
        sweep_cap = Text("Watching area accumulate", font_size=20, color=DIM)
        sweep_cap.to_edge(DOWN, buff=0.4)
        self.play(
            FadeOut(converge_cap), FadeIn(sweep_cap),
            FadeOut(area), FadeOut(result),
            run_time=0.5,
        )

        x_upper = ValueTracker(0.01)
        growing_area = always_redraw(
            lambda: axes.get_area(
                graph,
                x_range=[0, np.maximum(0.01, x_upper.get_value())],
                color=[BLUE, GREEN],
                opacity=0.5,
            )
        )
        area_val_label = always_redraw(
            lambda: MathTex(
                rf"A = {self._compute_area(x_upper.get_value()):.2f}",
                font_size=24, color=YELLOW,
            ).shift(UP * 1.2 + RIGHT * 3.5)
        )

        # Vertical line at x_upper
        sweep_line = always_redraw(
            lambda: DashedLine(
                axes.c2p(x_upper.get_value(), 0),
                axes.c2p(x_upper.get_value(), func(x_upper.get_value())),
                color=ORANGE,
                stroke_width=2,
            )
        )

        self.add(growing_area, sweep_line, area_val_label)
        self.play(
            x_upper.animate.set_value(4.0),
            run_time=4,
            rate_func=linear,
        )
        self.wait(1)

        self.play(FadeOut(VGroup(
            axes, labels, graph, integral_formula, growing_area,
            sweep_line, area_val_label, sweep_cap, title,
        )), run_time=0.6)

    @staticmethod
    def _compute_area(x):
        """Antiderivative of 0.25*x^2 + 0.5 evaluated from 0 to x."""
        return (x ** 3) / 12 + x / 2


# ═════════════════════════════════════════════════════════════════════════════════
# PATTERN INDEX
# Quick-reference for script_generator.py to select patterns by keyword.
# ═════════════════════════════════════════════════════════════════════════════════

PLOT_PATTERNS = {
    "function_trace": {
        "class": "AnimatedFunctionPlotting",
        "keywords": ["trace", "draw curve", "plot function", "graph drawing"],
        "description": "Trace a function being drawn with a moving dot and growing trail.",
    },
    "function_comparison": {
        "class": "MultipleFunctionComparison",
        "keywords": ["compare", "overlay", "multiple functions", "legend", "morph"],
        "description": "Overlay 2-3 functions with legend, morph between them, vary amplitude.",
    },
    "parametric_curves": {
        "class": "ParametricCurves",
        "keywords": ["parametric", "spiral", "lissajous", "heart", "polar"],
        "description": "Trace parametric curves: spiral, Lissajous, heart with animated parameter.",
    },
    "heatmap": {
        "class": "HeatmapContourPlot",
        "keywords": ["heatmap", "contour", "loss landscape", "2d surface", "gradient descent"],
        "description": "2D heatmap from colored squares with contour lines and gradient descent.",
    },
    "bar_chart": {
        "class": "BarChartAnimations",
        "keywords": ["bar chart", "histogram", "comparison chart", "bar graph"],
        "description": "Animated bar chart with grow-from-zero, value morphing, and highlighting.",
    },
    "linear_transform": {
        "class": "CoordinateSystemTransforms",
        "keywords": ["linear transform", "matrix", "basis vectors", "shear", "determinant"],
        "description": "3b1b-style linear transformation: grid, basis vectors, ghost plane.",
    },
    "vector_field": {
        "class": "VectorFieldVisualizations",
        "keywords": ["vector field", "gradient", "flow", "stream lines", "arrows"],
        "description": "ArrowVectorField + StreamLines with morphing and flow animation.",
    },
    "area_under_curve": {
        "class": "AreaUnderCurve",
        "keywords": ["integral", "riemann", "area", "sum", "calculus"],
        "description": "Riemann sums converging to integral with animated area sweep.",
    },
}
