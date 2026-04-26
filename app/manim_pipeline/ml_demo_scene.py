"""
Demo scene showcasing all ML visualization helpers from ml_visuals.py.

Render with:
    manim -qm app/manim_pipeline/ml_demo_scene.py MLVisualsDemo

This scene demonstrates:
    1. Neural network architecture drawing + forward pass + backprop
    2. Activation function comparison (ReLU, sigmoid, tanh, leaky ReLU)
    3. Gradient descent animation (1D and 2D)
    4. Loss curve / training loop animation
    5. Decision boundary morphing
    6. Weight matrix visualization
    7. Single neuron computation diagram
"""

from manim import *
import numpy as np

from app.manim_pipeline.styles import (
    BG_COLOR, TEXT_PRIMARY, TEXT_SECONDARY, TEXT_DIM,
    ACCENT_BLUE, ACCENT_ORANGE, ACCENT_GREEN, ACCENT_RED,
    ACCENT_PURPLE, ACCENT_YELLOW, ACCENT_CYAN, ACCENT_PINK,
    TITLE_SIZE, SUBTITLE_SIZE, BODY_SIZE, LABEL_SIZE,
)

from app.manim_pipeline.ml_visuals import (
    # Network
    draw_neural_network,
    animate_network_creation,
    animate_forward_pass,
    animate_backpropagation,
    # Activation functions
    draw_activation_functions,
    animate_activation_comparison,
    # Gradient descent
    animate_gradient_descent,
    draw_loss_landscape_contour,
    animate_gradient_descent_2d,
    quadratic_loss, quadratic_loss_deriv,
    bumpy_loss, bumpy_loss_deriv,
    bowl_2d, bowl_2d_grad,
    # Loss curves
    draw_loss_curve,
    animate_training_loop,
    draw_dual_curves,
    # Decision boundary
    animate_decision_boundary,
    # Weight matrix
    draw_weight_matrix,
    # Single neuron
    draw_single_neuron,
    # Pre-built sections
    build_nn_overview_section,
    build_gradient_descent_section,
    build_activation_comparison_section,
)


class MLVisualsDemo(Scene):
    """Full demo of all ML visualization helpers."""

    def construct(self):
        self.camera.background_color = BG_COLOR

        # ====================================================================
        # TITLE
        # ====================================================================
        title = Text("ML Visualization Toolkit Demo",
                     font_size=TITLE_SIZE, color=TEXT_PRIMARY, weight="BOLD")
        title.to_edge(UP, buff=0.3)

        subtitle = Text("Neural Networks  |  Gradient Descent  |  Training Loops",
                        font_size=LABEL_SIZE, color=ACCENT_CYAN)
        subtitle.next_to(title, DOWN, buff=0.3)

        self.play(FadeIn(title), FadeIn(subtitle), run_time=1.5)
        self.wait(1)
        self.play(FadeOut(subtitle), run_time=0.5)

        # ====================================================================
        # SECTION 1: Neural Network Architecture
        # ====================================================================
        section_label = Text("1. Neural Network Architecture",
                            font_size=BODY_SIZE, color=ACCENT_GREEN)
        section_label.to_edge(DOWN, buff=0.4)
        self.play(FadeIn(section_label), run_time=0.5)

        # Draw network
        net_data = draw_neural_network(
            layer_sizes=[3, 5, 5, 2],
            neuron_radius=0.18,
            layer_labels=["Input", "Hidden 1", "Hidden 2", "Output"],
        )
        net_data["network"].scale(0.75).shift(DOWN * 0.3)

        animate_network_creation(self, net_data, run_time=2.5)
        self.wait(0.5)

        # Forward pass
        cap2 = Text("Forward Pass", font_size=LABEL_SIZE, color=TEXT_SECONDARY)
        cap2.to_edge(DOWN, buff=0.4)
        self.play(ReplacementTransform(section_label, cap2), run_time=0.3)

        animate_forward_pass(self, net_data, input_values=[1.0, 0.5, -0.3], run_time=3.5)
        self.wait(0.3)

        # Backpropagation
        cap3 = Text("Backpropagation", font_size=LABEL_SIZE, color=TEXT_SECONDARY)
        cap3.to_edge(DOWN, buff=0.4)
        self.play(ReplacementTransform(cap2, cap3), run_time=0.3)

        animate_backpropagation(self, net_data, run_time=3)

        self.play(FadeOut(net_data["network"]), FadeOut(cap3), run_time=0.5)

        # ====================================================================
        # SECTION 2: Single Neuron Diagram
        # ====================================================================
        cap4 = Text("2. Single Neuron Computation",
                    font_size=BODY_SIZE, color=ACCENT_GREEN)
        cap4.to_edge(DOWN, buff=0.4)
        self.play(FadeIn(cap4), run_time=0.5)

        neuron_data = draw_single_neuron(
            num_inputs=3,
            input_labels=["x_1", "x_2", "x_3"],
            activation="ReLU",
        )
        neuron_data["diagram"].scale(0.85).shift(DOWN * 0.3)
        self.play(FadeIn(neuron_data["diagram"]), run_time=2)
        self.wait(1.5)

        self.play(FadeOut(neuron_data["diagram"]), FadeOut(cap4), run_time=0.5)

        # ====================================================================
        # SECTION 3: Activation Function Comparison
        # ====================================================================
        cap5 = Text("3. Activation Functions",
                    font_size=BODY_SIZE, color=ACCENT_GREEN)
        cap5.to_edge(DOWN, buff=0.4)
        self.play(FadeIn(cap5), run_time=0.5)

        act_data = draw_activation_functions(
            functions=["relu", "sigmoid", "tanh", "leaky_relu"],
            axes_width=3.2,
            axes_height=1.8,
            arrangement="grid",
        )
        act_data["group"].scale(0.8).shift(DOWN * 0.2)

        animate_activation_comparison(self, act_data, run_time=6)
        self.wait(1)
        self.play(FadeOut(act_data["group"]), FadeOut(cap5), run_time=0.5)

        # ====================================================================
        # SECTION 4: 1D Gradient Descent
        # ====================================================================
        cap6 = Text("4. Gradient Descent (1D)",
                    font_size=BODY_SIZE, color=ACCENT_GREEN)
        cap6.to_edge(DOWN, buff=0.4)
        self.play(FadeIn(cap6), run_time=0.5)

        gd_eq = MathTex(
            r"\theta_{t+1} = \theta_t - \alpha \nabla L(\theta_t)",
            font_size=32, color=TEXT_PRIMARY,
        )
        gd_eq.shift(UP * 2.2)
        self.play(Write(gd_eq), run_time=1)

        gd_data = animate_gradient_descent(
            self,
            loss_func=bumpy_loss,
            loss_func_deriv=bumpy_loss_deriv,
            start_x=3.0,
            learning_rate=0.15,
            num_steps=12,
            x_range=(-4, 4, 1),
            y_range=(-1, 12, 2),
            show_tangent=True,
        )

        self.wait(0.5)
        all_gd = VGroup(gd_eq, gd_data["axes"], gd_data["ax_labels"],
                        gd_data["curve"], gd_data["dot"])
        if gd_data["lr_label"]:
            all_gd.add(gd_data["lr_label"])
        self.play(FadeOut(all_gd), FadeOut(cap6), run_time=0.5)

        # ====================================================================
        # SECTION 5: 2D Gradient Descent (Contour Plot)
        # ====================================================================
        cap7 = Text("5. 2D Loss Landscape",
                    font_size=BODY_SIZE, color=ACCENT_GREEN)
        cap7.to_edge(DOWN, buff=0.4)
        self.play(FadeIn(cap7), run_time=0.5)

        contour_data = draw_loss_landscape_contour(
            loss_func_2d=bowl_2d,
            x_range=(-3, 3),
            y_range=(-3, 3),
            axes_width=5.5,
            axes_height=4.5,
            num_contours=8,
        )
        contour_data["group"].shift(DOWN * 0.2)
        self.play(Create(contour_data["axes"]), Write(contour_data["ax_labels"]), run_time=1)
        self.play(Create(contour_data["contours"]), run_time=1.5)

        animate_gradient_descent_2d(
            self,
            contour_data=contour_data,
            loss_func_2d=bowl_2d,
            grad_func_2d=bowl_2d_grad,
            start_point=(2.5, 2.0),
            learning_rate=0.15,
            num_steps=12,
            step_run_time=0.35,
        )

        self.wait(0.5)
        self.play(FadeOut(contour_data["group"]), FadeOut(cap7), run_time=0.5)
        # Clean up any remaining dots/lines
        self.clear()
        self.camera.background_color = BG_COLOR
        self.add(title)

        # ====================================================================
        # SECTION 6: Training Loss Curve
        # ====================================================================
        cap8 = Text("6. Training Loss Curve",
                    font_size=BODY_SIZE, color=ACCENT_GREEN)
        cap8.to_edge(DOWN, buff=0.4)
        self.play(FadeIn(cap8), run_time=0.5)

        loss_data = draw_loss_curve(
            num_epochs=60,
            axes_width=6.5,
            axes_height=3.0,
        )
        loss_data["group"].shift(DOWN * 0.3)

        animate_training_loop(self, loss_data, reveal_speed=3.5)
        self.wait(1)

        self.play(FadeOut(loss_data["group"]), FadeOut(cap8), run_time=0.5)

        # ====================================================================
        # SECTION 7: Train vs Validation Curves (Overfitting)
        # ====================================================================
        cap9 = Text("7. Overfitting: Train vs Validation",
                    font_size=BODY_SIZE, color=ACCENT_GREEN)
        cap9.to_edge(DOWN, buff=0.4)
        self.play(FadeIn(cap9), run_time=0.5)

        dual_data = draw_dual_curves(
            num_epochs=60,
            axes_width=6.5,
            axes_height=3.0,
        )
        dual_data["group"].shift(DOWN * 0.3)

        self.play(
            Create(dual_data["axes"]),
            Write(dual_data["ax_labels"]),
            FadeIn(dual_data["legend"]),
            run_time=1,
        )
        self.play(
            Create(dual_data["train_curve"]),
            Create(dual_data["val_curve"]),
            run_time=3,
            rate_func=linear,
        )
        self.wait(1)

        self.play(FadeOut(dual_data["group"]), FadeOut(cap9), run_time=0.5)

        # ====================================================================
        # SECTION 8: Decision Boundary
        # ====================================================================
        cap10 = Text("8. Decision Boundary Evolution",
                     font_size=BODY_SIZE, color=ACCENT_GREEN)
        cap10.to_edge(DOWN, buff=0.4)
        self.play(FadeIn(cap10), run_time=0.5)

        db_data = animate_decision_boundary(
            self,
            boundary_func_initial=lambda x: 0.5,
            boundary_func_final=lambda x: 0.3 * x ** 2 - 0.5,
            axes_width=5.5,
            axes_height=4.0,
            morph_run_time=3.0,
        )
        self.wait(0.5)

        db_group = VGroup(db_data["axes"], db_data["ax_labels"],
                         db_data["dots"]["all_dots"])
        self.play(FadeOut(db_group), FadeOut(cap10), run_time=0.5)

        # ====================================================================
        # SECTION 9: Weight Matrix
        # ====================================================================
        cap11 = Text("9. Weight Matrix Visualization",
                     font_size=BODY_SIZE, color=ACCENT_GREEN)
        cap11.to_edge(DOWN, buff=0.4)
        self.play(FadeIn(cap11), run_time=0.5)

        w_data = draw_weight_matrix(rows=4, cols=5)
        w_label = MathTex(r"W \in \mathbb{R}^{4 \times 5}",
                         font_size=28, color=TEXT_PRIMARY)
        w_data["matrix"].shift(DOWN * 0.3)
        w_label.next_to(w_data["matrix"], UP, buff=0.3)

        self.play(FadeIn(w_data["matrix"]), Write(w_label), run_time=2)
        self.wait(1)

        self.play(FadeOut(w_data["matrix"]), FadeOut(w_label), FadeOut(cap11), run_time=0.5)

        # ====================================================================
        # OUTRO
        # ====================================================================
        self.play(FadeOut(title), run_time=0.5)
        outro = Text("ML Visualization Toolkit", font_size=SUBTITLE_SIZE,
                     color=ACCENT_CYAN, weight="BOLD")
        self.play(FadeIn(outro, shift=UP * 0.3), run_time=0.8)
        self.wait(1)
        self.play(FadeOut(outro), run_time=0.8)


# ============================================================================
# STANDALONE MINI-SCENES (one visualization each)
# ============================================================================

class NeuralNetworkDemo(Scene):
    """Just the neural network architecture + forward/backward pass."""
    def construct(self):
        self.camera.background_color = BG_COLOR
        build_nn_overview_section(self, layer_sizes=[4, 6, 6, 3])


class GradientDescentDemo(Scene):
    """Just gradient descent on a 1D loss landscape."""
    def construct(self):
        self.camera.background_color = BG_COLOR
        build_gradient_descent_section(self)


class ActivationFunctionsDemo(Scene):
    """Just the activation function comparison grid."""
    def construct(self):
        self.camera.background_color = BG_COLOR
        build_activation_comparison_section(self)


class TrainingLoopDemo(Scene):
    """Just the loss curve being drawn progressively."""
    def construct(self):
        self.camera.background_color = BG_COLOR
        loss_data = draw_loss_curve(num_epochs=80)
        loss_data["group"].shift(DOWN * 0.3)
        animate_training_loop(self, loss_data, reveal_speed=5)
        self.wait(1)
        self.play(FadeOut(loss_data["group"]), run_time=0.5)


class DecisionBoundaryDemo(Scene):
    """Just the decision boundary morphing animation."""
    def construct(self):
        self.camera.background_color = BG_COLOR
        db = animate_decision_boundary(
            self,
            boundary_func_initial=lambda x: 0.0,
            boundary_func_final=lambda x: 0.5 * np.sin(2 * x) + 0.3 * x,
            morph_run_time=4,
        )
        self.wait(1)
