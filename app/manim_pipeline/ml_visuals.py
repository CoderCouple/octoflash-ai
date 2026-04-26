"""
Reusable Manim CE helper functions for machine learning and neural network
visualizations.  All functions return VGroup/Mobject objects or animate
directly on a scene, and are compatible with OctoflashScene and plain Scene.

Includes:
    - Neural network architecture drawing (layers, neurons, connections)
    - Forward pass animation (signals flowing through a network)
    - Backpropagation animation (gradient flow in reverse)
    - Activation function comparison (ReLU, sigmoid, tanh, leaky ReLU)
    - Gradient descent animation on a loss landscape
    - Loss curve / training loop visualization
    - Decision boundary animation
    - Loss landscape surface (2D contour)

Usage in a Manim scene:
    from app.manim_pipeline.ml_visuals import (
        draw_neural_network,
        animate_forward_pass,
        animate_backpropagation,
        draw_activation_functions,
        animate_gradient_descent,
        draw_loss_curve,
        animate_decision_boundary,
        draw_loss_landscape_contour,
    )

All helpers use Octoflash brand colors from styles.py.
"""

import numpy as np
from manim import (
    VGroup, VMobject, Circle, Line, Arrow, Dot,
    Text, MathTex, Axes, DashedLine,
    FadeIn, FadeOut, Create, Write, ShowPassingFlash,
    Transform, ReplacementTransform,
    UP, DOWN, LEFT, RIGHT, ORIGIN,
    GRAY, WHITE, YELLOW,
    ValueTracker, always_redraw,
    rate_functions,
    RoundedRectangle, SurroundingRectangle,
    ParametricFunction,
    NumberPlane,
    LaggedStartMap, LaggedStart,
    Succession,
    linear, smooth,
)

from app.manim_pipeline.styles import (
    ACCENT_BLUE, ACCENT_ORANGE, ACCENT_GREEN, ACCENT_RED,
    ACCENT_PURPLE, ACCENT_YELLOW, ACCENT_CYAN, ACCENT_PINK,
    TEXT_PRIMARY, TEXT_SECONDARY, TEXT_DIM,
    LABEL_SIZE, BODY_SIZE,
    BG_COLOR,
)


# ============================================================================
# 1. NEURAL NETWORK ARCHITECTURE
# ============================================================================

def draw_neural_network(
    layer_sizes: list[int],
    x_span: float = 8.0,
    y_span: float = 5.0,
    neuron_radius: float = 0.2,
    neuron_color: str = ACCENT_BLUE,
    connection_color: str = TEXT_DIM,
    connection_opacity: float = 0.4,
    layer_labels: list[str] | None = None,
    show_bias: bool = False,
    max_neurons_display: int = 8,
) -> dict:
    """Draw a fully-connected neural network diagram.

    Args:
        layer_sizes: Number of neurons per layer, e.g. [3, 4, 4, 1].
        x_span: Total horizontal width of the network.
        y_span: Total vertical height available for neurons.
        neuron_radius: Radius of each neuron circle.
        neuron_color: Fill/stroke color for neurons.
        connection_color: Color for connection lines.
        connection_opacity: Opacity of connection lines.
        layer_labels: Optional list of layer name strings.
        show_bias: If True, add a bias neuron (+1) to each hidden layer.
        max_neurons_display: Max neurons to show per layer (ellipsis for more).

    Returns:
        dict with keys:
            'network': VGroup  -- the entire network (add to scene)
            'layers': list[VGroup]  -- list of neuron VGroups per layer
            'neurons': list[list[Circle]]  -- nested list of neuron circles
            'connections': list[VGroup]  -- connections between layer i and i+1
            'labels': VGroup  -- layer label texts (if provided)
    """
    num_layers = len(layer_sizes)
    x_positions = np.linspace(-x_span / 2, x_span / 2, num_layers)

    all_layers = []       # list of VGroup per layer
    all_neurons = []      # list of list of Circle per layer
    all_connections = []  # connections between adjacent layers
    all_labels_group = VGroup()

    for layer_idx, size in enumerate(layer_sizes):
        # Handle large layers: show subset with ellipsis
        display_size = min(size, max_neurons_display)
        truncated = size > max_neurons_display

        y_positions = np.linspace(
            -y_span / 2 * (display_size / max(layer_sizes)),
            y_span / 2 * (display_size / max(layer_sizes)),
            display_size,
        )

        layer_neurons = []
        layer_group = VGroup()

        for i, y_pos in enumerate(y_positions):
            neuron = Circle(
                radius=neuron_radius,
                fill_color=neuron_color,
                fill_opacity=0.25,
                stroke_color=neuron_color,
                stroke_width=2,
            )
            neuron.move_to([x_positions[layer_idx], y_pos, 0])
            layer_neurons.append(neuron)
            layer_group.add(neuron)

        if truncated:
            dots = MathTex(r"\vdots", font_size=24, color=TEXT_DIM)
            dots.move_to([x_positions[layer_idx], 0, 0])
            layer_group.add(dots)

        if show_bias and 0 < layer_idx < num_layers - 1:
            bias = Circle(
                radius=neuron_radius * 0.8,
                fill_color=ACCENT_YELLOW,
                fill_opacity=0.3,
                stroke_color=ACCENT_YELLOW,
                stroke_width=2,
            )
            bias_y = y_positions[-1] + neuron_radius * 3 if len(y_positions) > 0 else 0
            bias.move_to([x_positions[layer_idx], bias_y, 0])
            bias_label = Text("+1", font_size=12, color=ACCENT_YELLOW)
            bias_label.move_to(bias.get_center())
            layer_group.add(bias, bias_label)

        all_neurons.append(layer_neurons)
        all_layers.append(layer_group)

        # Layer labels
        if layer_labels and layer_idx < len(layer_labels):
            label = Text(
                layer_labels[layer_idx],
                font_size=LABEL_SIZE,
                color=TEXT_SECONDARY,
            )
            label.next_to(layer_group, DOWN, buff=0.4)
            all_labels_group.add(label)

    # Create connections between adjacent layers
    for i in range(num_layers - 1):
        conn_group = VGroup()
        for n1 in all_neurons[i]:
            for n2 in all_neurons[i + 1]:
                line = Line(
                    n1.get_right(),
                    n2.get_left(),
                    stroke_color=connection_color,
                    stroke_width=1.2,
                    stroke_opacity=connection_opacity,
                )
                conn_group.add(line)
        all_connections.append(conn_group)

    # Assemble full network
    network = VGroup()
    for conn in all_connections:
        network.add(conn)
    for layer in all_layers:
        network.add(layer)
    if all_labels_group:
        network.add(all_labels_group)

    return {
        "network": network,
        "layers": all_layers,
        "neurons": all_neurons,
        "connections": all_connections,
        "labels": all_labels_group,
    }


def animate_network_creation(scene, net_data: dict, run_time: float = 3.0):
    """Animate the network appearing layer by layer with connections.

    Args:
        scene: The Manim scene instance.
        net_data: The dict returned by draw_neural_network().
        run_time: Total animation duration.
    """
    layers = net_data["layers"]
    connections = net_data["connections"]
    labels = net_data["labels"]
    num_layers = len(layers)

    per_layer = run_time / (num_layers + len(connections))

    # First layer
    scene.play(FadeIn(layers[0]), run_time=per_layer)

    # Remaining layers with incoming connections
    for i in range(1, num_layers):
        scene.play(
            Create(connections[i - 1]),
            FadeIn(layers[i]),
            run_time=per_layer,
        )

    # Labels
    if labels:
        scene.play(FadeIn(labels), run_time=per_layer)


# ============================================================================
# 2. FORWARD PASS ANIMATION
# ============================================================================

def animate_forward_pass(
    scene,
    net_data: dict,
    input_values: list[float] | None = None,
    signal_color: str = ACCENT_CYAN,
    run_time: float = 4.0,
):
    """Animate a forward pass through the network with glowing signals.

    Creates colored pulses that travel along connections from input to output,
    with neurons lighting up as signals arrive.

    Args:
        scene: The Manim scene instance.
        net_data: The dict returned by draw_neural_network().
        input_values: Optional list of input values to display on input neurons.
        signal_color: Color of the travelling signal.
        run_time: Total animation duration.
    """
    layers = net_data["layers"]
    neurons = net_data["neurons"]
    connections = net_data["connections"]
    num_layers = len(layers)
    per_layer = run_time / num_layers

    # Show input values if provided
    input_labels = VGroup()
    if input_values:
        for j, val in enumerate(input_values):
            if j < len(neurons[0]):
                lbl = Text(
                    f"{val:.1f}",
                    font_size=14,
                    color=signal_color,
                )
                lbl.move_to(neurons[0][j].get_center())
                input_labels.add(lbl)
        scene.play(FadeIn(input_labels), run_time=0.5)

    # Propagate signals through each layer
    for i in range(num_layers - 1):
        # Create signal dots at source neurons
        signals = VGroup()
        for n in neurons[i]:
            dot = Dot(n.get_center(), radius=0.08, color=signal_color)
            signals.add(dot)

        scene.play(FadeIn(signals), run_time=0.2)

        # Flash connections
        flash_anims = []
        for conn_line in connections[i]:
            flash = ShowPassingFlash(
                conn_line.copy().set_stroke(signal_color, width=4, opacity=0.8),
                time_width=0.4,
                run_time=per_layer * 0.6,
            )
            flash_anims.append(flash)

        # Light up target neurons
        target_anims = []
        for n in neurons[i + 1]:
            target_anims.append(
                n.animate.set_fill(signal_color, opacity=0.5)
            )

        scene.play(*flash_anims, run_time=per_layer * 0.6)
        scene.play(*target_anims, run_time=per_layer * 0.2)

        # Reset source neurons and remove signal dots
        reset_anims = []
        for n in neurons[i]:
            reset_anims.append(
                n.animate.set_fill(ACCENT_BLUE, opacity=0.25)
            )
        scene.play(*reset_anims, FadeOut(signals), run_time=per_layer * 0.2)

    # Reset final layer
    final_reset = []
    for n in neurons[-1]:
        final_reset.append(
            n.animate.set_fill(ACCENT_BLUE, opacity=0.25)
        )
    scene.play(*final_reset, FadeOut(input_labels), run_time=0.3)


# ============================================================================
# 3. BACKPROPAGATION ANIMATION
# ============================================================================

def animate_backpropagation(
    scene,
    net_data: dict,
    gradient_color: str = ACCENT_RED,
    run_time: float = 4.0,
):
    """Animate gradient signals flowing backward through the network.

    Red pulses travel from output to input along connections, representing
    the chain rule gradient computation.

    Args:
        scene: The Manim scene instance.
        net_data: The dict returned by draw_neural_network().
        gradient_color: Color for gradient signals.
        run_time: Total animation duration.
    """
    neurons = net_data["neurons"]
    connections = net_data["connections"]
    num_layers = len(neurons)
    per_layer = run_time / num_layers

    # Start from output: show loss
    loss_label = MathTex(r"\frac{\partial L}{\partial \hat{y}}", font_size=24, color=gradient_color)
    loss_label.next_to(neurons[-1][0] if neurons[-1] else ORIGIN, RIGHT, buff=0.3)
    scene.play(Write(loss_label), run_time=0.5)

    # Propagate backward
    for i in range(num_layers - 1, 0, -1):
        # Light up current layer neurons in gradient color
        light_anims = [
            n.animate.set_fill(gradient_color, opacity=0.5)
            for n in neurons[i]
        ]
        scene.play(*light_anims, run_time=per_layer * 0.2)

        # Flash connections backward (reverse direction)
        flash_anims = []
        for conn_line in connections[i - 1]:
            # Create reversed line for backward flash
            rev_line = Line(
                conn_line.get_end(),
                conn_line.get_start(),
                stroke_color=gradient_color,
                stroke_width=4,
                stroke_opacity=0.8,
            )
            flash = ShowPassingFlash(rev_line, time_width=0.4, run_time=per_layer * 0.5)
            flash_anims.append(flash)

        scene.play(*flash_anims, run_time=per_layer * 0.5)

        # Reset current layer
        reset_anims = [
            n.animate.set_fill(ACCENT_BLUE, opacity=0.25)
            for n in neurons[i]
        ]
        scene.play(*reset_anims, run_time=per_layer * 0.15)

    # Reset first layer
    scene.play(
        *[n.animate.set_fill(ACCENT_BLUE, opacity=0.25) for n in neurons[0]],
        FadeOut(loss_label),
        run_time=0.3,
    )


# ============================================================================
# 4. ACTIVATION FUNCTION COMPARISON
# ============================================================================

def draw_activation_functions(
    functions: list[str] | None = None,
    x_range: tuple = (-5, 5, 1),
    y_range: tuple = (-1.5, 5, 1),
    axes_width: float = 5.0,
    axes_height: float = 3.0,
    arrangement: str = "grid",
) -> dict:
    """Create side-by-side activation function plots.

    Args:
        functions: List of function names from ['relu', 'sigmoid', 'tanh',
            'leaky_relu', 'elu', 'swish']. Default: ['relu', 'sigmoid', 'tanh'].
        x_range: x-axis range tuple (min, max, step).
        y_range: y-axis range tuple (min, max, step).
        axes_width: Width of each axes.
        axes_height: Height of each axes.
        arrangement: 'grid' for 2x2 layout, 'row' for horizontal.

    Returns:
        dict with:
            'group': VGroup  -- everything (add to scene)
            'plots': list[dict]  -- per-function dicts with 'axes', 'graph',
                                    'equation', 'title'
    """
    if functions is None:
        functions = ["relu", "sigmoid", "tanh"]

    # Activation function definitions
    func_defs = {
        "relu": {
            "fn": lambda x: np.maximum(0, x),
            "eq": r"\text{ReLU}(x) = \max(0, x)",
            "title": "ReLU",
            "color": ACCENT_GREEN,
            "y_range": (-0.5, 4, 1),
        },
        "sigmoid": {
            "fn": lambda x: 1 / (1 + np.exp(-x)),
            "eq": r"\sigma(x) = \frac{1}{1 + e^{-x}}",
            "title": "Sigmoid",
            "color": ACCENT_BLUE,
            "y_range": (-0.2, 1.3, 0.5),
        },
        "tanh": {
            "fn": lambda x: np.tanh(x),
            "eq": r"\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}",
            "title": "Tanh",
            "color": ACCENT_PURPLE,
            "y_range": (-1.3, 1.3, 0.5),
        },
        "leaky_relu": {
            "fn": lambda x: np.where(x > 0, x, 0.01 * x),
            "eq": r"\text{LeakyReLU}(x) = \max(0.01x, x)",
            "title": "Leaky ReLU",
            "color": ACCENT_ORANGE,
            "y_range": (-0.5, 4, 1),
        },
        "elu": {
            "fn": lambda x: np.where(x > 0, x, np.exp(x) - 1),
            "eq": r"\text{ELU}(x) = \begin{cases} x & x > 0 \\ e^x - 1 & x \le 0 \end{cases}",
            "title": "ELU",
            "color": ACCENT_CYAN,
            "y_range": (-1.5, 4, 1),
        },
        "swish": {
            "fn": lambda x: x / (1 + np.exp(-x)),
            "eq": r"\text{Swish}(x) = \frac{x}{1 + e^{-x}}",
            "title": "Swish",
            "color": ACCENT_PINK,
            "y_range": (-0.5, 4, 1),
        },
    }

    plots = []
    full_group = VGroup()

    for func_name in functions:
        if func_name not in func_defs:
            continue
        fdef = func_defs[func_name]

        axes = Axes(
            x_range=list(x_range),
            y_range=list(fdef.get("y_range", y_range)),
            x_length=axes_width,
            y_length=axes_height,
            axis_config={"color": TEXT_DIM, "stroke_width": 1.5},
        )

        graph = axes.plot(fdef["fn"], color=fdef["color"], stroke_width=3)
        equation = MathTex(fdef["eq"], font_size=22, color=TEXT_PRIMARY)
        title_text = Text(fdef["title"], font_size=LABEL_SIZE, color=fdef["color"])
        title_text.next_to(axes, UP, buff=0.15)
        equation.next_to(axes, DOWN, buff=0.15)

        plot_group = VGroup(axes, graph, equation, title_text)
        plots.append({
            "axes": axes,
            "graph": graph,
            "equation": equation,
            "title": title_text,
            "group": plot_group,
            "color": fdef["color"],
        })
        full_group.add(plot_group)

    # Arrange
    n = len(plots)
    if arrangement == "grid" and n >= 4:
        # 2x2 grid
        rows = []
        for i in range(0, n, 2):
            row = VGroup(*[p["group"] for p in plots[i:i + 2]])
            row.arrange(RIGHT, buff=0.5)
            rows.append(row)
        grid = VGroup(*rows)
        grid.arrange(DOWN, buff=0.6)
        full_group = grid
    else:
        full_group.arrange(RIGHT, buff=0.5)

    return {"group": full_group, "plots": plots}


def animate_activation_comparison(
    scene,
    act_data: dict,
    run_time: float = 6.0,
):
    """Animate activation functions appearing one at a time with their graphs.

    Args:
        scene: The Manim scene instance.
        act_data: The dict returned by draw_activation_functions().
        run_time: Total duration.
    """
    plots = act_data["plots"]
    per_func = run_time / len(plots)

    for p in plots:
        scene.play(
            Create(p["axes"]),
            run_time=per_func * 0.3,
        )
        scene.play(
            Create(p["graph"]),
            Write(p["title"]),
            run_time=per_func * 0.4,
        )
        scene.play(
            Write(p["equation"]),
            run_time=per_func * 0.3,
        )


# ============================================================================
# 5. GRADIENT DESCENT ANIMATION
# ============================================================================

def animate_gradient_descent(
    scene,
    loss_func,
    loss_func_deriv,
    start_x: float = 3.0,
    learning_rate: float = 0.3,
    num_steps: int = 8,
    x_range: tuple = (-4, 4, 1),
    y_range: tuple = (-1, 10, 2),
    axes_width: float = 7.0,
    axes_height: float = 3.5,
    axes_shift=None,
    curve_color: str = ACCENT_BLUE,
    dot_color: str = ACCENT_RED,
    show_tangent: bool = True,
    show_learning_rate: bool = True,
    step_run_time: float = 0.6,
) -> dict:
    """Animate gradient descent on a 1D loss landscape.

    Shows a ball rolling down a curve, with tangent lines and parameter
    update annotations at each step.

    Args:
        scene: The Manim scene instance.
        loss_func: Callable f(x) -> y for the loss landscape.
        loss_func_deriv: Callable f'(x) -> dy/dx (gradient).
        start_x: Starting x position.
        learning_rate: Step size for parameter updates.
        num_steps: Number of gradient descent steps.
        x_range, y_range: Axes ranges.
        axes_width, axes_height: Axes dimensions.
        axes_shift: Optional shift vector for axes position.
        curve_color: Color of the loss curve.
        dot_color: Color of the moving point.
        show_tangent: Show tangent line at each step.
        show_learning_rate: Show learning rate annotation.
        step_run_time: Duration per step.

    Returns:
        dict with 'axes', 'curve', 'trajectory' (list of (x, y) points).
    """
    if axes_shift is None:
        axes_shift = DOWN * 0.4

    axes = Axes(
        x_range=list(x_range),
        y_range=list(y_range),
        x_length=axes_width,
        y_length=axes_height,
        axis_config={"color": TEXT_DIM, "stroke_width": 2},
    )
    axes.shift(axes_shift)

    ax_labels = axes.get_axis_labels(
        x_label=MathTex(r"\theta", font_size=24),
        y_label=MathTex(r"L(\theta)", font_size=24),
    )

    curve = axes.plot(loss_func, color=curve_color, stroke_width=3)

    scene.play(Create(axes), Write(ax_labels), run_time=1)
    scene.play(Create(curve), run_time=1.5)

    # Learning rate label
    lr_label = None
    if show_learning_rate:
        lr_label = MathTex(
            rf"\alpha = {learning_rate}",
            font_size=22,
            color=ACCENT_YELLOW,
        )
        lr_label.to_corner(UP + RIGHT, buff=0.8)
        scene.play(Write(lr_label), run_time=0.5)

    # Gradient descent steps
    x_val = start_x
    trajectory = [(x_val, loss_func(x_val))]

    dot = Dot(axes.c2p(x_val, loss_func(x_val)), radius=0.1, color=dot_color)
    scene.play(FadeIn(dot), run_time=0.3)

    tangent_line = None

    for step in range(num_steps):
        grad = loss_func_deriv(x_val)
        new_x = x_val - learning_rate * grad
        # Clamp to visible range
        new_x = max(x_range[0] + 0.5, min(x_range[1] - 0.5, new_x))
        new_y = loss_func(new_x)

        # Optional tangent line
        if show_tangent:
            if tangent_line:
                scene.play(FadeOut(tangent_line), run_time=0.15)
            slope = grad
            t_x1 = x_val - 0.8
            t_x2 = x_val + 0.8
            t_y1 = loss_func(x_val) + slope * (t_x1 - x_val)
            t_y2 = loss_func(x_val) + slope * (t_x2 - x_val)
            tangent_line = DashedLine(
                axes.c2p(t_x1, t_y1),
                axes.c2p(t_x2, t_y2),
                stroke_color=ACCENT_ORANGE,
                stroke_width=2,
                dash_length=0.08,
            )
            scene.play(Create(tangent_line), run_time=step_run_time * 0.3)

        # Move dot
        scene.play(
            dot.animate.move_to(axes.c2p(new_x, new_y)),
            run_time=step_run_time * 0.7,
            rate_func=smooth,
        )

        x_val = new_x
        trajectory.append((x_val, new_y))

    # Remove tangent
    if tangent_line:
        scene.play(FadeOut(tangent_line), run_time=0.2)

    return {
        "axes": axes,
        "ax_labels": ax_labels,
        "curve": curve,
        "dot": dot,
        "lr_label": lr_label,
        "trajectory": trajectory,
    }


def draw_loss_landscape_contour(
    loss_func_2d,
    x_range: tuple = (-3, 3),
    y_range: tuple = (-3, 3),
    resolution: int = 20,
    axes_width: float = 6.0,
    axes_height: float = 5.0,
    num_contours: int = 10,
    color_low: str = ACCENT_BLUE,
    color_high: str = ACCENT_RED,
) -> dict:
    """Create a 2D contour plot of a loss landscape.

    Draws concentric level curves of a 2D function, useful for
    visualizing optimization landscapes.

    Args:
        loss_func_2d: Callable f(x, y) -> z.
        x_range: (min, max) for x axis.
        y_range: (min, max) for y axis.
        resolution: Grid density for contour computation.
        axes_width, axes_height: Plot dimensions.
        num_contours: Number of contour levels.
        color_low, color_high: Color range for contours.

    Returns:
        dict with 'axes', 'contours' (VGroup), 'group' (VGroup of all).
    """
    axes = Axes(
        x_range=[x_range[0], x_range[1], 1],
        y_range=[y_range[0], y_range[1], 1],
        x_length=axes_width,
        y_length=axes_height,
        axis_config={"color": TEXT_DIM, "stroke_width": 2},
    )

    ax_labels = axes.get_axis_labels(
        x_label=MathTex(r"w_1", font_size=24),
        y_label=MathTex(r"w_2", font_size=24),
    )

    # Compute contour levels
    xs = np.linspace(x_range[0], x_range[1], resolution)
    ys = np.linspace(y_range[0], y_range[1], resolution)
    zz = np.array([[loss_func_2d(x, y) for x in xs] for y in ys])
    z_min, z_max = zz.min(), zz.max()

    levels = np.linspace(z_min + (z_max - z_min) * 0.05, z_max * 0.9, num_contours)

    contours = VGroup()
    for k, level in enumerate(levels):
        # Approximate contour by sampling a parametric circle approach
        # For simple bowl-shaped functions, use ellipse approximation
        t_frac = k / max(1, num_contours - 1)
        r_hex_low = color_low.lstrip("#")
        r_hex_high = color_high.lstrip("#")
        r_low = tuple(int(r_hex_low[i:i+2], 16) for i in (0, 2, 4))
        r_high = tuple(int(r_hex_high[i:i+2], 16) for i in (0, 2, 4))
        interp = tuple(int(r_low[j] + t_frac * (r_high[j] - r_low[j])) for j in range(3))
        hex_color = "#{:02x}{:02x}{:02x}".format(*interp)

        # Trace contour using parametric sampling
        contour_points = []
        for theta in np.linspace(0, 2 * np.pi, 80):
            # Binary search for the radius at this angle that hits the level
            r_min_search, r_max_search = 0.01, max(x_range[1], y_range[1]) * 1.5
            for _ in range(20):
                r_mid = (r_min_search + r_max_search) / 2
                px = r_mid * np.cos(theta)
                py = r_mid * np.sin(theta)
                if x_range[0] <= px <= x_range[1] and y_range[0] <= py <= y_range[1]:
                    val = loss_func_2d(px, py)
                    if val < level:
                        r_min_search = r_mid
                    else:
                        r_max_search = r_mid
                else:
                    r_max_search = r_mid

            px = r_min_search * np.cos(theta)
            py = r_min_search * np.sin(theta)
            if x_range[0] <= px <= x_range[1] and y_range[0] <= py <= y_range[1]:
                contour_points.append(axes.c2p(px, py))

        if len(contour_points) > 3:
            contour_line = VMobject()
            contour_line.set_points_smoothly(contour_points + [contour_points[0]])
            contour_line.set_stroke(hex_color, width=1.5, opacity=0.7)
            contours.add(contour_line)

    group = VGroup(axes, ax_labels, contours)

    return {
        "axes": axes,
        "ax_labels": ax_labels,
        "contours": contours,
        "group": group,
    }


def animate_gradient_descent_2d(
    scene,
    contour_data: dict,
    loss_func_2d,
    grad_func_2d,
    start_point: tuple = (2.5, 2.5),
    learning_rate: float = 0.1,
    num_steps: int = 15,
    dot_color: str = ACCENT_RED,
    path_color: str = ACCENT_YELLOW,
    step_run_time: float = 0.4,
):
    """Animate 2D gradient descent on a contour plot.

    Args:
        scene: The Manim scene instance.
        contour_data: Dict returned by draw_loss_landscape_contour().
        loss_func_2d: f(x, y) -> z.
        grad_func_2d: f(x, y) -> (dz/dx, dz/dy).
        start_point: Starting (x, y).
        learning_rate: Step size.
        num_steps: Number of iterations.
        dot_color: Color of the optimization point.
        path_color: Color of the path trace.
        step_run_time: Duration per step.
    """
    axes = contour_data["axes"]
    x, y = start_point

    dot = Dot(axes.c2p(x, y), radius=0.1, color=dot_color)
    scene.play(FadeIn(dot), run_time=0.3)

    path_points = [axes.c2p(x, y)]

    for _ in range(num_steps):
        gx, gy = grad_func_2d(x, y)
        x -= learning_rate * gx
        y -= learning_rate * gy
        new_pos = axes.c2p(x, y)
        path_points.append(new_pos)

        # Draw path segment
        segment = Line(
            path_points[-2], path_points[-1],
            stroke_color=path_color, stroke_width=2,
        )
        scene.play(
            Create(segment),
            dot.animate.move_to(new_pos),
            run_time=step_run_time,
        )


# ============================================================================
# 6. LOSS CURVE / TRAINING LOOP VISUALIZATION
# ============================================================================

def draw_loss_curve(
    loss_values: list[float] | None = None,
    num_epochs: int = 50,
    x_range: tuple | None = None,
    y_range: tuple | None = None,
    axes_width: float = 7.0,
    axes_height: float = 3.2,
    curve_color: str = ACCENT_GREEN,
    show_convergence_line: bool = True,
) -> dict:
    """Create a training loss curve plot.

    If loss_values is not provided, generates a realistic exponential
    decay curve with noise.

    Args:
        loss_values: Optional list of loss values per epoch.
        num_epochs: Number of epochs (used if loss_values not provided).
        x_range: Optional x-axis range.
        y_range: Optional y-axis range.
        axes_width, axes_height: Plot dimensions.
        curve_color: Color of the loss curve.
        show_convergence_line: Show a dashed line at the convergence value.

    Returns:
        dict with 'axes', 'curve', 'convergence_line', 'group',
               'loss_values', 'loss_func'.
    """
    if loss_values is None:
        # Generate realistic loss curve: exponential decay + noise
        epochs = np.arange(num_epochs)
        base = 2.5 * np.exp(-0.06 * epochs) + 0.3
        # Deterministic "noise" using sine waves
        noise = 0.1 * np.sin(0.5 * epochs) + 0.05 * np.sin(1.3 * epochs)
        loss_values = list(base + noise)

    n = len(loss_values)
    max_loss = max(loss_values) * 1.1
    min_loss = min(loss_values) * 0.9

    if x_range is None:
        x_range = (0, n, max(1, n // 10))
    if y_range is None:
        step = (max_loss - min_loss) / 5
        y_range = (min_loss - step, max_loss, step)

    axes = Axes(
        x_range=list(x_range),
        y_range=list(y_range),
        x_length=axes_width,
        y_length=axes_height,
        axis_config={"color": TEXT_DIM, "stroke_width": 2},
    )

    ax_labels = axes.get_axis_labels(
        x_label=Text("Epoch", font_size=18, color=TEXT_SECONDARY),
        y_label=Text("Loss", font_size=18, color=TEXT_SECONDARY),
    )

    # Create interpolated function from loss values
    def loss_func(x):
        idx = int(np.clip(x, 0, n - 1))
        if idx >= n - 1:
            return loss_values[-1]
        frac = x - idx
        return loss_values[idx] * (1 - frac) + loss_values[min(idx + 1, n - 1)] * frac

    curve = axes.plot(loss_func, x_range=[0, n - 1], color=curve_color, stroke_width=3)

    group = VGroup(axes, ax_labels, curve)
    convergence_line = None

    if show_convergence_line:
        final_val = loss_values[-1]
        convergence_line = DashedLine(
            axes.c2p(0, final_val),
            axes.c2p(n, final_val),
            stroke_color=ACCENT_RED,
            stroke_width=1.5,
            dash_length=0.1,
        )
        conv_label = MathTex(
            rf"L^* \approx {final_val:.2f}",
            font_size=18,
            color=ACCENT_RED,
        )
        conv_label.next_to(convergence_line, RIGHT, buff=0.1)
        group.add(convergence_line, conv_label)

    return {
        "axes": axes,
        "ax_labels": ax_labels,
        "curve": curve,
        "convergence_line": convergence_line,
        "group": group,
        "loss_values": loss_values,
        "loss_func": loss_func,
    }


def animate_training_loop(
    scene,
    loss_data: dict,
    reveal_speed: float = 4.0,
    show_epoch_counter: bool = True,
):
    """Animate the loss curve being drawn progressively (like a training run).

    Args:
        scene: The Manim scene instance.
        loss_data: Dict returned by draw_loss_curve().
        reveal_speed: Duration for the curve reveal animation.
        show_epoch_counter: Show an epoch counter during animation.
    """
    axes = loss_data["axes"]
    ax_labels = loss_data["ax_labels"]
    curve = loss_data["curve"]
    loss_values = loss_data["loss_values"]
    n = len(loss_values)

    scene.play(Create(axes), Write(ax_labels), run_time=1)

    # Epoch counter
    epoch_label = None
    if show_epoch_counter:
        epoch_tracker = ValueTracker(0)
        epoch_label = always_redraw(lambda: Text(
            f"Epoch: {int(epoch_tracker.get_value())}",
            font_size=LABEL_SIZE,
            color=ACCENT_CYAN,
        ).to_corner(UP + RIGHT, buff=0.8))
        scene.play(FadeIn(epoch_label), run_time=0.3)

    # Animate curve being drawn
    if show_epoch_counter:
        scene.play(
            Create(curve),
            epoch_tracker.animate.set_value(n),
            run_time=reveal_speed,
            rate_func=linear,
        )
    else:
        scene.play(Create(curve), run_time=reveal_speed, rate_func=linear)

    # Show convergence line
    if loss_data.get("convergence_line"):
        convergence_items = [
            m for m in loss_data["group"]
            if m is not axes and m is not ax_labels and m is not curve
        ]
        if convergence_items:
            scene.play(*[FadeIn(item) for item in convergence_items], run_time=0.5)


def draw_dual_curves(
    train_losses: list[float] | None = None,
    val_losses: list[float] | None = None,
    num_epochs: int = 50,
    axes_width: float = 7.0,
    axes_height: float = 3.2,
    train_color: str = ACCENT_BLUE,
    val_color: str = ACCENT_ORANGE,
) -> dict:
    """Create overlaid training and validation loss curves.

    Useful for illustrating overfitting (val loss diverging from train loss).

    Returns:
        dict with 'axes', 'train_curve', 'val_curve', 'legend', 'group'.
    """
    if train_losses is None:
        epochs = np.arange(num_epochs)
        train_losses = list(2.5 * np.exp(-0.08 * epochs) + 0.2 + 0.05 * np.sin(0.7 * epochs))
    if val_losses is None:
        epochs = np.arange(num_epochs)
        val_losses = list(
            2.5 * np.exp(-0.05 * epochs) + 0.4 +
            0.15 * np.sin(0.5 * epochs) +
            0.005 * epochs  # slight overfit divergence
        )

    n = len(train_losses)
    all_vals = train_losses + val_losses
    max_val = max(all_vals) * 1.1

    axes = Axes(
        x_range=[0, n, max(1, n // 10)],
        y_range=[0, max_val, max_val / 5],
        x_length=axes_width,
        y_length=axes_height,
        axis_config={"color": TEXT_DIM, "stroke_width": 2},
    )

    ax_labels = axes.get_axis_labels(
        x_label=Text("Epoch", font_size=18, color=TEXT_SECONDARY),
        y_label=Text("Loss", font_size=18, color=TEXT_SECONDARY),
    )

    def train_fn(x):
        idx = int(np.clip(x, 0, n - 1))
        if idx >= n - 1:
            return train_losses[-1]
        frac = x - idx
        return train_losses[idx] * (1 - frac) + train_losses[min(idx + 1, n - 1)] * frac

    def val_fn(x):
        idx = int(np.clip(x, 0, n - 1))
        if idx >= n - 1:
            return val_losses[-1]
        frac = x - idx
        return val_losses[idx] * (1 - frac) + val_losses[min(idx + 1, n - 1)] * frac

    train_curve = axes.plot(train_fn, x_range=[0, n - 1], color=train_color, stroke_width=3)
    val_curve = axes.plot(val_fn, x_range=[0, n - 1], color=val_color, stroke_width=3)

    # Legend
    legend = VGroup(
        VGroup(
            Line(ORIGIN, RIGHT * 0.5, stroke_color=train_color, stroke_width=3),
            Text("Train", font_size=16, color=train_color),
        ).arrange(RIGHT, buff=0.15),
        VGroup(
            Line(ORIGIN, RIGHT * 0.5, stroke_color=val_color, stroke_width=3),
            Text("Val", font_size=16, color=val_color),
        ).arrange(RIGHT, buff=0.15),
    ).arrange(RIGHT, buff=0.5)
    legend.next_to(axes, UP, buff=0.2)

    group = VGroup(axes, ax_labels, train_curve, val_curve, legend)

    return {
        "axes": axes,
        "ax_labels": ax_labels,
        "train_curve": train_curve,
        "val_curve": val_curve,
        "legend": legend,
        "group": group,
    }


# ============================================================================
# 7. DECISION BOUNDARY ANIMATION
# ============================================================================

def draw_data_points(
    axes,
    class_0_points: list[tuple],
    class_1_points: list[tuple],
    class_0_color: str = ACCENT_BLUE,
    class_1_color: str = ACCENT_RED,
    radius: float = 0.06,
) -> dict:
    """Draw 2D classification data points on axes.

    Args:
        axes: Manim Axes object.
        class_0_points: List of (x, y) for class 0.
        class_1_points: List of (x, y) for class 1.
        class_0_color, class_1_color: Colors for each class.
        radius: Dot radius.

    Returns:
        dict with 'dots_0', 'dots_1', 'all_dots'.
    """
    dots_0 = VGroup(*[
        Dot(axes.c2p(x, y), radius=radius, color=class_0_color)
        for x, y in class_0_points
    ])
    dots_1 = VGroup(*[
        Dot(axes.c2p(x, y), radius=radius, color=class_1_color)
        for x, y in class_1_points
    ])
    all_dots = VGroup(dots_0, dots_1)

    return {"dots_0": dots_0, "dots_1": dots_1, "all_dots": all_dots}


def animate_decision_boundary(
    scene,
    boundary_func_initial,
    boundary_func_final,
    class_0_points: list[tuple] | None = None,
    class_1_points: list[tuple] | None = None,
    x_range: tuple = (-3, 3, 1),
    y_range: tuple = (-3, 3, 1),
    axes_width: float = 6.0,
    axes_height: float = 5.0,
    axes_shift=None,
    num_morph_steps: int = 10,
    morph_run_time: float = 3.0,
) -> dict:
    """Animate a decision boundary evolving from initial to final.

    The boundary is a curve y = f(x) that separates two classes.
    Uses ValueTracker to smoothly morph between initial and final.

    Args:
        scene: The Manim scene instance.
        boundary_func_initial: f(x) -> y (initial boundary).
        boundary_func_final: f(x) -> y (final boundary).
        class_0_points, class_1_points: Optional data point lists.
        x_range, y_range: Axes ranges.
        axes_width, axes_height: Axes dimensions.
        axes_shift: Optional shift for axes.
        num_morph_steps: Not used directly; morphing is continuous.
        morph_run_time: Duration for the boundary morphing.

    Returns:
        dict with 'axes', 'boundary', 'dots'.
    """
    if axes_shift is None:
        axes_shift = DOWN * 0.4

    if class_0_points is None:
        # Generate deterministic sample data
        class_0_points = [
            (-2, -1), (-1.5, -0.5), (-1, -1.5), (-0.5, -0.8),
            (-2, 0.5), (-1.2, 0.2), (-0.8, -0.3), (-1.8, -0.2),
        ]
    if class_1_points is None:
        class_1_points = [
            (1, 1), (1.5, 0.8), (2, 1.5), (0.8, 0.5),
            (1.2, 2), (2, 0.5), (1.8, 1.2), (0.5, 1.8),
        ]

    axes = Axes(
        x_range=list(x_range),
        y_range=list(y_range),
        x_length=axes_width,
        y_length=axes_height,
        axis_config={"color": TEXT_DIM, "stroke_width": 2},
    )
    axes.shift(axes_shift)

    ax_labels = axes.get_axis_labels(
        x_label=MathTex("x_1", font_size=24),
        y_label=MathTex("x_2", font_size=24),
    )

    # Draw data
    dots_data = draw_data_points(axes, class_0_points, class_1_points)

    # Animated boundary using ValueTracker for interpolation
    t = ValueTracker(0)  # 0 = initial, 1 = final

    boundary = always_redraw(lambda: axes.plot(
        lambda x: (1 - t.get_value()) * boundary_func_initial(x) + t.get_value() * boundary_func_final(x),
        color=ACCENT_GREEN,
        stroke_width=3,
    ))

    scene.play(Create(axes), Write(ax_labels), run_time=1)
    scene.play(FadeIn(dots_data["all_dots"]), run_time=1)
    scene.play(Create(boundary), run_time=1)

    # Morph boundary
    scene.play(t.animate.set_value(1), run_time=morph_run_time, rate_func=smooth)

    return {
        "axes": axes,
        "ax_labels": ax_labels,
        "boundary": boundary,
        "dots": dots_data,
        "t_tracker": t,
    }


# ============================================================================
# 8. WEIGHT MATRIX VISUALIZATION
# ============================================================================

def draw_weight_matrix(
    rows: int = 3,
    cols: int = 4,
    values: list[list[float]] | None = None,
    cell_size: float = 0.6,
    positive_color: str = ACCENT_BLUE,
    negative_color: str = ACCENT_RED,
    zero_color: str = TEXT_DIM,
    show_values: bool = True,
) -> dict:
    """Draw a weight matrix as a colored grid.

    Cell colors interpolate between negative_color (for negative weights)
    and positive_color (for positive weights).

    Args:
        rows, cols: Matrix dimensions.
        values: Optional 2D list of weight values.
        cell_size: Size of each cell.
        positive_color, negative_color: Color extremes.
        zero_color: Color for near-zero weights.
        show_values: Display numeric values in cells.

    Returns:
        dict with 'matrix' (VGroup), 'cells' (2D list), 'bracket' labels.
    """
    if values is None:
        # Deterministic pseudo-random values
        values = []
        for i in range(rows):
            row_vals = []
            for j in range(cols):
                val = np.sin(i * 3.7 + j * 2.3) * 0.8
                row_vals.append(round(val, 2))
            values.append(row_vals)

    cells_2d = []
    matrix_group = VGroup()

    max_abs = max(abs(v) for row in values for v in row) or 1.0

    for i in range(rows):
        cell_row = []
        for j in range(cols):
            val = values[i][j]
            norm = val / max_abs  # -1 to 1

            if norm > 0:
                opacity = abs(norm) * 0.6
                fill = positive_color
            elif norm < 0:
                opacity = abs(norm) * 0.6
                fill = negative_color
            else:
                opacity = 0.1
                fill = zero_color

            cell = RoundedRectangle(
                corner_radius=0.05,
                width=cell_size,
                height=cell_size,
                fill_color=fill,
                fill_opacity=opacity,
                stroke_color=TEXT_DIM,
                stroke_width=1,
            )
            cell.move_to([j * cell_size - (cols - 1) * cell_size / 2,
                         -i * cell_size + (rows - 1) * cell_size / 2, 0])

            cell_group = VGroup(cell)

            if show_values:
                val_text = Text(f"{val:.1f}", font_size=12, color=TEXT_PRIMARY)
                val_text.move_to(cell.get_center())
                cell_group.add(val_text)

            cell_row.append(cell_group)
            matrix_group.add(cell_group)

        cells_2d.append(cell_row)

    # Brackets
    left_bracket = MathTex(r"\left[", font_size=rows * 30)
    left_bracket.next_to(matrix_group, LEFT, buff=0.1)
    right_bracket = MathTex(r"\right]", font_size=rows * 30)
    right_bracket.next_to(matrix_group, RIGHT, buff=0.1)

    full = VGroup(left_bracket, matrix_group, right_bracket)

    return {
        "matrix": full,
        "grid": matrix_group,
        "cells": cells_2d,
        "left_bracket": left_bracket,
        "right_bracket": right_bracket,
    }


# ============================================================================
# 9. SINGLE-NEURON COMPUTATION VISUAL
# ============================================================================

def draw_single_neuron(
    num_inputs: int = 3,
    input_labels: list[str] | None = None,
    weight_labels: list[str] | None = None,
    activation: str = "relu",
    neuron_radius: float = 0.5,
) -> dict:
    """Draw a detailed single-neuron computation diagram.

    Shows inputs on the left, weighted arrows converging on the neuron,
    the summation symbol, activation function, and output.

    Args:
        num_inputs: Number of input connections.
        input_labels: Optional labels for each input (e.g. ['x1', 'x2', 'x3']).
        weight_labels: Optional labels for each weight (e.g. ['w1', 'w2', 'w3']).
        activation: Name of activation function to display.
        neuron_radius: Radius of the neuron body.

    Returns:
        dict with 'diagram' (VGroup), 'inputs', 'weights', 'neuron', 'output'.
    """
    if input_labels is None:
        input_labels = [f"x_{i+1}" for i in range(num_inputs)]
    if weight_labels is None:
        weight_labels = [f"w_{i+1}" for i in range(num_inputs)]

    diagram = VGroup()

    # Input nodes
    y_positions = np.linspace(
        -(num_inputs - 1) * 0.8 / 2,
        (num_inputs - 1) * 0.8 / 2,
        num_inputs,
    )

    input_nodes = VGroup()
    input_texts = VGroup()
    weight_texts = VGroup()
    arrows = VGroup()

    for i in range(num_inputs):
        # Input circle
        inp = Circle(
            radius=0.2,
            fill_color=ACCENT_BLUE,
            fill_opacity=0.3,
            stroke_color=ACCENT_BLUE,
            stroke_width=2,
        )
        inp.move_to([-3, y_positions[i], 0])
        input_nodes.add(inp)

        # Input label
        inp_lbl = MathTex(input_labels[i], font_size=20, color=TEXT_PRIMARY)
        inp_lbl.move_to(inp.get_center())
        input_texts.add(inp_lbl)

        # Arrow to neuron
        arr = Arrow(
            inp.get_right(),
            [0 - neuron_radius, y_positions[i] * 0.3, 0],
            buff=0.05,
            stroke_color=TEXT_DIM,
            stroke_width=2,
            max_tip_length_to_length_ratio=0.15,
        )
        arrows.add(arr)

        # Weight label
        w_lbl = MathTex(weight_labels[i], font_size=16, color=ACCENT_ORANGE)
        w_lbl.next_to(arr, UP, buff=0.05)
        weight_texts.add(w_lbl)

    # Neuron body
    neuron_body = Circle(
        radius=neuron_radius,
        fill_color=ACCENT_GREEN,
        fill_opacity=0.2,
        stroke_color=ACCENT_GREEN,
        stroke_width=2,
    )
    neuron_body.move_to(ORIGIN)

    # Summation + activation inside neuron
    sigma = MathTex(r"\sigma(\sum)", font_size=24, color=TEXT_PRIMARY)
    sigma.move_to(neuron_body.get_center())

    # Bias
    bias_label = MathTex("+b", font_size=18, color=ACCENT_YELLOW)
    bias_label.next_to(neuron_body, DOWN, buff=0.15)

    # Output arrow
    output_arrow = Arrow(
        neuron_body.get_right(),
        [3, 0, 0],
        buff=0.05,
        stroke_color=TEXT_DIM,
        stroke_width=2,
    )
    output_label = MathTex(r"\hat{y}", font_size=24, color=TEXT_PRIMARY)
    output_label.next_to(output_arrow, RIGHT, buff=0.1)

    # Activation label
    act_label = Text(activation, font_size=14, color=ACCENT_GREEN)
    act_label.next_to(neuron_body, UP, buff=0.15)

    diagram.add(
        input_nodes, input_texts, arrows, weight_texts,
        neuron_body, sigma, bias_label,
        output_arrow, output_label, act_label,
    )

    return {
        "diagram": diagram,
        "inputs": input_nodes,
        "input_texts": input_texts,
        "weights": weight_texts,
        "arrows": arrows,
        "neuron": neuron_body,
        "sigma": sigma,
        "output_arrow": output_arrow,
        "output_label": output_label,
    }


# ============================================================================
# 10. CONVENIENCE: COMMON LOSS FUNCTIONS
# ============================================================================

def quadratic_loss(x):
    """Simple quadratic loss: L(x) = x^2."""
    return x ** 2

def quadratic_loss_deriv(x):
    """Derivative of quadratic loss: L'(x) = 2x."""
    return 2 * x

def bumpy_loss(x):
    """Non-convex loss with local minima: good for illustrating SGD issues."""
    return 0.5 * x ** 2 + 2 * np.sin(2 * x) + 3

def bumpy_loss_deriv(x):
    """Derivative of bumpy_loss."""
    return x + 4 * np.cos(2 * x)

def bowl_2d(x, y):
    """Simple 2D bowl: L(w1, w2) = w1^2 + w2^2."""
    return x ** 2 + y ** 2

def bowl_2d_grad(x, y):
    """Gradient of bowl_2d."""
    return (2 * x, 2 * y)

def rosenbrock_2d(x, y, a=1, b=5):
    """Rosenbrock function: classic non-convex optimization test."""
    return (a - x) ** 2 + b * (y - x ** 2) ** 2

def rosenbrock_2d_grad(x, y, a=1, b=5):
    """Gradient of Rosenbrock function."""
    dx = -2 * (a - x) - 4 * b * x * (y - x ** 2)
    dy = 2 * b * (y - x ** 2)
    return (dx, dy)


# ============================================================================
# 11. FULL DEMO SCENES (can be rendered directly)
# ============================================================================

def build_nn_overview_section(scene, layer_sizes=None):
    """Build and animate a complete neural network overview section.

    Draws the network, runs a forward pass, then a backward pass.
    Designed to be called inside a scene's construct() method.

    Args:
        scene: The Manim scene instance.
        layer_sizes: Network architecture, default [3, 5, 5, 2].
    """
    if layer_sizes is None:
        layer_sizes = [3, 5, 5, 2]

    net_data = draw_neural_network(
        layer_sizes=layer_sizes,
        neuron_radius=0.18,
        layer_labels=["Input", "Hidden 1", "Hidden 2", "Output"],
    )

    # Scale to fit
    net_data["network"].scale(0.8).shift(DOWN * 0.3)

    animate_network_creation(scene, net_data, run_time=3)
    scene.wait(0.5)
    animate_forward_pass(scene, net_data, input_values=[1.0, 0.5, -0.3], run_time=4)
    scene.wait(0.5)
    animate_backpropagation(scene, net_data, run_time=3)
    scene.wait(0.5)

    scene.play(FadeOut(net_data["network"]), run_time=0.5)


def build_gradient_descent_section(scene):
    """Build and animate a gradient descent demonstration section.

    Shows 1D gradient descent on a quadratic loss and on a bumpy loss.

    Args:
        scene: The Manim scene instance.
    """
    # Title
    gd_title = MathTex(
        r"\theta_{t+1} = \theta_t - \alpha \nabla L(\theta_t)",
        font_size=36,
        color=TEXT_PRIMARY,
    )
    gd_title.shift(UP * 2.5)
    scene.play(Write(gd_title), run_time=1.5)

    # Quadratic loss
    gd_data = animate_gradient_descent(
        scene,
        loss_func=quadratic_loss,
        loss_func_deriv=quadratic_loss_deriv,
        start_x=3.5,
        learning_rate=0.2,
        num_steps=10,
        x_range=(-4, 4, 1),
        y_range=(-1, 16, 4),
    )

    scene.wait(1)

    # Cleanup
    all_gd = VGroup(
        gd_title, gd_data["axes"], gd_data["ax_labels"],
        gd_data["curve"], gd_data["dot"],
    )
    if gd_data["lr_label"]:
        all_gd.add(gd_data["lr_label"])
    scene.play(FadeOut(all_gd), run_time=0.5)


def build_activation_comparison_section(scene):
    """Build and animate an activation function comparison section.

    Shows ReLU, sigmoid, tanh, and leaky ReLU in a 2x2 grid.

    Args:
        scene: The Manim scene instance.
    """
    act_data = draw_activation_functions(
        functions=["relu", "sigmoid", "tanh", "leaky_relu"],
        axes_width=3.5,
        axes_height=2.0,
        arrangement="grid",
    )

    act_data["group"].scale(0.85).shift(DOWN * 0.3)
    animate_activation_comparison(scene, act_data, run_time=8)
    scene.wait(1)
    scene.play(FadeOut(act_data["group"]), run_time=0.5)
