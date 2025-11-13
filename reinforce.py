"""Implementation of REINFORCE on the short corridor example.

The script trains two policy parameterisations on the short corridor with
switched actions and compares their learning curves.  The first model shares
parameters across most states via hand-crafted features, while the second uses
full tabular (one-hot) features.
"""
from __future__ import annotations

import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Tuple
from xml.etree.ElementTree import Element, SubElement, tostring


Action = int
State = int
FeatureVector = Tuple[float, ...]
FeatureFn = Callable[[State], FeatureVector]


class ShortCorridorEnv:
    """Short corridor with switched actions (Sutton & Barto, Example 13.1).

    The environment consists of four positions (0--3).  The agent starts in the
    left-most position (0) and aims to reach the right terminal (3).  Actions
    are interpreted as ``0 -> left`` and ``1 -> right``.  In position ``1`` the
    actions are swapped: selecting ``left`` moves the agent to the right and
    selecting ``right`` moves it to the left.  Every transition yields a reward
    of ``-1`` until the terminal state is reached.
    """

    LEFT = 0
    RIGHT = 1

    def __init__(self) -> None:
        self.start_state = 0
        self.terminal_state = 3
        self._state = self.start_state

    def reset(self) -> State:
        self._state = self.start_state
        return self._state

    def step(self, action: Action) -> Tuple[State, float, bool]:
        if action not in (self.LEFT, self.RIGHT):
            msg = f"Invalid action {action}. Valid actions are 0 (left) and 1 (right)."
            raise ValueError(msg)

        if self._state == self.terminal_state:
            msg = "Cannot step from the terminal state. Call reset() first."
            raise RuntimeError(msg)

        reward = -1.0
        next_state = self._transition(self._state, action)
        self._state = next_state
        done = next_state == self.terminal_state
        return next_state, reward, done

    @staticmethod
    def _transition(state: State, action: Action) -> State:
        if state == 0:
            return 0 if action == ShortCorridorEnv.LEFT else 1
        if state == 1:
            # Switched actions in the second cell of the corridor.
            return 2 if action == ShortCorridorEnv.LEFT else 0
        if state == 2:
            return 1 if action == ShortCorridorEnv.LEFT else 3
        if state == 3:
            return 3
        msg = f"Unknown state {state}."
        raise ValueError(msg)


class LogisticPolicy:
    """Binary-action soft policy parameterised by a linear logit."""

    def __init__(self, feature_fn: FeatureFn) -> None:
        feature_vector = feature_fn(0)
        self.feature_fn = feature_fn
        self.theta = [0.0 for _ in feature_vector]

    def action_probability(self, state: State) -> float:
        """Probability of taking the RIGHT action."""

        phi = self.feature_fn(state)
        logit = sum(theta_i * phi_i for theta_i, phi_i in zip(self.theta, phi))
        return 1.0 / (1.0 + math.exp(-logit))

    def sample_action(self, state: State, rng: random.Random) -> Tuple[Action, FeatureVector, float]:
        phi = self.feature_fn(state)
        logit = sum(theta_i * phi_i for theta_i, phi_i in zip(self.theta, phi))
        p_right = 1.0 / (1.0 + math.exp(-logit))
        action = int(rng.random() < p_right)
        return action, phi, p_right

    @staticmethod
    def grad_log_probability(phi: FeatureVector, action: Action, prob_right: float) -> FeatureVector:
        # Bernoulli log-likelihood gradient under logistic parameterisation.
        return tuple((action - prob_right) * value for value in phi)


@dataclass
class ReinforceConfig:
    alpha: float
    gamma: float
    episodes: int
    max_steps: int | None = None


@dataclass
class ReinforceHistory:
    returns: Tuple[float, ...]
    lengths: Tuple[float, ...]


def reinforce(
    env: ShortCorridorEnv,
    policy: LogisticPolicy,
    config: ReinforceConfig,
    rng: random.Random,
) -> ReinforceHistory:
    returns = [0.0 for _ in range(config.episodes)]
    lengths = [0.0 for _ in range(config.episodes)]

    for episode in range(config.episodes):
        state = env.reset()
        done = False
        trajectory: list[tuple[FeatureVector, Action, float, float]] = []

        steps = 0
        while not done:
            action, phi, prob_right = policy.sample_action(state, rng)
            next_state, reward, done = env.step(action)
            trajectory.append((phi, action, reward, prob_right))
            state = next_state
            steps += 1

            if config.max_steps is not None and steps >= config.max_steps:
                break

        G = 0.0
        for phi, action, reward, prob_right in reversed(trajectory):
            G = reward + config.gamma * G
            grad = policy.grad_log_probability(phi, action, prob_right)
            policy.theta = [theta_i + config.alpha * G * grad_i for theta_i, grad_i in zip(policy.theta, grad)]

        returns[episode] = sum(item[2] for item in trajectory)
        lengths[episode] = float(len(trajectory))

    return ReinforceHistory(returns=tuple(returns), lengths=tuple(lengths))


def shared_feature_vector(state: State) -> FeatureVector:
    """Feature vector suggested in lecture slides.

    The vector has two components:
    - A bias term shared by all states.
    - An indicator for the second cell, which is the only state with switched
      actions.  This allows the policy to learn a correction for the unusual
      transition dynamics while generalising across the rest of the corridor.
    """

    return (1.0, 1.0 if state == 1 else 0.0)


def tabular_feature_vector(state: State) -> FeatureVector:
    """One-hot feature vector for each non-terminal state."""

    if state == 3:
        # The terminal state is never used for action selection.
        return (0.0, 0.0, 0.0)
    one_hot = [0.0, 0.0, 0.0]
    one_hot[state] = 1.0
    return tuple(one_hot)


def aggregate_runs(
    feature_fn: FeatureFn,
    config: ReinforceConfig,
    runs: int,
    seed: int,
) -> ReinforceHistory:
    aggregated_returns = [0.0 for _ in range(config.episodes)]
    aggregated_lengths = [0.0 for _ in range(config.episodes)]

    for run in range(runs):
        rng = random.Random(seed + run)
        env = ShortCorridorEnv()
        policy = LogisticPolicy(feature_fn)
        history = reinforce(env, policy, config, rng)
        aggregated_returns = [acc + value for acc, value in zip(aggregated_returns, history.returns)]
        aggregated_lengths = [acc + value for acc, value in zip(aggregated_lengths, history.lengths)]

    aggregated_returns = [value / runs for value in aggregated_returns]
    aggregated_lengths = [value / runs for value in aggregated_lengths]

    return ReinforceHistory(returns=tuple(aggregated_returns), lengths=tuple(aggregated_lengths))


def save_learning_curves(
    shared: ReinforceHistory,
    tabular: ReinforceHistory,
    config: ReinforceConfig,
    output_path: Path,
) -> None:
    episodes = range(1, config.episodes + 1)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "episode",
                "shared_return",
                "tabular_return",
                "shared_length",
                "tabular_length",
            ]
        )
        for episode, shared_return, tabular_return, shared_length, tabular_length in zip(
            episodes,
            shared.returns,
            tabular.returns,
            shared.lengths,
            tabular.lengths,
        ):
            writer.writerow([episode, shared_return, tabular_return, shared_length, tabular_length])


def plot_learning_curves(
    shared: ReinforceHistory,
    tabular: ReinforceHistory,
    config: ReinforceConfig,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    width, height = 960, 360
    margin = 60
    panel_gap = 40
    panel_width = (width - 3 * margin - panel_gap) / 2
    panel_height = height - 2 * margin

    svg = Element(
        "svg",
        {
            "xmlns": "http://www.w3.org/2000/svg",
            "width": str(width),
            "height": str(height),
            "viewBox": f"0 0 {width} {height}",
        },
    )

    def add_text(x: float, y: float, text: str, size: int = 14, anchor: str = "middle") -> None:
        SubElement(
            svg,
            "text",
            {
                "x": f"{x:.2f}",
                "y": f"{y:.2f}",
                "font-size": str(size),
                "text-anchor": anchor,
                "font-family": "sans-serif",
            },
        ).text = text

    def add_panel(
        origin_x: float,
        title: str,
        y_label: str,
        series: tuple[tuple[str, tuple[float, ...]], ...],
    ) -> None:
        origin_y = margin
        plot_origin_y = origin_y + panel_height

        # Axis lines
        SubElement(
            svg,
            "line",
            {
                "x1": f"{origin_x:.2f}",
                "y1": f"{plot_origin_y:.2f}",
                "x2": f"{origin_x + panel_width:.2f}",
                "y2": f"{plot_origin_y:.2f}",
                "stroke": "#000",
                "stroke-width": "1",
            },
        )
        SubElement(
            svg,
            "line",
            {
                "x1": f"{origin_x:.2f}",
                "y1": f"{origin_y:.2f}",
                "x2": f"{origin_x:.2f}",
                "y2": f"{plot_origin_y:.2f}",
                "stroke": "#000",
                "stroke-width": "1",
            },
        )

        add_text(origin_x + panel_width / 2, origin_y - 15, title, size=16)
        add_text(origin_x - 45, origin_y + panel_height / 2, y_label, size=14, anchor="middle")

        # Determine value range across all series for consistent scaling.
        values = [value for _, data in series for value in data]
        min_value = min(values)
        max_value = max(values)
        if math.isclose(max_value, min_value):
            max_value = min_value + 1.0

        def scale_x(index: int) -> float:
            if config.episodes == 1:
                return origin_x + panel_width / 2
            return origin_x + (index / (config.episodes - 1)) * panel_width

        def scale_y(value: float) -> float:
            relative = (value - min_value) / (max_value - min_value)
            return plot_origin_y - relative * panel_height

        # Horizontal grid lines at quartiles
        for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
            y = plot_origin_y - fraction * panel_height
            SubElement(
                svg,
                "line",
                {
                    "x1": f"{origin_x:.2f}",
                    "y1": f"{y:.2f}",
                    "x2": f"{origin_x + panel_width:.2f}",
                    "y2": f"{y:.2f}",
                    "stroke": "#cccccc" if fraction not in (0.0, 1.0) else "#000000",
                    "stroke-width": "0.5" if fraction not in (0.0, 1.0) else "1",
                },
            )
            label_value = min_value + fraction * (max_value - min_value)
            add_text(origin_x - 10, y + 4, f"{label_value:.1f}", size=12, anchor="end")

        # X-axis tick labels
        for tick_episode in (1, config.episodes // 2, config.episodes):
            if tick_episode < 1:
                continue
            x = scale_x(tick_episode - 1)
            SubElement(
                svg,
                "line",
                {
                    "x1": f"{x:.2f}",
                    "y1": f"{plot_origin_y:.2f}",
                    "x2": f"{x:.2f}",
                    "y2": f"{plot_origin_y + 5:.2f}",
                    "stroke": "#000",
                    "stroke-width": "1",
                },
            )
            add_text(x, plot_origin_y + 20, str(tick_episode), size=12)

        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
        legend_y = origin_y + 20
        legend_x = origin_x + panel_width - 10

        for idx, (label, data) in enumerate(series):
            color = colors[idx % len(colors)]
            points = " ".join(
                f"{scale_x(i):.2f},{scale_y(value):.2f}" for i, value in enumerate(data)
            )
            SubElement(
                svg,
                "polyline",
                {
                    "points": points,
                    "fill": "none",
                    "stroke": color,
                    "stroke-width": "2",
                },
            )

            # Legend entry
            legend_group = SubElement(
                svg,
                "g",
                {"transform": f"translate({legend_x - 120},{legend_y + idx * 20})"},
            )
            SubElement(
                legend_group,
                "line",
                {
                    "x1": "0",
                    "y1": "0",
                    "x2": "18",
                    "y2": "0",
                    "stroke": color,
                    "stroke-width": "2",
                },
            )
            SubElement(
                legend_group,
                "text",
                {
                    "x": "22",
                    "y": "4",
                    "font-size": "12",
                    "font-family": "sans-serif",
                },
            ).text = label

    add_panel(
        margin,
        "Average return per episode",
        "Return",
        (
            ("Shared features", shared.returns),
            ("Tabular features", tabular.returns),
        ),
    )
    add_panel(
        margin + panel_width + panel_gap,
        "Average episode length",
        "Steps",
        (
            ("Shared features", shared.lengths),
            ("Tabular features", tabular.lengths),
        ),
    )

    svg_bytes = tostring(svg, encoding="unicode")
    output_path.write_text(svg_bytes)


def main() -> None:
    config = ReinforceConfig(alpha=0.05, gamma=1.0, episodes=200, max_steps=200)
    runs = 200
    seed = 7

    shared_history = aggregate_runs(shared_feature_vector, config, runs=runs, seed=seed)
    tabular_history = aggregate_runs(tabular_feature_vector, config, runs=runs, seed=seed + 10_000)

    output_path = Path("results/reinforce_learning_curves.csv")
    save_learning_curves(shared_history, tabular_history, config, output_path)

    figure_path = Path("results/reinforce_learning_curves.svg")
    plot_learning_curves(shared_history, tabular_history, config, figure_path)

    print("Saved aggregated learning curves to", output_path)
    print("Saved learning curve figure to", figure_path)
    print(f"Final average return (shared features): {shared_history.returns[-1]:.2f}")
    print(f"Final average return (tabular features): {tabular_history.returns[-1]:.2f}")
    print(f"Final average length (shared features): {shared_history.lengths[-1]:.2f}")
    print(f"Final average length (tabular features): {tabular_history.lengths[-1]:.2f}")


if __name__ == "__main__":
    main()
