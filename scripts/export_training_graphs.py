"""Export publication-style training graphs from TensorBoard event files.

Example:
    python scripts/export_training_graphs.py
    python scripts/export_training_graphs.py --logdir results/multiseed_s1_M/logs/PPO_0 --output training_graphs
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import PercentFormatter
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


DEFAULT_LOGDIR = Path("results/multiseed_s1_M/logs/PPO_0")
DEFAULT_OUTPUT = Path("training_graphs")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logdir", type=Path, default=DEFAULT_LOGDIR, help="TensorBoard run directory containing event files.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Directory where PNG plots will be written.")
    parser.add_argument("--window", type=int, default=11, help="Rolling window used to smooth dense scalar series.")
    parser.add_argument("--dpi", type=int, default=220, help="Output DPI for saved figures.")
    return parser.parse_args()


def load_scalars(logdir: Path) -> dict[str, pd.DataFrame]:
    if not logdir.exists():
        raise FileNotFoundError(f"TensorBoard logdir not found: {logdir}")

    accumulator = EventAccumulator(str(logdir))
    accumulator.Reload()

    frames: dict[str, pd.DataFrame] = {}
    for tag in accumulator.Tags().get("scalars", []):
        scalars = accumulator.Scalars(tag)
        if not scalars:
            continue
        frame = pd.DataFrame(
            {
                "step": [item.step for item in scalars],
                "value": [item.value for item in scalars],
            }
        )
        frame["step_millions"] = frame["step"] / 1_000_000
        frames[tag] = frame
    return frames


def smooth_values(frame: pd.DataFrame, window: int) -> pd.Series:
    if len(frame) < 4:
        return frame["value"]
    effective_window = max(3, min(window, len(frame)))
    if effective_window % 2 == 0:
        effective_window -= 1
    return frame["value"].rolling(effective_window, min_periods=1, center=True).mean()


def style_axis(ax: plt.Axes, ylabel: str, xlabel: str = "Training steps (millions)") -> None:
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.grid(True, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_scalar(
    ax: plt.Axes,
    frame: pd.DataFrame,
    *,
    title: str,
    ylabel: str,
    color: str,
    window: int,
    raw_alpha: float = 0.22,
    add_final_label: bool = True,
) -> None:
    smoothed = smooth_values(frame, window)
    ax.plot(frame["step_millions"], frame["value"], color=color, alpha=raw_alpha, linewidth=1.0)
    ax.plot(frame["step_millions"], smoothed, color=color, linewidth=2.4)
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold")
    style_axis(ax, ylabel)

    if add_final_label:
        final_x = frame["step_millions"].iloc[-1]
        final_y = smoothed.iloc[-1]
        ax.scatter([final_x], [final_y], color=color, s=22, zorder=3)
        ax.annotate(
            f"{final_y:.2f}",
            xy=(final_x, final_y),
            xytext=(8, 6),
            textcoords="offset points",
            fontsize=9,
            color=color,
        )


def require_tags(frames: dict[str, pd.DataFrame], tags: list[str]) -> None:
    missing = [tag for tag in tags if tag not in frames]
    if missing:
        raise KeyError(f"Missing TensorBoard scalar tags: {', '.join(missing)}")


def save_figure(fig: plt.Figure, output_path: Path, dpi: int) -> None:
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def make_progress_overview(frames: dict[str, pd.DataFrame], output_dir: Path, window: int, dpi: int) -> None:
    tags = ["mario/flag_capture_rate", "mario/mean_x_pos", "rollout/ep_rew_mean"]
    require_tags(frames, tags)

    fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=False)
    fig.suptitle("Phase M Training Progress Overview", fontsize=16, fontweight="bold", y=0.995)

    plot_scalar(
        axes[0],
        frames["mario/flag_capture_rate"],
        title="Task Success Trend",
        ylabel="Flag capture rate",
        color="#0f766e",
        window=window,
    )
    axes[0].yaxis.set_major_formatter(PercentFormatter(xmax=1.0))

    plot_scalar(
        axes[1],
        frames["mario/mean_x_pos"],
        title="Task Progression",
        ylabel="Mean final x position",
        color="#1d4ed8",
        window=window,
    )

    plot_scalar(
        axes[2],
        frames["rollout/ep_rew_mean"],
        title="Return Improvement",
        ylabel="Episode reward",
        color="#b45309",
        window=window,
    )

    save_figure(fig, output_dir / "01_phase_m_progress_overview.png", dpi)


def make_policy_stability(frames: dict[str, pd.DataFrame], output_dir: Path, window: int, dpi: int) -> None:
    tags = ["train/entropy_loss", "train/approx_kl", "train/clip_fraction"]
    require_tags(frames, tags)

    fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=False)
    fig.suptitle("Phase M Policy Optimization Stability", fontsize=16, fontweight="bold", y=0.995)

    plot_scalar(
        axes[0],
        frames["train/entropy_loss"],
        title="Policy Entropy",
        ylabel="Entropy loss",
        color="#7c3aed",
        window=window,
    )

    plot_scalar(
        axes[1],
        frames["train/approx_kl"],
        title="KL Divergence",
        ylabel="Approx KL",
        color="#dc2626",
        window=window,
    )

    plot_scalar(
        axes[2],
        frames["train/clip_fraction"],
        title="Update Clipping Activity",
        ylabel="Clip fraction",
        color="#2563eb",
        window=window,
    )

    save_figure(fig, output_dir / "02_phase_m_policy_stability.png", dpi)


def make_collapse_diagnostics(frames: dict[str, pd.DataFrame], output_dir: Path, window: int, dpi: int) -> None:
    tags = ["collapse/entropy", "collapse/flag_rate", "collapse/entropy_vel_rolling", "collapse/kl_clip_frac"]
    require_tags(frames, tags)

    fig, axes = plt.subplots(4, 1, figsize=(12, 13), sharex=False)
    fig.suptitle("Phase M Collapse Diagnostics", fontsize=16, fontweight="bold", y=0.995)

    plot_scalar(
        axes[0],
        frames["collapse/entropy"],
        title="Captured Entropy Signal",
        ylabel="Entropy",
        color="#9333ea",
        window=window,
    )

    plot_scalar(
        axes[1],
        frames["collapse/flag_rate"],
        title="Diagnostic Flag Rate",
        ylabel="Flag rate",
        color="#059669",
        window=window,
    )
    axes[1].yaxis.set_major_formatter(PercentFormatter(xmax=1.0))

    plot_scalar(
        axes[2],
        frames["collapse/entropy_vel_rolling"],
        title="Rolling Entropy Velocity",
        ylabel="Velocity",
        color="#ea580c",
        window=window,
    )
    axes[2].axhline(0.0, color="#6b7280", linestyle="--", linewidth=1.0, alpha=0.7)

    plot_scalar(
        axes[3],
        frames["collapse/kl_clip_frac"],
        title="KL Over-Threshold Fraction",
        ylabel="KL clip fraction",
        color="#0f766e",
        window=window,
    )

    save_figure(fig, output_dir / "03_phase_m_collapse_diagnostics.png", dpi)


def make_eval_curve(frames: dict[str, pd.DataFrame], output_dir: Path, window: int, dpi: int) -> None:
    tags = ["eval/mean_reward", "eval/mean_ep_length"]
    require_tags(frames, tags)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8.5), sharex=False)
    fig.suptitle("Phase M EvalCallback Trends", fontsize=16, fontweight="bold", y=0.995)

    plot_scalar(
        axes[0],
        frames["eval/mean_reward"],
        title="Periodic Evaluation Reward",
        ylabel="Mean reward",
        color="#0284c7",
        window=max(5, window // 2),
        raw_alpha=0.12,
    )

    plot_scalar(
        axes[1],
        frames["eval/mean_ep_length"],
        title="Periodic Evaluation Episode Length",
        ylabel="Mean episode length",
        color="#ca8a04",
        window=max(5, window // 2),
        raw_alpha=0.12,
    )

    save_figure(fig, output_dir / "04_phase_m_eval_callback.png", dpi)


def write_manifest(frames: dict[str, pd.DataFrame], output_dir: Path, source_logdir: Path) -> None:
    lines = [
        "Project Mario training graph export",
        f"source_logdir: {source_logdir.as_posix()}",
        "",
        "available_scalar_tags:",
    ]
    lines.extend(f"- {tag} ({len(frame)} points)" for tag, frame in sorted(frames.items()))
    (output_dir / "manifest.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def configure_plot_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#d1d5db",
            "axes.labelcolor": "#111827",
            "axes.titlecolor": "#111827",
            "xtick.color": "#374151",
            "ytick.color": "#374151",
            "grid.color": "#cbd5e1",
            "font.size": 10,
        }
    )


def main() -> None:
    args = parse_args()
    configure_plot_style()

    frames = load_scalars(args.logdir)
    args.output.mkdir(parents=True, exist_ok=True)

    make_progress_overview(frames, args.output, args.window, args.dpi)
    make_policy_stability(frames, args.output, args.window, args.dpi)
    make_collapse_diagnostics(frames, args.output, args.window, args.dpi)
    make_eval_curve(frames, args.output, args.window, args.dpi)
    write_manifest(frames, args.output, args.logdir)

    print(f"Wrote plots to {args.output}")


if __name__ == "__main__":
    main()