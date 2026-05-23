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
FIGURE_FACE = "#f8fafc"
AXES_FACE = "#fcfdff"


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


def summarize_series(frame: pd.DataFrame) -> tuple[float, float, float, float]:
    start_step = frame["step_millions"].iloc[0]
    end_step = frame["step_millions"].iloc[-1]
    start_value = frame["value"].iloc[0]
    end_value = frame["value"].iloc[-1]
    return start_step, end_step, start_value, end_value


def format_delta(start_value: float, end_value: float, *, percent: bool = False) -> str:
    if percent:
        return f"{end_value - start_value:+.1%}"
    return f"{end_value - start_value:+.2f}"


def add_figure_text(fig: plt.Figure, title: str, subtitle: str, caption: str) -> None:
    fig.text(0.06, 0.975, title, ha="left", va="top", fontsize=18, fontweight="bold", color="#0f172a")
    fig.text(0.06, 0.948, subtitle, ha="left", va="top", fontsize=10.5, color="#475569")
    fig.text(0.06, 0.028, caption, ha="left", va="bottom", fontsize=9.5, color="#475569")


def style_axis(ax: plt.Axes, ylabel: str, xlabel: str = "Training steps (millions)") -> None:
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.grid(True, alpha=0.22, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#94a3b8")
    ax.spines["bottom"].set_color("#94a3b8")
    ax.tick_params(length=0)


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
    value_format: str = ".2f",
) -> None:
    smoothed = smooth_values(frame, window)
    ax.set_facecolor(AXES_FACE)
    ax.plot(frame["step_millions"], frame["value"], color=color, alpha=raw_alpha, linewidth=1.2)
    ax.plot(frame["step_millions"], smoothed, color=color, linewidth=2.6)
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold", pad=10)
    style_axis(ax, ylabel)

    if add_final_label:
        final_x = frame["step_millions"].iloc[-1]
        final_y = smoothed.iloc[-1]
        ax.scatter([final_x], [final_y], color=color, s=28, zorder=3)
        ax.annotate(
            format(final_y, value_format),
            xy=(final_x, final_y),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=9,
            color=color,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "none", "alpha": 0.9},
        )


def require_tags(frames: dict[str, pd.DataFrame], tags: list[str]) -> None:
    missing = [tag for tag in tags if tag not in frames]
    if missing:
        raise KeyError(f"Missing TensorBoard scalar tags: {', '.join(missing)}")


def save_figure(fig: plt.Figure, output_path: Path, dpi: int) -> None:
    fig.tight_layout(rect=(0.04, 0.06, 0.98, 0.92))
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def make_progress_overview(frames: dict[str, pd.DataFrame], output_dir: Path, window: int, dpi: int) -> None:
    tags = ["mario/flag_capture_rate", "mario/mean_x_pos", "rollout/ep_rew_mean"]
    require_tags(frames, tags)

    fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=False)
    fig.set_facecolor(FIGURE_FACE)
    start_step, end_step, start_flag, end_flag = summarize_series(frames["mario/flag_capture_rate"])
    _, _, start_x, end_x = summarize_series(frames["mario/mean_x_pos"])
    _, _, start_reward, end_reward = summarize_series(frames["rollout/ep_rew_mean"])
    add_figure_text(
        fig,
        "Phase M Learning Curves",
        f"Seed 1 resume window from {start_step:.1f}M to {end_step:.1f}M steps; smoothed traces emphasize trend while preserving raw variance.",
        "Caption: Task-facing metrics improve together across the final 4M training steps, with higher flag completion, deeper stage penetration, and stronger episodic return.",
    )

    plot_scalar(
        axes[0],
        frames["mario/flag_capture_rate"],
        title=f"Flag capture rate  |  {start_flag:.1%} -> {end_flag:.1%} ({format_delta(start_flag, end_flag, percent=True)})",
        ylabel="Flag capture rate",
        color="#0f766e",
        window=window,
        value_format=".0%",
    )
    axes[0].yaxis.set_major_formatter(PercentFormatter(xmax=1.0))

    plot_scalar(
        axes[1],
        frames["mario/mean_x_pos"],
        title=f"Mean terminal x-position  |  {start_x:.0f} -> {end_x:.0f} ({end_x - start_x:+.0f})",
        ylabel="Mean final x position",
        color="#1d4ed8",
        window=window,
        value_format=".0f",
    )

    plot_scalar(
        axes[2],
        frames["rollout/ep_rew_mean"],
        title=f"Episode reward  |  {start_reward:.0f} -> {end_reward:.0f} ({end_reward - start_reward:+.0f})",
        ylabel="Episode reward",
        color="#b45309",
        window=window,
        value_format=".0f",
    )

    save_figure(fig, output_dir / "01_phase_m_progress_overview.png", dpi)


def make_policy_stability(frames: dict[str, pd.DataFrame], output_dir: Path, window: int, dpi: int) -> None:
    tags = ["train/entropy_loss", "train/approx_kl", "train/clip_fraction"]
    require_tags(frames, tags)

    fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=False)
    fig.set_facecolor(FIGURE_FACE)
    _, _, start_ent, end_ent = summarize_series(frames["train/entropy_loss"])
    _, _, start_kl, end_kl = summarize_series(frames["train/approx_kl"])
    _, _, start_clip, end_clip = summarize_series(frames["train/clip_fraction"])
    add_figure_text(
        fig,
        "PPO Stability Diagnostics",
        "Optimization-side curves reveal whether policy improvement came from steady refinement rather than unstable late-stage updates.",
        "Caption: Entropy, KL, and clip fraction jointly summarize exploration pressure and PPO update behavior during the successful Phase M continuation run.",
    )

    plot_scalar(
        axes[0],
        frames["train/entropy_loss"],
        title=f"Policy entropy  |  {start_ent:.3f} -> {end_ent:.3f} ({end_ent - start_ent:+.3f})",
        ylabel="Entropy loss",
        color="#7c3aed",
        window=window,
        value_format=".3f",
    )

    plot_scalar(
        axes[1],
        frames["train/approx_kl"],
        title=f"Approximate KL  |  {start_kl:.3f} -> {end_kl:.3f} ({end_kl - start_kl:+.3f})",
        ylabel="Approx KL",
        color="#dc2626",
        window=window,
        value_format=".3f",
    )

    plot_scalar(
        axes[2],
        frames["train/clip_fraction"],
        title=f"Clip fraction  |  {start_clip:.3f} -> {end_clip:.3f} ({end_clip - start_clip:+.3f})",
        ylabel="Clip fraction",
        color="#2563eb",
        window=window,
        value_format=".3f",
    )

    save_figure(fig, output_dir / "02_phase_m_policy_stability.png", dpi)


def make_collapse_diagnostics(frames: dict[str, pd.DataFrame], output_dir: Path, window: int, dpi: int) -> None:
    tags = ["collapse/entropy", "collapse/flag_rate", "collapse/entropy_vel_rolling", "collapse/kl_clip_frac"]
    require_tags(frames, tags)

    fig, axes = plt.subplots(4, 1, figsize=(12, 13), sharex=False)
    fig.set_facecolor(FIGURE_FACE)
    _, _, start_entropy, end_entropy = summarize_series(frames["collapse/entropy"])
    _, _, start_flag, end_flag = summarize_series(frames["collapse/flag_rate"])
    _, _, start_vel, end_vel = summarize_series(frames["collapse/entropy_vel_rolling"])
    _, _, start_klfrac, end_klfrac = summarize_series(frames["collapse/kl_clip_frac"])
    add_figure_text(
        fig,
        "Entropy-Collapse Monitoring",
        "Custom diagnostics track whether strong returns are being purchased by brittle entropy collapse, a recurring failure mode in long Mario PPO runs.",
        "Caption: The collapse monitor complements standard PPO metrics by exposing entropy drift, KL threshold pressure, and task success in the same diagnostic frame.",
    )

    plot_scalar(
        axes[0],
        frames["collapse/entropy"],
        title=f"Captured entropy  |  {start_entropy:.3f} -> {end_entropy:.3f} ({end_entropy - start_entropy:+.3f})",
        ylabel="Entropy",
        color="#9333ea",
        window=window,
        value_format=".3f",
    )

    plot_scalar(
        axes[1],
        frames["collapse/flag_rate"],
        title=f"Diagnostic flag rate  |  {start_flag:.1%} -> {end_flag:.1%} ({format_delta(start_flag, end_flag, percent=True)})",
        ylabel="Flag rate",
        color="#059669",
        window=window,
        value_format=".0%",
    )
    axes[1].yaxis.set_major_formatter(PercentFormatter(xmax=1.0))

    plot_scalar(
        axes[2],
        frames["collapse/entropy_vel_rolling"],
        title=f"Rolling entropy velocity  |  {start_vel:.3f} -> {end_vel:.3f} ({end_vel - start_vel:+.3f})",
        ylabel="Velocity",
        color="#ea580c",
        window=window,
        value_format=".3f",
    )
    axes[2].axhline(0.0, color="#6b7280", linestyle="--", linewidth=1.0, alpha=0.7)

    plot_scalar(
        axes[3],
        frames["collapse/kl_clip_frac"],
        title=f"KL over-threshold fraction  |  {start_klfrac:.3f} -> {end_klfrac:.3f} ({end_klfrac - start_klfrac:+.3f})",
        ylabel="KL clip fraction",
        color="#0f766e",
        window=window,
        value_format=".3f",
    )

    save_figure(fig, output_dir / "03_phase_m_collapse_diagnostics.png", dpi)


def make_eval_curve(frames: dict[str, pd.DataFrame], output_dir: Path, window: int, dpi: int) -> None:
    tags = ["eval/mean_reward", "eval/mean_ep_length"]
    require_tags(frames, tags)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8.5), sharex=False)
    fig.set_facecolor(FIGURE_FACE)
    _, _, start_reward, end_reward = summarize_series(frames["eval/mean_reward"])
    _, _, start_length, end_length = summarize_series(frames["eval/mean_ep_length"])
    add_figure_text(
        fig,
        "Deterministic EvalCallback Snapshot",
        "This figure is useful for internal comparison, but it should be interpreted as a training-time checkpoint signal rather than the final stochastic benchmark.",
        "Caption: EvalCallback metrics here reflect deterministic policy rollouts; the final 87.5% project result comes from separate stochastic evaluation of the 12.0M checkpoint.",
    )

    plot_scalar(
        axes[0],
        frames["eval/mean_reward"],
        title=f"Periodic evaluation reward  |  {start_reward:.0f} -> {end_reward:.0f} ({end_reward - start_reward:+.0f})",
        ylabel="Mean reward",
        color="#0284c7",
        window=max(5, window // 2),
        raw_alpha=0.12,
        value_format=".0f",
    )

    plot_scalar(
        axes[1],
        frames["eval/mean_ep_length"],
        title=f"Periodic evaluation episode length  |  {start_length:.0f} -> {end_length:.0f} ({end_length - start_length:+.0f})",
        ylabel="Mean episode length",
        color="#ca8a04",
        window=max(5, window // 2),
        raw_alpha=0.12,
        value_format=".0f",
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
            "figure.facecolor": FIGURE_FACE,
            "axes.facecolor": AXES_FACE,
            "savefig.facecolor": FIGURE_FACE,
            "axes.edgecolor": "#cbd5e1",
            "axes.labelcolor": "#0f172a",
            "axes.titlecolor": "#0f172a",
            "xtick.color": "#374151",
            "ytick.color": "#374151",
            "grid.color": "#dbe4ee",
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
            "axes.titleweight": "bold",
            "axes.labelsize": 10.5,
            "axes.titlesize": 12,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
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