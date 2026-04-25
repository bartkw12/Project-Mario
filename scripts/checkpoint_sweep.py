"""Sweep checkpoints with stochastic evaluation to find the most robust model.

Iterates over checkpoint .zip files in a directory (and optionally best_model /
final_model), runs N stochastic episodes on each, and prints a ranked summary
table sorted by flag_capture_rate.

Usage:
    # Sweep all checkpoints + best_model + final_model for ablation_d
    python -m scripts.checkpoint_sweep \
        --config configs/experiments/ablation_d.yaml \
        --checkpoints results/ablation_d/models/checkpoints \
        --extra results/ablation_d/models/best_model/best_model.zip \
               results/ablation_d/models/final_model.zip \
        --episodes 50

    # Sweep only specific checkpoints from ablation_j golden window
    python -m scripts.checkpoint_sweep \
        --config configs/experiments/ablation_j.yaml \
        --models results/ablation_j/models/checkpoints/ppo_mario_8000000_steps.zip \
                 results/ablation_j/models/checkpoints/ppo_mario_8500000_steps.zip \
                 results/ablation_j/models/checkpoints/ppo_mario_9000000_steps.zip \
                 results/ablation_j/models/checkpoints/ppo_mario_9500000_steps.zip \
                 results/ablation_j/models/best_model/best_model.zip \
        --episodes 50
"""

import argparse
import sys
import time
from pathlib import Path

from src.config import Config, set_global_seed
from src.evaluate import evaluate


def _collect_models(args: argparse.Namespace) -> list[Path]:
    """Build the ordered list of model paths to sweep."""
    models: list[Path] = []

    # --models: explicit list of .zip paths
    if args.models:
        for p in args.models:
            path = Path(p)
            if not path.exists():
                print(f"[sweep] WARNING: {path} does not exist, skipping")
                continue
            models.append(path)

    # --checkpoints: directory of .zip files (sorted by timestep)
    if args.checkpoints:
        ckpt_dir = Path(args.checkpoints)
        if not ckpt_dir.is_dir():
            print(f"[sweep] ERROR: {ckpt_dir} is not a directory")
            sys.exit(1)
        zips = sorted(ckpt_dir.glob("*.zip"), key=_sort_key)
        models.extend(zips)

    # --extra: additional paths (best_model, final_model, etc.)
    if args.extra:
        for p in args.extra:
            path = Path(p)
            if not path.exists():
                print(f"[sweep] WARNING: {path} does not exist, skipping")
                continue
            models.append(path)

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for m in models:
        resolved = m.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(m)

    return unique


def _sort_key(path: Path) -> int:
    """Extract timestep number from checkpoint filename for sorting."""
    # e.g. ppo_mario_1000000_steps.zip → 1000000
    stem = path.stem  # ppo_mario_1000000_steps
    parts = stem.split("_")
    for part in parts:
        if part.isdigit():
            return int(part)
    return 0


def _short_name(path: Path) -> str:
    """Create a short display name from the model path."""
    if path.stem == "best_model":
        return "best_model"
    if path.stem == "final_model":
        return "final_model"
    # Checkpoint: extract timestep
    stem = path.stem
    parts = stem.split("_")
    for part in parts:
        if part.isdigit():
            steps = int(part)
            if steps >= 1_000_000:
                return f"{steps / 1_000_000:.1f}M"
            return f"{steps / 1_000:.0f}K"
    return stem


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep checkpoints with stochastic evaluation",
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to YAML config (determines env/reward wrappers)",
    )
    parser.add_argument(
        "--checkpoints", type=str, default=None,
        help="Directory containing checkpoint .zip files",
    )
    parser.add_argument(
        "--models", type=str, nargs="+", default=None,
        help="Explicit list of model .zip paths to evaluate",
    )
    parser.add_argument(
        "--extra", type=str, nargs="+", default=None,
        help="Additional model paths (best_model, final_model, etc.)",
    )
    parser.add_argument(
        "--episodes", type=int, default=50,
        help="Number of stochastic episodes per checkpoint (default: 50)",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Override seed from config",
    )
    args = parser.parse_args()

    models = _collect_models(args)
    if not models:
        print("[sweep] ERROR: No models to evaluate. Use --checkpoints, --models, or --extra.")
        sys.exit(1)

    cfg = Config.from_yaml(args.config)
    if args.seed is not None:
        cfg.seed = args.seed
    set_global_seed(cfg.seed)

    print(f"[sweep] Config: {args.config}")
    print(f"[sweep] Episodes per model: {args.episodes} (stochastic)")
    print(f"[sweep] Models to evaluate: {len(models)}")
    print()

    results = []
    for i, model_path in enumerate(models, 1):
        name = _short_name(model_path)
        print(f"[sweep] ({i}/{len(models)}) Evaluating {name} — {model_path}")

        t0 = time.time()
        res = evaluate(
            cfg,
            episodes=args.episodes,
            model_path=str(model_path),
            stochastic=True,
            quiet=True,
        )
        elapsed = time.time() - t0

        flags = res["flags"]
        flag_rate = res["flag_rate"]
        mean_x = res["mean_x_pos"]
        max_x = res["max_x_pos"]
        mean_rew = res["mean_reward"]
        mean_steps = res["mean_steps"]

        print(
            f"         flag_rate={flag_rate:.0%} ({flags}/{args.episodes})  "
            f"mean_x={mean_x:.0f}  max_x={max_x}  "
            f"reward={mean_rew:.0f}  steps={mean_steps:.0f}  "
            f"({elapsed:.0f}s)"
        )

        results.append({
            "name": name,
            "path": str(model_path),
            "flag_rate": flag_rate,
            "flags": flags,
            "mean_x_pos": mean_x,
            "max_x_pos": max_x,
            "mean_reward": mean_rew,
            "mean_steps": mean_steps,
        })

    # Sort by flag_rate descending, then mean_x_pos descending
    results.sort(key=lambda r: (r["flag_rate"], r["mean_x_pos"]), reverse=True)

    # Print ranked summary table
    print(f"\n{'=' * 95}")
    print(f"  CHECKPOINT SWEEP RESULTS — {args.episodes} stochastic episodes per model")
    print(f"  Config: {args.config}")
    print(f"{'=' * 95}")
    print(
        f"  {'Rank':<5} {'Model':<14} {'Flag%':>6} {'Flags':>6} "
        f"{'mean_x':>8} {'max_x':>7} {'reward':>8} {'steps':>7}"
    )
    print(f"  {'-' * 5} {'-' * 14} {'-' * 6} {'-' * 6} {'-' * 8} {'-' * 7} {'-' * 8} {'-' * 7}")

    for rank, r in enumerate(results, 1):
        marker = " ★" if rank == 1 else ""
        print(
            f"  {rank:<5} {r['name']:<14} {r['flag_rate']:>5.0%} "
            f"{r['flags']:>5}/{args.episodes}  "
            f"{r['mean_x_pos']:>7.0f} {r['max_x_pos']:>7} "
            f"{r['mean_reward']:>8.0f} {r['mean_steps']:>6.0f}{marker}"
        )

    print(f"{'=' * 95}")

    # Print the winner
    best = results[0]
    print(f"\n  Best checkpoint: {best['name']} — {best['flag_rate']:.0%} flag rate ({best['flags']}/{args.episodes})")
    print(f"  Path: {best['path']}")


if __name__ == "__main__":
    main()
