"""Evaluation entry point for Project Mario.

Usage:
    python -m src.evaluate                     # run random agent, print stats
    python -m src.evaluate --record            # record video to results/videos/
    python -m src.evaluate --episodes 10       # run 10 episodes
    python -m src.evaluate --config path.yaml  # custom config
"""

import os

from src.config import Config, parse_args, set_global_seed
from src.envs import make_single_env


def evaluate(cfg: Config, episodes: int = 5, record: bool = False) -> None:
    """Run random policy for N episodes on a single env, print per-episode stats."""
    video_dir = "results/videos"
    if record:
        os.makedirs(video_dir, exist_ok=True)

    env = make_single_env(cfg, seed=cfg.seed, record_video=record, video_dir=video_dir)

    for ep in range(episodes):
        obs, info = env.reset()
        total_reward = 0.0
        steps = 0

        while True:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            if terminated or truncated:
                break

        print(
            f"  Episode {ep + 1}/{episodes}: "
            f"x_pos={info.get('x_pos', '?'):>5}  "
            f"reward={total_reward:>8.1f}  "
            f"steps={steps:>4}  "
            f"flag={info.get('flag_get', False)}"
        )

    env.close()
    if record:
        print(f"\nVideos saved to {video_dir}/")


def main() -> None:
    args = parse_args()
    cfg = Config.from_yaml(args.config)

    if args.seed is not None:
        cfg.seed = args.seed

    set_global_seed(cfg.seed)

    episodes = getattr(args, "episodes", 5)
    record = getattr(args, "record", False)
    evaluate(cfg, episodes=episodes, record=record)


if __name__ == "__main__":
    main()
