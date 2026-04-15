"""Evaluation entry point for Project Mario.

Usage:
    python -m src.evaluate                                    # random agent
    python -m src.evaluate --model results/models/best_model/best_model.zip
    python -m src.evaluate --model best_model.zip --episodes 50 --record
"""

import os

import numpy as np
from stable_baselines3 import PPO

from src.config import Config, parse_args, set_global_seed
from src.envs import make_vec_env


def evaluate(cfg: Config, episodes: int = 5, record: bool = False, model_path: str | None = None) -> None:
    """Run evaluation for N episodes, print per-episode and summary stats.

    If model_path is provided, loads a trained PPO model and uses
    deterministic predictions. Otherwise falls back to random actions.
    """
    video_dir = "results/videos"
    if record:
        os.makedirs(video_dir, exist_ok=True)

    # Use a 1-env VecEnv so obs shape matches what PPO expects
    env = make_vec_env(cfg, use_subproc=False, num_envs=1)

    model = None
    if model_path:
        model = PPO.load(model_path, device=cfg.device)
        print(f"[eval] Loaded model from {model_path}")
    else:
        print("[eval] No model provided — using random actions")

    print(f"[eval] Running {episodes} episodes...\n")

    x_positions = []
    flag_gets = []
    rewards = []
    episode_lengths = []

    for ep in range(episodes):
        obs = env.reset()
        total_reward = 0.0
        steps = 0
        done = False
        info = {}

        while not done:
            if model:
                action, _ = model.predict(obs, deterministic=True)
            else:
                action = [env.action_space.sample()]
            obs, reward, dones, infos = env.step(action)
            total_reward += reward[0]
            steps += 1
            done = dones[0]
            info = infos[0]

        x_pos = info.get("x_pos", 0)
        flag = bool(info.get("flag_get", False))

        x_positions.append(x_pos)
        flag_gets.append(flag)
        rewards.append(total_reward)
        episode_lengths.append(steps)

        print(
            f"  Episode {ep + 1:>{len(str(episodes))}}/{episodes}: "
            f"x_pos={x_pos:>5}  "
            f"reward={total_reward:>8.1f}  "
            f"steps={steps:>4}  "
            f"flag={flag}"
        )

    env.close()

    # Summary
    flag_rate = np.mean(flag_gets)
    print(f"\n{'='*55}")
    print(f"  Summary ({episodes} episodes):")
    print(f"    mean_x_pos:         {np.mean(x_positions):.0f}")
    print(f"    max_x_pos:          {np.max(x_positions):.0f}")
    print(f"    mean_reward:        {np.mean(rewards):.1f}")
    print(f"    mean_steps:         {np.mean(episode_lengths):.0f}")
    print(f"    flag_capture_rate:  {flag_rate:.1%} ({int(np.sum(flag_gets))}/{episodes})")
    print(f"{'='*55}")

    if record:
        print(f"\nVideos saved to {video_dir}/")


def main() -> None:
    args = parse_args()
    cfg = Config.from_yaml(args.config)

    if args.seed is not None:
        cfg.seed = args.seed

    set_global_seed(cfg.seed)

    evaluate(
        cfg,
        episodes=args.episodes,
        record=args.record,
        model_path=args.model,
    )


if __name__ == "__main__":
    main()
