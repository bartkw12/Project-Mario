"""Evaluation entry point for Project Mario.

Usage:
    python -m src.evaluate                                    # random agent
    python -m src.evaluate --model results/models/best_model/best_model.zip
    python -m src.evaluate --model best_model.zip --episodes 5 --record
"""

import os

import cv2
import numpy as np
from stable_baselines3 import PPO

from src.config import Config, parse_args, set_global_seed
from src.envs import make_vec_env

# NES native resolution
_NES_H, _NES_W = 240, 256
_SCALE = 3  # 3x upscale → 720×768 (good for pixel art)
_FPS = 15  # effective FPS after frame_skip=4 (60 NES fps / 4)


def _draw_stats(frame: np.ndarray, ep: int, total_eps: int, x_pos: int,
                reward: float, steps: int, flag: bool) -> np.ndarray:
    """Overlay live stats on the bottom of a video frame."""
    h, w = frame.shape[:2]

    # Semi-transparent black bar at the bottom
    bar_h = 36 * _SCALE // 2
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - bar_h), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Stats text
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.45 * _SCALE / 2
    color = (255, 255, 255)
    thickness = max(1, _SCALE // 2)
    y = h - bar_h // 3

    flag_str = "YES" if flag else "no"
    text = f"Ep {ep}/{total_eps}  |  x_pos: {x_pos}  |  reward: {reward:.0f}  |  steps: {steps}  |  flag: {flag_str}"
    cv2.putText(frame, text, (8, y), font, scale, color, thickness, cv2.LINE_AA)

    return frame


def _save_video(frames: list[np.ndarray], path: str) -> None:
    """Write a list of BGR frames to an mp4 file."""
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, _FPS, (w, h))
    for f in frames:
        writer.write(f)
    writer.release()


def evaluate(cfg: Config, episodes: int = 5, record: bool = False, model_path: str | None = None, stochastic: bool = False) -> None:
    """Run evaluation for N episodes, print per-episode and summary stats.

    If model_path is provided, loads a trained PPO model. Uses deterministic
    (argmax) predictions by default, or stochastic (sampled) if stochastic=True.
    Stochastic eval produces varied trajectories across episodes, giving a
    meaningful flag_capture_rate over N episodes.
    When record=True, captures the raw NES render (upscaled 3x) with a
    stats overlay bar at the bottom and saves one mp4 per episode.
    """
    video_dir = "results/videos"
    if record:
        os.makedirs(video_dir, exist_ok=True)

    # Use a 1-env VecEnv so obs shape matches what PPO expects
    # render_mode="rgb_array" enables raw NES frame capture for video
    rm = "rgb_array" if record else None
    env = make_vec_env(cfg, use_subproc=False, num_envs=1, render_mode=rm)

    model = None
    deterministic = not stochastic
    if model_path:
        model = PPO.load(model_path, device=cfg.device)
        print(f"[eval] Loaded model from {model_path}")
    else:
        print("[eval] No model provided — using random actions")

    mode_str = "stochastic (sampled)" if stochastic else "deterministic (argmax)"
    print(f"[eval] Running {episodes} episodes ({mode_str})...\n")

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
        frames = []

        while not done:
            if model:
                action, _ = model.predict(obs, deterministic=deterministic)
            else:
                action = [env.action_space.sample()]
            obs, reward, dones, infos = env.step(action)
            total_reward += reward[0]
            steps += 1
            done = dones[0]
            info = infos[0]

            if record:
                # Get raw NES frame (240×256 RGB) via render
                raw_frame = env.render()
                # Upscale with nearest-neighbor (crisp pixel art)
                big = cv2.resize(raw_frame, (_NES_W * _SCALE, _NES_H * _SCALE),
                                 interpolation=cv2.INTER_NEAREST)
                # RGB → BGR for cv2
                big = cv2.cvtColor(big, cv2.COLOR_RGB2BGR)
                # Overlay stats
                big = _draw_stats(
                    big, ep + 1, episodes,
                    info.get("x_pos", 0), total_reward, steps,
                    bool(info.get("flag_get", False)),
                )
                frames.append(big)

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

        if record and frames:
            vid_path = os.path.join(video_dir, f"episode_{ep + 1:03d}.mp4")
            _save_video(frames, vid_path)
            print(f"           → saved {vid_path}")

    env.close()

    # Summary
    flag_rate = np.mean(flag_gets)
    print(f"\n{'='*55}")
    print(f"  Summary ({episodes} episodes, {mode_str}):")
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
        stochastic=args.stochastic,
    )


if __name__ == "__main__":
    main()
