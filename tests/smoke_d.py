"""Smoke Test D: RecordVideo on a single env.

Verifies that make_single_env with record_video=True produces a playable
.mp4 file in results/videos/.

Phase 1 exit criterion #2: random-agent video saved.
"""

import os
from src.config import Config
from src.envs import make_single_env

cfg = Config.from_yaml("configs/default.yaml")

video_dir = "results/videos"
os.makedirs(video_dir, exist_ok=True)

# Create single env with video recording
env = make_single_env(cfg, seed=cfg.seed, record_video=True, video_dir=video_dir)

print(f"Env type: {type(env)}")
print(f"Recording to: {video_dir}")

# Run 1 episode (or max 500 steps as safety cap)
obs, info = env.reset()
total_reward = 0.0
steps = 0

for i in range(500):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    steps += 1
    if terminated or truncated:
        break

print(f"\nEpisode finished:")
print(f"  Steps: {steps}")
print(f"  Total reward: {total_reward:.2f}")
print(f"  Final x_pos: {info.get('x_pos')}")
print(f"  Terminated: {terminated}, Truncated: {truncated}")

env.close()

# Check that video file was created
files = [f for f in os.listdir(video_dir) if f.endswith(".mp4")]
print(f"\nVideo files in {video_dir}/: {files}")
assert len(files) > 0, f"FAIL: no .mp4 files found in {video_dir}/"

print(f"\nSmoke Test D: PASSED  ({len(files)} video(s) recorded)")
