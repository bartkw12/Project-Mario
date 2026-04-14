"""Smoke Test E: DummyVecEnv + VecFrameStack shape verification.

Verifies that make_vec_env produces the correct observation shape
for SB3's CnnPolicy: (num_envs, 4, 84, 84).

Decision point: if shape is (num_envs, 84, 84, 4) instead,
VecTransposeImage needs to be added to make_vec_env.
"""

from src.config import Config
from src.envs import make_vec_env

cfg = Config.from_yaml("configs/default.yaml")

print(f"Creating DummyVecEnv with {cfg.env.num_envs} envs...")
vec_env = make_vec_env(cfg, use_subproc=False)

print(f"Vec env type: {type(vec_env)}")
print(f"Observation space: {vec_env.observation_space}")
print(f"Action space: {vec_env.action_space}")

# Reset and check shape
obs = vec_env.reset()
print(f"\nAfter reset:")
print(f"  obs.shape: {obs.shape}")
print(f"  obs.dtype: {obs.dtype}")
print(f"  obs min/max: {obs.min()}/{obs.max()}")

# Run 10 steps
for i in range(10):
    actions = [vec_env.action_space.sample() for _ in range(cfg.env.num_envs)]
    obs, rewards, dones, infos = vec_env.step(actions)

print(f"\nAfter 10 steps:")
print(f"  obs.shape: {obs.shape}")
print(f"  obs.dtype: {obs.dtype}")
print(f"  rewards.shape: {rewards.shape}")
print(f"  dones.shape: {dones.shape}")

vec_env.close()

# Verify shape
expected = (cfg.env.num_envs, cfg.env.frame_stack, cfg.env.obs_size, cfg.env.obs_size)
actual = obs.shape

if actual == expected:
    print(f"\nShape is (num_envs, frames, H, W) = {actual}")
    print("VecTransposeImage is NOT needed.")
else:
    print(f"\nWARNING: expected {expected}, got {actual}")
    if len(actual) == 4 and actual[1] == cfg.env.obs_size:
        print("Shape is channels-last — VecTransposeImage IS needed.")
    else:
        print("Unexpected layout — investigate manually.")

assert actual == expected, f"FAIL: expected {expected}, got {actual}"
print(f"\nSmoke Test E: PASSED  (obs shape: {actual})")
