"""Smoke Test B: Shimmy bridge — old gym env to Gymnasium API.

This is the highest-risk step in Phase 1. Verifies that shimmy can convert
the gym-super-mario-bros env (old gym 0.21 API) to Gymnasium's 5-tuple API.

Expected output:
- reset() returns 2-tuple: (obs, info)
- step() returns 5-tuple: (obs, reward, terminated, truncated, info)
- obs.shape still (240, 256, 3)
- info still contains x_pos, life, flag_get, etc.
"""

import gymnasium
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from shimmy import GymV21CompatibilityV0

# 1. Create old-gym env + JoypadSpace (same as Smoke A)
old_env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0")
old_env = JoypadSpace(old_env, SIMPLE_MOVEMENT)
print(f"Old env action space: {old_env.action_space}")

# 2. Wrap with shimmy compatibility bridge
env = GymV21CompatibilityV0(env=old_env)
print(f"Shimmy-wrapped env type: {type(env)}")
print(f"Action space: {env.action_space}")
print(f"Observation space: {env.observation_space}")

# 3. Reset — Gymnasium API returns (obs, info)
result = env.reset()
print(f"\nreset() return type: {type(result)}")
print(f"reset() return length: {len(result)}")
obs, info = result
print(f"  obs.shape: {obs.shape}")
print(f"  obs.dtype: {obs.dtype}")
print(f"  info type: {type(info)}")
print(f"  info keys: {sorted(info.keys())}")

# 4. Step — Gymnasium API returns 5-tuple
result = env.step(1)  # move right
print(f"\nstep() return length: {len(result)}")
obs, reward, terminated, truncated, info = result
print(f"  obs.shape: {obs.shape}")
print(f"  reward: {reward} (type: {type(reward)})")
print(f"  terminated: {terminated} (type: {type(terminated)})")
print(f"  truncated: {truncated} (type: {type(truncated)})")
print(f"  info keys: {sorted(info.keys())}")
print(f"  info['x_pos']: {info.get('x_pos')}")
print(f"  info['life']: {info.get('life')}")
print(f"  info['flag_get']: {info.get('flag_get')}")
print(f"  info['time']: {info.get('time')}")

# 5. Run 20 steps with proper terminated/truncated handling
for i in range(20):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()

print(f"\n20 random steps completed. Final x_pos: {info.get('x_pos')}")

env.close()
print("\nSmoke Test B: PASSED")
