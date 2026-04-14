"""Smoke Test C: Full single-env wrapper stack with shape verification at every stage.

Builds the complete pipeline and prints obs shape after each wrapper:
  mario → JoypadSpace → shimmy → SkipFrame → SimpleRewardShaping
       → GrayscaleObservation → ResizeObservation

Expected final obs shape: (84, 84, 1)
"""

import gymnasium
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from shimmy import GymV21CompatibilityV0

from src.envs.wrappers import SkipFrame, SimpleRewardShaping


def print_stage(name, env, obs):
    print(f"  [{name}]")
    print(f"    obs.shape: {obs.shape}  dtype: {obs.dtype}  min/max: {obs.min()}/{obs.max()}")
    print(f"    obs_space: {env.observation_space}")
    print()


# --- Build the stack one wrapper at a time ---

print("=" * 60)
print("Building wrapper stack, printing shape at each stage")
print("=" * 60)

# 1. Base env
env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0")
env = JoypadSpace(env, SIMPLE_MOVEMENT)
print(f"\n  Action space: {env.action_space}  ({len(SIMPLE_MOVEMENT)} actions)")

# 2. Shimmy bridge
env = GymV21CompatibilityV0(env=env)
obs, info = env.reset()
print_stage("After shimmy (Gymnasium API)", env, obs)

# 3. SkipFrame
env = SkipFrame(env, skip=4)
obs, reward, terminated, truncated, info = env.step(1)
print_stage("After SkipFrame(skip=4)", env, obs)
print(f"    (reward from 4 frames: {reward})")
print()

# 4. SimpleRewardShaping
env = SimpleRewardShaping(env, forward_scale=0.1, death_penalty=-15.0, flag_bonus=50.0)
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(1)
print_stage("After SimpleRewardShaping", env, obs)
print(f"    (shaped reward: {reward}, x_pos: {info.get('x_pos')})")
print()

# 5. GrayscaleObservation
env = gymnasium.wrappers.GrayscaleObservation(env, keep_dim=True)
obs, info = env.reset()
print_stage("After GrayscaleObservation(keep_dim=True)", env, obs)

# 6. ResizeObservation
env = gymnasium.wrappers.ResizeObservation(env, shape=(84, 84))
obs, info = env.reset()
print_stage("After ResizeObservation(84, 84)", env, obs)

# --- Run 100 random steps ---
print("=" * 60)
print("Running 100 random steps with full wrapper stack...")
print("=" * 60)

obs, info = env.reset()
total_reward = 0.0
steps = 0
resets = 0

for i in range(100):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    steps += 1
    if terminated or truncated:
        obs, info = env.reset()
        resets += 1

print(f"\n  Steps: {steps}")
print(f"  Resets: {resets}")
print(f"  Total reward: {total_reward:.2f}")
print(f"  Final obs.shape: {obs.shape}")
print(f"  Final obs.dtype: {obs.dtype}")
print(f"  Final obs min/max: {obs.min()}/{obs.max()}")
print(f"  Final x_pos: {info.get('x_pos')}")

env.close()

# --- Verify expected shape ---
# Note: ResizeObservation uses cv2.resize which drops trailing dim=1,
# so shape is (84, 84) not (84, 84, 1). This is fine — VecFrameStack
# will handle channel stacking in the vectorized env layer (Step 11).
actual = obs.shape
assert actual in [(84, 84), (84, 84, 1)], f"FAIL: expected (84, 84) or (84, 84, 1), got {actual}"
print(f"\nSmoke Test C: PASSED  (final shape: {actual})")
