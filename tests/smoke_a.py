"""Smoke Test A: Bare Mario env + JoypadSpace (old gym 0.21 API).

Verifies that gym-super-mario-bros and nes-py work on this Python version
before adding any compatibility layers.

Expected output:
- obs.shape = (240, 256, 3)
- step returns 4-tuple (obs, reward, done, info)
- info contains: x_pos, life, flag_get, time, etc.
- action_space = Discrete(7) (SIMPLE_MOVEMENT)
"""

import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

# 1. Create base env
env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0")
print(f"Base env created: {env.spec.id}")
print(f"Observation space: {env.observation_space}")
print(f"Action space (raw): {env.action_space}")

# 2. Wrap with JoypadSpace (discrete action mapping)
env = JoypadSpace(env, SIMPLE_MOVEMENT)
print(f"Action space (SIMPLE_MOVEMENT): {env.action_space}")
print(f"Actions: {SIMPLE_MOVEMENT}")

# 3. Reset — old gym 0.21 returns just obs
obs = env.reset()
print(f"\nAfter reset:")
print(f"  obs type: {type(obs)}")
print(f"  obs.shape: {obs.shape}")
print(f"  obs.dtype: {obs.dtype}")
print(f"  obs min/max: {obs.min()} / {obs.max()}")

# 4. Step — old gym 0.21 returns 4-tuple
obs, reward, done, info = env.step(1)  # action 1 = move right
print(f"\nAfter step(1) [move right]:")
print(f"  obs.shape: {obs.shape}")
print(f"  reward: {reward} (type: {type(reward)})")
print(f"  done: {done} (type: {type(done)})")
print(f"  info keys: {sorted(info.keys())}")
print(f"  info['x_pos']: {info.get('x_pos')}")
print(f"  info['life']: {info.get('life')}")
print(f"  info['flag_get']: {info.get('flag_get')}")
print(f"  info['time']: {info.get('time')}")
print(f"  info['world']: {info.get('world')}")
print(f"  info['stage']: {info.get('stage')}")

# 5. Run 10 steps to make sure it doesn't crash
for i in range(10):
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    if done:
        obs = env.reset()

print(f"\n10 random steps completed. Final x_pos: {info.get('x_pos')}")

env.close()
print("\nSmoke Test A: PASSED")
