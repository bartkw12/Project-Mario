import torch
import os
import gym
import gym_super_mario_bros

from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecNormalize, VecTransposeImage, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

from gym import RewardWrapper
from gym.wrappers import GrayScaleObservation, ResizeObservation

from SMv1_config import (
    framestack,
    lr_schedule,
    update_steps,
    total_timesteps,
    batch_size,
    n_epochs,
    gamma,
    gae_lambda,
    clip_range,
    ent_coef,
)

# verify CUDA
print("CUDA available:", torch.cuda.is_available())

# Reward shaping
class CustomReward(RewardWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.last_x_pos = 0
        self.last_y_pos = 79
        self.jump_start_x = None
        self.consecutive_jumps = 0

    def step(self, action):
        state, reward, done, info = self.env.step(action)

        # Death penalty
        if info["life"] < 2:
            reward -= 30  # Extreme penalty

        # Progress rewards
        delta_x = info["x_pos"] - self.last_x_pos
        reward += delta_x * 0.3  # Strong progress incentive

        # Jump system v2
        if action == 2:
            reward += 4.0
            self.jump_start_x = info["x_pos"]
            self.consecutive_jumps += 1
            # Bonus for consecutive jumps
            reward += self.consecutive_jumps * 0.5
        else:
            self.consecutive_jumps = 0

        # Air-time and obstacle clearance bonus
        if info["y_pos"] > 82:  # Significant jump height
            reward += 2.0
            # If jumping over obstacle (x progressed without ground contact)
            if delta_x > 0 and info["y_pos"] > 85:
                reward += 3.0

        # Pipe collision detection
        if self.last_x_pos == info["x_pos"] and info["time"] < self.last_time:
            reward -= 8.0  # Heavy penalty for pipe stalling

        # Momentum maintenance
        if delta_x > 1.5:  # Good running speed
            reward += 1.0

        # Update trackers
        self.last_x_pos = info["x_pos"]
        self.last_time = info["time"]
        self.last_y_pos = info["y_pos"]

        return state, reward, done, info

class SimpleShape(RewardWrapper):
    def step(self, action):
        state, reward, done, info = self.env.step(action)

        # Strongly reward forward movement:
        reward += 0.1 * (info['x_pos'] - getattr(self, 'last_x', info['x_pos']))
        self.last_x = info['x_pos']

        # Penalty on death
        if info['life'] < 2:
            reward -= 10

        # Big bonus at end of level
        if info.get('flag_get', False):
            reward += 100

        return state, reward, done, info

class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip=4):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = False
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

# Create single-env
def create_env():
    # Base env + discrete actions
    env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')  # Train on just Level 1-1
    env = JoypadSpace(env, SIMPLE_MOVEMENT)

    # Preprocessing
    env = SkipFrame(env, skip=4)
    env = SimpleShape(env)  # Apply custom rewards
    env = GrayScaleObservation(env, keep_dim=True)
    env = ResizeObservation(env, shape=84)
    env = Monitor(env, './logs/')

    # Vectorize, stack and Norm
    #env = DummyVecEnv([lambda: env])
    #env = VecTransposeImage(env)
    #env = VecFrameStack(env, framestack, channels_order='first')
    #env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_reward=1.0)

    # return a single, un‐vectorized gym.Env
    return env

# Main Training Script
# --------------------
if __name__ == "__main__":
    os.makedirs("./train/", exist_ok=True)
    os.makedirs("./logs/", exist_ok=True)

    # Parallelize across 8 processes
    num_envs = 8
    vec_env = SubprocVecEnv([create_env for _ in range(num_envs)])

    # Apply vectorized wrappers
    vec_env = VecTransposeImage(vec_env)
    vec_env = VecFrameStack(vec_env, framestack, channels_order='first')
    vec_env = VecNormalize(
        vec_env,
        norm_obs=True,
        norm_reward=True,
        clip_reward=1.0,
    )


    # Setup evaluation callback to save best model
    eval_callback = EvalCallback(
        vec_env,
        best_model_save_path="./train/",
        log_path="./logs/",
        eval_freq=5000,  # Evaluate every 5000 steps
        deterministic=True,
        render=False
    )

    # Initialize model
    model = PPO(
        "CnnPolicy",
        vec_env,
        verbose=1,
        tensorboard_log="./logs/",
        learning_rate=lr_schedule,
        n_steps=update_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        device="cuda"
    )

    # Train the model
    model.learn(
        total_timesteps=total_timesteps,
        callback=eval_callback  # save best model
    )

    # Save the final model
    model.save('final_mario_model_v4')
    vec_env.close()
    #eval_env.close()
