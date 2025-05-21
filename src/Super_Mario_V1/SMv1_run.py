# Use the trained model from SMv1_model

import os
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym.wrappers import GrayScaleObservation, ResizeObservation

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage, VecFrameStack

# Import SkipFrame wrapper from your model file
from SMv1_model import SkipFrame
from SMv1_config import framestack

# Recreate the environment
def make_env():
    """
    Returns a vectorized, frame-stacked env suitable for inference.
    """
    # 1) Base Gym environment
    env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
    # 2) Discrete action space
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    # 3) Frame skipping
    env = SkipFrame(env, skip=4)
    # 4) Grayscale + resize
    env = GrayScaleObservation(env, keep_dim=True)
    env = ResizeObservation(env, shape=84)
    # 5) Vectorize (single process)
    env = DummyVecEnv([lambda: env])
    # 6) Transpose for PyTorch (channels first)
    env = VecTransposeImage(env)
    # 7) Frame stack
    env = VecFrameStack(env, framestack, channels_order='first')
    return env

if __name__ == '__main__':
    # create the inference environment
    env = make_env()

    # load trained model
    model_path = './train/best_model_v4.zip'
    model = PPO.load(model_path, env=env)

    obs = env.reset()
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render('human')
        if done:
            obs = env.reset()
