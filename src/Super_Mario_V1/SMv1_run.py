# Use the trained model from SMv1_model

import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from stable_baselines3 import PPO
from SMv1_config import framestack
from gym.wrappers import GrayScaleObservation
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack


# Recreate the environment
def create_env():
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = GrayScaleObservation(env, keep_dim=True)
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, framestack, channels_order='last')
    return env


# Load the best model (no need to pass environment for inference)
model = PPO.load('./train/best_model.zip')

# Run the agent
env = create_env()
state = env.reset()
while True:
    action, _ = model.predict(state, deterministic=True)
    state, reward, done, _ = env.step(action)
    env.render("human")

    if done:
        state = env.reset()


