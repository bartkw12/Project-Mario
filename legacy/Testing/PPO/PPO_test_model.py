# PPO agent to play Cart pole

import gym
import numpy as np

env = gym.make('CartPole-v1')  # default reward threshold is 500 for v1

print("observation space: ", env.observation_space)


