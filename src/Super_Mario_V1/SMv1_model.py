# 1) Setup Mario Environment
import torch
import os
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from SMv1_config import framestack, lr, update_steps, total_timesteps, batch_size, n_epochs, gamma, gae_lambda, clip_range, ent_coef
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from gym.wrappers import GrayScaleObservation
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.monitor import Monitor
from gym import RewardWrapper

# verify CUDA
print(torch.cuda.is_available())  # should return True

# Reward shaping implementation
class CustomReward(RewardWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.last_x_pos = 0  # Initialize tracking variable
        self.last_time = 400 # start time in mario
        self.in_air = False # track the jump state

    def step(self, action):
        state, reward, done, info = self.env.step(action)

        # Penalize death heavily
        if info["life"] < 2:  # Mario died
            reward -= 25

        # Reward forward progress (delta x_pos)
        delta_x = info["x_pos"] - self.last_x_pos
        reward += delta_x * 0.2  # Scale to avoid reward explosion (inc scaling from 0.1 prev)

        # Reward jumping (action 2 in SIMPLE_MOVEMENT)
        if action == 2:
            reward += 3.0 # inc reward from 2
            self.in_air = True
        elif self.in_air:
            reward += 1.5 # reward air time
            if info["y_pos"] > 79: # mario mid-jump (y pos is greater than ground lvl)
                reward += 1.0

        # Penalize standing still (no x_pos change)
        if delta_x == 0:
            reward -= 1.0 # inc penalty from -0.5
        if self.last_time - info["time"] == 0: # time frozen ex. pipe collision
            reward -= 5.0

        # Update trackers
        self.last_x_pos = info["x_pos"]
        self.last_time = info["time"]
        self.in_air = (info["y_pos"] > 79)

        return state, reward, done, info

def create_env():
    env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')  # Train on just Level 1-1
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = CustomReward(env)  # Apply custom rewards
    env = GrayScaleObservation(env, keep_dim=True)
    env = Monitor(env, './logs/')
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, framestack, channels_order='last')
    return env

# Create training and evaluation environments
train_env = create_env()
eval_env = create_env()

# Set up directories
CHECKPOINT_DIR = './train/'
LOG_DIR = './logs/'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Setup evaluation callback to save best model
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=CHECKPOINT_DIR,
    log_path=LOG_DIR,
    eval_freq=5000,  # Evaluate every 5000 steps
    deterministic=True,
    render=False
)

# Initialize PPO model
model = PPO(
    "CnnPolicy",
    train_env,
    verbose=1,
    tensorboard_log=LOG_DIR,
    learning_rate=lr,
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
    callback=eval_callback  # Uses EvalCallback to save best model
)

# Save the final model
model.save('final_mario_model_w_reward_shaping')
train_env.close()
eval_env.close()

