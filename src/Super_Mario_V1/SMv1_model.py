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

# verify CUDA
print(torch.cuda.is_available())  # should return True

def create_env():
    # Initialize base environment
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)

    # Preprocess
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
    device="cuda"
)

# Train the model
model.learn(
    total_timesteps=total_timesteps,
    callback=eval_callback  # Uses EvalCallback to save best model
)

# Save the final model
model.save('final_mario_model')
train_env.close()
eval_env.close()

