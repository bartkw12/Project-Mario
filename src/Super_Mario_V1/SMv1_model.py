# 1) Setup Mario Environment
import torch
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from SMv1_config import framestack, lr, update_steps, total_timesteps, batch_size, n_epochs

# verify cuda
print(torch.cuda.is_available())  # should return True

# Initialize environment
env = gym_super_mario_bros.make('SuperMarioBros-v0')  # v0 is the standard Mario version
env = JoypadSpace(env, SIMPLE_MOVEMENT)               # reduces action space to Discrete(7) from (256)

# print(env.observation_space.shape)
# print(env.step(1)[3])

# 2) Preprocess Environment
import os
from gym.wrappers import GrayScaleObservation
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from stable_baselines3.common.monitor import Monitor  # reward tracking

# Set up directories
CHECKPOINT_DIR = './train/'
LOG_DIR = './logs/'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Apply preprocessing
env = GrayScaleObservation(env, keep_dim=True)
env = Monitor(env, './logs/')
env = DummyVecEnv([lambda: env])
env = VecFrameStack(env, framestack, channels_order='last')

# 3) Train the RL Model
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback


class SaveBestModelCallback(BaseCallback):
    def __init__(self, save_path, verbose=1):
        super().__init__(verbose)
        self.save_path = save_path
        self.best_reward = -float('inf')

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        # Check if any episodes are completed
        if len(self.model.ep_info_buffer) > 0:

            # Get the latest episode reward
            latest_reward = max([ep['r'] for ep in self.model.ep_info_buffer])

            # Save model if it's the best so far
            if latest_reward > self.best_reward:
                self.best_reward = latest_reward
                self.model.save(os.path.join(self.save_path, 'best_model'))
                print(f"Saved new best model with reward: {self.best_reward}")

        return True

# Initialize callback (save only best model)
callback = SaveBestModelCallback(save_path=CHECKPOINT_DIR)

# Proximal Policy Optimization Algorithm
model = PPO(
    "CnnPolicy",
    env,
    verbose=1,
    tensorboard_log=LOG_DIR,
    learning_rate=lr,
    n_steps=update_steps,
    batch_size=batch_size,
    n_epochs=n_epochs,
    device="cuda"
)

# Train the model
model.learn(total_timesteps=total_timesteps, callback=callback)

# Save the final model
model.save('final_mario_model')
env.close()
