# Config file for Super Mario AI v1 - May 12th, 2025
from stable_baselines3.common.utils import get_schedule_fn

# Hyperparameters
framestack = 4

lr = 2.5e-4               # Increased from 0.000001
lr_schedule = get_schedule_fn(lr, 'linear')

update_steps = 2048       # Reduced from 512
total_timesteps = 1000000
batch_size = 64
n_epochs = 8
gamma = 0.99              # Need long‐term credit for clearing gaps.
gae_lambda = 0.95         # Balance bias/variance in advantage estimation
clip_range = 0.1          # Tighter clipping stabilizes training.
ent_coef = 0.005           # Encourage exploration w/o chaos


