# Config file for Super Mario AI v1 - May 12th, 2025
# Hyperparameters
framestack = 4
lr = 0.00025               # Increased from 0.000001
update_steps = 4096       # Reduced from 512
total_timesteps = 300000  # Reduced from 1,000,000
batch_size = 256
n_epochs = 12
gamma = 0.8              # Discount factor (prioritize short-term rewards)
gae_lambda = 0.9         # Balance bias/variance in advantage estimation
clip_range = 0.4          # Standard PPO clipping for policy updates
ent_coef = 0.06           # Encourage exploration


