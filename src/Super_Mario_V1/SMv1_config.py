# Config file for Super Mario AI v1 - May 12th, 2025
# Hyperparameters
framestack = 4
lr = 0.0003               # Increased from 0.000001
update_steps = 2048       # Reduced from 512
total_timesteps = 200000  # Reduced from 1,000,000
batch_size = 128
n_epochs = 10
gamma = 0.85              # Discount factor (prioritize short-term rewards)
gae_lambda = 0.95         # Balance bias/variance in advantage estimation
clip_range = 0.35          # Standard PPO clipping for policy updates
ent_coef = 0.04           # Encourage exploration


