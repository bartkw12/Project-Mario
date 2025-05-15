# Use the trained model from SMv1_model
from stable_baselines3 import PPO
from SMv1_model import env

# Load model from memory
model = PPO.load('./train/best_model', env=env)

# Visualize results

state = env.reset()
while True:
    action, _ = model.predict(state, deterministic=True)
    state, reward, done, info = env.step(action)

    # Access underlying environment for rendering
    env.render("human")

    # Handle episode completion
    if done:
        state = env.reset()



