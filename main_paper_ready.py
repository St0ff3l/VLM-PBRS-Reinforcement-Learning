import gymnasium as gym
from stable_baselines3 import PPO
from envs.visual_wrapper import AdaptiveVisualPBRS_Wrapper
import os

def create_env(env_id="MountainCar-v0", use_vlm=True):
    env = gym.make(env_id, render_mode="rgb_array")
    
    if use_vlm:
        # P2 Fix: With generic float output, we can easily change prompts for other environments!
        # Example for CartPole: "Evaluate the stability and visual uprightness of the pole on a scale of 0.0 to 1.0"
        
        # We define a prompt specifically for this environment without ANY "LEFT/RIGHT" code logic
        if env_id == "MountainCar-v0":
            prompt = "Evaluate the position of the car. 0.0 means the bottom of the valley, and 1.0 means it has climbed the right slope and reached the flag. Output only a float."
        else:
            prompt = "Evaluate the current progression towards success from 0.0 to 1.0. Output only a float."
            
        print(f"Wrapping {env_id} with Adaptive VLM + PBRS.")
        env = AdaptiveVisualPBRS_Wrapper(env, prompt=prompt, mse_threshold=15.0, gamma=0.99)
        
    return env

if __name__ == "__main__":
    print("Starting Rigorous Scientific RL Training (VLM-PPO)...")
    
    # Use "True" to invoke our Novelty-Triggered Online VLM Evaluation
    env = create_env("MountainCar-v0", use_vlm=True)
    
    model = PPO("MlpPolicy", env, verbose=1, n_steps=2048, batch_size=64)
    
    # Start learning
    # With Adaptive MSE Caching, it gracefully avoids calling VLM thousands of times 
    # while still being strictly an "Online-RL" setup (P3 Fix)
    print("Training...")
    model.learn(total_timesteps=100_000)
    print("Done!")
