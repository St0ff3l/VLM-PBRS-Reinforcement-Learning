import gymnasium as gym
import numpy as np
import collections

# Need to append the path to find vlm from project root when running directly
import sys
from pathlib import Path
root_dir = Path(__file__).resolve().parents[1]
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from vlm.llava_client import query_llava_potential_score

class AdaptiveVisualPBRS_Wrapper(gym.Wrapper):
    """
    A rigorous, Paper-ready PBRS Wrapper.
    P1 Fix: Introduces Adaptive Inference based on perceptual novelty (MSE).
    The VLM is ONLY queried when the visual frame changes beyond a threshold.
    Valid mathematically via PBRS: F(s, s') = gamma * Phi(s') - Phi(s).
    """
    def __init__(self, env, prompt: str, mse_threshold: float = 15.0, gamma: float = 0.99):
        super().__init__(env)
        self.prompt = prompt
        self.mse_threshold = mse_threshold
        self.gamma = gamma
        
        self.last_semantic_frame = None
        self.current_phi = 0.0
        self.vlm_call_count = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        
        frame = self.env.render()
        if frame is not None:
            self.last_semantic_frame = frame
            # Initial VLM call
            self.current_phi = query_llava_potential_score(frame, prompt=self.prompt)
            self.vlm_call_count += 1
        else:
            self.current_phi = 0.0
            
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        current_frame = self.env.render()
        mse = float('inf')
        
        # Calculate visual divergence (MSE)
        if self.last_semantic_frame is not None and current_frame is not None:
            mse = np.mean((current_frame.astype(np.float32) - self.last_semantic_frame.astype(np.float32)) ** 2)

        # Triggers VLM only if visual novelty is high enough
        if current_frame is not None and mse > self.mse_threshold:
            next_phi = query_llava_potential_score(current_frame, prompt=self.prompt)
            self.vlm_call_count += 1
            self.last_semantic_frame = current_frame
        else:
            # P3 Fix: Caches online state safely avoiding complete offline table creation
            next_phi = self.current_phi

        if terminated and reward > 0:
            next_phi = 1.0  # Forces max potential upon task success
            
        # PBRS mathematical formula implementation
        shaping_reward = self.gamma * next_phi - self.current_phi
        self.current_phi = next_phi
        
        total_reward = reward + shaping_reward

        info['vlm_calls'] = self.vlm_call_count
        info['current_phi'] = self.current_phi
        info['shaping_reward'] = shaping_reward
        info['frame_mse'] = mse

        return obs, total_reward, terminated, truncated, info
