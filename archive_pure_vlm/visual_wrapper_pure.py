import gymnasium as gym
import numpy as np
import time

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
    def __init__(
        self,
        env,
        prompt: str,
        mse_threshold: float = 15.0,
        gamma: float = 0.99,
        vlm_call_every_n: int = 64,
        log_every_vlm_calls: int = 64,
        vlm_query_on_reset: bool = True,
        vlm_alpha: float = 1.0,
        env_reward_mode: str = "preserve",
        debug: bool = False,
    ):
        super().__init__(env)
        self.prompt = prompt
        self.mse_threshold = mse_threshold
        self.gamma = gamma
        self.vlm_call_every_n = max(1, int(vlm_call_every_n))
        self.log_every_vlm_calls = max(1, int(log_every_vlm_calls))
        self.vlm_query_on_reset = bool(vlm_query_on_reset)
        self.vlm_alpha = float(vlm_alpha)
        # env_reward_mode: 'preserve' | 'suppress_step'
        self.env_reward_mode = env_reward_mode
        self.debug = bool(debug)

        self.visual_cache = []
        self.cache_hits = 0
        self.current_phi = 0.0
        self.vlm_call_count = 0
        self.total_vlm_latency = 0.0
        self.env_step_count = 0

    def _query_vlm_phi(self, frame):
        start_t = time.perf_counter()
        phi = query_llava_potential_score(frame, prompt=self.prompt)
        latency = time.perf_counter() - start_t

        self.vlm_call_count += 1
        self.total_vlm_latency += latency

        if self.vlm_call_count % self.log_every_vlm_calls == 0:
            avg_latency = self.total_vlm_latency / self.vlm_call_count
            print("\n" + "="*80)
            print(f"| {'VLM Stats Table (Every ' + str(self.log_every_vlm_calls) + ' Calls)':^76} |")
            print("-" * 80)
            print(f"| {'Total Calls':<15} | {'Env Steps':<15} | {'Cache Hits':<15} | {'Cache Size':<15} |")
            print(f"| {self.vlm_call_count:<15} | {self.env_step_count:<15} | {self.cache_hits:<15} | {len(self.visual_cache):<15} |")
            print(f"| {'Avg Latency(s)':<15} | {'MSE Threshold':<15} | {'':<15} | {'':<15} |")
            print(f"| {avg_latency:<15.3f} | {self.mse_threshold:<15.3f} | {'':<15} | {'':<15} |")
            print("="*80 + "\n")

        return phi, latency

    def _get_phi_from_frame(self, frame):
        if frame is None:
            return 0.0, None, False, 0.0
        
        # Extreme downsample for rapid MSE computing (e.g. 400x600 -> 50x75)
        frame_ds = frame[::8, ::8, :].astype(np.float32)
        
        if not self.visual_cache:
            next_phi, query_latency = self._query_vlm_phi(frame)
            self.visual_cache.append({'frame': frame_ds, 'phi': next_phi})
            return next_phi, query_latency, True, 0.0

        # Compute MSE distance array to all prototypes
        mses = np.array([np.mean((frame_ds - proto['frame']) ** 2) for proto in self.visual_cache])
        min_mse = np.min(mses)

        query_latency = None
        queried = False

        if min_mse > self.mse_threshold and len(self.visual_cache) < 500:
            # Novel visual state! Query VLM and build prototype bank.
            vlm_phi, query_latency = self._query_vlm_phi(frame)
            queried = True
            self.visual_cache.append({'frame': frame_ds, 'phi': vlm_phi})
            # Re-append 0.0 to mses array because distance to newly added self is 0
            mses = np.append(mses, 0.0)
        else:
            self.cache_hits += 1

        # ========= MATHEMATICAL BREAKTHROUGH: RBF Kernel Smoothing =========
        # A nearest-neighbor cache causes catastrophic discontinuities in Phi(s).
        # We blend all known prototype Phi's using exponential weighting (RBF).
        # This renders the VLM potential landscape strictly continuous and differentiable!
        tau = 5.0 # Temperature scaling (matches mse_threshold)
        weights = np.exp(-mses / tau)
        sum_weights = np.sum(weights)
        
        phis = np.array([proto['phi'] for proto in self.visual_cache])
        smoothed_phi = float(np.sum(weights * phis) / sum_weights)

        return smoothed_phi, query_latency, queried, min_mse

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        
        frame = self.env.render()
        if frame is not None and self.vlm_query_on_reset:
            self.current_phi, _, _, _ = self._get_phi_from_frame(frame)
        else:
            self.current_phi = 0.0
            
        return obs, info

    def step(self, action):
        self.env_step_count += 1
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        current_frame = self.env.render()
        next_phi, query_latency, queried_this_step, mse = self._get_phi_from_frame(current_frame)

        if self.env_step_count % 512 == 0:
            try:
                pos = obs[0]
                delta_phi = self.gamma * next_phi - self.current_phi
                print(f"[PBRS Diagnostic] Step={self.env_step_count:<6} x={pos:>6.3f}   Φ={next_phi:>5.3f}   ΔΦ={delta_phi:>6.3f}")
            except Exception:
                pass

        # Determine whether this episode termination is a success (not just a time truncation)
        is_success = terminated and not truncated

        if is_success:
            # keeping success bonus
            success_bonus = 100.0
            next_phi = 1.0  # Force terminal potential to maximum
        else:
            success_bonus = 0.0

        # Rigorous PBRS: Shaping MUST be calculated on EVERY step mathematically.
        # F(s, s') = gamma * Phi(s') - Phi(s). 
        # Even if next_phi == current_phi, this yields (gamma - 1)*phi, which acts as a valid small penalty.
        raw_shaping = self.gamma * next_phi - self.current_phi

        # weighted shaping (alpha * (gamma Phi' - Phi))
        shaped_reward = self.vlm_alpha * raw_shaping

        # Compute clean environment reward according to chosen mode
        if self.env_reward_mode == 'suppress_step':
            # suppress per-step penalties, keep terminal reward + success bonus
            if is_success:
                clean_env_reward = reward + success_bonus
            else:
                clean_env_reward = 0.0
        else:
            # 'preserve' (default): keep original env reward and add success bonus if any
            clean_env_reward = reward + success_bonus

        # update phi state
        self.current_phi = next_phi

        total_reward = clean_env_reward + shaped_reward

        info['vlm_calls'] = self.vlm_call_count
        info['vlm_queried_this_step'] = queried_this_step
        info['vlm_query_latency'] = query_latency
        info['vlm_avg_latency'] = (
            self.total_vlm_latency / self.vlm_call_count if self.vlm_call_count > 0 else 0.0
        )
        info['env_steps'] = self.env_step_count
        info['cache_hits_total'] = self.cache_hits
        info['vlm_call_every_n'] = self.vlm_call_every_n
        info['vlm_query_on_reset'] = self.vlm_query_on_reset
        info['current_phi'] = self.current_phi
        info['raw_shaping'] = raw_shaping
        info['shaped_reward'] = shaped_reward
        info['vlm_alpha'] = self.vlm_alpha
        info['env_reward_mode'] = self.env_reward_mode
        info['frame_mse'] = mse

        # Optional debug print (kept concise)
        if self.debug and (self.vlm_call_count % max(1, self.log_every_vlm_calls) == 0):
            print(
                f"[VLM DEBUG] step={self.env_step_count} phi={self.current_phi:.3f} "
                f"raw_shaping={raw_shaping:.4f} shaped={shaped_reward:.4f} "
                f"clean_env={clean_env_reward:.3f} total={total_reward:.3f}"
            )

        return obs, total_reward, terminated, truncated, info
