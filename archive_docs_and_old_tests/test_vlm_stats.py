# test_vlm_stats.py
# Samples frames from the environment, computes frame-to-frame MSE distribution,
# measures a few VLM (llava) latencies, and writes a recommendation for mse_threshold.

import time
import json
import numpy as np
import gymnasium as gym
from vlm.llava_client import query_llava_potential_score

# Config
N_FRAMES = 800        # number of frames to sample for MSE distribution
LATENCY_SAMPLES = 3   # how many frames to query to estimate VLM latency
N_STEPS = 2048        # PPO n_steps (used to compute fraction)
T_ALLOW = 120.0       # allowed seconds per PPO update to spend on VLM


def main():
    env = gym.make('MountainCar-v0', render_mode='rgb_array')
    obs, info = env.reset()
    f = env.render()
    frames = [f]

    mses = []
    for i in range(N_FRAMES - 1):
        a = env.action_space.sample()
        obs, r, term, trunc, info = env.step(a)
        frame = env.render()
        if frame is None:
            break
        mse = float(np.mean((frame.astype(np.float32) - frames[-1].astype(np.float32)) ** 2))
        mses.append(mse)
        frames.append(frame)
        if term or trunc:
            obs, info = env.reset()
            frame = env.render()
            frames[-1] = frame

    env.close()

    # measure latencies
    latencies = []
    sample_idxs = [len(frames) // 4, len(frames) // 2, len(frames) - 1]
    for idx in sample_idxs[:LATENCY_SAMPLES]:
        f = frames[idx]
        t0 = time.time()
        try:
            _ = query_llava_potential_score(f)
        except Exception:
            # even if the request fails, measure elapsed time to reflect timeout
            pass
        latencies.append(time.time() - t0)

    vlm_avg_latency = float(np.mean(latencies)) if len(latencies) else 0.0

    percentiles = {p: float(np.percentile(mses, p)) for p in [90,95,97,98,99,99.5,99.9]} if mses else {}

    calls_per_update = max(1, int(T_ALLOW // vlm_avg_latency)) if vlm_avg_latency > 0 else 1
    fraction_allowed = calls_per_update / float(N_STEPS)
    recommended_percentile = max(0.0, min(99.999, 100.0 * (1.0 - fraction_allowed)))
    recommended_threshold = float(np.percentile(mses, recommended_percentile)) if mses else 0.0

    out = {
        "mses_count": len(mses),
        "mses_percentiles": percentiles,
        "vlm_latencies": latencies,
        "vlm_avg_latency": vlm_avg_latency,
        "n_steps": N_STEPS,
        "T_allow_seconds": T_ALLOW,
        "calls_per_update_allowed": calls_per_update,
        "fraction_of_frames_allowed": fraction_allowed,
        "recommended_percentile": recommended_percentile,
        "recommended_mse_threshold": recommended_threshold,
    }

    with open("vlm_measure.json", "w") as fp:
        json.dump(out, fp, indent=2)

    print("Wrote vlm_measure.json")
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
