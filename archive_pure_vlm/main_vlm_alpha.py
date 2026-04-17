import argparse
import os
import sys

from main import run_experiment, DEFAULT_VLM_MSE_THRESHOLD, ensure_python_environment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run VLM-PPO test profile with amplified shaping (vlm_alpha)."
    )
    parser.add_argument("--env-id", type=str, default="MountainCar-v0")
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--run-name-prefix", type=str, default="vlm_alpha_test")
    parser.add_argument(
        "--mse-threshold",
        type=float,
        default=DEFAULT_VLM_MSE_THRESHOLD,
        help="MSE novelty threshold for VLM query trigger.",
    )
    parser.add_argument("--vlm-call-every-n", type=int, default=2048)
    parser.add_argument("--log-every-vlm-calls", type=int, default=64)
    parser.add_argument(
        "--vlm-query-on-reset",
        action="store_true",
        help="If set, query VLM once on every env reset (slower but denser shaping).",
    )
    parser.add_argument(
        "--vlm-alpha",
        type=float,
        default=10.0,
        help="Scaling factor for VLM shaping (alpha * (gamma Phi' - Phi)).",
    )
    parser.add_argument(
        "--env-reward-mode",
        type=str,
        choices=["preserve", "suppress_step"],
        default="suppress_step",
        help="How to treat original environment reward: preserve or suppress per-step penalties.",
    )
    parser.add_argument(
        "--expected-conda-env",
        type=str,
        default="lab",
        help="Expected conda env name. Use empty string to disable the check.",
    )
    parser.add_argument(
        "--run-eval",
        action="store_true",
        help="Run final evaluation and save video after training (slower).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_python_environment(args.expected_conda_env)

    # Direct training flow here so Monitor records the wrapper-returned (shaped) rewards.
    import datetime
    import json
    import gymnasium as gym
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv
    from envs.visual_wrapper import AdaptiveVisualPBRS_Wrapper
    from main import MultiThresholdCallback

    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f'{args.run_name_prefix}_{timestamp}'

    run_dir = f'./logs_and_results/{run_name}'
    video_dir = os.path.join(run_dir, 'videos')
    model_dir = os.path.join(run_dir, 'models')
    tb_dir = os.path.join(run_dir, 'tensorboard')

    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(tb_dir, exist_ok=True)

    print(f'\n{"="*60}\n[PHASE 1] Training {run_name}\n{"="*60}')

    # Build env -> wrap with VLM wrapper -> Monitor (so Monitor sees shaped rewards) -> DummyVecEnv
    env = gym.make(args.env_id, render_mode='rgb_array')
    prompt = (
        "Analyze the environment image. "
        "Evaluate how closely the car aligns with this strategy: "
        "'First, fully move left to build momentum on the left slope, "
        "then accelerate right to reach the goal flag.' "
        "Output ONLY a float between 0.0 (failing) and 1.0 (perfectly executing the strategy)."
    )

    env = AdaptiveVisualPBRS_Wrapper(
        env,
        prompt=prompt,
        mse_threshold=args.mse_threshold,
        gamma=0.99,
        vlm_call_every_n=args.vlm_call_every_n,
        log_every_vlm_calls=args.log_every_vlm_calls,
        vlm_query_on_reset=args.vlm_query_on_reset,
        vlm_alpha=args.vlm_alpha,
        env_reward_mode=args.env_reward_mode,
        debug=False,
    )

    monitor_file = os.path.join(run_dir, 'monitor.csv')
    env = Monitor(env, filename=monitor_file)

    vec_env = DummyVecEnv([lambda: env])

    # Train
    threshold_cb = MultiThresholdCallback(verbose=1)
    model = PPO('MlpPolicy', vec_env, verbose=1, n_steps=2048, batch_size=64, ent_coef=0.05, tensorboard_log=tb_dir)
    model.learn(total_timesteps=args.timesteps, tb_log_name=run_name, callback=threshold_cb)

    model_path = os.path.join(model_dir, 'final_model.zip')
    model.save(model_path)
    vec_env.close()

    with open(os.path.join(run_dir, 'thresholds.json'), 'w') as f:
        json.dump(threshold_cb.reached_steps, f, indent=4)

    if args.run_eval:
        print(f'\n[PHASE 2] Evaluation & Final Video Recording...')
        from gymnasium.wrappers import RecordVideo

        eval_env = gym.make(args.env_id, render_mode='rgb_array')
        eval_env = RecordVideo(eval_env, video_folder=video_dir, episode_trigger=lambda x: True)
        eval_model = PPO.load(model_path)
        obs, info = eval_env.reset()
        done = False

        while not done:
            action, _ = eval_model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            done = terminated or truncated

        eval_env.close()
        print(f'>>> Evaluation Video saved to {video_dir} <<<\n')
    else:
        print('\n[PHASE 2] Skipped evaluation/video by configuration.\n')


if __name__ == "__main__":
    main()
