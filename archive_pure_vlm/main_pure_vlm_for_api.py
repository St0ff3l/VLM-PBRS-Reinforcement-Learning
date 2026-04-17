import argparse
import sys
import subprocess
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from envs.visual_wrapper import AdaptiveVisualPBRS_Wrapper
from gymnasium.wrappers import RecordVideo
import os
import json
import datetime

DEFAULT_THRESHOLDS = [-195.0, -190.0, -185.0, -180.0, -175.0, -170.0, -165.0, -160.0]
DEFAULT_VLM_MSE_THRESHOLD = 69.84652538597584
DEFAULT_RUN_MODE = "combo"

class MultiThresholdCallback(BaseCallback):
    """监控并记录达到多个成功阈值所需步数的专属回调"""
    def __init__(self, thresholds=None, verbose=0):
        super().__init__(verbose)
        if thresholds is None:
            thresholds = DEFAULT_THRESHOLDS
        self.thresholds = sorted(thresholds)
        self.reached_steps = {str(t): None for t in self.thresholds}
        
    def _on_step(self) -> bool:
        infos = self.locals.get('infos')
        if infos and 'episode' in infos[0]:
            ep_reward = infos[0]['episode']['r']
            for t in self.thresholds:
                if ep_reward >= t and self.reached_steps[str(t)] is None:
                    self.reached_steps[str(t)] = self.num_timesteps
                    print(f'\n>>> [PROGRESS] Threshold {t} reached at step {self.num_timesteps} <<<\n')
        return True

def create_env(
    env_id='MountainCar-v0',
    use_vlm=True,
    log_dir=None,
    is_eval=False,
    video_dir=None,
    vlm_mse_threshold=DEFAULT_VLM_MSE_THRESHOLD,
    vlm_call_every_n=2048,
    vlm_log_every_n_calls=64,
    vlm_query_on_reset=False,
    vlm_alpha: float = 1.0,
    env_reward_mode: str = 'preserve',
    debug_vlm: bool = False,
):
    env = gym.make(env_id, render_mode='rgb_array')
    
    if log_dir and not is_eval:
        env = Monitor(env, filename=log_dir)
        
    if is_eval and video_dir:
        # 你的需求：只在最后测评环节才录制一次最终完整的视频
        env = RecordVideo(env, video_folder=video_dir, episode_trigger=lambda x: True)
        
    if use_vlm and not is_eval:
        # VLM只在训练时做Reward Shaping辅助！测评时不加载VLM，这是向导师证明Agent“出师”的最好方式
        # New VLM Prompt: Option C - Multiple Choice Question (MCQ) Format!
        # This is the industry standard for robustly parsing 7B VLM outputs.
        prompt = (
            "Analyze the MountainCar environment. Look at the lowest point (the dip) of the black curved track. "
            "Note that this lowest point is spatially on the left side of the image.\n"
            "Question: Where is the car located? Select the best option:\n"
            "A. The car is on the steeper slope behind the lowest dip (Far Left).\n"
            "B. The car is resting at the lowest dip (Bottom).\n"
            "C. The car is on the slope curving up toward the yellow flag (Right).\n"
            "Reply strictly with only one letter: A, B, or C."
        )
        print(f'Wrapping {env_id} with VLM.')
        env = AdaptiveVisualPBRS_Wrapper(
            env,
            prompt=prompt,
            mse_threshold=vlm_mse_threshold,
            gamma=0.99,
            vlm_call_every_n=vlm_call_every_n,
            log_every_vlm_calls=vlm_log_every_n_calls,
            vlm_query_on_reset=vlm_query_on_reset,
            vlm_alpha=vlm_alpha,
            env_reward_mode=env_reward_mode,
            debug=debug_vlm,
        )
        
    return env

def run_experiment(
    env_id='MountainCar-v0',
    use_vlm=False,
    total_timesteps=60_000,
    run_name_prefix='run',
    vlm_mse_threshold=DEFAULT_VLM_MSE_THRESHOLD,
    vlm_call_every_n=2048,
    vlm_log_every_n_calls=64,
    vlm_query_on_reset=False,
    vlm_alpha: float = 1.0,
    env_reward_mode: str = 'preserve',
    run_evaluation=False,
):
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f'{run_name_prefix}_{timestamp}'
    
    run_dir = f'./logs_and_results/{run_name}'
    video_dir = os.path.join(run_dir, 'videos')
    model_dir = os.path.join(run_dir, 'models')
    tb_dir = os.path.join(run_dir, 'tensorboard')
    
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    print(f'\n{"="*60}\n[PHASE 1] Training {run_name}\n{"="*60}')

    # 1. 记录超参数
    params = {
        'env_id': env_id,
        'use_vlm': use_vlm,
        'total_timesteps': total_timesteps,
        'vlm_mse_threshold': vlm_mse_threshold,
        'vlm_call_every_n': vlm_call_every_n,
        'vlm_log_every_n_calls': vlm_log_every_n_calls,
        'vlm_query_on_reset': vlm_query_on_reset,
        'run_evaluation': run_evaluation,
    }
    with open(os.path.join(run_dir, 'params.json'), 'w') as f:
        json.dump(params, f, indent=4)

    # 2. 训练
    env = create_env(
        env_id,
        use_vlm=use_vlm,
        log_dir=run_dir,
        is_eval=False,
        vlm_mse_threshold=vlm_mse_threshold,
        vlm_call_every_n=vlm_call_every_n,
        vlm_log_every_n_calls=vlm_log_every_n_calls,
        vlm_query_on_reset=vlm_query_on_reset,
        vlm_alpha=vlm_alpha,
        env_reward_mode=env_reward_mode,
    )
    threshold_cb = MultiThresholdCallback(thresholds=DEFAULT_THRESHOLDS, verbose=1)

    # 强化探索，防止策略坍缩
    model = PPO('MlpPolicy', env, verbose=1, n_steps=2048, batch_size=64, ent_coef=0.05, tensorboard_log=tb_dir)
    model.learn(total_timesteps=total_timesteps, tb_log_name=run_name, callback=threshold_cb)
    
    model_path = os.path.join(model_dir, 'final_model.zip')
    model.save(model_path)
    env.close()
    
    with open(os.path.join(run_dir, 'thresholds.json'), 'w') as f:
        json.dump(threshold_cb.reached_steps, f, indent=4)
        
    if run_evaluation:
        print(f'\n[PHASE 2] Evaluation & Final Video Recording...')
        # 3. 评测及视频录制 (用最终模型玩游戏，不带VLM)
        eval_env = create_env(env_id, use_vlm=False, log_dir=None, is_eval=True, video_dir=video_dir)
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

def ensure_python_environment(expected_conda_env: str) -> None:
    if not expected_conda_env:
        return

    active_env = os.environ.get("CONDA_DEFAULT_ENV")
    expected_path_token = f"\\envs\\{expected_conda_env}\\"
    interpreter_ok = expected_path_token.lower() in sys.executable.lower()

    if active_env == expected_conda_env or interpreter_ok:
        return

    print("[ENV CHECK] Wrong Python environment detected.")
    print(f"[ENV CHECK] Current interpreter: {sys.executable}")
    print(f"[ENV CHECK] CONDA_DEFAULT_ENV: {active_env}")
    print(f"[ENV CHECK] Expected conda env: {expected_conda_env}")
    print("[ENV CHECK] Please run with your conda env interpreter, for example:")
    print(r"C:\Users\Stoffel\.conda\envs\lab\python.exe main.py")
    raise SystemExit(1)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run experiments (combo/fast/slow).")
    # ========================= IMPORTANT =========================
    # You can switch workflow by changing --mode:
    # 1) combo (default): baseline(video) -> fast VLM(video)
    # 2) fast: fast VLM only (optional video via --run-eval)
    # 3) slow: run original backup script (main_slow.py)
    # ============================================================
    parser.add_argument(
        "--mode",
        choices=["combo", "fast", "slow"],
        default=DEFAULT_RUN_MODE,
        help=(
            "combo: baseline(video) then fast VLM(video, default); "
            "fast: only fast VLM; "
            "slow: run original backup script"
        ),
    )
    parser.add_argument("--env-id", type=str, default="MountainCar-v0")
    parser.add_argument("--baseline-timesteps", type=int, default=600_000)
    parser.add_argument("--baseline-run-name-prefix", type=str, default="baseline_ppo")
    parser.add_argument("--timesteps", type=int, default=150_000)
    parser.add_argument("--run-name-prefix", type=str, default="ours_vlm_fast")
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

if __name__ == '__main__':
    args = parse_args()
    ensure_python_environment(args.expected_conda_env)
    print(f"[MAIN] Selected mode: {args.mode} (default: {DEFAULT_RUN_MODE})")
    if args.mode == 'slow':
        print('[MAIN] Running original slow-mode script (main_slow.py).')
        script_path = os.path.join(os.path.dirname(__file__), 'main_slow.py')
        subprocess.check_call([sys.executable, script_path])
    elif args.mode == 'combo':
        print('[MAIN] Running combo mode: baseline(video) -> fast VLM(video).')
        run_experiment(
            env_id=args.env_id,
            use_vlm=False,
            total_timesteps=args.baseline_timesteps,
            run_name_prefix=args.baseline_run_name_prefix,
            run_evaluation=True,
        )
        run_experiment(
            env_id=args.env_id,
            use_vlm=True,
            total_timesteps=args.timesteps,
            run_name_prefix=args.run_name_prefix,
            vlm_mse_threshold=args.mse_threshold,
            vlm_call_every_n=args.vlm_call_every_n,
            vlm_log_every_n_calls=args.log_every_vlm_calls,
            vlm_query_on_reset=args.vlm_query_on_reset,
            run_evaluation=True,
        )
    else:
        run_experiment(
            env_id=args.env_id,
            use_vlm=True,
            total_timesteps=args.timesteps,
            run_name_prefix=args.run_name_prefix,
            vlm_mse_threshold=args.mse_threshold,
            vlm_call_every_n=args.vlm_call_every_n,
            vlm_log_every_n_calls=args.log_every_vlm_calls,
            vlm_query_on_reset=args.vlm_query_on_reset,
            run_evaluation=args.run_eval,
        )
