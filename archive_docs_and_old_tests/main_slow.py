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
    vlm_call_every_n=64,
    vlm_log_every_n_calls=64,
    vlm_query_on_reset=True,
):
    env = gym.make(env_id, render_mode='rgb_array')
    
    if log_dir and not is_eval:
        env = Monitor(env, filename=log_dir)
        
    if is_eval and video_dir:
        # 你的需求：只在最后测评环节才录制一次最终完整的视频
        env = RecordVideo(env, video_folder=video_dir, episode_trigger=lambda x: True)
        
    if use_vlm and not is_eval:
        # VLM只在训练时做Reward Shaping辅助！测评时不加载VLM，这是向导师证明Agent“出师”的最好方式
        prompt = 'Evaluate progression towards flag [0.0, 1.0]. Float only.'
        print(f'Wrapping {env_id} with VLM.')
        env = AdaptiveVisualPBRS_Wrapper(
            env,
            prompt=prompt,
            mse_threshold=vlm_mse_threshold,
            gamma=0.99,
            vlm_call_every_n=vlm_call_every_n,
            log_every_vlm_calls=vlm_log_every_n_calls,
            vlm_query_on_reset=vlm_query_on_reset,
        )
        
    return env

def run_experiment(
    env_id='MountainCar-v0',
    use_vlm=False,
    total_timesteps=60_000,
    run_name_prefix='run',
    vlm_mse_threshold=DEFAULT_VLM_MSE_THRESHOLD,
    vlm_call_every_n=64,
    vlm_log_every_n_calls=64,
    vlm_query_on_reset=True,
    run_evaluation=True,
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
    )
    threshold_cb = MultiThresholdCallback(thresholds=DEFAULT_THRESHOLDS, verbose=1)

    model = PPO('MlpPolicy', env, verbose=1, n_steps=2048, batch_size=64, tensorboard_log=tb_dir)
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

if __name__ == '__main__':
    run_experiment(use_vlm=False, total_timesteps=600_000, run_name_prefix='baseline_ppo')
    run_experiment(
        use_vlm=True,
        total_timesteps=150_000,
        run_name_prefix='ours_vlm_ppo',
        vlm_mse_threshold=DEFAULT_VLM_MSE_THRESHOLD,
        vlm_call_every_n=64,
        vlm_log_every_n_calls=64,
        vlm_query_on_reset=True,
    )
