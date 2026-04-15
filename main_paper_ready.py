import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from envs.visual_wrapper import AdaptiveVisualPBRS_Wrapper
from gymnasium.wrappers import RecordVideo
import os
import json
import datetime

class ThresholdCallback(BaseCallback):
    """监控并记录达到成功阈值所需步数的专属回调"""
    def __init__(self, threshold=-199.0, verbose=0):
        super().__init__(verbose)
        self.threshold = threshold
        self.reached_step = None
        
    def _on_step(self) -> bool:
        infos = self.locals.get('infos')
        if infos and 'episode' in infos[0]:
            ep_reward = infos[0]['episode']['r']
            if ep_reward >= self.threshold and self.reached_step is None:
                self.reached_step = self.num_timesteps
                print(f'\n>>> [SUCCESS] Threshold {self.threshold} reached at step {self.reached_step} <<<\n')
        return True

def create_env(env_id='MountainCar-v0', use_vlm=True, log_dir=None, is_eval=False, video_dir=None):
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
        env = AdaptiveVisualPBRS_Wrapper(env, prompt=prompt, mse_threshold=15.0, gamma=0.99)
        
    return env

def run_experiment(env_id='MountainCar-v0', use_vlm=False, total_timesteps=60_000, run_name_prefix='run'):
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
    params = {'env_id': env_id, 'use_vlm': use_vlm, 'total_timesteps': total_timesteps}
    with open(os.path.join(run_dir, 'params.json'), 'w') as f:
        json.dump(params, f, indent=4)

    # 2. 训练
    env = create_env(env_id, use_vlm=use_vlm, log_dir=run_dir, is_eval=False)
    threshold_cb = ThresholdCallback(threshold=-199.0, verbose=1)

    model = PPO('MlpPolicy', env, verbose=1, n_steps=2048, batch_size=64, tensorboard_log=tb_dir)
    model.learn(total_timesteps=total_timesteps, tb_log_name=run_name, callback=threshold_cb)
    
    model_path = os.path.join(model_dir, 'final_model.zip')
    model.save(model_path)
    env.close()
    
    threshold_data = {
        'reached': threshold_cb.reached_step is not None,
        'step_to_threshold': threshold_cb.reached_step if threshold_cb.reached_step is not None else -1
    }
    with open(os.path.join(run_dir, 'thresholds.json'), 'w') as f:
        json.dump(threshold_data, f, indent=4)
        
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

if __name__ == '__main__':
    run_experiment(use_vlm=False, total_timesteps=300_000, run_name_prefix='baseline_ppo')
    run_experiment(use_vlm=True, total_timesteps=150_000, run_name_prefix='ours_vlm_ppo')
