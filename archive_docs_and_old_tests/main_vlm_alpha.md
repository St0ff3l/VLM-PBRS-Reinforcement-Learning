# main_vlm_alpha 测试说明

说明：这是一个用于快速验证“放大 VLM 塑形奖励 (vlm_alpha)”效果的独立入口与说明文档。该入口不会替换主线代码，默认行为与主代码兼容，但允许在运行时使用两种奖励融合策略（保留原始环境奖励或抑制每步负惩罚）。

运行示例（在项目根且已激活 `lab` 环境下）：

```powershell
conda activate lab
python main_vlm_alpha.py --timesteps 1000000 --vlm-alpha 10.0 --env-reward-mode suppress_step
```

参数说明：

- `--vlm-alpha`: VLM 塑形奖励的放大系数 alpha（默认 10.0）。
- `--env-reward-mode`: 原始环境奖励如何处理，取值：
  - `preserve`：保留原始环境回报（默认行为），只在终止时额外加 success bonus。
  - `suppress_step`：抑制每步负惩罚，仅在成功时保留终止奖励（实验用，改变原始 MDP，需在论文中明确标注）。

设计理念：

- 使用 PBRS 形式的塑形奖励 F(s,s') = gamma * Phi(s') - Phi(s)，并通过 `alpha` 缩放为 `alpha * F(s,s')`，这在数学上等价于缩放势函数 Φ，因此仍属 PBRS 类方法。
- `suppress_step` 模式为工程/诊断快捷方式，用于快速验证 VLM 信号是否被原始环境步惩罚淹没；如果用于论文，必须作为一个单独的 ablation（并明确标注方法学差别）。

仓库内已使用的“加速 / 取巧”方法清单（说明位置与性质）：

- 双门控节流（Dual gating） — 在 `envs/visual_wrapper.py` 中：
  - 视觉新颖度门控：只有当帧间 MSE 超过 `mse_threshold` 时才考虑调用 VLM；
  - 时间节流门控：只有当 `env_step_count % vlm_call_every_n == 0` 时才允许真实调用 VLM；
  - 目的：显著降低 VLM 调用频率，减少在线推理对训练 FPS 的影响。

- 工程快速配置（Fast profile） — 在运行脚本与配置中：
  - 默认 `vlm_call_every_n=2048`、`vlm_query_on_reset=False`、`run_evaluation=False`（在 `main_vlm_only.py` 与 `main.py` 的 fast/combo 配置中体现）；
  - 目的：用于吞吐/调试评估，而非主实验结论。

- VLM 本地化运行（Ollama / 本地 HTTP） — 在 `vlm/llava_client.py`：
  - 使用 `host='http://localhost:11434'` 与低温度 `temperature=0.0`，以降低网络延迟与输出随机性；
  - 目的：稳定化 VLM 输出并减少远端调用开销。

- 打分式替代离散标签（Scalar scoring） — 在 `vlm/llava_client.py` 与 wrapper：
  - VLM 输出为 0.0~1.0 的连续分数而非离散 LEFT/RIGHT 标签，便于直接作为势函数 Φ 的值。

- 缓存与复用（phi 缓存） — 在 `envs/visual_wrapper.py`：
  - 未调用 VLM 时复用 `self.current_phi` 与 `self.last_semantic_frame`，避免重复推理；

- 日志与统计节流 — 在 `envs/visual_wrapper.py`：
  - 只在累计 `log_every_vlm_calls` 次调用后打印统计信息，减少 I/O 干扰训练速度。

- 跳过训练期视频录制 — 在 `main.py`：
  - 仅在 `run_evaluation=True`（评测阶段）时录制视频，避免录制视频导致的磁盘 I/O 与帧率下降。

- 增强探索（entropy coefficient 增大） — 在 `main.py`：
  - 将 PPO 的 `ent_coef` 提高到 `0.05`，用于防止早期策略坍缩并更快观察到策略变化（这不是“加速”计算，但加速了收敛行为）。

- 删除/移除冗余 Notebook 与文件（减小杂项干扰） — 已移除 `algorithms/*.ipynb` 中不再使用的实验文件。

哪些属于“非标准/需谨慎”的取巧手段（论文写作时必须标注）

- `suppress_step`（抑制 per-step 负惩罚）：改变了原始 MDP 的回报结构，应仅作为消融或工程对照项，并在论文中清楚说明其与原始任务的差别。
- 大幅放大 `vlm_alpha`：数学上等价于缩放势函数，但若在论文中只报告放大过后的效果而不对比 `alpha=1` 的情况，会被质疑为“调参取巧”。务必做 alpha 值敏感性消融。

建议的实验对照（必须列入论文/报告）

1. Baseline：纯 PPO，无 VLM；
2. PBRS（alpha=1，preserve env reward）：标准 VLM-PBRS；
3. PBRS（alpha tuned，preserve env reward）：展示 alpha 的提升效果；
4. 工程对照（suppress_step + alpha）：仅用于展示工程可行性与系统吞吐加速，不作为主结论；

文件位置参考：

- `envs/visual_wrapper.py` — VLM 调用、MSE 门控、phi 缓存、shaping 实现；
- `vlm/llava_client.py` — VLM 本地调用与解析逻辑；
- `main.py` / `main_vlm_only.py` / `main_vlm_alpha.py` — 训练入口与不同 profile；
- `研究课题提案_VLM奖励塑形RL.md` — 已更新的论文提案文本。

如果你同意，我会：

1. 把这个 `main_vlm_alpha.md` 的内容加入到仓库（我已创建文件）；
2. 你可以直接运行上面的命令来测试 `suppress_step + alpha=10` 的效果；
3. 我也可以帮你跑一次小规模本地测试（例如 10000 步）并回传日志摘录，如果你允许我在当前环境执行训练。
