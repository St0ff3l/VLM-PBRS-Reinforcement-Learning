# VLM-Guided RL: 论文实验运行终极指南 (Paper Experiment Guide)

## ⚠️ 重要澄清：Python 脚本 vs. Notebook

`main_paper_ready.py` **绝对不是**用来调用 `algorithms/` 目录下那些 `.ipynb` 文件的！
相反，它是那些 Notebook 的**彻底替代品 (Complete Replacement)**。

### 为什么我们要在写论文时抛弃 Notebook？
1. **稳定性**：在 Jupyter Notebook 中连续跑 60 万步（可能几个小时）极易引起浏览器内存泄漏、页面卡死或输出日志断流。
2. **严谨的日志隔离**：每次 Notebook 重新运行变量容易串。新脚本确保每一次运行都生成带独立时间戳的沙盒文件夹。
3. **架构升级**：`main_paper_ready.py` 直接调用了我们在 `envs/visual_wrapper.py` 中新写的**自适应图像触发（Adaptive Inference MSE）PBRS 机制**，以及修复了提示词的 `vlm/llava_client.py`，而旧的 Notebook 里还充斥着人工写的 `if LEFT / RIGHT` 硬逻辑。

> **提示**：你现在完全可以把 `algorithms/*.ipynb` 视为之前“小组作业阶段”的**历史废弃档案 (Legacy Archive)**。现在的论文实验**全部**且**仅**依赖 `main_paper_ready.py`。

---

## 🚀 如何运行实验出图表

### 1. 确保后台正开启大模型服务
由于我们的 VLM Wrapper 会实时向本地 Ollama 发数据，请先新开一个终端窗口挂起大模型：
```bash
ollama run llava:7b
```

### 2. 一键启动对比实验
在项目根目录运行：
```bash
python main_paper_ready.py
```
执行后，您可以去休息。脚本会自动按顺序执行：
- **阶段一 (Baseline PPO)**：不带视觉指导，跑 60 0000 步（模拟传统 RL 的探索挣扎）。
- **阶段二 (Ours VLM-PPO)**：搭载大模型语义势能，跑 15 0000 步（验证大模型如何加速收敛）。

### 3. 去哪里找论文需要的数据？
所有的输出都会被严格隔离在 `logs_and_results/` 下。你会看到类似这样的目录结构：
```text
logs_and_results/
├── baseline_ppo_20260415_183000/    # 你的纯PPO对照组数据
│   ├── monitor.csv                  # 【画训练曲线图】专用的奖励历史记录
│   ├── params.json                  # 【写Settings表格】用的环境超参数
│   ├── thresholds.json              # 【画效率柱状图】首次达到成功的确切步数
│   ├── models/                      # 最终训练好的模型权重 .zip
│   ├── tensorboard/                 # 给 tensorboard 看的日志
│   └── videos/                      # 每 500 个 Episode 自动录制的 MP4 证明视频
└── ours_vlm_ppo_20260415_214500/    # 你的大模型引导组数据（结构同上）
```

### 4. 论文绘图建议
在拿到两个文件夹的 `monitor.csv` 后，你可以写一个简单的 Python matplotlib 脚本，将两个 `.csv` 的 `r` (奖励) 随 `l` (周期步数) 的变化画在同一张折线图上，这将是证明你论文核心 Contribution 的最美图片。
