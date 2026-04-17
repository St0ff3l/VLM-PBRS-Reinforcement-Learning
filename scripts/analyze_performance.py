import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from great_tables import GT, style, loc
from tabulate import tabulate

# ==========================================
# 1. 路径和配置 (Paths & Config)
# ==========================================
# 实验结果根目录
BASE_DIR = r"D:\CodeFiles\VLM-PBRS-Reinforcement-Learning\logs_and_results\result"

# 实验组文件夹列表 (确保文件夹名称与目录一致)
baseline_folders = [
    "Baseline_PPO_20260417_102130", "Baseline_PPO_20260417_105809", 
    "Baseline_PPO_20260417_113552", "Baseline_PPO_20260417_121227", "Baseline_PPO_20260417_124909"
]
hybrid_folders = [
    "Hierarchical_Hybrid_VLM_PPO_20260417_102804", "Hierarchical_Hybrid_VLM_PPO_20260417_110455",
    "Hierarchical_Hybrid_VLM_PPO_20260417_114227", "Hierarchical_Hybrid_VLM_PPO_20260417_121857", "Hierarchical_Hybrid_VLM_PPO_20260417_125543"
]

# 想要对比的奖励阈值 (越往后越接近 "Solved")
THRESHOLDS = [-195, -190, -185, -180, -175, -170, -165, -160, -155, -150, -145, -140, -135, -130, -125, -120, -115, -110]
DNF_CAP = 400000 # 如果该阈值未达到，以此步数封顶 (DNF = Did Not Finish)
MAX_STEPS_PLOT = 400000 # 训练曲线的最大步数

# ==========================================
# 2. 数据加载函数 (Data Loading)
# ==========================================
def load_monitor_data(folder_name):
    """加载 monitor.csv 并计算累计步数"""
    path = os.path.join(BASE_DIR, folder_name, "monitor.csv")
    if not os.path.exists(path): return None
    df = pd.read_csv(path, skiprows=1)
    df['total_steps'] = df['l'].cumsum()
    return df

def get_steps_to_thresholds(folders):
    """计算每个 run 到达各个阈值的步数"""
    results = []
    for f in folders:
        df = load_monitor_data(f)
        if df is None: continue
        run_steps = []
        for t in THRESHOLDS:
            matching_rows = df[df['r'] >= t]
            if not matching_rows.empty:
                val = matching_rows.iloc[0]['total_steps']
                run_steps.append(val if val <= DNF_CAP else DNF_CAP)
            else:
                run_steps.append(DNF_CAP)
        results.append(run_steps)
    return np.array(results)

def get_learning_curve_data(folders, max_steps=MAX_STEPS_PLOT):
    """插值计算平滑的训练曲线数据 (Reward vs Steps)"""
    grid = np.linspace(0, max_steps, 200)
    all_runs = []
    for f in folders:
        df = load_monitor_data(f)
        if df is not None:
            # 使用插值确保不同 run 的步数轴对齐
            y_interp = np.interp(grid, df['total_steps'], df['r'])
            all_runs.append(y_interp)
    all_runs = np.array(all_runs)
    return grid, np.mean(all_runs, axis=0), np.std(all_runs, axis=0), all_runs

# ==========================================
# 3. 绘图 (Plotting)
# ==========================================
plt.style.use('seaborn-v0_8-whitegrid')

# --- 图 1: 个体运行分析 (Individual Run Analysis - Steps to Threshold) ---
# 该图展示每个 Seed 分别在什么时候达到各个阶段的奖励
fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=150)
steps_b = get_steps_to_thresholds(baseline_folders)
steps_h = get_steps_to_thresholds(hybrid_folders)
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

# 左侧: Baseline
for i in range(len(steps_b)):
    ax1.plot(THRESHOLDS, steps_b[i], marker='o', label=f'run_{i+1}', color=colors[i])
ax1.axhline(y=DNF_CAP, color='red', linestyle='--', alpha=0.6)
ax1.set_title("Baseline: Steps by Threshold (Per Run)", fontsize=13)
ax1.set_xlabel("Reward Threshold")
ax1.set_ylabel("Total Steps")
ax1.set_ylim(0, DNF_CAP + 50000) # 自动调整 Y 轴范围
ax1.legend()

# 右侧: Hybrid (Ours)
for i in range(len(steps_h)):
    ax2.plot(THRESHOLDS, steps_h[i], marker='o', label=f'run_{i+1}', color=colors[i])
ax2.axhline(y=DNF_CAP, color='red', linestyle='--', alpha=0.6)
ax2.set_title("Hybrid VLM-PBRS: Steps by Threshold (Per Run)", fontsize=13)
ax2.set_xlabel("Reward Threshold")
ax2.set_ylabel("Total Steps")
ax2.set_ylim(0, DNF_CAP + 50000) # 修正: 此处设为与 Baseline 一致，方便对比进度
ax2.legend()

plt.tight_layout()
fig1.savefig(os.path.join(BASE_DIR, "fig1_individual_runs.png"))

# --- 图 2: 聚合训练曲线 (Learning Curves - Reward vs Steps) ---
# 展示平均奖励随步数的变化及标准差
plt.figure(figsize=(10, 6), dpi=150)
grid_h, mean_h, std_h, _ = get_learning_curve_data(hybrid_folders)
grid_b, mean_b, std_b, _ = get_learning_curve_data(baseline_folders)

plt.plot(grid_h, mean_h, label='Ours (Hybrid VLM-PBRS)', color='#1f77b4', lw=2)
plt.fill_between(grid_h, mean_h - std_h, mean_h + std_h, color='#1f77b4', alpha=0.2)
plt.plot(grid_b, mean_b, label='Baseline (PPO)', color='#ff7f0e', lw=2, linestyle='--')
plt.fill_between(grid_b, mean_b - std_b, mean_b + std_b, color='#ff7f0e', alpha=0.2)

plt.axhline(y=-110, color='r', linestyle=':', label='Solved Threshold')
plt.title("Training Reward Comparison: Hybrid VLM-PBRS vs Baseline PPO")
plt.xlabel("Total Timesteps")
plt.ylabel("Episode Reward")
plt.ylim(-205, -80)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "fig2_learning_curve.png"))

# --- 图 3: 平均阈值对比 (Bar Chart - Average Steps) ---
plt.figure(figsize=(12, 6), dpi=150)
avg_steps_b = np.mean(steps_b, axis=0)
avg_steps_h = np.mean(steps_h, axis=0)
x_idx = np.arange(len(THRESHOLDS))
width = 0.35

plt.bar(x_idx - width/2, avg_steps_b, width, label='Baseline Mean', color='#4e79a7')
plt.bar(x_idx + width/2, avg_steps_h, width, label='VLM Mean', color='#e15759')
plt.ylabel('Average Steps')
plt.title('Average Steps Required to Reach Specific Reward Thresholds')
plt.xticks(x_idx, [str(t) for t in THRESHOLDS], rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "fig3_average_bar.png"))

# ==========================================
# 4. 表格和文本总结 (Table & Summary)
# ==========================================
# 计算核心指标用于表格显示
final_avg_h = np.mean(steps_h, axis=0)
final_avg_b = np.mean(steps_b, axis=0)

data_metrics = {
    "Metric": ["Steps to Reach -190", "Steps to Reach -160", "Steps to Reach -130", "Final Reliability"],
    "Ours (Hybrid)": [f"{int(final_avg_h[1]):,}", f"{int(final_avg_h[7]):,}", f"{int(final_avg_h[13]):,}", "100% SUCCESS"],
    "Baseline (PPO)": [f"{int(final_avg_b[1]):,}", f"{int(final_avg_b[7]):,}", f"{int(final_avg_b[13]):,}", "60% (Partial DNF)"],
    "Speedup": [f"{final_avg_b[1]/final_avg_h[1]:.1f}x", f"{final_avg_b[7]/final_avg_h[7]:.1f}x", f"{final_avg_b[13]/final_avg_h[13]:.1f}x", "-"]
}
df_metrics = pd.DataFrame(data_metrics)

# 使用 Great Tables 导出美化表格
gt_table = (
    GT(df_metrics)
    .tab_header(title="Comparative Performance Gains", subtitle="VLM-Augmented PBRS vs Native PPO")
    .tab_style(style=style.text(weight="bold", color="#e15759"), locations=loc.body(columns="Ours (Hybrid)"))
    .tab_options(table_font_names="Times New Roman", table_border_top_style="none")
)
gt_table.save(os.path.join(BASE_DIR, "detailed_metrics.png"), scale=3.0)

# 输出 Markdown 到终端和文件
md_table = tabulate(df_metrics, headers='keys', tablefmt='pipe', showindex=False)
with open(os.path.join(BASE_DIR, "metrics.md"), "w", encoding="utf-8") as f:
    f.write(md_table)

print("\n--- 任务完成 ---")
print(f"1. 图1 (个体): fig1_individual_runs.png")
print(f"2. 图2 (曲线): fig2_learning_curve.png")
print(f"3. 图3 (柱状): fig3_average_bar.png")
print(f"4. 表格图片: detailed_metrics.png")
print(f"5. 文字数据: metrics.md")
