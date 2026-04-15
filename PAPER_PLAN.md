# 课程论文计划（目标：英文会议；初稿截止：2026-04-26）

## 一、概要

目标：以 "VLM-guided PBRS + PPO" 为核心，撰写一篇面向英文会议的论文（6–8 页会议稿风格）。初稿目标日期：2026-04-26（4月26日）。硬件：单台 NVIDIA GeForce RTX 3070（8GB）。

本计划给出：写作与实验任务分配、查文献流程、到 4/26 的逐日执行计划、实验优先级与可行性建议、以及交付物清单。

---

## 二、交付物（到 4/26 要交的最小集合）

- 英文初稿 PDF（包含方法、结果、图表、讨论）
- 关键图表（高分辨率 PNG/PDF）：训练曲线、steps-to-threshold、消融与延迟分析
- results JSON/CSV（每次运行的原始数值，含 seed 信息）
- 用于制作图表的脚本（plot_scripts/）和统计检验脚本
- 简短的 Reproducibility 文档（如何跑实验、依赖、GPU 信息）

---

## 三、总体时间与工作量分配建议（用于规划优先级）

（针对短时冲刺；总时间按 100% 计）

- 实验数据采集与跑实验：50%（包含 debug、重跑、预处理 VLM 标注）
- 写作（论文主体 + 图注）：30%
- 查文献与整理引用：10%
- 数据分析与统计检验（包括画图）：7%
- 最后润色、格式/排版：3%

注：如果你已经有部分实验结果（如 repo 中所示），把实验比例适当下调，写作比例上调。

---

## 四、论文内容占比建议（面向 6–8 页英文会议稿）

- 引言 + 相关工作：20–25%
- 方法（含 PBRS 证明、VLM 映射与衰减策略、伪代码/图示）：30–35%
- 实验（主结果 + 消融 + 延迟/成本分析）：35–45%
- 讨论 + 结论：5–10%

说明：会议稿篇幅有限，实验部分应当“精而准”——挑几个最能说明问题的图表（训练曲线、steps-to-threshold、消融柱状图、延迟折线），其余细节放 appendix / 补充材料。

---

## 五、实验优先级（初稿阶段优先级 P0/P1/P2）

P0（必须）
- 主对比：Baseline PPO vs VLM-PPO（decay）
- 环境：MountainCar-v0（首选）
- seeds：3 个隨機种子（初稿可用 3；最终稿建议 5）
- timesteps：VLM-PPO：150k（或可缩短为 100k 做快速验证），Baseline：200k–600k（若时间受限，改为 200k 并在文中说明为初步结果）
- sample_every_n：128（默认）

P1（强烈建议，但可并行或次阶段完成）
- 消融：NoDecay, NoProgress（每项 2–3 个 seed，较短步数）
- VLM 频率敏感性：sample_every_n ∈ {64,128,256}（只跑 1–2 个额外点用于曲线）
- VLM 精度评估：对一小批帧计算混淆矩阵/准确率

P2（可选/资源允许）
- 泛化到 CartPole-v1 / Acrobot-v1（各 2–3 seed）
- 与其它基线（RND/Curiosity/Hand-crafted shaping）比较

工程提示（为适配单卡 8GB）
- 强烈建议先做离线 VLM 标注（见下），把 VLM 推理从训练环节分离：先采集环境帧/状态并批量调用 VLM 得到 φ(s)，保存 JSON；训练时直接读取 φ 替代实时推理。这样可以把耗时的 VLM 调用移到单独批处理上，既保证可重复性也节省 GPU/时间。

---

## 六、查文献：一步步做法（我不太懂怎么查文献）

1) 明确关键词（英文）：
- "vision-language model" "reinforcement learning"
- "reward shaping" "potential-based reward shaping" PBRS
- "VLM" "LLaVA" "CLIP" "visual reward" "visual reward function" "language-guided RL"

2) 搜索渠道与查询模板：
- Google Scholar：搜索上面的关键词组合；使用时间过滤（近 5 年）
- arXiv：同样关键词，优先查看最新预印本
- Semantic Scholar / ResearchGate / dblp
- 会议检索：NeurIPS, ICML, ICLR, CVPR, ECCV, ACL（跨模态相关）

3) 筛选与阅读顺序：
- 先读经典与基础：PPO（Schulman 2017）、PBRS（Ng 1999）、Curiosity/RND（Pathak 2017, Burda 2019）
- 读 VLM 近作（如 LLaVA、BLIP-2、Qwen-VL）理解能力与限制
- 读 VLM+RL 相关工作（CLIP+RL 系列、VIP、其它视觉语义奖励方法）
- 最后读最近一年（最新方法、实现细节、evaluation protocols）

4) 管理文献（推荐工具）：
- 用 Zotero / Mendeley / Paperpile 收集并导出 BibTeX
- 把每篇论文做一条笔记，记录：引用（BibTeX）、3-4 行摘要、关键方法、优缺点、可复现性（代码地址）、与本课题的关系（一句话）
- 建议把这些笔记存为 CSV 或 markdown（便于 later paste into paper）

5) 实战示例（每天 1 小时高效做法）：
- 第 1 天：用关键词抓 20 篇候选（Google Scholar 快速浏览标题/摘要）
- 第 2 天：把 20 篇缩减到 8–12 篇（阅读全文并做笔记）
- 持续每天 30–60 分钟补充并把重要句子摘录为引用语句

---

## 七、统计检验与图表（初稿需要至少完成）

- 主指标：steps-to-threshold（到达某个 episode reward 的步数）、最终平均回合奖励、训练 wall-time
- 统计方法：若成对实验（同一 seed 下对照）用配对 t-test 或 Wilcoxon signed-rank；若不成对用 Mann-Whitney U
- DNF 处理：把 DNF 标为 cap（例如 Baseline cap=600k），并在图中标注 DNF；若可行，用生存分析（Kaplan–Meier）更严谨
- 置信区间：使用 bootstrap（10000 次）来估计 mean 的 95% CI

---

## 八、到 4/26 的逐日计划（紧凑版，今日 = 4/14）

注：每天结束时把结果/图表/JSON 推到 `logs_and_results/` 下的对应子目录，保证可复制。

2026-04-14（Day 0）
- 确认并固定实验配置文件（学习率、n_steps、sample_every_n、decay schedule、timesteps）
- 在 repo 增加 `experiments/README.md`（记录要跑的实验与命名规则）
- 写一个小脚本 `scripts/precompute_vlm.py`（或至少计划步骤）来实现离线标注
- 开始做快速 smoke-run（1 seed，短步数）验证 pipeline 可跑

2026-04-15（Day 1）
- 收集并保存少量环境帧用于 VLM 离线标注（例如 5k 帧）
- 用 Ollama/LLaVA（或替代方法）对这些帧标注 φ，生成 `vlm_cache.json`
- 运行 Baseline PPO（short run）确认 monitor CSV 生成

2026-04-16（Day 2）
- 使用离线标注训练 VLM-PPO（3 seeds，timesteps=100k 或 150k 视进度）
- 同步开始 Baseline PPO（3 seeds，timesteps=200k，如果太慢则 100k）

2026-04-17（Day 3）
- 等待训练完成／监控训练曲线并修正 bug
- 生成初版训练曲线图（移动平均窗 50）和 steps-to-threshold 表

2026-04-18（Day 4）
- 运行消融实验（NoDecay, NoProgress，1–2 seeds，较短步数）
- 开始统计分析脚本（bootstrap + 配对检验）

2026-04-19（Day 5）
- 整理所有 CSV/JSON，运行统计检验并形成表格
- 生成所有最终图（高分辨率）并写图注（英文）

2026-04-20（Day 6）
- 写 Methods 和 Experiments 部分草稿（英文），包含实验设置表格
- 写图表 embed 到初稿目录，保证论文可直接引用图

2026-04-21（Day 7）
- 写 Introduction + Related Work 草稿（英文），至少引用 8–12 篇关键文献
- 把参考文献导出为 `refs.bib`

2026-04-22（Day 8）
- 合并前面章节，生成论文初稿（LaTeX 或 Word/Markdown 转 PDF），先不要追求完美
- 把初稿发给导师/同学快速审阅（请求 24–48 小时反馈）

2026-04-23（Day 9）
- 根据收到的反馈修正图表与实验说明；若需要，补一两个短实验

2026-04-24（Day 10）
- 完善 Discussion 与 Limitations 部分（包括 VLM 延迟、可靠性、泛化性限制）
- 开始论文语言润色（英文）

2026-04-25（Day 11）
- 最后一遍排版校对，检查图注、表格、参考文献格式
- 生成最终 PDF

2026-04-26（Day 12）
- 输出并提交英文初稿（或把 PDF 发给导师），准备接收导师反馈并进入下一轮完善

---

## 九、投稿与后续里程碑（建议）

- 初稿后（4/26）：根据导师反馈补足实验（目标把 seed 增到 5，补充泛化环境）——1–2 周
- 完成最终实验和润色：再 1 周
- 选择目标会议并按其模板格式化稿件：0.5–1 周

建议投稿目标（视结果成熟度）：
- 方向方向短论文或 workshop：易投稿，时间窗口更灵活
- 若结果充分，目标顶会（ICLR/NeurIPS/ICML/CVPR）或 domain workshop

---

## 十、我可以替你做的具体事情（请选择）

1. 写一个 `scripts/aggregate_results.py`：把 monitor CSV → JSON，总结每个 seed 的 steps/最终 reward 并计算 bootstrap CI
2. 写 `scripts/precompute_vlm.py`：采集帧、批量调用 VLM（或读取本地 Ollama）、保存 `vlm_cache.json`
3. 自动化绘图脚本，生成 publication-ready PNG/PDF
4. 生成 LaTeX 论文骨架并把当前图表自动插入（Overleaf 可直接用）
5. 帮你做一次快速的文献检索并生成 `refs.bib`（8–12 篇关键引用）

在你选定后我会直接在仓库里添加相应脚本或文件并运行必要的操作。

---

如果你现在希望我做第一件事，告诉我：
- 要我先写 aggregation + plotting 脚本，还是
- 要我先做文献检索并生成 refs.bib，还是
- 要我先实现 precompute_vlm 的脚本（以节省训练时间）

你也可以直接告诉我哪一天你想先看到什么结果，我会按该优先级开始实现。 
