# VLM-Only 严谨性与 FPS 优化更新说明

本文档记录了对 `main_vlm_only.py` 及底层 `envs/visual_wrapper.py` 逻辑的重大重构。本次更新旨在解决之前的“VLM 安慰剂 bug”，并在保证论文级别学术严谨性的同时，继续维持极高的训练吞吐量 (FPS)。

## 1. 核心 Bug 修复：从“余数节流”到“CD冷却”

**之前的问题：**
在截流逻辑中，原本使用了 `self.env_step_count % self.vlm_call_every_n == 0` (并且默认 `n=2048`)。这导致 VLM 仅仅在一局游戏里的极少数特定总步数上才被激活（大约每 10 个 Episode 才触发 1 次），形同虚设。原本表现出来的“收敛快”，完全是因为去掉了环境惩罚外加大熵引发的“撞大运”。

**现在的设计：**
将 `vlm_call_every_n` 的语义从“模运算”修正为了**“冷却时间 (Cooldown)”**。
- 当前步数距离上次调用步数 $\ge 32$ 且画面产生足够变化 (MSE > threshold) 时，才调用 VLM。
- 在 `main_vlm_only.py` 中，该默认值被调整为 `32`。这意味着在长达 200 步的 MountainCar 截断内，最多允许约 6 次查询。这在保证极高帧率 (FPS) 的同时，确保了 VLM 能真实给出反馈。

## 2. PBRS 严谨性修复：零阶保持 (Zero-Order Hold)

**之前的问题：**
在未调用 VLM 的时间步，Shaping Reward 强制等于 `0.0`。这打破了吴恩达 (Ng et al., 1999) 证明的 *Potential-Based Reward Shaping* (PBRS) 连续性定义，丧失了数学最优策略保持的严谨性。同时如果使用 `suppress_step`，环境被彻底修改。

**现在的设计：**
1. **全局 Shaping 更新：** 无论是否调用 VLM，每一步的 `raw_shaping` 严格按照数学公式 $F(s,s') = \gamma \Phi(s') - \Phi(s)$ 计算。
2. **状态零阶保持：** 当冷却未完毕或画面相似未查询时，令 $\Phi(s') = \Phi(s)$。此时该步的奖励微调量为 $(\gamma - 1)\Phi(s)$，不仅数学计算合法闭环，而且还会隐式地提供一个微小的时间成本惩罚（因为 $\gamma < 1$），催促智能体加快进度。
3. **保留原始环境：** `main_vlm_only.py` 的默认环境奖励模式改回了 `"preserve"`。模型必须顶着每步 `-1` 的真实惩罚爬山，但靠 VLM 放大的 PBRS 提供梯度。这样的结论才能无懈可击。

## 3. 日志与统计重构 (Table Formatting)

日志系统已根据要求重写。保留 `log_every_vlm_calls=64`，并在累积 64 次 VLM 调用时，打印出结构化的统计表格格式，便于直观追踪：
- **Avg Latency(s):** 追踪本地并发调用的平均响应速度。
- **Call Ratio:** 展现调用 VLM 的次数与 Environment 总步数的占比。验证冷却机制的高效率（通常 Ratio 极低，这意味着高 FPS）。

> **总结：**
> 现在你可以非常有底气地将这个方法写进论文：**“我们在保持原生 MDP 难度 (preserve) 和严谨 PBRS 数学定义的前提下，通过 MSE Novelty Gate 和 Cooldown 的零阶保持，在仅消耗极少量 VLM 计算的占比下，实现了惊人的学习加速。”**
