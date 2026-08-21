---
title: "Zetta ζ: An Efficient Closed-Loop Embodied Harness for Self-Evolving Physical Intelligence"
method_name: "Zetta"
authors: [Xin Ding, Liang Mi, Mingzhe Huang, Zixuan Wang, Chao Zhang, Zixu Hao, Fu Chen, Xiangyu Li, Yikai Zheng, Yaoyu Guo, Weijun Wang, Kun Li, Hao Wu, Yunxin Liu, Ting Cao]
year: 2026
venue: arXiv
tags: [embodied-ai, closed-loop-control, self-evolving, vla, robot-learning, runtime-critics, policy-evolution]
zotero_collection: Robotics
image_source: online
arxiv_html: https://arxiv.org/html/2608.16590
created: 2026-08-21
---

# 论文笔记：Zetta ζ — 自演化具身智能的闭环封套

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Institute for AI Industry Research (AIR), Tsinghua University; Z-Trans AI |
| 日期 | August 2026 |
| 项目主页 | [air-embodied-brain.github.io/zetta](https://air-embodied-brain.github.io/zetta) |
| 对比基线 | Pure-VLA, RPent |
| 链接 | [arXiv](https://arxiv.org/abs/2608.16590) |

---

## 一句话总结

> Zetta 引入可演化封套 (Evolvable Harness) 在不修改冻结基础策略的前提下，通过三时间尺度的[[Closed-Loop Control|闭环]]批评-恢复循环，实现机器人在部署期间的在线自我演化。

---

## 核心贡献

1. **三时间尺度闭环框架**: 动作级 [[Runtime Critics|运行时批评器]]、轨迹级恢复优化、验证门控技能更新三层循环协同，在执行中实时介入而非事后反思
2. **可演化封套架构 (Evolvable Harness)**: 将 Critics、Recovery Playbook、Tools 封装为独立于策略参数的运行时组件 $\mathcal{H}$，策略本身完全冻结 ($\nabla\theta = 0$)
3. **Z-Infra 高吞吐量基础设施**: 解耦环境与模型 worker 池，异步批量推理 + 模型分区，实现 **20.6× 吞吐提升** 和 **11.1× 推理加速**（vs RPent）

---

## 问题背景

### 要解决的问题

[[VLA]] (Vision-Language-Action Model) 等基础策略在部署时面临「语义-物理鸿沟」：策略在高层语义上理解任务，但无法感知物理执行中的细微偏差（如抓取滑落、接触力异常）。现有方案要么等待 episode 结束后再反思改进，要么频繁微调模型权重（代价高昂且易过拟合）。

### 现有方法的局限

- **Post-hoc Reflection** (如 ReAct、Reflexion)：只在 episode 失败后回顾，无法在执行中实时干预
- **参数微调**：修改基础模型权重，在新任务上易过拟合，且需要大量数据和计算
- **手工调试循环**：依赖人类专家逐步排错，无法自动化扩展

### 本文的动机

将「批评」与「恢复」能力从策略参数中解耦，以**运行时封套** (harness) 的形式累积，让系统无需重训基础模型即可通过「Critic + Recovery」的积累实现物理智能扩展。

---

## 方法详解

### 系统架构总览

Zetta 采用**双代理 + 可演化封套**架构：

- **冻结组件**: 动作策略 $\pi$ (Action Policy) + 调度代理 $\mathcal{A}_{orch}$ (Orchestrator Agent)，参数不变 ($\nabla\theta = 0$)
- **可演化封套** $\mathcal{H}$: 由 Critics $\mathcal{C}$、Recovery Playbook $\mathcal{R}$、工具集 $\mathcal{T}$ 组成
- **演化代理** $\mathcal{A}_{evo}$: 离线阶段分析失败轨迹，更新 $\mathcal{H}$

### 核心模块

#### 模块1: 运行时批评器 (Runtime Critics)

[[Runtime Critics|运行时批评器]] 在**动作频率**运行，实现高频状态监控：

$$
P_t = \mathcal{C}(\tau_{0:t}) = \langle e_t, \hat{\sigma}_t \rangle
$$

每个批评器输出**结构化提案** $P_t$，包含：
- $e_t$：可审计的失败证据（视觉、物理信号）
- $\hat{\sigma}_t$：建议的执行模式（继续 / 触发恢复）

**高频状态治理 (High-Frequency State Governance)** 是关键设计原则：语义-物理鸿沟要求在物理状态偏离正常分布时**立即**干预，而非等待 LLM 推理窗口。

#### 模块2: 调度代理 (Orchestrator Agent)

调度代理整合批评器提案，做出最终执行模式决策：

$$
\sigma_t = \mathcal{A}_{orch}(P_t, \mathcal{R}, \mathcal{T}, \mathcal{K})
$$

其中 $\mathcal{K}$ 为任务知识上下文。调度代理在较低频率运行（不是动作频率），负责选择恢复方案或工具调用。

#### 模块3: 演化代理 (Evolutionary Agent)

离线演化循环，从失败轨迹中蒸馏可复用的 Critic-Recovery 对：

$$
\mathcal{H}^{(k+1)} \leftarrow \mathcal{A}_{evo}\!\left(\mathcal{D}_{fail}^{(k)},\, \mathcal{H}^{(k)}\right)
$$

优化目标为最大化任务成功率：

$$
\max_{\mathcal{H}}\; J(\mathcal{H}) = \mathbb{E}_{g,s_0 \sim \mathcal{D}}\!\left[\mathrm{Success}(\tau)\,|\,\pi,\,\mathcal{A}_{orch},\,\mathcal{H}\right]
$$

### 三阶段演化周期

**Phase I — 经验失败剖析 (Empirical Failure Profiling)**

确定性可复现的 rollout 采集，构建多模态证据库：

$$
\tau^{(j)} = \{(s_t, a_t, \mu_t, \varphi_t)\}_{t=0}^{T}
$$

其中 $\mu_t$ 为任务里程碑完成状态，$\varphi_t$ 为物理辅助信号（碰撞、接触力）。

定义有效 rollout 集合：

$$
\mathcal{V} = \{\mathrm{seed}_j \in \mathcal{D}_{dev} \mid \mathrm{Valid}(\mathrm{seed}_j) = \mathrm{True}\}
$$

**Phase II — 层次化因果诊断 (Hierarchical Causal Diagnosis)**

[[Hierarchical Causal Diagnosis|层次化因果诊断]] 采用自顶向下六层遍历定位根因：

```
Evaluation → Critic → State → Planning → Recovery → Parameter
```

首先定位第一个未达成里程碑：

$$
m^* = \min\{m_k \in \mathcal{M} \mid m_k \notin \{\mu_t\}_{t=0}^T\}
$$

再找到最早可观测偏差点 (Earliest Observable Divergence):

$$
t_{EOD} = \min\{t \mid \mathrm{dist}(s_t,\, s_t^{ref}) > \varepsilon,\; s_t^{ref} \in \mathcal{I}_{succ}(\mu_t)\}
$$

其中成功参考索引定义为：

$$
\mathcal{I}_{succ}(\mu) = \{s_t \mid \tau \in \mathcal{V}_{succ},\; \mu_t = \mu\}
$$

**Phase III — 验证门控技能更新 (Validation-Gated Skill Update)**

合并 patch 版本为统一封套：

$$
\mathcal{H}_{merged} = \mathrm{Consolidate}\!\left(\{\mathcal{H}_{patch,j} \mid \mathrm{seed}_j \in K_i\}\right)
$$

历史回归测试（必须 100% 通过）：

$$
\forall\,\mathrm{seed}_j \in K_i,\quad \mathrm{Success}(\mathrm{seed}_j \mid \mathcal{H}_{merged}) = 1
$$

Held-out 泛化验证：

$$
\Delta SR = SR(\mathcal{D}_{held\text{-}out} \mid \mathcal{H}_{merged}) - SR(\mathcal{D}_{held\text{-}out} \mid \pi_{VLA})
$$

VLA 重入合约（安全交接条件）：

$$
\Psi(s_t) = \mathbf{1}(\mathrm{FailureCleared}) \wedge \mathbf{1}(\mathrm{Stability}(s_t) > \gamma)
$$

---

## 关键图表

### Figure 1: 系统总览 — 闭环自演化

![Figure 1: Zetta closes the loop](https://arxiv.org/html/2608.16590/2608.16590v1/teaser_v3.png)

**说明**: Zetta 通过高频 [[Runtime Critics|运行时批评器]] 在执行中触发恢复，并将验证过的失败案例蒸馏为跨 rollout 可复用的 Critic-Recovery 技能对。

### Figure 2: Zetta 演化框架总览

![Figure 2: Zetta evolutionary framework](https://arxiv.org/html/2608.16590/2608.16590v1/agent_framework_v2.png)

**说明**: 系统在线执行（左）与离线演化（右）持续循环。在线阶段，动作策略 $\pi$ 在 [[Evolvable Harness|可演化封套]] $\mathcal{H}$ 的监控下执行任务；离线阶段，演化代理 $\mathcal{A}_{evo}$ 分析失败数据更新 $\mathcal{H}$。

### Figure 3: Z-Infra 三层基础设施架构

![Figure 3: Z-Infra architecture](https://arxiv.org/html/2608.16590/2608.16590v1/infra_arch_v2.png)

**说明**: Control Plane 路由代理请求至专属 Env Workers 和 Rollout Workers。Env Workers 管理环境仿真；Rollout Workers 分区处理 VL 编码（计算密集）和 Action Expert 生成（延迟敏感），实现 20.6× 吞吐提升。

### Figure 4 & 5: LIBERO-Pro 与 RoboCasa 上的 "Aha moment"

**说明**: 成功率在早期阶段停滞（v0 → v1 为症状性修复，如分阶段或局部门控），随后出现突变式跃升 — 如 Wine Bottle in Bowl 任务从 15% 突破至 95%。这种非线性演化是 Critic-Recovery 积累规律性的体现。

### Figure 6: LIBERO-Pro Goal 上的累积物理智能扩展

![Figure 6: Cumulative scaling](https://arxiv.org/html/2608.16590/2608.16590v1/Libero-T-scaling-font-2.5x-cropped.png)

**说明**: 每个数据点代表一个选定的累积封套版本，展示跨 Goal-T 和 Goal-S 扰动下的平均性能随 harness 版本积累的持续提升。

### Figure 7: 跨任务技能迁移

**说明**: 从 Goal-T8 任务发现的三种 Critic-Recovery 能力（抓取保持机制等）零样本迁移至 Goal-T2、Goal-T6、Goal-S3，验证了 harness 的跨任务泛化能力。

### Table 1: Z-Infra 核心 API 原语

| API 原语 | 功能描述 |
|----------|----------|
| `create_sessions(requests)` | 批量创建 session，指定 env_family、env_config、lease_seconds；返回 SessionHandle |
| `renew_sessions(session_ids)` | 延长活跃 session 的租约时长 |
| `close_sessions(session_ids)` | 关闭 session 并释放资源（幂等操作） |
| `reset(session_ids, reset_spec)` | 以 task_id、seed、instruction 开启新 episode；返回 episode_id 和初始观测 |
| `observe(session_ids)` | 读取当前观测（无副作用） |
| `action_step(session_ids, actions)` | 执行动作，返回观测、reward、terminated、truncated |
| `policy_step(session_ids, policy_req)` | 原子化 observe→inference→step；返回步骤结果和执行动作 |
| `policy_infer(session_ids, policy_req)` | 仅推理不步进，返回供代理后处理的动作 |
| `run_episode(session_ids, episode_req)` | 在 worker 中完整执行 episode；返回摘要（步数、reward、stop_reason） |

---

## 实验

### 基准测试平台

| 数据集 | 任务数 | 特点 | 用途 |
|--------|--------|------|------|
| [[LIBERO-PRO]] | 10 (Goal-T) + 10 (Goal-S) | 高精度操作，Goal-T/S 扰动模式 | 主要评测 |
| [[RoboCasa]] | 多任务 | 大厨房场景，多物体交互 | 泛化性评测 |

### 主要结果

| 方法 | LIBERO-Pro 成功率 | RoboCasa 成功率 |
|------|-------------------|-----------------|
| Pure-VLA (v0) | 34.5% | 73.6% |
| **Zetta (最终版)** | **90.8%** | **93.6%** |

**推理效率**: 相比 RPent 基线，推理延迟降低 **11.1×**；Z-Infra 吞吐提升 **20.6×**。

### 涌现能力

- **"Aha Moment"**: 成功率出现突变式跃升（如 Wine Bottle in Bowl: 15% → 95%），非线性缘于根因修复而非渐进调整
- **零样本技能迁移**: 针对特定任务种子发现的 Critic-Recovery 能力自动迁移到同族任务，无需额外适配
- **累积扩展**: 通过 Critic-Recovery 对的积累而非模型重训实现持续性能提升

---

## 关键公式汇总

### [[Runtime Critics|批评器输出]] (Eq. 2)

$$
P_t = \mathcal{C}(\tau_{0:t}) = \langle e_t,\; \hat{\sigma}_t \rangle
$$

**含义**: 批评器在每个时间步将历史轨迹 $\tau_{0:t}$ 映射为结构化提案，包含失败证据 $e_t$ 和建议执行模式 $\hat{\sigma}_t$。

**符号说明**:
- $\tau_{0:t}$: 从起始到当前时刻的完整轨迹历史
- $e_t$: 可审计的失败证据（支持人工复查）
- $\hat{\sigma}_t$: 建议执行模式（继续正常执行 / 触发恢复）

### [[Evolvable Harness|封套定义]] (Eq. 1)

$$
\mathcal{H} = \{\mathcal{C},\; \mathcal{R},\; \mathcal{T}\}
$$

**含义**: 可演化封套是独立于策略参数的运行时组件集合。

**符号说明**:
- $\mathcal{C}$: 运行时批评器集合（高频运行）
- $\mathcal{R}$: 恢复 Playbook（结构化恢复方案库）
- $\mathcal{T}$: 异构工具集（视觉工具、物理信号工具等）

### [[Hierarchical Causal Diagnosis|最早偏差点]] (Eq. 10)

$$
t_{EOD} = \min\!\left\{t \;\middle|\; \mathrm{dist}(s_t,\, s_t^{ref}) > \varepsilon,\;\; s_t^{ref} \in \mathcal{I}_{succ}(\mu_t)\right\}
$$

**含义**: 定位物理状态首次偏离成功参考分布的最早时间步，精确标定根因发生时刻。

**符号说明**:
- $t_{EOD}$: 最早可观测偏差时间步 (Earliest Observable Divergence)
- $\mathrm{dist}(\cdot,\cdot)$: 状态空间距离度量
- $\varepsilon$: 偏差阈值
- $\mathcal{I}_{succ}(\mu_t)$: 在里程碑 $\mu_t$ 处的成功参考状态索引

### [[Closed-Loop Control|演化优化目标]] (Eq. 5)

$$
\max_{\mathcal{H}}\; J(\mathcal{H}) = \mathbb{E}_{g,s_0 \sim \mathcal{D}}\!\left[\mathrm{Success}(\tau)\;\middle|\;\pi,\, \mathcal{A}_{orch},\, \mathcal{H}\right]
$$

**含义**: 在冻结策略 $\pi$ 和调度代理 $\mathcal{A}_{orch}$ 条件下，通过优化封套 $\mathcal{H}$ 最大化期望任务成功率。

### [[Evolvable Harness|VLA 重入合约]] (Eq. 12)

$$
\Psi(s_t) = \mathbf{1}(\mathrm{FailureCleared}) \wedge \mathbf{1}(\mathrm{Stability}(s_t) > \gamma)
$$

**含义**: 从恢复模式向策略正常执行模式切换的安全条件 — 故障必须已清除且系统处于稳定状态。

**符号说明**:
- $\mathbf{1}(\cdot)$: 指示函数
- $\gamma$: 稳定性阈值
- $\Psi(s_t)$: 重入谓词（True 时允许 VLA 接管）

---

## 批判性思考

### 优点

1. **策略解耦思路清晰**: 将「怎么做」(冻结策略) 与「做得好不好」(运行时批评) 完全解耦，避免了微调的脆弱性
2. **因果诊断有工程价值**: 六层自顶向下诊断框架兼顾理论完整性与工程可操作性，"最小干预"原则防止过度修改
3. **基础设施贡献扎实**: Z-Infra 解耦 VL 编码与 Action Expert 的分区推理设计有实际工程价值，20.6× 吞吐提升数据可信

### 局限性

1. **演化代理依赖 LLM 推理**: $\mathcal{A}_{evo}$ 本身依赖大模型做因果诊断，其推理可靠性未被充分分析；当失败模式超出 LLM 知识边界时行为未知
2. **物理信号覆盖有限**: 当前 $\varphi_t$ 主要靠仿真器提供的碰撞/接触力；真实硬件上传感器噪声和时延对 $t_{EOD}$ 估计的影响未讨论
3. **Benchmark 局限**: LIBERO-Pro 和 RoboCasa 均为仿真环境，真实世界验证缺失

### 潜在改进方向

1. 将演化代理的诊断决策也纳入验证框架，防止 LLM 幻觉引入错误 patch
2. 研究真实物理传感器噪声对 Critic 触发率的影响及鲁棒化方案
3. 探索 Critic-Recovery 对在跨形态机器人（不同自由度）之间的迁移

### 可复现性评估

- [ ] 代码开源（未见明确声明）
- [ ] 预训练模型（未提供）
- [x] 实验细节完整（seed/rollout 设置详细）
- [x] 数据集可获取（LIBERO / RoboCasa 均为公开 benchmark）

---

## 关联笔记

### 基于
- [[Closed-Loop Control]]: 核心设计原则，在动作执行频率运行批评器
- [[VLM Critic]]: 早期使用 VLM 做事后批评的工作；Zetta 将其推进到运行时高频

### 对比
- Pure-VLA: 无任何运行时干预的基础 VLA，从 34.5% 提升至 90.8%（LIBERO-Pro）

### 方法相关
- [[Runtime Critics]]: 核心运行时组件
- [[Evolvable Harness]]: 可演化封套整体架构
- [[Hierarchical Causal Diagnosis]]: Phase II 诊断框架

### 数据集相关
- [[LIBERO-PRO]]: 主要评测 benchmark
- [[RoboCasa]]: 泛化性验证 benchmark

---

## 速查卡片

> [!summary] Zetta ζ (2026)
> - **核心**: 通过可演化封套 $\mathcal{H}$ 在不修改冻结策略的前提下实现具身 AI 在线自演化
> - **方法**: 三时间尺度闭环 (Runtime Critics + Orchestrator + Evolutionary Agent) + Z-Infra 高吞吐基础设施
> - **结果**: LIBERO-Pro 34.5%→90.8%，RoboCasa 73.6%→93.6%，推理加速 11.1×
> - **项目**: [air-embodied-brain.github.io/zetta](https://air-embodied-brain.github.io/zetta)

---

*笔记创建时间: 2026-08-21*
