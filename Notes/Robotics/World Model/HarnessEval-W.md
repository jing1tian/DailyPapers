---
title: "HarnessEval-W: Agentifying the Evaluation of Visual Worlds"
method_name: "HarnessEval-W"
authors: [Weiliang Chen, Haowen Sun, Jun Gao, Jiawei Chi, Hanyang Wang, Qiyu Dai, Yihao Li, Hao Li]
year: 2026
venue: arXiv
tags: [benchmark, world-model, agentic-evaluation, video-generation, evaluation-framework]
zotero_collection: N/A
image_source: online
arxiv_html: https://arxiv.org/html/2608.16859
created: 2026-08-23
---

# 论文笔记：HarnessEval-W: Agentifying the Evaluation of Visual Worlds

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | MirroS Lab |
| 日期 | August 2026 |
| 项目主页 | [mirros-lab.github.io/HarnessEval-W](https://mirros-lab.github.io/HarnessEval-W) |
| 对比基线 | [[WBench]] |
| 链接 | [arXiv](https://arxiv.org/abs/2608.16859) / [Code](https://github.com/mirros-lab/harnesseval-w) |

---

## 一句话总结

> HarnessEval-W 将 [[交互式世界模型]] 的评估"智能体化"，用分层 Agent 替代固定 rubric，对 18 个世界模型进行 330 个案例的三维八轴评估，与人类偏好 Spearman 相关系数达 0.93。

---

## 核心贡献

1. **Agentic 评估框架**: 提出 [[Agentic评估|Agentic Evaluation]] 框架，用上下文感知的分层 Agent 替代固定评分指标，自动将问题分解为可测量的子问题并汇聚结果
2. **三维八轴评估体系**: 围绕 Observation Quality、Transition Correctness、World Persistence 三大轴线构建 8 个具体评估维度，覆盖 [[交互式世界模型]] 的全部核心能力
3. **自动化案例构建流水线**: 通过场景分类采样 + Agent 创作 + 验证，自动构建 330 个多样化评估案例，无需人工标注

---

## 问题背景

### 要解决的问题

[[交互式世界模型]] 的评估长期依赖固定的数值指标（如 FVD、SSIM），这些指标无法捕捉模型在"观测质量""状态转移正确性""世界持久性"等核心能力上的细粒度差异，也无法提供可追溯的评估理由。

### 现有方法的局限

- [[WBench]] 等现有 benchmark 采用静态 rubric，评估方差大（HarnessEval-W 方差比 WBench 窄 4.9 倍）
- 固定指标无法区分不同交互接口（Prompt I2V vs. 原生动作 vs. 相机姿态）的能力差异
- 评估结果缺乏可解释性，无法追溯到具体失败的子问题

### 本文的动机

用 [[Agentic评估|Agentic Evaluation]] 的思路解决评估问题：让 Agent 像人类评测者一样理解评估上下文，按需分解子问题，调用专用工具获取视觉证据，最终生成可追溯的 [[证据树|Evidence Tree]]。

---

## 方法详解

### 系统架构

HarnessEval-W 采用 **分层 Agent** 架构评估 [[交互式世界模型]]：

- **输入**: 评估案例（包含生成的视频 rollout + 问题描述）
- **第一层 — [[技能路由|Skill Routing]]**: 根据案例上下文将其路由到适用的评估技能
- **第二层 — 技能分解**: 每个技能将评估问题分解为若干可测量的子问题，分配给专用子 Agent
- **第三层 — 证据聚合**: 父 Agent 验证子 Agent 的结果，合并为结构化评分并生成 [[证据树|Evidence Tree]]
- **输出**: 每个维度的分数 + 完整推理链（记录每个子问题的测试内容与视觉证据来源）

### 核心模块

#### 模块 1：[[技能路由|Skill Routing]] 系统

**设计动机**: 不同评估案例激活不同的技能子集（例如，相机运动案例不激活物理转变技能）

**具体实现**:
- Agent 读取案例描述后，判断哪些技能适用于当前案例
- 未激活的技能在推理链中仍记录"跳过原因"，保持透明度

#### 模块 2：Intentional Change Verifier 技能（示例）

**设计动机**: 验证有意图的实体状态变化（颜色、位置、形状等）需要细粒度的语义理解

**具体实现**:
- 分解为 8 个可测量子问题（如"目标实体是否发生了变化？""变化方向是否正确？"）
- 每个子问题由一个专用子 Agent 负责，配备对应的视觉 grounding 工具

#### 模块 3：自动化案例构建流水线

**设计动机**: 手工构建评估案例费时且难以保证多样性

**具体实现**:
- 场景分类采样（环境 / 前景 / 中景 / 密度 / 外观 / 视角）
- Probe family 分配（确定交互类型）
- Agentic 创作（图像生成 + 动作规划 + 案例验证）

---

## 关键公式

### 公式 1：[[交互式世界模型|Interactive World Model]] 因式分解

$$
P(o_1, \ldots, o_t \mid o_{-t}, \ldots, o_0; a_0, \ldots, a_{t-1}) \;\propto\; P(s_0 \mid o_{-t}, \ldots, o_0) \prod_{i=1}^{t} S(o_i \mid s_i)\, T(s_i \mid s_{i-1}, a_{i-1})
$$

**含义**: 将未来观测的条件概率通过边际化隐藏状态进行因式分解，将能力分解为三个独立的可测量组件。

**符号说明**:
- $o_{-t}, \ldots, o_0$: 历史观测序列（上下文帧）
- $a_0, \ldots, a_{t-1}$: 动作序列
- $s_i$: 第 $i$ 步隐藏状态
- $P(s_0 \mid \cdot)$: 初始状态估计分布（World Persistence 能力的来源）
- $S(o_i \mid s_i)$: 观测似然函数（Observation Quality 能力的来源）
- $T(s_i \mid s_{i-1}, a_{i-1})$: 状态转移函数（Transition Correctness 能力的来源）

---

## 关键图表

### Figure 1：系统概览（Teaser）

![Figure 1 - HarnessEval-W Teaser](https://arxiv.org/html/2608.16859v1/fig_teaser.png)

**说明**: HarnessEval-W 的整体评估流程。给定一个评估案例，系统将其路由到适用的技能，分解为子问题，聚合验证后的证据得出最终分数，分数可追溯回具体失败的子问题。

### Figure 2：评估流水线

![Figure 2 - Evaluation Pipeline](https://arxiv.org/html/2608.16859v1/fig_evaluation_pipeline.png)

**说明**: HarnessEval-W 评估流水线全貌，展示案例如何路由到适用技能，激活技能和跳过技能分别产生基于证据的推理，最终汇聚为各维度分数。

### Figure 3：子 Agent 层次结构

![Figure 3 - Sub-agent Hierarchy](https://arxiv.org/html/2608.16859v1/fig_subagents.png)

**说明**: 以 Intentional Change Verifier 技能为例，展示高层技能如何分解为 8 个可测量子问题，每个子问题由专用子 Agent 处理。

### Figure 4：数据构建流水线

![Figure 4 - Data Construction Pipeline](https://arxiv.org/html/2608.16859v1/fig_data_construction_pipeline.png)

**说明**: HarnessEval-W 330 个评估案例的自动化构建流程，从场景分类采样到 Agentic 创作（世界生成、动作规划、案例验证）。

### Figure 5：案例统计分布

![Figure 5 - Case Statistics](https://arxiv.org/html/2608.16859v1/case_figure.png)

**说明**: (a) 案例分类分布，(b) 场景描述关键词频率，(c) Probe family 分布，展示 330 个案例的多样性。

### Figure 6：与人类偏好的对齐

![Figure 6 - Human Alignment](https://arxiv.org/html/2608.16859v1/human-align.png)

**说明**: (a) 模型级人类对齐曲线，(b) 与 [[WBench]] 的受控对比（pairwise accuracy、draw rate、Brier score）。HarnessEval-W 在有意图转变维度上 [[Spearman相关系数]] ρ=0.93。

### Figure 7：完整推理轨迹示例

![Figure 7 - Reasoning Example](https://arxiv.org/html/2608.16859v1/fig_reasoning_example.png)

**说明**: HarnessEval-W 生成的完整 [[证据树|Evidence Tree]]，记录案例规格、生成的 rollout、评估路由和聚合结果，覆盖两个不同评估场景。

### Figure 8：评估鲁棒性

![Figure 8 - Robustness](https://arxiv.org/html/2608.16859v1/robustness.png)

**说明**: 三次运行的拟合曲线对比，HarnessEval-W 的置信包络比 [[WBench]] 窄 **4.9 倍**，体现更高的评估稳定性。

### Figure 9：各维度相关性矩阵

![Figure 9 - Correlation Matrix](https://arxiv.org/html/2608.16859v1/capability_correlation_8x8_pinkpurple_arial.png)

**说明**: 18 个被评估模型在 8 个维度上的 [[Pearson Correlation]] 矩阵，揭示有意图转变与物理转变高度相关（r=0.98），探索性转变与语义理解几乎不相关（r≈−0.15）。

### Figure 10：Fine-tuning 前后能力迁移（图片 URL 未获取）

**说明**: 对比两对模型（Wan 2.2→DreamX-World，HunyuanVideo 1.5→HY-WorldPlay 1.5）在微调前后各轴能力的差异。核心发现：将 [[视频生成 (Video Generation)|视频生成]] 模型微调为世界模型后，Revisit Consistency 提升，但有意图/物理转变性能下降。

---

### Table 1：HarnessEval-W 评估维度

| 评估轴 | 维度 | 核心问题 |
|--------|------|----------|
| **Observation Quality** | Render Quality (Obs-R) | 渲染帧是否视觉可靠、可读？ |
| **Observation Quality** | Physical Observation (Obs-P) | 物理现象是否被正确观测？ |
| **Transition Correctness** | Exploratory (Trans-E) | 视点变化后场景是否正确更新？ |
| **Transition Correctness** | Intentional (Trans-I) | 有意图的实体修改是否被正确执行？ |
| **Transition Correctness** | Physical (Trans-P) | 物理动力学响应是否正确？ |
| **World Persistence** | Drift Resistance (Pers-D) | 长时间内场景是否保持一致性？ |
| **World Persistence** | Revisit Consistency (Pers-R) | 重访同一位置时场景是否一致？ |
| **World Persistence** | Offscreen Evolution (Pers-O) | 离屏过程在重新观测时是否合理延续？ |

### Table 2：HarnessEval-W 主排行榜（18 个模型）

| 模型 | 接口类型 | Obs-R | Obs-P | Trans-E | Trans-I | Trans-P | Pers-D | Pers-R | Pers-O | Overall |
|------|----------|-------|-------|---------|---------|---------|--------|--------|--------|---------|
| **Seedance 2.0*** | Prompt I2V | 83.6 | 61.8 | 80.2 | 81.8 | 63.5 | 79.8 | 76.8 | 68.9 | **75.5** |
| Wan 2.7* | Prompt I2V | 80.9 | 58.8 | 78.7 | 83.6 | 71.1 | 74.3 | 68.5 | 65.9 | 75.0 |
| Kling 3.0* | Prompt I2V | 82.1 | 60.6 | 79.1 | 82.6 | 63.2 | 77.2 | 75.4 | 66.2 | 74.4 |
| MiniMax H3 | Prompt I2V | 81.5 | 61.3 | 77.7 | 81.7 | 66.9 | 77.0 | 72.3 | 67.2 | 74.3 |
| Grok Imagine 1.5* | Prompt I2V | 85.1 | 60.6 | 76.4 | 80.2 | 66.7 | 79.1 | 70.9 | 64.5 | 73.4 |
| FLUX 3* | Prompt I2V | 81.7 | 61.6 | 76.4 | 79.0 | 63.2 | 77.1 | 70.2 | 67.4 | 72.2 |
| Cosmos3-Super | Prompt I2V | 83.5 | 61.0 | 77.9 | 75.1 | 60.2 | 77.6 | 70.6 | 66.8 | 71.9 |
| HunyuanVideo 1.5 | Prompt I2V | 80.0 | 59.0 | 77.6 | 73.2 | 57.2 | 76.5 | 69.5 | 63.2 | 70.3 |
| Wan 2.2 | Prompt I2V | 80.4 | 58.6 | 77.2 | 62.0 | 55.1 | 76.0 | 66.5 | 63.4 | 67.7 |
| LTX-2.3 | Prompt I2V | 78.4 | 54.5 | 73.8 | 60.4 | 53.3 | 72.4 | 63.3 | 57.7 | 64.6 |
| SANA-WM | Native action | 81.0 | 62.3 | 82.5 | 50.6 | 47.6 | 78.9 | 78.8 | 72.3 | 68.7 |
| ABot-World | Native action | 77.7 | 60.8 | 83.5 | 49.0 | 45.7 | 76.3 | 72.0 | 60.8 | 66.1 |
| DreamX-World | Native action | 78.5 | 60.6 | 81.8 | 50.0 | 47.4 | 77.5 | 73.1 | 65.4 | 66.8 |
| LingBot World v2 | Camera pose | 80.5 | 63.3 | 81.5 | 56.8 | 49.8 | 79.5 | 75.8 | 66.1 | 68.8 |
| Lyra 2 | Camera pose | 79.5 | 64.0 | 80.7 | 48.9 | 45.1 | 77.6 | 79.7 | 56.4 | 65.5 |
| Fantasy-World | Camera pose | 74.1 | 62.7 | 73.2 | 52.4 | 48.9 | 69.9 | 68.4 | 53.8 | 62.1 |
| HY-WorldPlay 1.5 | Camera pose | 79.4 | 64.6 | 82.0 | 49.9 | 45.5 | 79.3 | 81.9 | 61.1 | 67.1 |
| InSpatio-World | Camera pose | 79.0 | 63.0 | 70.7 | 48.6 | 45.6 | 74.5 | 77.0 | 54.0 | 61.4 |

*闭源模型。Obs-R = Render Quality；Obs-P = Physical Observation；Trans-E/I/P = Exploratory/Intentional/Physical Transition；Pers-D/R/O = Drift Resistance/Revisit Consistency/Offscreen Evolution。

**关键发现**:
- Prompt I2V 模型（Seedance 2.0 等）在 Trans-I 和 Trans-P 上远超原生动作模型（~80 vs ~50）
- 原生动作模型在 Trans-E（探索性转变）上表现最好（~82-84）
- Camera pose 模型在 Pers-R（重访一致性）上有优势（Lyra 2 达 79.7，HY-WorldPlay 1.5 达 81.9）

---

## 实验

### 评估数据集

| 数据集 | 规模 | 特点 |
|--------|------|------|
| HarnessEval-W cases | 330 个案例 | 自动构建，覆盖 6 维场景分类 × 多种 Probe family |

### 被评估模型

18 个代表性世界模型，涵盖三种控制接口：
- **Prompt I2V** (10 个): Seedance 2.0, Wan 2.7, Kling 3.0, MiniMax H3, Grok Imagine 1.5, FLUX 3, Cosmos3-Super, HunyuanVideo 1.5, Wan 2.2, LTX-2.3
- **Native action** (3 个): SANA-WM, ABot-World, DreamX-World
- **Camera pose** (5 个): LingBot World v2, Lyra 2, Fantasy-World, HY-WorldPlay 1.5, InSpatio-World

### 主要实验结果

- **人类对齐**: Trans-I 维度 [[Spearman相关系数]] ρ=0.93，Trans-P 维度 ρ=0.87
- **鲁棒性**: 三次运行置信包络比 [[WBench]] 窄 4.9 倍
- **维度相关性**: Trans-I 与 Trans-P 强相关（r=0.98）；Trans-E 与语义理解几乎不相关（r≈−0.15）
- **微调效应**: 将 [[视频生成 (Video Generation)|视频生成]] 模型微调为世界模型后 Pers-R 提升，但 Trans-I/P 下降

---

## 批判性思考

### 优点

1. **框架可扩展**: [[技能路由|Skill Routing]] + 分层 Agent 设计使得新的评估维度可以轻松添加为新技能
2. **透明可追溯**: [[证据树|Evidence Tree]] 记录完整推理链，评估失败可定位到具体子问题
3. **高鲁棒性**: 评估方差比现有 benchmark 小 4.9 倍，更适合模型排名

### 局限性

1. **计算成本高**: 每个评估案例都需要多层 Agent 调用，单次评估成本可能远高于传统指标
2. **Agent 本身的偏差**: 评估 Agent 的质量依赖底层 LLM，可能引入 LLM 自身的偏见
3. **案例覆盖有限**: 330 个案例虽经过精心设计，但仍难以覆盖真实世界的全部长尾场景

### 潜在改进方向

1. **Test-time scaling**: 通过增加评估 Agent 的计算量进一步提升准确性（论文中已提出）
2. **自扩展技能库**: 当遇到分布外场景时，Agent 自动学习新技能（论文中已提出）
3. **跨模型集成评估**: 用多个不同 LLM 的 Agent 评估结果取平均，降低单一 LLM 偏差

### 可复现性评估

- [x] 代码开源（GitHub: mirros-lab/harnesseval-w）
- [ ] 预训练模型（仅评估框架，无需预训练模型）
- [x] 训练细节完整（案例构建流程完整描述）
- [x] 数据集可获取（评估案例通过流水线生成）

---

## 关联笔记

### 被评估对象

- [[Cosmos3]]: 被评估的 Prompt I2V 世界模型
- [[HunyuanVideo 1.5]]: 被评估的 Prompt I2V 世界模型，微调后变为 [[HY-WorldPlay 1.5]]

### 对比

- [[WBench]]: 现有世界模型评估基线，HarnessEval-W 在鲁棒性上比其高 4.9 倍

### 方法相关

- [[交互式世界模型]]: 评估的核心对象，通过 Equation 1 进行能力因式分解
- [[Agentic评估]]: 核心方法论，用 Agent 替代固定 rubric
- [[证据树]]: 输出格式，记录评估推理链
- [[技能路由]]: 系统核心组件，根据上下文路由评估技能

### 统计方法

- [[Spearman相关系数]]: 用于量化与人类偏好的对齐程度（ρ=0.93）
- [[Pearson Correlation]]: 用于分析各评估维度之间的相关性

---

## 速查卡片

> [!summary] HarnessEval-W
> - **核心**: 用分层 Agentic 评估框架替代固定指标，评估交互式世界模型
> - **方法**: Skill Routing + 子 Agent 分解 + Evidence Tree 汇聚
> - **结果**: 18 模型评估，ρ=0.93 人类对齐，方差比 WBench 小 4.9 倍；Seedance 2.0 排名第一（75.5）
> - **代码**: https://github.com/mirros-lab/harnesseval-w

---

*笔记创建时间: 2026-08-23*
