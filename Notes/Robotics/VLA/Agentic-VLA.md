---
title: "Agentic-VLA: Efficient Online Adaptation for Vision-Language-Action Models"
method_name: "Agentic-VLA"
authors: [Ruofan Jin, Zaixi Zhang]
year: 2026
venue: arXiv
tags: [vla, online-adaptation, reinforcement-learning, curriculum-learning, memory-augmented, exploration, long-horizon]
zotero_collection: Robotics/VLA
image_source: online
arxiv_html: https://arxiv.org/html/2605.22896v1
created: 2026-05-26
---

# 论文笔记：Agentic-VLA: Efficient Online Adaptation for Vision-Language-Action Models

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | 未公开 |
| 日期 | May 2026 |
| 项目主页 | N/A |
| 对比基线 | [[EVOLVE-VLA]], [[OpenVLA-OFT]] |
| 链接 | [arXiv](https://arxiv.org/abs/2605.22896) / Code: N/A |

---

## 一句话总结

> Agentic-VLA 通过自适应奖励合成、语言引导探索和经验记忆三个模块，使 VLA 模型在部署时高效在线自适应，在 LIBERO 长任务上提升 12.3%，收敛速度提升 2.4×。

---

## 核心贡献

1. **自适应奖励合成 (ARS)**: 将复杂任务自动分解为子目标，并根据智能体当前能力动态调整奖励权重，形成自然的 [[课程学习|自动课程]]
2. **语言引导探索 (LGE)**: 利用预训练 [[视觉语言模型|VLM]] 生成结构化探索建议，通过 Prompt 增强引导策略的多样性行为发现
3. **经验记忆 (EM)**: 存储任务嵌入与适应后策略参数，通过余弦相似度检索相关任务并进行软插值权重初始化，实现跨任务快速迁移

---

## 问题背景

### 要解决的问题

[[视觉语言动作模型]] 在训练数据分布之外的新任务上泛化能力差，部署后需要针对目标任务进行有效的在线适应，同时保证样本效率。

### 现有方法的局限

- 基于 [[监督微调]] (SFT) 的方法需要大量示范数据，难以快速适应新任务
- 现有 RL 在线适应方法（如 [[EVOLVE-VLA]]）缺乏 **自适应奖励机制**、**结构化探索引导** 和 **跨任务知识共享**，导致样本效率低（约需 53.8k rollouts），收敛慢
- 稀疏奖励信号难以指导长视野任务的学习

### 本文的动机

将大型语言模型的 Agentic 能力（任务分解、推理、记忆）引入 VLA 在线适应，形成密集、动态的课程式奖励，并利用历史经验加速新任务学习。

---

## 方法详解

### 模型架构

Agentic-VLA 采用 **三模块 Agentic 在线适应框架**，基于 [[OpenVLA-OFT]] 作为基础策略模型：

- **输入**: 语言任务指令 $l_{task}$ + 图像观测 $o_t$ + 探索建议 $s_{explore}$（可选）
- **Backbone**: [[OpenVLA-OFT]]（动作 tokenization + 自回归生成）
- **核心模块**: [[自适应奖励合成|ARS]] + [[语言引导探索|LGE]] + [[经验记忆|EM]]
- **优化算法**: [[GRPO]] (Group Relative Policy Optimization)
- **Critic 模型**: [[VLAC]] (Vision-Language-Action-Critic) 用于进度估计
- **探索 VLM**: Qwen3-VL-8B-Instruct

### 核心模块

#### 模块 1: 自适应奖励合成 (Adaptive Reward Synthesis, ARS)

**设计动机**: 利用 [[课程学习]] 原理，让模型先掌握简单子目标再挑战困难子目标，提供密集奖励信号

**具体实现**:

1. **任务分解**: 使用语言模型将任务分解为有序子目标集合

$$
\mathcal{G} = \{g_1, g_2, \ldots, g_K\} = \text{LM\_decompose}(l_{task})
$$

2. **能力追踪**: 用指数移动平均追踪每个子目标的完成能力

$$
\hat{c}_k^{(t+1)} = \alpha \cdot \hat{c}_k^{(t)} + (1 - \alpha) \cdot \mathbb{1}[\text{success at } g_k]
$$

其中平滑系数 $\alpha = 0.9$

3. **能力感知权重**: 已掌握子目标权重降低，困难子目标权重保持高位

$$
w_k = 1 - \hat{c}_k
$$

**自然形成 Auto-Curriculum**: 随着能力提升，重点自动从简单子目标迁移到困难子目标

#### 模块 2: 语言引导探索 (Language-Guided Exploration, LGE)

**设计动机**: 利用 [[视觉语言模型]] 的推理能力提供空间和任务感知的探索建议，避免盲目随机探索

**具体实现**:
- 使用 Qwen3-VL-8B-Instruct 基于当前观测生成可执行建议（如空间位置引导、抓取点优化）
- 通过 Prompt 拼接将建议融入策略输入（见公式 2）
- **自适应建议频率**: 随策略改善降低建议频率，避免过度依赖

#### 模块 3: 经验记忆 (Experience Memory, EM)

**设计动机**: 跨任务知识共享，避免从零开始训练，实现快速热启动

**具体实现**:
- 存储三元组：(任务嵌入 $e_j$, 适应参数 $\theta_j$, 元数据)
- 容量上限 100 条，基于语义重要性的优先级管理
- 检索时取 $k=3$ 个最相似任务（余弦相似度），用温度 $\tau=0.1$ 的软插值初始化

---

## 关键公式

### 公式 1: [[自适应奖励合成|ARS 奖励计算]]

$$
R(\tau) = \sum_{k=1}^{K} w_k \cdot \Delta_k(\tau)
$$

**含义**: 整体任务奖励是各子目标能力加权进度之和

**符号说明**:
- $\tau$: 轨迹
- $K$: 子目标数量（平均 4.2 个）
- $w_k = 1 - \hat{c}_k$: 子目标 $k$ 的能力感知权重
- $\Delta_k(\tau) = C_\phi(o_{start}^k, o_{end}^k, g_k)$: VLAC Critic 估计的子目标 $k$ 进度增量

### 公式 2: [[语言引导探索|建议条件化动作生成]]

$$
a_t \sim \pi_\theta(a \mid s_t, l_{task} \oplus s_{explore})
$$

**含义**: 策略以任务指令与探索建议的拼接作为条件生成动作

**符号说明**:
- $s_t$: 当前状态
- $l_{task}$: 任务语言指令
- $s_{explore}$: VLM 生成的探索建议文本
- $\oplus$: 文本拼接操作

### 公式 3: [[课程学习|自适应建议频率]]

$$
p_{suggest}(t) = p_{max} \cdot \exp(-\lambda \cdot \bar{R}^{(t)})
$$

**含义**: 随着平均奖励 $\bar{R}$ 提升，VLM 建议的触发概率指数衰减

**符号说明**:
- $p_{max}$: 最大建议概率
- $\lambda$: 衰减率超参数
- $\bar{R}^{(t)}$: $t$ 时刻的滑动平均奖励

### 公式 4: [[经验记忆|温启动权重初始化]]

$$
\theta_{init} = \sum_j \frac{\exp(\cos(e_{new}, e_j) / \tau)}{\sum_{j'} \exp(\cos(e_{new}, e_{j'}) / \tau)} \cdot \theta_j
$$

**含义**: 通过对历史任务参数的 softmax 加权平均初始化当前任务策略参数

**符号说明**:
- $e_{new}$: 新任务的嵌入向量
- $e_j$: 记忆库中第 $j$ 个任务嵌入
- $\tau = 0.1$: 温度参数
- $\theta_j$: 第 $j$ 个任务的适应后策略参数

### 公式 5: [[能力追踪|指数移动平均能力估计]]

$$
\hat{c}_k^{(t+1)} = \alpha \cdot \hat{c}_k^{(t)} + (1 - \alpha) \cdot \mathbb{1}[\text{success at } g_k]
$$

**含义**: 用 EMA 平滑子目标完成率以稳定能力估计

**符号说明**:
- $\alpha = 0.9$: 平滑系数
- $\hat{c}_k^{(t)}$: 第 $t$ 步对子目标 $k$ 的能力估计
- $\mathbb{1}[\cdot]$: 指示函数（子目标是否成功完成）

---

## 关键图表

### Figure 1: Framework Overview / 系统框架

![Figure 1](https://arxiv.org/html/2605.22896v1/x1.png)

**说明**: Agentic-VLA 的整体适应循环。经验记忆 (EM) 检索相关参数做热启动 → VLA 与环境交互时接收 LGE 结构化引导建议 → 轨迹由 ARS 评估（基于能力动态加权子目标奖励）→ GRPO 优化策略。

### Figure 2: 自适应奖励动态调整

![Figure 2](https://arxiv.org/html/2605.22896v1/fig2.png)

**说明**: 在 "turn on the stove and put the moka pot on it" 任务上，各子目标奖励随训练进程动态调整——已掌握的子目标权重下降，困难子目标保持高权重，自然形成课程学习节奏。

### Figure 3: LIBERO 各套件学习曲线

![Figure 3](https://arxiv.org/html/2605.22896v1/fig3.png)

**说明**: Agentic-VLA 在 LIBERO 全部四个套件（Spatial/Object/Goal/Long）上的学习曲线，阴影区域为 5 个随机种子的 ±1 标准差。相比 EVOLVE-VLA 收敛速度提升 2.4×（700 vs 1680 iterations）。

### Figure 4: 经验记忆分析

![Figure 4](https://arxiv.org/html/2605.22896v1/fig4.png)

**说明**: 任务嵌入空间可视化与检索模式分析，展示跨任务语义相似性及记忆检索效果。

### Figure 5: 涌现能力展示

![Figure 5](https://arxiv.org/html/2605.22896v1/figure_qualitative.png)

**说明**: 在线适应后涌现的四类能力：(a) **错误恢复** — 抓取失败后自主重新调整夹爪；(b) **自适应物体处理** — 旋钮位移后轨迹自适应；(c) **新策略发现** — LGE 引导发现示范中不存在的侧面抓取方式；(d) **探索建议** — Critic 建议先开抽屉再抓取，规避常见失败模式。

### Table 1: LIBERO 基准总体性能

| Method | Spatial | Object | Goal | Long | **Average** |
|--------|---------|--------|------|------|-------------|
| Octo | - | - | - | - | 75.1% |
| OpenVLA | - | - | - | - | 76.5% |
| π₀ | - | - | - | - | 94.2% |
| OpenVLA-OFT | - | - | - | - | 89.2% |
| VLA-RL | - | - | - | - | 81.0% |
| SimpleVLA-RL | - | - | - | - | 91.2% |
| EVOLVE-VLA | - | - | - | - | 95.8% |
| **Agentic-VLA** | **97.2%** | **98.6%** | **97.4%** | **98.1%** | **97.8%** |

**关键发现**: Agentic-VLA 以 97.8% 平均成功率超越所有基线，相比 OpenVLA-OFT 提升 +8.6%，相比 EVOLVE-VLA 提升 +2.0%。

### Table 2: 单样本学习结果 (1-Shot)

| Method | Spatial | Object | Goal | Long | **Average** |
|--------|---------|--------|------|------|-------------|
| OpenVLA-OFT | - | - | - | - | 43.6% |
| EVOLVE-VLA | - | - | - | - | 61.3% |
| **Agentic-VLA** | **79.8%** | **78.4%** | **72.6%** | **51.2%** | **70.5%** |

**关键发现**: 仅用 1 条示范时，Agentic-VLA 较 OpenVLA-OFT 提升 +26.9%（+28.5% 原文总结数字包含四舍五入），较 EVOLVE-VLA 提升 +9.2%。

### Table 3: 跨任务泛化结果

| Method | Success Rate | Progress |
|--------|-------------|---------|
| EVOLVE-VLA | 20.8% | 54.2% |
| **Agentic-VLA** | **31.2%** | **68.7%** |

**关键发现**: 无任务示范的跨任务迁移（LIBERO-Long → LIBERO-Object）成功率从 20.8% 提升至 31.2%（+10.4%）。

### Table 4: 训练效率对比 (LIBERO-Long)

| Method | 达到 90% 成功的迭代数 | Rollouts | 训练时间 |
|--------|---------------------|---------|---------|
| EVOLVE-VLA | 1,680 iterations | 53.8k | ~19 hours |
| **Agentic-VLA** | **700 iterations** | **22.4k** | **~8 hours** |

**关键发现**: 2.4× 更快收敛，Rollout 数量减少 58%。

### Table 5: 消融实验 (LIBERO-Long 成功率)

| 配置 | 成功率 | 说明 |
|------|--------|------|
| SFT Baseline | 85.8% | 无在线适应 |
| + ARS | 94.6% | +8.8% |
| + ARS + LGE | 96.2% | +1.6% |
| + ARS + LGE + EM | **98.1%** | +1.9%，收敛快 50% |

**关键发现**: ARS 贡献最大（+8.8%），三个模块协同提升效果最优；EM 主要贡献在收敛速度而非最终精度。

### Table 6: 课程与探索替代方案对比

| 方法 | LIBERO-Long 成功率 |
|------|-------------------|
| 均匀权重 | 92.7% |
| 固定调度 | 93.4% |
| 学习进度采样 | 94.0% |
| **能力感知 ARS（本文）** | **94.6%** |

### Table 7: 任务分解质量分析

| 分解方式 | 语义覆盖率 | 平均子目标数 | 成功率 |
|---------|-----------|------------|--------|
| Oracle 分解 | - | - | 98.4% |
| LM 自动分解（本文） | 0.91 | 4.2 | 98.1% |

### Table 8: RoboTwin 2.0 双臂操作结果

| Method | Easy | Hard |
|--------|------|------|
| π₀ | 46.4% | 16.3% |
| **Agentic-VLA** | **62.5%** | **34.7%** |

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| LIBERO-Spatial | 50 demos/task | 空间关系操作 | 训练/测试 |
| LIBERO-Object | 50 demos/task | 多物体操作 | 训练/测试 |
| LIBERO-Goal | 50 demos/task | 目标导向任务 | 训练/测试 |
| LIBERO-Long | 50 demos/task | 长视野复合任务 | 训练/测试 |
| RoboTwin 2.0 | 50 dual-arm tasks | 双臂 Aloha AgileX，含域随机化 | 验证泛化 |

### 实现细节

- **Backbone**: OpenVLA-OFT（动作 tokenization + 自回归）
- **Critic**: VLAC（Vision-Language-Action-Critic）
- **探索 VLM**: Qwen3-VL-8B-Instruct
- **优化器**: GRPO，学习率 $1 \times 10^{-5}$
- **Batch Size**: 32，Group Size: 8
- **Horizon**: 500 步
- **记忆容量**: 100 条任务经验
- **检索 top-k**: $k=3$，温度 $\tau=0.1$
- **EMA 平滑系数**: $\alpha=0.9$
- **硬件**: 4× NVIDIA A100 80GB
- **训练时长**: ~8 hours/LIBERO suite

### 可视化结果

在线适应涌现了训练数据中未见的行为：抓取失败后自主重抓、物体移位后轨迹重规划、以及 Critic 引导下发现的新抓取策略。这表明框架具备真正的探索与泛化能力而非单纯的模式记忆。

---

## 批判性思考

### 优点

1. **三模块设计互补性强**: ARS 提供密集奖励信号，LGE 引导探索方向，EM 加速跨任务迁移，三者协同解决在线适应的核心挑战
2. **高样本效率**: 相比 EVOLVE-VLA 减少 58% Rollouts，对计算资源受限的真实机器人场景友好
3. **接近人类级分解质量**: LM 自动任务分解的语义覆盖率达 0.91，与 Oracle 分解性能差距仅 0.3%
4. **跨平台验证**: 在 LIBERO 和 RoboTwin 2.0（双臂）两个基准上均显著优于基线

### 局限性

1. **奖励 Hacking 问题**: 约 12% 的失败案例源于进度估计与真实成功标准不一致，VLAC Critic 的奖励建模仍存在漏洞
2. **多模块系统复杂性**: ARS + LGE + EM + VLAC 多个外部组件的集成在真实机器人部署中工程复杂度高
3. **记忆扩展性隐患**: 固定容量 100 条的记忆库对于大规模任务集可能不足，语义优先级策略的长期效果待验证
4. **自主探索的安全性**: 在真实机器人上运行无约束探索阶段存在安全风险，论文未提供安全约束机制

### 潜在改进方向

1. 引入基于置信度的 Critic 不确定性估计，缓解奖励 Hacking 问题
2. 设计弹性记忆扩展机制（如层次化聚类）以支持大规模任务集
3. 在 LGE 探索阶段加入安全约束层（如工作空间限制），提高真实部署可行性

### 可复现性评估

- [ ] 代码开源（论文未提供代码链接）
- [ ] 预训练模型（未提供）
- [x] 训练细节完整（附录 A-E 包含完整超参数）
- [x] 数据集可获取（LIBERO 和 RoboTwin 2.0 均公开）

---

## 关联笔记

### 基于

- [[OpenVLA-OFT]]: 基础策略模型，提供动作 tokenization 和自回归生成
- [[GRPO]]: 组相对策略优化算法，用于在线适应

### 对比

- [[EVOLVE-VLA]]: 最主要对比基线，在样本效率和最终性能上均被超越
- [[π0]]: SFT 强基线，Agentic-VLA 在 LIBERO 上超越其 +3.6%

### 方法相关

- [[课程学习]]: ARS 模块的核心思想来源
- [[VLAC]]: 用于子目标进度估计的 Critic 模型
- [[视觉语言模型]]: LGE 模块使用 Qwen3-VL-8B-Instruct 生成探索建议
- [[经验回放]]: EM 模块借鉴经验回放与参数迁移思想

### 硬件/数据相关

- [[LIBERO]]: 主要评测基准，包含四个任务套件
- [[RoboTwin 2.0]]: 双臂操作验证平台，使用 Aloha AgileX 机器人

---

## 速查卡片

> [!summary] Agentic-VLA: Efficient Online Adaptation for VLAs
> - **核心**: 三模块 Agentic 框架（ARS + LGE + EM）实现 VLA 高效在线适应
> - **方法**: 自适应课程奖励 + VLM 探索引导 + 跨任务记忆热启动，GRPO 优化
> - **结果**: LIBERO 平均 97.8%（+8.6% vs OFT），1-shot +26.9%，收敛 2.4× 更快
> - **代码**: 暂未开源

---

*笔记创建时间: 2026-05-26*
