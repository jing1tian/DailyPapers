---
title: "V-VLAPS: Value-Guided Planning for Vision-Language-Action Models"
method_name: "V-VLAPS"
authors: [Ke Ren, Ali Salamatian, Kieran Pattison, Cyrus Neary]
year: 2026
venue: arXiv
tags: [vla, mcts, value-function, robot-manipulation, tree-search, planning, imitation-learning]
zotero_collection: Robotics/VLA
image_source: local
arxiv_html: https://arxiv.org/html/2601.00969
created: 2026-07-11
---

# 论文笔记：V-VLAPS: Value-Guided Planning for Vision-Language-Action Models

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | 未在论文中明确列出 |
| 日期 | January 2026（最终版 July 2026） |
| 项目主页 | N/A |
| 对比基线 | [[VLAPS]] |
| 链接 | [arXiv](https://arxiv.org/abs/2601.00969) |

---

## 一句话总结

> V-VLAPS 为 [[VLA（视觉-语言-动作模型）]] 引导的树搜索加入轻量级值函数头（~2.4M 参数），用离线 rollout 训练的 [[蒙特卡洛回报]] 预测来引导 MCTS 优先探索高价值分支，在 LIBERO 长时域任务上实现最高 +6pp 的提升。

---

## 核心贡献

1. **值函数头（Value Head）**: 在冻结的 [[Octo]] backbone 特征之上训练一个轻量 MLP，预测连续的蒙特卡洛折扣回报 $G_t$，而非二元成功/失败探针（SAFE）
2. **值引导 MCTS 评分**: 将值函数估计 $Q_\theta(v, a^i)$ 与 [[VLAPS]] 的先验探索项线性组合，在节点选择阶段实现值引导的树搜索
3. **数据重均衡策略**: 将 Monte Carlo 回报按 10 个等宽 bin 分桶并对底部 bin 降采样，防止值头退化为全零预测

---

## 问题背景

### 要解决的问题

[[VLA（视觉-语言-动作模型）]] 通过[[行为克隆]]训练，在分布内表现良好，但在分布偏移和 out-of-distribution 状态下表现脆弱。以 [[VLAPS]] 为代表的 VLA 引导规划方法通过 [[MCTS]] 改善执行效果，但缺乏**显式的值信号**来纠正策略偏差——只依靠策略自身的分布先验来引导搜索。

### 现有方法的局限

- **[[VLAPS]]**：使用 VLA 策略分布引导树搜索，但无显式值函数；停在 "学习一个先验" 这一步
- **SAFE（Gu et al. 2026）**：训练成功/失败二元探针（probes），用于检测失败，而非积极引导搜索
- 两者都止步于：VLA latent features 中已经编码了任务成功信息，但未将其转化为连续值估计

### 本文的动机

"If a binary success signal generalizes across tasks, the same features may also carry information useful for estimating continuous value." 既然 VLA latent 空间中存在可分离的成功/失败区域（如 [[t-SNE]] 可视化所示），就可以训练连续值头，并在 MCTS 选择阶段发挥引导作用。

---

## 方法详解

### 模型架构

V-VLAPS 以 **[[VLAPS]] + 值函数头** 的架构为核心：

- **输入**: 视觉观测 $o_t$ + 任务语言指令 $L_\mathcal{T}$
- **Backbone**: 冻结的 [[Octo]]（不更新参数），输出 latent readout $h_t \in \mathbb{R}^d$
- **值函数头**: 三层 MLP（~2.4M 参数），结构为 768→3072(ReLU)→scalar
- **输出**: 标量值估计 $\hat{v}_t = V_\theta(h_t)$，用于 [[MCTS]] 节点评分
- **搜索机制**: 基于 [[PUCT]] 的树搜索，结合 VLAPS 先验与值头 Q 估计

### 核心模块

#### 模块1：数据收集与值目标定义

**设计动机**: 利用 [[行为克隆]] 策略（未规划）的 rollout 建立训练集，保留有限初始状态泛化能力

**具体实现**:
- 每个 suite 中每个任务使用 4 个初始状态（indices 0, 33, 66, 99）进行 [[Octo]] 无规划 rollout
- Episode 终止时得到二元结果 $R \in \{0, 1\}$
- 采用折扣蒙特卡洛回报作为值目标：$G_t = \gamma^{T-t}$（成功）或 $G_t = 0$（失败）

**数据重均衡**:
- 原始分布严重偏向 $G_t=0$（最高 92.8% 为零目标）
- 将目标按 10 个等宽 bin 分桶，对底部 bin 进行**降采样**，使其与其他 9 个 bin 大小之和相当
- 避免值头退化为恒零预测

#### 模块2：值函数头训练

**设计动机**: 在冻结 [[Octo]] backbone 上 overhead 最小的方式引入值估计能力

**具体实现**:
- 值头架构：3 层 MLP，$V_\theta: \mathbb{R}^{768} \to \mathbb{R}$
- 使用 Adam 优化器、MSE 损失、cosine 学习率衰减
- **仅更新值头参数 $\theta$**，Octo 全程冻结
- 每个 suite 单独训练一个值头（90/10 train/val split）

#### 模块3：值引导 MCTS 选择

**设计动机**: 将 VLA 先验探索与值引导评分融合，在有足够搜索时间时优先走向高价值分支

**具体实现**:
- 在节点 $v$ 处采样候选动作块 $a^i$，先用 VLAPS 计算先验探索项
- 将候选动作块在模拟器中前向执行，到达后继状态 $s'$，用 [[Octo]] 计算 $h(s')$
- 值头估计后继状态的 Q 值：$Q_\theta(v, a^i) = V_\theta(h(s'))$
- 最终 V-VLAPS 选择分数：$\lambda_V$ 固定为 1，贯穿所有实验

---

## 关键公式

### 公式1: [[蒙特卡洛回报|蒙特卡洛折扣回报]]

$$
G_t = \begin{cases}
\gamma^{T-t}, & \text{if episode ends in success}\ (R=1) \\
0, & \text{if episode ends in failure}\ (R=0)
\end{cases}
$$

**含义**: 对每个时间步 $t$，其值目标为折扣因子 $\gamma$ 的幂次（成功）或零（失败）

**符号说明**:
- $G_t$: 时间步 $t$ 的 Monte Carlo 值目标
- $\gamma = 0.99$: 折扣因子
- $T$: episode 总时间步数
- $R \in \{0, 1\}$: episode 终止时的二元成功信号

### 公式2: [[均方误差|值头训练目标（MSE）]]

$$
\mathcal{L}(\theta) = \mathbb{E}_{(h_t, G_t)}\left[(V_\theta(h_t) - G_t)^2\right]
$$

**含义**: 值头以 MSE 损失拟合 Monte Carlo 回报，backbone 参数全程冻结

**符号说明**:
- $\theta$: 值头（MLP）参数
- $h_t \in \mathbb{R}^{768}$: Octo 最后一层 readout 向量
- $G_t$: Monte Carlo 值目标
- $V_\theta(h_t)$: 值头输出的标量值估计

### 公式3: [[PUCT|VLAPS 先验探索项]]

$$
U_{\text{VLAPS}}(v, a^i) = \psi_{\Phi_v}(a^i \mid I_t, L_\mathcal{T}) \times \frac{\sqrt{N(v, a^i)}}{1 + N(v, a^i)}
$$

**含义**: 将 VLA 策略先验分布与 UCB 探索项相乘，平衡利用与探索

**符号说明**:
- $\psi_{\Phi_v}$: 节点 $v$ 处 VLAPS 对候选动作块的先验分布（softmax，温度 $\alpha=5$）
- $I_t$: 当前视觉观测
- $L_\mathcal{T}$: 任务语言指令
- $N(v, a^i)$: 候选 $a^i$ 在节点 $v$ 被选择的次数

### 公式4: [[值函数|后继状态 Q 值估计]]

$$
Q_\theta(v, a^i) = V_\theta(h(s'))
$$

**含义**: 将候选动作块在模拟器中执行后，用值头评估后继状态的期望回报

**符号说明**:
- $s'$: 执行动作块 $a^i$ 后的后继状态
- $h(s')$: 后继状态通过 Octo 得到的 latent readout
- $V_\theta$: 训练好的值函数头

### 公式5: [[值引导规划|V-VLAPS 最终选择分数]]

$$
\text{SCORE}(v, a^i) = \lambda_V \cdot Q_\theta(v, a^i) + U_{\text{VLAPS}}(v, a^i)
$$

**含义**: 将值引导项与 VLAPS 先验探索项线性组合，得到 MCTS 节点选择的最终分数

**符号说明**:
- $\lambda_V = 1.0$: 值引导权重系数（全实验固定）
- $Q_\theta(v, a^i)$: 后继状态 Q 值估计
- $U_{\text{VLAPS}}(v, a^i)$: VLAPS 先验探索项

---

## 关键图表

### Figure 1: V-VLAPS 系统概览

![[V-VLAPS_fig1_overview.png]]

**说明**: V-VLAPS 的整体架构。在每个 [[MCTS]] 节点处，当前视觉观测和语言指令（如 "Move ketchup to basket"）通过冻结的 [[VLA（视觉-语言-动作模型）]] backbone 和值头 MLP 产生标量值估计。高预测值（绿色）的分支在搜索中被优先选择；低值（红色）分支被压制。值估计挂载到节点并进入 VLAPS 评分规则。

### Figure 2: Octo Readout 的 t-SNE 投影

![[V-VLAPS_fig2_tsne.png]]

**说明**: [[t-SNE]] 投影 libero_object 上 [[Octo]] 最后一层 readout $h_t$，按 Monte Carlo 值目标 $G_t$ 着色。成功和失败 rollout 在 embedding 空间中占据视觉上可分离的不同区域，与 SAFE 的发现一致：VLA latent features 携带任务结果的判别性信号。

### Table 1: 主要结果——各方法各 Suite 成功率（%）

| Suite | VLA（无规划） | VLAPS (600s) | V-VLAPS (600s) | VLAPS (1800s) | V-VLAPS (1800s) |
|-------|------------|--------------|----------------|---------------|-----------------|
| libero_object | 37 | 82 | 85 | 87 | **93** |
| libero_spatial | 81 | 96 | 95 | 96 | 97 |
| libero_goal | 88 | 92 | 93 | 90 | 93 |
| libero_10 | 38 | 77 | 75 | 81 | **85** |
| libero_90 | 57 | 90 | 89 | 88 | 90 |
| **Average** | **60.2** | **87.4** | **87.4** | **88.4** | **91.6** |

**关键发现**: 600s 时 V-VLAPS 与 VLAPS 持平（均 87.4%）；1800s 时 V-VLAPS 达到 91.6%（+3.2pp），最大收益在 libero_object（+6pp）和 libero_10（+4pp）。

### Table 2: 失败模式分析——libero_object 任务 6-8（600s）

| Method | 总失败次数 | Root Timeout 次数 | Root Timeout 占比 |
|--------|-----------|-----------------|-----------------|
| VLAPS | 18 | 14 | 77.8% |
| V-VLAPS | 15 | 13 | 86.7% |

**关键发现**: Root-level timeout（MCTS 始终停在根节点直到时间耗尽）是主要失败模式，说明搜索未到达值函数能发挥判别作用的深层状态，这解释了为何 600s 下增益有限。

### Table 3: 各 Suite 重均衡后训练数据量（Appendix A）

| Suite | 原始样本数 | 零目标占比 (%) | 重均衡后 | 训练集 | 验证集 |
|-------|-----------|--------------|---------|-------|-------|
| libero_object | 60,695 | 92.8 | 8,700 | 7,830 | 870 |
| libero_spatial | 28,580 | 63.5 | 20,880 | 18,792 | 2,088 |
| libero_goal | 34,155 | 70.3 | 20,300 | 18,270 | 2,030 |
| libero_10 | 11,020 | 50.0 | 11,020 | 9,918 | 1,102 |
| libero_90 | 63,595 | 86.8 | 16,810 | 15,129 | 1,681 |

**说明**: libero_object 的 92.8% 零目标反映了 Octo 无规划时的极低成功率；libero_10 本已接近均衡（50%），无需降采样。

### Table 4: 值头训练超参数（Appendix B）

| 设置 | 值 |
|------|-----|
| 架构 | 768→3072 (ReLU) → scalar |
| 参数量 | ~2.4M |
| 优化器 | Adam |
| 初始学习率 | 1×10⁻⁴ |
| 学习率调度 | Cosine decay (α=0.1) |
| Batch Size | 64 |
| 训练轮数 | 100 |
| Train/Val 分割 | 90/10（seed 42） |
| 损失 | MSE on Monte Carlo returns $G_t$ |
| 折扣因子 | 0.99 |
| 重均衡策略 | 10-bin [0,1] 等宽，底部 bin 降采样 |

### Table 5: MCTS 超参数（Appendix B，VLAPS/V-VLAPS 共享）

| 设置 | 值 |
|------|-----|
| PUCT 探索权重 ψ | 1.0 |
| 每节点动作样本数 | 300 |
| 每节点扩展数 | 10 |
| 每步 MCTS 扩展数 | 1 |
| 最大树深度 | 80 |
| 最大模拟 rollout 长度 | 300 |
| 每次扩展 chunk 步数 | 4 |
| 每 chunk 后重新规划 | 是 |
| Wall-time 上限 | 600s（默认），1800s（扩展） |
| 动作 chunk 库大小 | 2,000 medoids |
| $\beta_{\Phi}$ 分布 | Softmax (α=10, ε=0.1) |
| $\psi_{\Phi_v}$ 分布 | Softmax (α=5) |
| VLA 温度 | 0.0 |
| 评估 GPU | 单块 NVIDIA H100 |

### Table 6: libero_object 逐任务结果（Appendix C）

| 任务 | VLAPS (600s) | V-VLAPS (600s) | VLAPS (1800s) | V-VLAPS (1800s) |
|------|-------------|----------------|---------------|-----------------|
| 0 | 10/10 | 10/10 | 10/10 | 10/10 |
| 1 | 10/10 | 10/10 | 10/10 | 10/10 |
| 2 | 10/10 | 10/10 | 10/10 | 10/10 |
| 3 | 10/10 | 10/10 | 10/10 | 10/10 |
| 4 | 8/10 | 9/10 | 9/10 | 9/10 |
| 5 | 9/10 | 10/10 | 9/10 | 10/10 |
| 6 | 5/10 | 3/10 | 6/10 | 5/10 |
| 7 | 8/10 | 5/10 | 6/10 | **10/10** |
| 8 | 3/10 | 8/10 | 8/10 | 9/10 |
| 9 | 9/10 | 10/10 | 9/10 | 10/10 |
| **Total** | **82/100** | **85/100** | **87/100** | **93/100** |

**关键发现**: 方差集中在任务 6-8，其余任务几乎达到天花板。任务 7 在 1800s 时 V-VLAPS 从 5/10 飙升至 10/10，最能体现值引导在深层搜索时的效果。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| [[LIBERO]]-Object | 10 任务 | pick-and-place，物体级变化 | 训练+评估 |
| LIBERO-Spatial | 10 任务 | 空间推理 | 训练+评估 |
| LIBERO-Goal | 10 任务 | 目标条件 | 训练+评估 |
| LIBERO-10 | 10 长时域任务 | 多步骤操作 | 训练+评估 |
| LIBERO-90 | 90 多样任务 | 最广泛覆盖 | 训练+评估（取 10 任务） |

### 实现细节

- **Backbone**: [[Octo]]（冻结，不更新）
- **值头**: 3 层 MLP，768→3072(ReLU)→scalar，~2.4M 参数
- **优化器**: Adam，初始 lr=1×10⁻⁴，cosine decay
- **Batch Size**: 64
- **训练轮数**: 100
- **硬件**: 单 NVIDIA H100（每 100-episode cell 约需 3.5–4.5 小时 @ 600s budget）
- **评估协议**: 每个（方法, suite, budget）组合评估 100 episodes；值头在 4 个初始状态上训练，全部 10 个初始状态上评估

### 可视化结果

t-SNE 投影（Figure 2）证实 Octo latent space 中成功/失败 rollout 占据明显可分离区域，为值函数训练提供了信号基础。Wall-time 扩展实验（600s→1800s）清晰展示了值引导的生效条件：搜索到达更深分支时，V-VLAPS 的优势持续扩大。

---

## 批判性思考

### 优点

1. **设计简洁高效**: 仅添加 ~2.4M 参数的 MLP 值头，无需修改 VLA backbone，几乎不增加推理负担
2. **基于实证观察**: t-SNE 可视化验证了 VLA latent 空间中确实存在可判别的值信号，方法有坚实的动机
3. **详细的失败模式分析**: Root-level timeout 分析诚实揭示了值函数生效的边界条件，为后续工作指明方向

### 局限性

1. **依赖模拟器**: 需要与规划/评估环境匹配的仿真器，无法直接迁移到真实机器人部署
2. **Off-policy 训练偏差**: 值头基于无规划 Octo rollout 训练，但被用于评估 V-VLAPS 自身搜索产生的状态，存在 off-policy gap
3. **未测试其他 VLA**: 所有实验均使用 [[Octo]] 作为骨干，迁移至 π₀ 等更强 VLA 的效果未验证
4. **数据集规模有限**: 每个 suite 训练数据最少仅 7,830 样本，大规模数据的影响未探索
5. **缺乏受控消融**: 没有打乱标签或随机初始化的消融实验，难以隔离值信号的真实贡献

### 潜在改进方向

1. **On-policy 值头迭代优化**: 在 V-VLAPS 自身的 rollout 上重新训练值头，缓解 off-policy 问题
2. **基于值的分支剪枝**: 用值头在扩展阶段裁剪低价值分支，将 root-level timeout 转化为高价值分支上的有效计算
3. **HL-Gauss 分类头替代 MSE 回归**: 论文尝试过但遇到类别不均衡问题，值得进一步探索
4. **多层聚合值头输入**: 聚合 Transformer 多层特征而非仅用最后一层 readout，避免平均化抹去有用信号

### 可复现性评估

- [ ] 代码开源（论文未提及代码发布）
- [ ] 预训练模型（未提供）
- [x] 训练细节完整（Appendix B 提供了完整超参数）
- [x] 数据集可获取（LIBERO 是公开 benchmark）

---

## 关联笔记

### 基于

- [[VLAPS]]: 本文的直接前驱，V-VLAPS 在其基础上添加值函数
- [[Octo]]: 固定使用的 VLA backbone
- [[MuZero]]: 将策略+值函数用于树搜索的游戏领域先驱（AlphaGo 同类）

### 对比

- [[VLAPS]]: 主要对比基线，探讨值函数的边际贡献
- [[V-GPS]]: 另一个将 VLA 与规划结合的相关方法

### 方法相关

- [[MCTS]]: 核心搜索框架
- [[PUCT]]: 节点选择的探索-利用平衡公式
- [[蒙特卡洛回报]]: 值头的训练目标
- [[行为克隆]]: Octo 预训练方式
- [[动作分块]]: VLA 输出格式

### 硬件/数据相关

- [[LIBERO]]: 评估 benchmark，5 个 suite
- [[t-SNE]]: 用于可视化 Octo latent space 分布

---

## 速查卡片

> [!summary] V-VLAPS: Value-Guided Planning for VLA Models
> - **核心**: 在冻结 VLA backbone 上加轻量值头（MLP），预测 Monte Carlo 回报来引导 MCTS
> - **方法**: 离线 rollout 收集值目标 + 数据重均衡 + 值引导 MCTS 评分（$\lambda_V \cdot Q_\theta + U_\text{VLAPS}$）
> - **结果**: LIBERO-Object +6pp、LIBERO-10 +4pp（1800s budget），600s 时与 VLAPS 持平
> - **代码**: 未发布

---

*笔记创建时间: 2026-07-11*
