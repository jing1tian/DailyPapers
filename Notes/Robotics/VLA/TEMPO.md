---
title: "TEMPO: Semantic-Action Decoupled RL Post-Training for Vision-Language-Action Models"
method_name: "TEMPO"
authors: [Ziheng Liu, Quantao Yang]
year: 2026
venue: arXiv
tags: [vla, rl-post-training, two-timescale-rl, manipulation, latent-action, robot-learning]
zotero_collection: Robotics/VLA
image_source: online
arxiv_html: https://arxiv.org/html/2608.07314v2
created: 2026-11-14
---

# 论文笔记：TEMPO: Semantic-Action Decoupled RL Post-Training for Vision-Language-Action Models

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | 未披露（匿名投稿） |
| 日期 | August 2026 |
| 项目主页 | [tempo-page](https://anonymous.4open.science/w/tempo-page/) |
| 对比基线 | [[FLOWER]] / [[DeFI]] |
| 链接 | [arXiv](https://arxiv.org/abs/2608.07314) / [Code](https://anonymous.4open.science/r/TEMPO-RL/README.md) |

---

## 一句话总结

> TEMPO 通过将 [[VLA]] 的语义投影层与动作专家解耦为两套独立 TD3 循环并以不同频率更新，有效抑制 [[Latent-Action|隐动作]] 漂移，在 CALVIN ABC→D 基准和真实操作任务上超越所有 RL 后训练基线。

---

## 核心贡献

1. **语义-动作解耦 RL 后训练**: 将 [[VLA]] 的两个可训练模块（语义投影层 & 动作专家）各自配备独立的 [[TD3]] 循环，而非共用一个优化器
2. **双时间尺度更新频率**: 语义投影层更新频率低于动作专家（5:1 比例最优），通过减少语义层更新次数来抑制[[Latent-Action|隐动作]]漂移
3. **Stop-Gradient 隔离**: 动作专家的 TD3 循环对 $z_t$ 施加 $\operatorname{sg}(\cdot)$，确保梯度不回传到语义层，两个循环完全解耦

---

## 问题背景

### 要解决的问题

现有 [[VLA]] 模型在部署后需要针对特定任务做[[强化学习]]后训练（RL post-training），但直接对整个可训练部分（语义投影层 + 动作专家）进行端到端 RL 更新存在性能瓶颈。

### 现有方法的局限

- 以 [[FLOWER]] 为代表的 RL 后训练基线（FLOWER-RL）对语义投影层和动作专家使用相同的更新频率
- 语义层更新过频导致 [[Latent-Action|隐动作]] $z_t$ 偏离预训练分布（漂移量高，FLOWER-RL 的 $D_{\mathrm{drift}}=0.611$），使动作专家的学习变得不稳定
- 已有工作（如 RL-Token、VLA-RL）对更新哪些参数各有侧重，但无人探索不同模块以不同时间尺度协同训练

### 本文的动机

两时间尺度随机近似（Two-Timescale Stochastic Approximation）理论表明，以不同速率更新不同参数块有助于收敛。TEMPO 将此理论引入 VLA RL 后训练：让动作专家高频学习控制反馈，同时让语义层低频更新以保持隐动作稳定，从而两者协同提升。

---

## 方法详解

### 模型架构

TEMPO 基于 [[FLOWER]] 架构，冻结 [[视觉语言模型|VLM]] 主干，仅训练两个轻量级模块：

- **输入**: 语言指令 $l_t$ + 视觉观测 $o_t$
- **Backbone（冻结）**: [[视觉语言模型|VLM]]，输出语境嵌入 $h_t$
- **可训练模块 1**: 语义投影层 $\pi^s_\theta$，将 $h_t$ 映射为[[Latent-Action|隐动作]] $z_t$
- **可训练模块 2**: 动作专家 $\pi^a_\phi$，将 $z_t$ 映射为[[Action Chunking|动作块]] $\mathbf{a}_t$
- **输出**: 可执行动作块 $\mathbf{a}_t$

### 核心模块

#### 模块1: 语义级 TD3 循环（低频更新）

**设计动机**: 利用[[Temporal Difference|时序差分]]学习直接优化 $\pi^s_\theta$，使语义投影层能适应目标任务的隐动作分布，同时更新频率低以减少[[Latent-Action|隐动作]]漂移。

**具体实现**:
- 维护独立语义回放缓冲区 $\mathcal{D}_S = \{(h_t, z_t, r_t, h_{t+1}, d_t)\}$
- 双 critic 网络 $Q^s_1, Q^s_2$ 评估 $(h_t, z_t)$ 对的价值
- Actor $\pi^s_\theta$ 以较低频率（每 $d_s$ 步更新一次）优化语义层

#### 模块2: 动作级 TD3 循环（高频更新）

**设计动机**: 动作专家需要从在线控制反馈中快速学习底层动作模式，高频更新使其充分利用[[经验回放]]数据。

**具体实现**:
- 维护独立动作回放缓冲区 $\mathcal{D}_A = \{(h_t, z_t, \mathbf{a}_t, r_t, h_{t+1}, d_t)\}$
- 双 critic 网络 $Q^a_1, Q^a_2$ 评估 $(z_t, \mathbf{a}_t)$ 对的价值
- Actor $\pi^a_\phi$ 以较高频率（每 $d_a$ 步更新一次）优化动作专家
- **关键**: 对 $z_t$ 施加 $\operatorname{sg}(\cdot)$ 阻止梯度回传至语义层

---

## 关键公式

### 公式1: [[Markov Decision Process|MDP 优化目标]]

$$
\max_{\omega} \mathbb{E}_{\tau \sim \pi_{\omega}} \left[ \sum_{t=0}^{H-1} \gamma^t r_t \right]
$$

**含义**: 在动作块级 MDP 上最大化折扣累积奖励，$\omega = (\theta, \phi)$ 为可训练参数。

**符号说明**:
- $\omega = (\theta, \phi)$: 语义投影层参数 $\theta$ 与动作专家参数 $\phi$ 的合集
- $\tau$: 轨迹采样自策略 $\pi_\omega$
- $H$: 片段长度
- $\gamma$: 折扣因子

### 公式2: [[Latent-Action|策略分解]]

$$
h_t = \mathrm{VL}(o_t, l_t), \qquad z_t = \pi^s_\theta(h_t), \qquad \mathbf{a}_t = \pi^a_\phi(z_t)
$$

**含义**: VLM 提取上下文嵌入，语义投影层生成隐动作，动作专家解码为可执行动作块的三级级联。

**符号说明**:
- $h_t$: VLM 输出的语境嵌入（冻结，不更新）
- $z_t$: 语义投影层输出的[[Latent-Action|隐动作]]
- $\mathbf{a}_t$: 动作专家输出的[[Action Chunking|动作块]]

### 公式3: [[Temporal Difference|语义 Bellman 目标]]

$$
\begin{aligned}
\tilde{z}_{t+1} &= \bar{\pi}^s_\theta(h_{t+1}) \\
y_t^S &= r_t + \gamma(1-d_t) \min_{i=1,2} \bar{Q}^s_i(h_{t+1}, \tilde{z}_{t+1})
\end{aligned}
$$

**含义**: 利用目标网络（带上划线的参数）计算语义层的 TD 目标值，双 critic 取最小以减少过估计。

**符号说明**:
- $\bar{\pi}^s_\theta$: 目标语义投影层（EMA 参数）
- $\bar{Q}^s_i$: 目标语义 critic（EMA 参数）
- $d_t$: 终止标志（1 表示片段结束）

### 公式4: [[TD3|语义 critic 与 actor 损失]]

$$
\mathcal{L}_{Q^s} = \sum_{i=1}^{2} \mathbb{E}_{\mathcal{D}_S} \left[ (Q^s_i(h_t, z_t) - y_t^S)^2 \right]
$$

$$
\mathcal{L}_s = -\mathbb{E}_{h_t \sim \mathcal{D}_S} \left[ Q^s_1(h_t, \pi^s_\theta(h_t)) \right]
$$

**含义**: 语义 critic 以 MSE 拟合 Bellman 目标；语义 actor 以最大化第一个 critic 的 Q 值为目标。

**符号说明**:
- $\mathcal{D}_S$: 语义回放缓冲区
- $y_t^S$: 语义 Bellman 目标

### 公式5: [[Temporal Difference|动作 Bellman 目标]]

$$
\begin{aligned}
\tilde{\mathbf{a}}_{t+1} &= \bar{\pi}^a_\phi(\tilde{z}_{t+1}) \\
y_t^A &= r_t + \gamma(1-d_t) \min_{i=1,2} \bar{Q}^a_i(\tilde{z}_{t+1}, \tilde{\mathbf{a}}_{t+1})
\end{aligned}
$$

**含义**: 利用目标网络计算动作层的 TD 目标值，输入为目标语义投影层产生的隐动作。

**符号说明**:
- $\tilde{z}_{t+1}$: 目标语义投影层在 $h_{t+1}$ 处的输出
- $\bar{Q}^a_i$: 目标动作 critic（EMA 参数）

### 公式6: [[TD3|动作 critic 与 actor 损失（含 stop-gradient）]]

$$
\mathcal{L}_{Q^a} = \sum_{i=1}^{2} \mathbb{E}_{\mathcal{D}_A} \left[ (Q^a_i(z_t, \mathbf{a}_t) - y_t^A)^2 \right]
$$

$$
\mathcal{L}_a = -\mathbb{E}_{h_t \sim \mathcal{D}_A} \left[ Q^a_1(z_t^{\mathrm{sg}}, \pi^a_\phi(z_t^{\mathrm{sg}})) \right], \quad z_t^{\mathrm{sg}} = \operatorname{sg}(\pi^s_\theta(h_t))
$$

**含义**: 动作 critic 以 MSE 拟合 Bellman 目标；动作 actor 对 $z_t$ 施加 stop-gradient，确保梯度不反向传播至语义投影层，实现两个循环的完全解耦。

**符号说明**:
- $z_t^{\mathrm{sg}}$: 对语义投影层输出施加 stop-gradient 后的隐动作
- $\operatorname{sg}(\cdot)$: stop-gradient 算子
- $\mathcal{D}_A$: 动作回放缓冲区

### 公式7: [[Two-Timescale Optimization|双时间尺度更新频率与比例]]

$$
f_a = \frac{N_a}{d_a}, \qquad f_s = \frac{N_s}{d_s}, \qquad \rho = \frac{f_a}{f_s} = \frac{N_a d_s}{N_s d_a}
$$

**含义**: 有效更新频率由梯度步数 $N$ 除以延迟步数 $d$ 决定，$\rho > 1$ 表示动作专家更新比语义层更频繁。

**符号说明**:
- $N_a, N_s$: 动作层、语义层的梯度步数
- $d_a, d_s$: 动作层、语义层的延迟因子（actor 每 $d$ 步更新一次）
- $\rho$: 动作-语义更新频率比，最优值为 5

### 公式8: [[Latent-Action|隐动作漂移度量]]

$$
D_{\mathrm{drift}} = \frac{1}{N} \sum_{i=1}^{N} \|\hat{\mathbf{z}}_i^{\mathrm{trained}} - \hat{\mathbf{z}}_i^{\mathrm{FLOWER}}\|_2
$$

**含义**: 量化 RL 后训练后语义投影层输出相对于预训练 FLOWER 的 L2 偏移量，越小表示隐动作越稳定。

**符号说明**:
- $\hat{\mathbf{z}}_i^{\mathrm{trained}}$: RL 后训练后第 $i$ 个样本的隐动作（归一化）
- $\hat{\mathbf{z}}_i^{\mathrm{FLOWER}}$: 预训练 FLOWER 在同一输入下的隐动作
- $N$: 评估样本数

---

## 关键图表

### Figure 1: Overview / 方法概览

![TEMPO Overview](https://anonymous.4open.science/w/tempo-page/assets/overview.png)

**说明**: TEMPO 保持 VLM 主干冻结，将语义投影层（低频更新）与动作专家（高频更新）解耦为两个独立 TD3 循环，通过 $\rho = f_a/f_s$ 控制更新频率比。

### Figure 2: Framework / 整体框架

![TEMPO Framework](https://anonymous.4open.science/w/tempo-page/assets/framework_overview.png)

**说明**: 展示冻结 VLM、语义投影层、动作专家以及两套独立 TD3 循环（各含独立回放缓冲区和 critic 网络）的完整框架。动作层 actor 更新时对 $z_t$ 施加 $\operatorname{sg}(\cdot)$，实现梯度隔离。

### Figure 3: 隐动作漂移可视化

![TEMPO Latent Action Analysis](https://arxiv.org/html/2608.07314v2/tempo_response_figures/representation_three_trajectories.png)

**说明**: 在三条 FLOWER 失败轨迹上对比 FLOWER、FLOWER-RL 和不同 $\rho$ 下的 TEMPO 的隐动作分布（PCA + t-SNE 投影，颜色表示环境步数）。高 $\rho$ 下 TEMPO 的隐动作分布更接近预训练 FLOWER，漂移更小。

### Figure 4a: 真实机器人在线 RL 奖励曲线（Block Sliding）

![TEMPO Real-world Online RL](https://anonymous.4open.science/w/tempo-page/assets/online_reinforcement.png)

**说明**: 在 Block Sliding 任务中，TEMPO（绿）在在线 RL 后期达到并维持更高奖励，FLOWER-RL（红）奖励在前期上升但后期稳定在更低水平。

### Figure 4b: 真实机器人操作对比（Block Sliding）

![TEMPO Real-world Comparison](https://anonymous.4open.science/w/tempo-page/assets/realworld_comparison.png)

**说明**: FLOWER-RL 仅将方块推向抽屉口但未成功推入；TEMPO 先打开抽屉再将方块滑入，演示了 TEMPO 在语义理解层面保持更完整的操作规划。

### Table I: CALVIN ABC→D 基准测试

| 方法 | SR1 | SR2 | SR3 | SR4 | SR5 | Avg. Len. |
|------|-----|-----|-----|-----|-----|-----------|
| OpenVLA | — | — | — | — | — | 3.27 |
| π0 | — | — | — | — | — | 3.84 |
| π0.5 | — | — | — | — | — | 3.85 |
| FLOWER | — | — | — | — | — | 4.49 |
| DeFI | — | — | — | — | 81.2% | 4.51 |
| FLOWER-RL | — | — | — | — | 78.70% | 4.51 |
| **TEMPO（ours）** | **100.0%** | **97.1%** | **92.9%** | **87.1%** | **81.7%** | **4.59** |

**表格说明**: TEMPO 在 SR5 和 Avg.Len 上均达到 SOTA，优于 RL 后训练基线 FLOWER-RL 和 DeFI。

### Table II: 组件消融实验

| 配置 | Avg. Len. | 说明 |
|------|-----------|------|
| w/o 语义投影更新 | 4.54 | 移除低频语义层 TD3 循环 |
| w/o 动作专家更新 | 4.55 | 移除高频动作层 TD3 循环 |
| **完整 TEMPO** | **4.59** | 两套循环均启用 |

**关键发现**: 两个 TD3 循环均有贡献，单独移除任意一个都会导致性能下降，说明语义层与动作层的协同训练不可缺失。

### Table III: 更新频率比 ρ 对 SR5 的影响

| 更新比 $\rho$（动作:语义）| SR5 |
|--------------------------|-----|
| 1:1 | ~79.5% |
| 5:1 | **81.7%** |
| 10:1 | ~81.5% |

**关键发现**: $\rho = 5$ 时 SR5 最高；1:1（等频更新）接近或低于 FLOWER baseline，验证了低频语义更新的重要性。

### Table IV: 更新频率比 ρ 对 Avg.Len 的影响

| 更新比 $\rho$（动作:语义）| Avg. Len. |
|--------------------------|-----------|
| 1:1 | 4.46 |
| 5:1 | **4.59** |
| 10:1 | 4.57 |

**关键发现**: 过低（1:1）或过高（10:1）的频率比均次优，5:1 为最优平衡点。

### Table V: 隐动作漂移分析

| 方法/配置 | $D_{\mathrm{drift}}$ | 目标任务成功率 |
|-----------|----------------------|--------------|
| FLOWER-RL | 0.611 | 68% |
| TEMPO（$\rho=4:1$）| 0.409 | 98% |
| TEMPO（$\rho=5:1$）| — | 96% |

**关键发现**: TEMPO 在 $\rho=4:1$ 时漂移量降低 33%（0.409 vs 0.611），目标任务成功率从 68% 提升至 98%，直接验证了抑制隐动作漂移的有效性。

### Table VI: RL 后训练任务集大小消融

| 任务集大小 | Avg. Len. |
|-----------|-----------|
| 1 任务 | 4.59（最优）|
| 多任务 | 4.54–4.57 |

**关键发现**: 增加后训练任务数量并不单调提升性能，单任务反而表现最佳，说明过多任务可能引入干扰。

### Table VII: 架构泛化性（VLA-Adapter-Pro）

| 方法 | SR5 |
|------|-----|
| VLA-Adapter-Pro（基线）| 77.23% |
| VLA-Adapter-Pro + TEMPO | **79.17%** |

**关键发现**: TEMPO 的双时间尺度框架可泛化到不同 VLA 架构，在 VLA-Adapter-Pro 上同样实现稳定提升。

---

## 实验

### 数据集

| 数据集/环境 | 规模 | 特点 | 用途 |
|------------|------|------|------|
| CALVIN ABC→D | ABC 训练，D 测试 | 跨分布长时序操控，5 步连续任务 | 主要 benchmark |
| 真实机器人（3 任务）| 60 条 demo/任务 | Synria Alicia-D 6-DoF 机械臂，多阶段操作 | 真实世界验证 |

### 实现细节

- **基础模型**: [[FLOWER]]（VLM 冻结，语义投影层 + 动作专家可训练）
- **RL 算法**: [[TD3]]（双 critic + actor 延迟更新）
- **更新频率比**: $\rho = 5$（最优）
- **相机**: 第三视角 Intel RealSense D435i + 腕部 D405
- **预训练数据**: 每任务 60 条演示（真实实验），总计 120 条
- **评估**: 每 checkpoint 20 次试验 × 3 种子，取均值（误差带为 1 个标准差）
- **稀疏奖励**: 二值成功奖励

### 可视化结果

- 在 FLOWER 失败的三条轨迹上，TEMPO（$\rho=4:1$ 或 $5:1$）的 PCA/t-SNE 隐动作投影与预训练 FLOWER 高度重合，而 FLOWER-RL 明显偏离
- 真实机器人 Car Retrieval 任务中，FLOWER-RL 无法完成取车并放置，TEMPO 能完整执行打开抽屉→取出玩具车→放置到桌面的完整流程

---

## 批判性思考

### 优点

1. **理论动机明确**: [[Two-Timescale Optimization|双时间尺度随机近似]]有成熟理论支撑，动机充分
2. **实现简洁**: 在 FLOWER 框架上仅添加两套独立 TD3 循环，无需改变网络结构
3. **泛化性好**: 在 VLA-Adapter-Pro 上同样有效，不局限于特定 VLA 架构

### 局限性

1. **基础模型依赖**: TEMPO 依赖具有明确隐动作接口的 VLA 架构（如 FLOWER），对无此设计的模型（如端到端 token 生成式 VLA）适用性待验证
2. **超参数敏感性**: 最优 $\rho$ 值（5:1）可能随任务和基础模型变化，需要额外调参
3. **任务集规模实验结论反直觉**: 单任务后训练最优与直觉不符，多任务泛化性有待深入分析

### 潜在改进方向

1. 自适应调整 $\rho$ 值（如根据隐动作漂移量动态调整更新频率）
2. 探索三级以上的时间尺度分层（如 VLM 解冻 fine-tune 也以极低频率参与更新）

### 可复现性评估

- [x] 代码开源（匿名链接）
- [ ] 预训练模型
- [x] 训练细节完整（消融 + 超参数）
- [x] 数据集可获取（CALVIN 公开）

---

## 关联笔记

### 基于

- [[FLOWER]]: 基础 VLA 模型，提供语义投影层 + 动作专家的两阶段架构
- [[TD3]]: Twin Delayed Deep Deterministic Policy Gradient，TEMPO 两套循环均采用此算法

### 对比

- [[DeFI]]: CALVIN 上的强 baseline，SR5 81.2%，TEMPO 以 81.7% 超越
- [[FLOWER]]: 预训练基线，Avg.Len 4.49，TEMPO（4.59）显著提升

### 方法相关

- [[Two-Timescale Optimization]]: 核心理论基础，不同参数块以不同学习速率更新
- [[Latent-Action]]: 语义投影层输出的中间表示，TEMPO 的关键保护对象
- [[Action Chunking]]: 动作专家输出格式，TEMPO 在动作块级建模 MDP
- [[Temporal Difference]]: TD 学习，两套 critic 均使用 Bellman 方程更新

### 硬件/数据相关

- [[Intel RealSense D435i]]: 第三视角 RGB-D 相机
- [[CALVIN]]: 多任务机器人操作 benchmark，4 个语言引导任务的长序列评估

---

## 速查卡片

> [!summary] TEMPO (arXiv 2608.07314)
> - **核心**: 语义层低频更新 + 动作层高频更新，两套独立 TD3 循环，抑制隐动作漂移
> - **方法**: 冻结 VLM，$\rho=5:1$ 双时间尺度 RL 后训练（FLOWER 架构）
> - **结果**: CALVIN SR5=81.7%、Avg.Len=4.59，真实机器人 3 任务全部优于 FLOWER-RL
> - **代码**: https://anonymous.4open.science/r/TEMPO-RL/README.md

---

*笔记创建时间: 2026-11-14*
