---
title: "DyGRO-VLA: Cross-Task Scaling of Vision-Language-Action Models via Dynamic Grouped Residual Optimization"
method_name: "DyGRO-VLA"
authors: [Sixu Lin, Yunpeng Qing, Litao Liu, Ming Zhou, Ruixing Jin, Xiaoyi Fan, Guiliang Liu]
year: 2026
venue: arXiv
tags: [vla, reinforcement-learning, multi-task-learning, mixture-of-experts, robot-manipulation, information-bottleneck, residual-policy]
zotero_collection: 3-Robotics/VLA
image_source: online
arxiv_html: https://arxiv.org/html/2605.17486
created: 2026-05-20
---

# 论文笔记：DyGRO-VLA

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | 未公开（论文中未明确列出机构） |
| 日期 | May 2026 |
| 项目主页 | N/A |
| 对比基线 | [[pi0]] / [[OpenVLA]] / [[SpatialVLA]] |
| 链接 | [arXiv](https://arxiv.org/abs/2605.17486) / Code N/A |

---

## 一句话总结

> DyGRO-VLA 通过两阶段框架（离线信息瓶颈表示学习 + 在线混合残差 RL 微调）解决多任务 VLA 强化学习中灾难性遗忘问题，在 LIBERO 上达到 97.1% 平均成功率。

---

## 核心贡献

1. **跨任务泛化分析**: 首次系统性实证分析 RL 后训练（RFT）在多任务 [[VLA]] 场景下导致灾难性遗忘的机制，通过 t-SNE 可视化揭示特征空间漂移现象。
2. **离线跨任务表示学习**: 基于[[信息瓶颈]]（Information Bottleneck）原理，在离线阶段学习压缩且行动相关的跨任务潜在表示，过滤无关干扰信息。
3. **动态分组残差优化（DyGRO）**: 在线阶段引入[[混合专家]]（MoE）残差结构，通过动态路由将任务分配给专门的 RL 残差专家，避免梯度冲突。

---

## 问题背景

### 要解决的问题

[[VLA]] 模型经过 RL 后训练（RFT）后，单任务性能提升但多任务泛化能力严重下降——针对任务 A 微调后，任务 B 的成功率往往大幅滑落（灾难性遗忘）。随着任务数量增加，RFT 变得越来越不稳定。

### 现有方法的局限

- 现有 VLA-RL 方法（TGRPO、GRAPE、VLA-RL 等）专注提升单个任务的 RL 优化效果，忽视多任务间的特征共享与梯度冲突问题。
- 简单地对所有任务联合做 RL 微调时，任务梯度相互干扰（余弦相似度为负），导致共享表示崩溃。
- [[混合专家]] 路由策略在 NLP 等领域有所研究，但在 VLA 多任务 RL 场景中尚未被有效利用。

### 本文的动机

跨任务共享的紧凑潜在表示是泛化能力的关键。RFT 破坏了这种共享结构。因此，若能在离线阶段固化跨任务表示，再在线阶段通过残差 MoE 进行任务特异性精细化，即可同时保留泛化能力与提升灵巧度。

---

## 方法详解

### 模型架构

DyGRO-VLA 采用**两阶段训练**架构：

- **输入**: 语言指令 $l$ + 多视角 RGB 观测 $o_t$ + 本体感知状态 $s_t$
- **视觉 Backbone**: [[DINOv2]] 提取图像特征，[[SigLIP]] 进行视觉-语言对齐
- **语言 Backbone**: [[Qwen2.5]]-0.5B LLM 处理指令 Token
- **核心模块 1（离线）**: [[信息瓶颈]] 目标学习压缩潜在表示 $z$
- **核心模块 2（在线）**: [[混合专家|混合残差专家]]（Mixture-of-RL-Residuals, MoRR）动态路由
- **输出**: [[Action Chunking|动作块]] $a_{t:t+k}$，基础策略输出叠加残差修正

### 核心模块

#### 模块 1: 离线跨任务表示学习（IB 阶段）

**设计动机**: 利用[[信息瓶颈]]原理，在保留行动预测能力的同时，压缩观测中的无关信息，学习对所有任务通用的紧凑表示 $z$。

**具体实现**:
- 使用神经网络 critic $T_\psi(o, z)$ 估计观测 $o$ 与潜在码 $z$ 之间的互信息上界
- 通过[[对比学习]]（Contrastive Learning）对齐任务嵌入 $e_\zeta$ 与任务级潜在表示 $z_T$
- 难度感知采样（Difficulty-Aware Sampling）让欠解决任务获得更多训练机会
- 离线训练完成后，冻结 VLM Backbone，防止在线阶段破坏已学表示

#### 模块 2: 在线混合残差 RL 优化（MoRR 阶段）

**设计动机**: 不同任务需要不同的精细化策略；通过[[混合专家]]路由，每个专家专注子任务集合，从根本上避免跨任务梯度冲突。

**具体实现**:
- 设置 $N=8$ 个独立残差 RL 策略（专家），每个专家 $\pi^\Delta_{i,\psi}$ 预测残差动作 $\tilde{a}_i \in [-\alpha, \alpha]$
- 动态路由器基于任务嵌入 $z_T$ 产生 Top-m 稀疏门控权重 $\omega_i(z_T)$
- 最终动作 = 基础策略输出 $a_\text{base}$ + 加权残差 $\Delta a$
- [[负载均衡正则化]]防止专家崩塌（所有请求集中到单一专家）
- [[Cal-QL]] 校准正则器平滑离线-在线过渡，使用 h 步 Bellman 回归

---

## 关键公式

### 公式 1: [[强化学习|多任务 RL 目标]]

$$
\pi^* = \arg\max_\pi \, \mathbb{E}_{\zeta \sim p(\zeta)} \mathbb{E}_{\tau \sim \pi, M^\mathcal{T}} \left[ \sum_{t=0}^{\infty} \gamma^t r_t \right]
$$

**含义**: 跨所有任务 $\zeta$ 的期望累积折扣回报最大化，是 DyGRO-VLA 的优化总目标。

**符号说明**:
- $\pi^*$: 最优策略
- $\zeta \sim p(\zeta)$: 从任务分布中采样任务
- $\tau$: 在任务 MDP $M^\mathcal{T}$ 下执行策略 $\pi$ 产生的轨迹
- $\gamma$: 折扣因子
- $r_t$: 时刻 $t$ 的即时奖励

---

### 公式 2: [[信息瓶颈|变分信息瓶颈目标]]

$$
\mathcal{L}_\text{base} = \mathbb{E}_{p_\theta(z|o)} \left[ -\log \pi_\theta(a|z) \right] + \lambda_\text{IB} \left[ \mathbb{E}_{P_{OZ}} \left[ T_\psi(o,z) \right] - \log \mathbb{E}_{P_O P_Z} \left[ e^{T_\psi(o,z)} \right] \right]
$$

**含义**: 第一项最大化动作预测似然（保留任务相关信息），第二项通过 MINE 估计压缩观测与潜在码的互信息（丢弃无关信息）。

**符号说明**:
- $z$: 潜在表示
- $o$: 观测
- $a$: 动作
- $T_\psi(o,z)$: 神经网络 critic，用于估计 $I(Z;O)$ 的上界
- $\lambda_\text{IB}$: 信息压缩强度超参数
- $P_{OZ}$: 联合分布；$P_O P_Z$: 边际分布之积

---

### 公式 3: [[对比学习|任务嵌入对比损失]]

$$
\mathcal{L}_\text{CL}(\psi) = -\mathbb{E}_{(z_T, \zeta)} \left[ \log \frac{\exp(\text{sim}(z_T, e_\zeta)/\tau)}{\sum_{\zeta'=1}^{N} \exp(\text{sim}(z_T, e_{\zeta'})/\tau)} \right]
$$

**含义**: InfoNCE 形式的对比损失，使同任务的潜在表示 $z_T$ 与任务原型 $e_\zeta$ 靠近，不同任务远离，从而学习可区分的任务嵌入空间。

**符号说明**:
- $z_T$: 任务级潜在表示（聚合跨时刻的特征）
- $e_\zeta$: 任务 $\zeta$ 的可学习原型嵌入
- $\tau$: 温度系数
- $\text{sim}(\cdot,\cdot)$: 点积相似度

---

### 公式 4: [[混合专家|混合残差动作]]

$$
\Delta a = \sum_{i=1}^{m} \omega_i(z_T) \, \tilde{a}_i \quad \text{s.t.} \quad \tilde{a}_i \sim \pi^\Delta_{i,\psi}(\tilde{a}_i | z, a_\text{base})
$$

**含义**: 最终残差修正量是 Top-m 激活专家的加权和；每个专家在基础策略输出 $a_\text{base}$ 的条件下生成有界残差 $\tilde{a}_i \in [-\alpha, \alpha]$。

**符号说明**:
- $m$: Top-m 稀疏激活专家数量
- $\omega_i(z_T)$: 动态路由器给第 $i$ 个专家分配的门控权重
- $\tilde{a}_i$: 第 $i$ 个专家产生的残差动作
- $a_\text{base}$: 冻结基础策略的输出动作

---

### 公式 5: [[负载均衡正则化|专家负载均衡正则化]]

$$
\mathcal{L}_\text{LB}(\psi) = \sum_{i=1}^{N} \hat{\omega}_{i,\text{avg}} \log(\hat{\omega}_{i,\text{avg}} + \epsilon)
$$

**含义**: 熵形式的正则化项，鼓励所有专家均匀使用，防止路由崩塌为单一专家。

**符号说明**:
- $\hat{\omega}_{i,\text{avg}}$: 第 $i$ 个专家在当前 batch 的平均门控权重
- $\epsilon$: 数值稳定常数

---

### 公式 6: [[Actor-Critic|在线训练总目标]]

$$
\mathcal{L}_\pi(\psi) = \mathcal{L}_\text{RL}(\psi) + \lambda_\text{CL} \mathcal{L}_\text{CL}(\psi) + \lambda_\text{LB} \mathcal{L}_\text{LB}(\psi)
$$

**含义**: 在线阶段的 Actor 总损失，同时优化 RL 策略性能、任务嵌入判别性和专家负载均衡。

**符号说明**:
- $\mathcal{L}_\text{RL}$: Actor-Critic RL 损失（基于 Q 值）
- $\lambda_\text{CL}, \lambda_\text{LB}$: 对比学习和负载均衡损失权重

---

### 公式 7: [[h步Bellman回归|h 步 Bellman 回归]]

$$
\mathcal{L}_\text{TD}(\theta_i) = \mathbb{E}_{(s_t,a_t,r_t^{(h)},s_{t+h}) \sim \mathcal{D}} \left[ \left( Q_{\theta_i}(s_t,a_t) - \mathcal{B}^\pi \bar{Q}_\text{min}(s_t,a_t) \right)^2 \right]
$$

$$
\mathcal{B}^\pi \bar{Q}_\text{min}(s_t,a_t) = r_t^{(h)} + \gamma^h \mathbb{E}_{a' \sim \pi(\cdot|s_{t+h})} \left[ \bar{Q}_\text{min}(s_{t+h}, a') \right]
$$

**含义**: 使用 h 步累积回报的 TD 误差训练 Critic，结合集成最小 Q 值提升稳定性，适配[[Action Chunking|动作块]]预测的多步结构。

**符号说明**:
- $r_t^{(h)} = \sum_{i=0}^{h-1} \gamma^i r_{t+i}$: h 步折扣累积回报
- $\bar{Q}_\text{min}$: 目标 Critic 集成的最小 Q 值
- $K=4$: Critic 集成数量

---

## 关键图表

### Figure 1: 灾难性遗忘现象

![Figure 1: 针对单一任务 RFT 导致其他任务性能下降](https://arxiv.org/html/2605.17486v1/figures/rft.png)

**说明**: 单任务 RFT 在目标任务上提升成功率，但对无关任务造成持续且严重的性能退化，直观展示了灾难性遗忘的存在。

---

### Figure 2: 多任务冲突随任务数扩展

![Figure 2: 任务数增加时 RFT 稳定性下降](https://arxiv.org/html/2605.17486v1/figures/success_curve.png)

**说明**: 随着联合 RFT 任务数量增加，平均成功率不升反降，说明简单的多任务 RL 无法扩展，需要专门的跨任务优化机制。

---

### Figure 3: t-SNE 特征空间可视化

![Figure 3a: RFT 前的特征分布（跨任务共享）](https://arxiv.org/html/2605.17486v1/figures/tsne_sft.png)

![Figure 3b: RFT 后的特征分布（任务孤立）](https://arxiv.org/html/2605.17486v1/figures/tsne_rft.png)

**说明**: 左图为 RFT 前——40 个任务的嵌入形成连续分布，说明存在跨任务共享结构；右图为 RFT 后——特征空间碎片化为孤立簇，跨任务泛化能力丧失。

---

### Figure 4: DyGRO-VLA 方法总览

![Figure 4: 两阶段训练流水线](https://arxiv.org/html/2605.17486v1/x1.png)

**说明**: 左侧为离线阶段——VLM Backbone 通过[[信息瓶颈]]目标学习压缩表示；右侧为在线阶段——冻结 Backbone，通过[[混合专家|混合残差专家]]（MoRR）动态路由精细化每个任务的策略。

---

### Figure 5: 梯度冲突热力图

![Figure 5: 单 MLP vs MoRR 的任务间梯度余弦相似度对比](https://arxiv.org/html/2605.17486v1/figures/gradient_conflicts.png)

**说明**: 单 MLP 头（左）任务间大量蓝色方块（负余弦相似度），即梯度冲突严重；MoRR（右）红色区域增多，专家路由有效隔离了冲突梯度。

---

### Figure 6: 在线微调策略消融

![Figure 6: 不同在线训练策略的成功率曲线](https://arxiv.org/html/2605.17486v1/figures/online_ablation.png)

**说明**: 对比不同在线策略在 LIBERO 上随迭代次数变化的成功率，DyGRO-VLA 曲线最高且最稳定，虚线为离线基线。

---

### Figure 7: 真实机器人平台

![Figure 7: 真实世界机器人操作平台](https://arxiv.org/html/2605.17486v1/figures/real2.png)

**说明**: 用于 Sim2Real 验证的真实世界机器人平台，包含 4 个 RoboTwin 操作任务。

---

### Table 1: LIBERO 基准测试成功率（%）

| 方法 | Spatial | Object | Goal | Long | Avg. |
|------|---------|--------|------|------|------|
| Diffusion Policy | 59.6 | 73.8 | 51.6 | 41.0 | 56.5 |
| MT-ACT | 50.2 | 72.0 | 60.2 | 50.2 | 58.2 |
| Octo | 78.7 | 85.5 | 84.2 | 51.0 | 74.9 |
| OpenVLA | 82.1 | 87.3 | 77.4 | 51.8 | 74.7 |
| SpatialVLA | 88.2 | 89.9 | 78.6 | 55.5 | 78.1 |
| π₀-FAST | 96.4 | 96.8 | 88.6 | 60.2 | 85.5 |
| π₀ | 98.0 | 96.8 | 94.4 | 88.4 | 94.4 |
| DyGRO-VLA (SFT) | 95.4 | 96.0 | 93.8 | 85.0 | 92.6 |
| DyGRO-VLA (Offline) | 95.6 | 96.0 | 94.0 | 85.2 | 92.7 |
| **DyGRO-VLA** | **97.6** | **98.6** | **97.2** | **95.0** | **97.1** |
| Δ vs Offline | +2.0 | +2.6 | +3.2 | **+9.8** | +4.4 |

**关键发现**: DyGRO-VLA 在所有子任务上超越离线基线，尤其在长时序 LIBERO-Long 上提升最大（+9.8%），在 Object 子集上甚至超过 π₀（98.6% vs 96.8%）。

---

### Table 2: LIBERO-Long RL 微调方法对比

| 方法 | LIBERO-Long 成功率 |
|------|------------------|
| TGRPO | 59.2% |
| GRAPE | 57.2% |
| VLA-RL | 59.8% |
| World-Env | 57.8% |
| RIPT-VLA | 93.8% |
| SimpleVLA-RL | 91.7% |
| RLinf | 94.0% |
| **DyGRO-VLA** | **95.2%** |

**关键发现**: DyGRO-VLA 在专注单任务优化的 RL 方法中同样排名第一，超过 RLinf（94.0%）和 RIPT-VLA（93.8%）。

---

### Table 3: RoboTwin2 仿真与真实世界成功率（%）

| 方法 | Beat Block Hammer |  | Pick Dual Bottles |  | Stack Bowls Two |  | Place Empty Cup |  | Avg |  |
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| | Sim | Real | Sim | Real | Sim | Real | Sim | Real | Sim | Real |
| OpenVLA-oft (SFT) | 54.0 | 15.0 | 32.0 | 25.0 | 88.0 | 70.0 | 50.0 | 60.0 | 56.0 | 42.5 |
| OpenVLA-oft (RFT) | 71.9 | 30.0 | 54.0 | 40.0 | 92.0 | 90.0 | 96.1 | 60.0 | 78.5 | 55.0 |
| **DyGRO-VLA** | **72.2** | 30.0 | **57.0** | 40.0 | 90.4 | 90.0 | **97.0** | **70.0** | **79.2** | **57.5** |

**关键发现**: 仿真平均成功率 79.2%，真实世界 57.5%，在 Place Empty Cup（最精细任务）上真实世界比 OpenVLA-RFT 高 10 个百分点。

---

### Table 4: 专家数量消融

| 方法 | Spatial | Object | Goal | Long | Avg. |
|------|---------|--------|------|------|------|
| DyGRO-VLA (1 expert) | 95.2 | 96.4 | 93.6 | 88.2 | 93.4 |
| DyGRO-VLA (4 experts) | 97.4 | 98.8 | 96.8 | 95.0 | 97.0 |
| **DyGRO-VLA (8 experts)** | **97.6** | **98.6** | **97.2** | **95.0** | **97.1** |

**关键发现**: 8 个专家为最优配置，从 1→4 专家提升显著（93.4%→97.0%），4→8 专家边际收益递减。

---

### Table 5: 组件消融实验

| 配置 | Spatial | Object | Goal | Long | Avg. |
|------|---------|--------|------|------|------|
| **DyGRO-VLA (Full)** | **97.6** | **98.6** | **97.2** | **95.0** | **97.1** |
| w/o IB Objective | 97.4 | 98.2 | 97.0 | 94.6 | 96.8 |
| w/o CL Objective | 95.2 | 97.2 | 94.2 | 90.4 | 94.3 |
| w/o DA Sampling | 97.4 | 98.4 | 97.0 | 93.0 | 96.4 |

**关键发现**: 对比学习目标贡献最大（去除后下降 2.8%），难度感知采样次之（-0.7%），信息瓶颈贡献最小但仍不可缺（-0.3%）。

---

## 实验结果

### 数据集与基准

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| LIBERO | 5 suites, 130 tasks | 多样化桌面操作任务，含长时序 LIBERO-Long | 主要训练/测试 |
| RoboTwin2.0 | 4 tasks | 仿真+真实世界 Sim2Real 验证 | 跨域泛化测试 |

### 实现细节

- **视觉 Backbone**: [[DINOv2]] + [[SigLIP]]
- **语言 Backbone**: [[Qwen2.5]]-0.5B
- **Critic 集成**: K=4 个独立 Q 网络
- **专家数量**: N=8
- **关键超参**: $\lambda_\text{IB}$ 控制压缩强度，$\lambda_\text{CL}$、$\lambda_\text{LB}$ 控制辅助损失权重

### 核心实验结论

- **LIBERO 总体**: 97.1% 平均成功率，较离线基线 +4.4%，较 π₀ +2.7%
- **LIBERO-Long（最难）**: 95.0%，提升最大 +9.8%，说明方法在长时序任务上效果显著
- **RoboTwin2 仿真**: 79.2%，**真实世界**: 57.5%（Sim2Real 迁移有效）
- **跨任务保持**: 在多任务场景下不发生灾难性遗忘，对比其他 RL 方法有明显优势

---

## 批判性思考

### 优点

1. **问题定义清晰**: 通过 t-SNE 和梯度冲突热力图对多任务 RL 失效机制的实证分析严谨，说服力强。
2. **设计与理论匹配**: 信息瓶颈目标与残差 MoE 的组合在直觉和理论上都有坚实依据，信息论框架给出了优化方向。
3. **实验覆盖全面**: 同时在 LIBERO 和 RoboTwin2 双基准上验证，还包括真实机器人 Sim2Real 测试，实验说服力较强。

### 局限性

1. **在线训练敏感性**: 在线 RL 阶段对超参数（特别是 $\lambda_\text{IB}$、学习率）敏感，工程调参成本高。
2. **任务依赖路由**: 路由机制依赖任务 ID 或任务嵌入，在推理阶段需要明确的任务标识，对任务未知场景适应性差。
3. **真实世界评估规模有限**: 只有 4 个 RoboTwin 任务的真实机器人实验，泛化到更多真实场景的能力尚不明确。

### 潜在改进方向

1. **无任务标签路由**: 探索基于观测内容自动推断任务归属的路由机制，移除对显式任务 ID 的依赖。
2. **移动操作扩展**: 将方法拓展到移动机器人（locomotion + manipulation）场景，论文已列为未来工作。

### 可复现性评估

- [ ] 代码开源（论文中未提供 GitHub 链接）
- [ ] 预训练模型（未发布）
- [x] 训练细节完整（论文中描述较详细）
- [x] 数据集可获取（LIBERO、RoboTwin2 均为公开基准）

---

## 关联笔记

### 基于

- [[pi0]]: 基础 VLA 模型，DyGRO-VLA 以其 SFT 版本为出发点进行离线预训练
- [[信息瓶颈]]: 离线阶段表示学习的核心理论框架
- [[Cal-QL]]: 离线到在线 RL 过渡的校准 Q 学习方法

### 对比

- [[OpenVLA]]: 主要对比基线之一（RoboTwin2 上的 SFT/RFT 版本）
- [[SpatialVLA]]: LIBERO 基准中的强力竞争者（78.1%）
- [[RIPT-VLA]]: LIBERO-Long 上最接近的 RL 微调竞争者（93.8%）

### 方法相关

- [[混合专家]]: 核心架构组件（MoRR 模块）
- [[对比学习]]: 任务嵌入学习方法
- [[Actor-Critic]]: 在线 RL 优化框架
- [[残差策略学习]]: DyGRO 设计思想来源
- [[负载均衡正则化]]: 防止专家崩塌的关键机制

### 硬件/数据相关

- [[LIBERO]]: 主要评测基准（5 suites, 130 tasks）
- [[RoboTwin2]]: Sim2Real 评测平台

---

## 速查卡片

> [!summary] DyGRO-VLA (2026)
> - **核心**: 用 IB 离线学跨任务表示 + MoRR 在线残差 RL 微调，解决多任务 VLA 的灾难性遗忘
> - **方法**: 两阶段：离线 IB 表示学习（冻结 Backbone）+ 在线 N=8 残差 MoE 动态路由
> - **结果**: LIBERO 97.1%（+4.4% vs 离线基线），LIBERO-Long 95.0%（+9.8%），RoboTwin2 真实 57.5%
> - **代码**: 未公开

---

*笔记创建时间: 2026-05-20*
