---
title: "Delta-JEPA: Learning Action-Sensitive World Models via Latent Difference Decoding"
method_name: "Delta-JEPA"
authors: [Zhenghao Zhang, Yuanxiang Wang, Zhenyu Guan, Yujia Yang, Bingkang Shi, Tianyu Zong, Hongzhu Yi, Guoqing Chao, Xingchen Chen, Tiankun Yang, Chenxi Bao, Tao Yu, Jingjing Zhou, Jungang Xu]
year: 2026
venue: arXiv
tags: [world-model, jepa, representation-learning, planning, self-supervised-learning, continuous-control, action-sensitivity]
zotero_collection: 3-Robotics/World Model
image_source: online
arxiv_html: https://arxiv.org/html/2606.31232v1
created: 2026-07-02
---

# 论文笔记：Delta-JEPA — Learning Action-Sensitive World Models via Latent Difference Decoding

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | 未详细列出（多机构合作，共 14 位作者） |
| 日期 | June 2026 |
| 项目主页 | — |
| 对比基线 | [[LeWM]] · [[PLDM]] · [[Sub-JEPA]] |
| 链接 | [arXiv](https://arxiv.org/abs/2606.31232) / [HTML](https://arxiv.org/html/2606.31232v1) |

---

## 一句话总结

> Delta-JEPA 提出 [[LDAD|潜在差分动作解码器（LDAD）]]，通过监督相邻潜变量之差 $\Delta z_t$ 来重建动作，使 [[JEPA]] 世界模型在无像素重建、无冻结编码器的情况下同时避免 [[Representation Collapse|表示坍缩]] 并对动作保持高度敏感，在四个视觉连续控制任务上规划成功率全面领先。

---

## 核心贡献

1. **[[LDAD|潜在差分动作解码器（LDAD）]]**: 不从拼接状态 $[z_t, z_{t+1}]$ 重建动作，而是仅从潜变量差 $\Delta z_t = z_{t+1} - z_t$ 解码动作，消除对单状态线索的依赖，驱动编码器学习动作敏感的转移几何。
2. **无需辅助正则化的端到端训练**: 仅用前向预测损失 $\mathcal{L}_\text{pred}$ 与动作重建损失 $\mathcal{L}_\text{action}$ 两项目标，无冻结编码器、无 EMA、无 [[VICReg]] 式多项分布匹配，即可防止 [[Representation Collapse|坍缩]]。
3. **多步动作解码扩展**: LDAD 以 $z_{t+N} - z_t$ 为输入、通过含可学习 Action Query 的 [[Transformer]] 解码 $N$ 步动作序列，原生支持 Action Chunking 场景。

---

## 问题背景

### 要解决的问题

在无奖励的离线数据集上，仅从原始像素学习紧凑的、**对执行动作敏感**的潜在动态模型，并用它进行规划，无需像素重建。

### 现有方法的局限

- **像素重建世界模型**（[[DreamerV3]]、[[PlaNet]]）：浪费容量在视觉细节上，成本高昂。
- **[[JEPA]] 类潜变量预测**（[[I-JEPA]]、[[V-JEPA]]）：端到端训练易发生 [[Representation Collapse|表示坍缩]]。
- **[[LeWM]]**：用 SIGReg 高斯正则化防止坍缩，但未显式约束潜空间对动作敏感。
- **[[PLDM]]**：用 VICReg + 逆动力学从 $[z_t, z_{t+1}]$ 拼接特征重建动作，但拼接方式允许模型利用单帧线索绕过对转移位移的建模。
- **[[DINO-WM]]**：冻结 DINOv2 特征避免坍缩，代价是无法针对任务自适应。

### 本文的动机

如果强制从 $\Delta z_t$（而非 $z_t$ 或 $[z_t, z_{t+1}]$）解码动作，则网络必须将动作信息编码进**转移方向**中：
- 若编码器坍缩，$\Delta z_t$ 趋近零向量，无法恢复动作 → 自然防坍缩。
- 因仅依赖位移，消除了对 $z_t$ 本身绝对位置的快捷依赖。
- 不同动作必须产生可区分的潜在位移 → 规划时可利用位移方向导航。

---

## 方法详解

### 问题设置

离线数据集 $\mathcal{D} = \{(o_1, a_1, \ldots, o_T)\}$，含图像观测 $o_t \in \mathbb{R}^{C \times H \times W}$ 与连续动作 $a_t \in \mathbb{R}^{d_a}$，**无奖励信号**，由未知行为策略采集。

目标：学习紧凑潜空间 $\mathcal{Z} \subseteq \mathbb{R}^d$，以及动作敏感的潜在动力学预测器，无需像素重建。

### 模型架构概览

Delta-JEPA 采用三组件 [[JEPA]] 架构：

- **编码器** $f_\theta$：[[ViT]]-Tiny，将图像 $o_t$ 映射为潜变量 $z_t = f_\theta(o_t)$
- **动力学预测器** $P_\phi$：6 层因果 [[Transformer]]（16 头，64 维，MLP 宽度 2048），预测下一潜变量 $\hat{z}_{t+1} = P_\phi(z_t, a_t)$
- **[[LDAD|潜在差分动作解码器（LDAD）]]** $D_\Theta$：3 层非因果 [[Transformer]]（8 头，64 维，FFN 512），从 $\Delta z_t$ 重建动作

训练完全端到端，**无冻结模块、无 stop-gradient、无分布匹配正则项**。

### 核心模块

#### 模块 1：前向潜在动力学预测器

**设计动机**：学习如何根据当前状态和动作预测未来潜变量，是利用世界模型进行规划的基础。

**具体实现**：
- 编码器 $f_\theta$ 输出 $z_t$
- 因果 [[Transformer]] 预测器 $P_\phi$ 以 $(z_t, a_t)$ 为条件预测 $\hat{z}_{t+1}$
- 单独使用此损失会陷入 [[Representation Collapse|表示坍缩]]（所有状态映射为常数）

#### 模块 2：[[LDAD|潜在差分动作解码器（LDAD）]]

**设计动机**：将逆动力学监督施加在**位移**而非拼接特征上，让转移几何本身承载动作信息，从而同时解决坍缩和动作不敏感问题。

**具体实现**：
- 计算潜变量位移：$\Delta z_t = z_{t+1} - z_t$
- 解码器仅以 $\Delta z_t$ 为输入，通过 [[AdaLN|自适应层归一化（AdaLN）]] 注入位移信息
- 输出重建动作 $\hat{a}_t$
- 解码目标为**原始动作**（优于状态差分代理目标，见消融实验）

**多步扩展**：解码器含 $N=5$ 个可学习 Action Query，输入为 $\Delta z_t^{t+N} = z_{t+N} - z_t$，同时预测 $N$ 步动作序列：

$$
\{\hat{a}_\tau\}_{\tau=t}^{t+N-1} = D_\Theta(z_{t+N} - z_t)
$$

**LDAD 防坍缩机制分析**：

1. **反坍缩**：若 $z_t \equiv z_{t+1}$（完全坍缩），则 $\Delta z_t = \mathbf{0}$，无法恢复任何动作，损失爆炸 → 训练信号强制编码器保持可区分表示。
2. **减少单帧快捷依赖**：拼接方案 $[z_t, z_{t+1}]$ 允许解码器利用 $z_t$ 本身的绝对信息作弊；仅用 $\Delta z_t$ 则强制动作信息必须体现在转移**变化量**中。
3. **动作敏感潜在几何**：要求不同动作从同一状态出发产生可区分的 $\Delta z_t$，即不同方向的潜在位移，使规划时可在潜空间中按"方向"导航。

---

## 关键公式

### 公式 1：[[JEPA|前向动力学预测]]

$$
\hat{z}_{t+1} = P_\phi(z_t, a_t)
$$

**含义**：动力学预测器以当前潜变量和动作为条件，预测下一时刻的潜变量。

**符号说明**：
- $z_t \in \mathbb{R}^d$：编码器输出的当前时刻潜变量
- $a_t \in \mathbb{R}^{d_a}$：执行的连续动作
- $\hat{z}_{t+1}$：预测的下一时刻潜变量

### 公式 2：[[Representation Collapse|前向预测损失]]

$$
\mathcal{L}_\text{pred} = \|\hat{z}_{t+1} - z_{t+1}\|_2^2
$$

**含义**：最小化预测潜变量与真实编码潜变量的 L2 距离，驱动预测器学习准确的动态。

**符号说明**：
- $z_{t+1} = f_\theta(o_{t+1})$：真实下一帧的编码表示（编码器输出）

### 公式 3：[[LDAD|潜变量差分]]

$$
\Delta z_t = z_{t+1} - z_t
$$

**含义**：LDAD 的核心输入，编码从状态 $t$ 到 $t+1$ 的转移方向与幅度。

**符号说明**：
- $\Delta z_t \in \mathbb{R}^d$：潜空间中的转移位移向量

### 公式 4：[[LDAD|动作重建]]

$$
\hat{a}_t = D_\Theta(\Delta z_t)
$$

**含义**：解码器仅从潜变量差分重建执行动作，强制转移位移承载完整的动作信息。

**符号说明**：
- $D_\Theta$：LDAD 解码器（3 层非因果 Transformer + AdaLN）
- $\hat{a}_t \in \mathbb{R}^{d_a}$：重建的动作

### 公式 5：[[LDAD|动作重建损失]]

$$
\mathcal{L}_\text{action} = \|\hat{a}_t - a_t\|_2^2
$$

**含义**：最小化重建动作与真实动作的 L2 误差，驱动编码器赋予转移位移以动作语义。

### 公式 6：[[LDAD|多步动作解码]]

$$
\{\hat{a}_\tau\}_{\tau=t}^{t+N-1} = D_\Theta(z_{t+N} - z_t)
$$

**含义**：将 LDAD 扩展至多步场景，以 $N$ 步累积位移为输入，同时预测 $N$ 步动作序列。

**符号说明**：
- $N$：预测步数（实验中取 5）
- $z_{t+N} - z_t$：$N$ 步累积潜变量位移

### 公式 7：[[JEPA|联合训练目标]]

$$
\mathcal{L} = \mathcal{L}_\text{pred} + \lambda \mathcal{L}_\text{action}
$$

**含义**：前向预测损失与动作重建损失的加权组合，$\lambda$ 平衡两者重要性。

**符号说明**：
- $\lambda > 0$：权重超参数（消融实验中最优值约为 $\lambda = 50.0$，默认训练取 10.0）
- $\mathcal{L}_\text{pred}$：前向预测损失，驱动准确动态建模
- $\mathcal{L}_\text{action}$：LDAD 动作重建损失，防坍缩 + 动作敏感

---

## 关键图表

### Figure 1: Framework Overview / Delta-JEPA 框架概览

![Figure 1](https://arxiv.org/html/2606.31232v1/framework.png)

**说明**：展示 Delta-JEPA 三组件架构。编码器 $f_\theta$ 将观测映射为潜变量；动力学预测器 $P_\phi$ 以 $(z_t, a_t)$ 预测 $\hat{z}_{t+1}$；[[LDAD]] 接收 $\Delta z_t = z_{t+1} - z_t$ 重建动作 $\hat{a}_t$。三者端到端联合训练，无任何冻结模块。

---

### Figure 2: Action-Sensitive Latent Geometry / LDAD 诱导的动作敏感潜在几何

![Figure 2](https://arxiv.org/html/2606.31232v1/path.png)

**说明**：说明 LDAD 如何改变潜空间几何。**左上**（无位移监督）：不同动作从同一状态出发可能产生相似的 $z_{t+1}$，潜在转移方向混叠。**右上**（有 LDAD）：不同动作诱导可区分的转移方向，使规划器可沿期望方向在潜空间中导航。

---

### Figure 3: Action Reconstruction Weight Ablation / 动作重建权重消融

![Figure 3](https://arxiv.org/html/2606.31232v1/action_weight_ablation.png)

**说明**：Push-T 上规划成功率随 $\lambda$ 变化的曲线。$\lambda=0$（移除 LDAD）导致近乎完全坍缩；$\lambda \geq 1.0$ 性能显著提升；$\lambda=50.0$ 时达到峰值；过大的 $\lambda$（如 1000.0）反而降低性能。

---

### Figure 4: PCA Visualization of Latent Space Evolution / 潜空间演化 PCA 可视化

![Figure 4](https://arxiv.org/html/2606.31232v1/pca_vision_feat.png)

**说明**：Push-T 任务上第 1、4、7、10 训练轮次的 PCA 投影。随着训练推进，潜空间从紧凑簇逐渐扩展并形成结构化几何，证明 LDAD 有效防止表示坍缩。

---

### Figure 5: Latent Trajectory Comparison / 潜在轨迹对比

![Figure 5](https://arxiv.org/html/2606.31232v1/delta-JEPA_lewm_two_trace_stacked.png)

**说明**：Two-Room 任务上相邻初始状态（但不同终态）的潜在轨迹 PCA 对比。**上方（Delta-JEPA）**：相似起点的轨迹随动作差异逐渐分叉，体现组合结构。**下方（[[LeWM]]）**：轨迹区分度较低，在同等初始条件下更难区分最终状态。

---

### Figure 6: Action-Conditioned Predictor Responses / 动作条件预测器响应

![Figure 6](https://arxiv.org/html/2606.31232v1/mean_delta_pca.png)

**说明**：Two-Room 上采样 512 条起始历史，施加不同动作后预测器响应的 PCA 可视化。Delta-JEPA 的响应呈现清晰的方向性分离（不同动作对应不同方向），而基线方法的响应几乎无法区分动作。

---

### Figure 7: Attention Rollout Visualizations / 注意力热力图可视化

![Figure 7](https://arxiv.org/html/2606.31232v1/attention_maps_combined.png)

**说明**：ViT-Tiny 编码器中间层（4–6 层）的 Attention Rollout 可视化，分别来自 Push-T（左）和 Two-Room（右）。Delta-JEPA 的注意力集中于与任务相关的可控元素（操作物体、智能体位置），而非背景纹理。

---

### Figure 8: Layer-wise Attention Specialization on OGB-Cube / OGB-Cube 逐层注意力特化

![Figure 8](https://arxiv.org/html/2606.31232v1/layer_specialization_ogb_cube.png)

**说明**：OGB-Cube 任务上不同 ViT 层的注意力分工。第 5 层主要关注目标立方体，第 7 层更突出地关注机械臂夹爪，展现出 Delta-JEPA 诱导的逐层语义分工。

---

### Table 1: Planning Success Rates / 规划成功率对比（%）

| Method | Two-Room | Reacher | Push-T | OGB-Cube |
|--------|----------|---------|--------|----------|
| [[PLDM]] | 93.73 ± 1.03 | 64.33 ± 2.14 | 76.13 ± 1.70 | 57.27 ± 1.53 |
| [[LeWM]] | 74.93 ± 0.42 | 79.87 ± 0.90 | 84.53 ± 1.50 | 64.13 ± 1.89 |
| [[Sub-JEPA]] | 90.60 ± 0.53 | 81.00 ± 2.40 | 63.73 ± 0.12 | 62.67 ± 1.45 |
| **Delta-JEPA** | **100.00 ± 0.00** | **81.33 ± 0.50** | **89.07 ± 1.90** | **79.27 ± 1.81** |

**关键发现**：Delta-JEPA 在所有四个环境均达到最优。OGB-Cube 提升最显著（+15.14 pp vs. LeWM），Two-Room 实现满分（+6.27 pp vs. PLDM）。Push-T 领先 LeWM +4.54 pp，Reacher 领先 Sub-JEPA +0.33 pp（小幅领先）。

---

### Table 2: Action-Decoder Input Ablation / 解码器输入消融

| Input | Two-Room | Reacher | Push-T | OGB-Cube |
|-------|----------|---------|--------|----------|
| $[z_t, z_{t+1}]$（拼接） | 95.93 ± 0.61 | 80.27 ± 0.81 | 76.47 ± 2.08 | 78.60 ± 3.29 |
| $\Delta z_t$（LDAD） | **100.00 ± 0.00** | **81.33 ± 0.50** | **89.07 ± 1.90** | **79.27 ± 1.81** |
| Gain | +4.07 | +1.07 | +12.60 | +0.67 |

**关键发现**：位移输入在所有环境全面超越拼接输入，Push-T 提升最大（+12.60 pp），证明仅依赖差分向量的核心设计的有效性。

---

### Table 3: LDAD Decoding Target Ablation on Reacher / LDAD 解码目标消融（Reacher）

| 解码目标 | 规划成功率 (%) |
|----------|---------------|
| 原始动作 $a_t$ | **81.33 ± 0.50** |
| $\Delta$ 指端位置 | 64.93 ± 1.10 |
| $\Delta$ 关节位置 | 80.47 ± 2.10 |
| $\Delta$ 指端 + 关节位置 | 76.40 ± 1.40 |

**关键发现**：直接解码原始动作效果最优，优于各种状态差分代理目标；$\Delta$ 关节位置接近最优，说明关节空间信息与动作信息高度相关。

---

### Table 4: Physical Latent Probing on Two-Room / 物理状态潜变量探测（Two-Room）

| 属性 | Method | Linear MSE↓ | Linear r↑ | MLP MSE↓ | MLP r↑ |
|------|--------|------------|-----------|----------|--------|
| Agent Pos. | PLDM | 0.078 | 0.960 | 0.002 | 0.999 |
| | Sub-JEPA | 0.179 | 0.907 | 0.006 | 0.997 |
| | LeWM | 0.085 | 0.950 | 0.002 | 0.999 |
| | **Delta-JEPA** | **0.004** | **0.998** | **0.000** | **1.000** |

**关键发现**：Delta-JEPA 在 Two-Room 上的线性与非线性探测均达最优，智能体位置以接近完美的精度（r=1.000）被编码进潜变量。

---

### Table 5: State-Delta Probing on Two-Room / 状态差分潜变量探测（Two-Room）

| 属性 | Method | Linear MSE↓ | Linear r↑ | MLP MSE↓ | MLP r↑ |
|------|--------|------------|-----------|----------|--------|
| Δ Agent Pos. | PLDM | 0.355 | 0.813 | 0.095 | 0.955 |
| | Sub-JEPA | 0.601 | 0.674 | 0.141 | 0.928 |
| | LeWM | 0.444 | 0.765 | 0.085 | 0.958 |
| | **Delta-JEPA** | **0.016** | **0.992** | **0.005** | **0.997** |

**关键发现**：Delta-JEPA 的潜变量差分高度编码了智能体的实际位移，远优于其他基线，直接体现了 LDAD 的差分监督效果。

---

### Table 6: Physical Latent Probing on Push-T / 物理状态潜变量探测（Push-T）

| 属性 | Method | Linear MSE↓ | Linear r↑ | MLP MSE↓ | MLP r↑ |
|------|--------|------------|-----------|----------|--------|
| Agent Location | PLDM | 0.007 | 0.996 | 0.000 | 1.000 |
| | Sub-JEPA | 0.094 | 0.955 | 0.003 | 0.999 |
| | LeWM | 0.017 | 0.991 | 0.000 | 1.000 |
| | **Delta-JEPA** | **0.004** | **0.998** | **0.000** | **1.000** |
| Block Location | PLDM | 0.055 | 0.974 | 0.006 | 0.997 |
| | Sub-JEPA | 0.250 | 0.895 | 0.006 | 0.997 |
| | LeWM | 0.041 | 0.979 | 0.002 | 0.999 |
| | Delta-JEPA | 0.189 | 0.929 | 0.013 | 0.994 |
| Block Angle | PLDM | 0.005 | 0.998 | 0.000 | 1.000 |
| | Sub-JEPA | 0.024 | 0.988 | 0.001 | 0.999 |
| | LeWM | 0.004 | 0.998 | 0.000 | 1.000 |
| | Delta-JEPA | 0.011 | 0.995 | 0.001 | 1.000 |

**关键发现**：Delta-JEPA 在智能体位置编码上达到最优，但在 Block Location 上线性探测不如 PLDM 和 LeWM（0.189 vs. 0.041），暗示对非直接操作的物体状态建模有待提升。

---

### Table 7: Physical Latent Probing on DMC Reacher / 物理状态潜变量探测（DMC Reacher）

| 属性 | Method | Linear MSE↓ | Linear r↑ | MLP MSE↓ | MLP r↑ |
|------|--------|------------|-----------|----------|--------|
| Finger Position | PLDM | **0.016** | **0.992** | **0.000** | **1.000** |
| | Sub-JEPA | 0.798 | 0.632 | 0.014 | 0.995 |
| | LeWM | 0.262 | 0.869 | 0.096 | 0.969 |
| | Delta-JEPA | 0.520 | 0.777 | 0.056 | 0.972 |
| Joint Position | PLDM | **0.215** | **0.879** | **0.133** | **0.928** |
| | Sub-JEPA | 0.760 | 0.446 | 0.545 | 0.673 |
| | LeWM | 0.630 | 0.586 | 0.789 | 0.576 |
| | Delta-JEPA | 0.622 | 0.537 | 0.555 | 0.651 |

**关键发现**：Reacher 上 PLDM 在绝对状态编码方面优于 Delta-JEPA，但 Delta-JEPA 规划成功率仍更高，暗示绝对状态精度并非规划成功的唯一决定因素。

---

### Table 8: Physical Latent Probing on OGB-Cube / 物理状态潜变量探测（OGB-Cube）

| 属性 | Method | Linear MSE↓ | Linear r↑ | MLP MSE↓ | MLP r↑ |
|------|--------|------------|-----------|----------|--------|
| Joint Position | PLDM | 0.545 | 0.595 | 0.304 | 0.813 |
| | Sub-JEPA | 0.619 | 0.482 | 0.928 | 0.511 |
| | LeWM | 0.817 | 0.470 | 1.494 | 0.518 |
| | **Delta-JEPA** | **0.378** | **0.674** | **0.621** | **0.652** |
| Joint Velocity | PLDM | **0.953** | **0.269** | **0.656** | **0.595** |
| | Sub-JEPA | 1.082 | 0.041 | 1.797 | 0.035 |
| | LeWM | 1.262 | 0.054 | 4.765 | 0.035 |
| | Delta-JEPA | 0.936 | 0.273 | 1.304 | 0.283 |
| End-Effector Position | PLDM | 0.025 | 0.988 | 0.003 | 0.998 |
| | Sub-JEPA | 0.226 | 0.909 | 0.027 | 0.987 |
| | LeWM | 0.515 | 0.739 | 0.256 | 0.897 |
| | **Delta-JEPA** | **0.007** | **0.997** | **0.001** | **1.000** |
| End-Effector Yaw | PLDM | 0.363 | 0.791 | 0.136 | 0.927 |
| | Sub-JEPA | 0.958 | 0.075 | 1.874 | -0.084 |
| | LeWM | 1.468 | -0.037 | 1.886 | -0.100 |
| | Delta-JEPA | 1.218 | -0.074 | 1.987 | -0.073 |
| Block Position | PLDM | 0.246 | 0.860 | 0.057 | 0.971 |
| | Sub-JEPA | 0.327 | 0.835 | 0.054 | 0.973 |
| | LeWM | 0.464 | 0.765 | 0.244 | 0.885 |
| | **Delta-JEPA** | **0.038** | **0.983** | **0.007** | **0.997** |
| Block Quaternion | PLDM | 0.635 | 0.577 | 0.296 | 0.839 |
| | Sub-JEPA | 1.180 | -0.030 | 2.412 | -0.062 |
| | LeWM | 1.803 | -0.090 | 3.670 | -0.057 |
| | **Delta-JEPA** | **1.053** | **0.273** | **1.619** | **0.205** |
| Block Yaw | PLDM | 0.462 | 0.742 | 0.190 | 0.904 |
| | Sub-JEPA | 1.696 | 0.166 | 3.156 | 0.038 |
| | LeWM | 2.927 | -0.294 | 2.945 | -0.076 |
| | **Delta-JEPA** | **1.758** | **0.241** | **1.896** | **0.232** |

**关键发现**：OGB-Cube 上 Delta-JEPA 在末端执行器位置（r=1.000）和方块位置（r=0.997）上达到接近完美的编码精度，但旋转量（Yaw/Quaternion）的编码整体较差，说明 SO(3) 方向的建模是未来改进方向。

---

### Table 9: State-Delta Probing on Push-T / 状态差分探测（Push-T）

| 属性 | Method | Linear MSE↓ | Linear r↑ | MLP MSE↓ | MLP r↑ |
|------|--------|------------|-----------|----------|--------|
| Δ Agent Location | PLDM | 0.061 | 0.969 | 0.012 | 0.994 |
| | Sub-JEPA | 0.291 | 0.846 | 0.037 | 0.981 |
| | LeWM | 0.091 | 0.954 | 0.017 | 0.991 |
| | **Delta-JEPA** | **0.018** | **0.995** | **0.001** | **1.000** |
| Δ Block Location | PLDM | 0.114 | 0.944 | 0.020 | 0.991 |
| | Sub-JEPA | 0.375 | 0.807 | 0.028 | 0.987 |
| | LeWM | **0.073** | **0.966** | **0.014** | **0.993** |
| | Delta-JEPA | 0.189 | 0.927 | 0.024 | 0.989 |
| Δ Block Angle | PLDM | 0.103 | 0.950 | 0.012 | 0.994 |
| | Sub-JEPA | 0.333 | 0.823 | 0.015 | 0.993 |
| | LeWM | **0.088** | **0.959** | **0.006** | **0.997** |
| | Delta-JEPA | 0.163 | 0.924 | 0.013 | 0.994 |

**关键发现**：Delta-JEPA 在自身位移（Δ Agent Location）上最优，但在物体位移上不及 LeWM，与绝对状态探测结果一致。

---

### Table 10: State-Delta Probing on DMC Reacher / 状态差分探测（DMC Reacher）

| 属性 | Method | Linear MSE↓ | Linear r↑ | MLP MSE↓ | MLP r↑ |
|------|--------|------------|-----------|----------|--------|
| Δ Finger Position | PLDM | **0.605** | **0.654** | **0.319** | **0.828** |
| | Sub-JEPA | 1.007 | 0.197 | 0.457 | 0.751 |
| | LeWM | 0.784 | 0.507 | 0.644 | 0.648 |
| | Delta-JEPA | 0.900 | 0.359 | 1.150 | 0.467 |
| Δ Joint Position | PLDM | 1.366 | 0.067 | 1.378 | 0.284 |
| | Sub-JEPA | 1.072 | -0.027 | 1.065 | 0.311 |
| | LeWM | 1.408 | 0.017 | 1.755 | 0.137 |
| | **Delta-JEPA** | **1.012** | **0.207** | **0.237** | **0.870** |

**关键发现**：Reacher 任务上各方法对状态差分的编码整体较弱，Delta-JEPA 在 Δ 关节位置的 MLP 探测上达最优（r=0.870）。

---

### Table 11: State-Delta Probing on OGB-Cube / 状态差分探测（OGB-Cube）

| 属性 | Method | Linear MSE↓ | Linear r↑ | MLP MSE↓ | MLP r↑ |
|------|--------|------------|-----------|----------|--------|
| Δ Joint Position | PLDM | 0.851 | 0.283 | 2.056 | 0.262 |
| | Sub-JEPA | 0.744 | 0.365 | 0.613 | 0.544 |
| | LeWM | 0.790 | 0.355 | 0.890 | 0.517 |
| | **Delta-JEPA** | **0.358** | **0.686** | **0.359** | **0.711** |
| Δ Joint Velocity | PLDM | 1.188 | 0.017 | 2.478 | 0.149 |
| | Sub-JEPA | 1.076 | 0.066 | 1.056 | 0.274 |
| | LeWM | 1.130 | 0.055 | 1.687 | 0.253 |
| | **Delta-JEPA** | **0.858** | **0.391** | **0.753** | **0.555** |
| Δ End-Effector Position | PLDM | 0.608 | 0.654 | 0.261 | 0.868 |
| | Sub-JEPA | 0.443 | 0.760 | 0.113 | 0.939 |
| | LeWM | 0.678 | 0.568 | 0.336 | 0.812 |
| | **Delta-JEPA** | **0.010** | **0.995** | **0.003** | **0.999** |
| Δ End-Effector Yaw | PLDM | 1.147 | -0.012 | 2.573 | -0.039 |
| | Sub-JEPA | 0.851 | 0.083 | 1.203 | 0.188 |
| | LeWM | 1.055 | 0.037 | 1.578 | -0.013 |
| | **Delta-JEPA** | **0.851** | **0.239** | **0.160** | **0.910** |
| Δ Block Position | PLDM | 0.845 | 0.417 | 1.047 | 0.420 |
| | Sub-JEPA | 0.614 | 0.590 | 0.337 | 0.790 |
| | LeWM | 0.675 | 0.526 | 0.539 | 0.646 |
| | **Delta-JEPA** | **0.198** | **0.886** | **0.024** | **0.987** |
| Δ Block Quaternion | PLDM | 0.927 | 0.027 | 1.921 | 0.087 |
| | Sub-JEPA | 0.827 | -0.004 | 1.290 | 0.028 |
| | LeWM | 0.911 | 0.023 | 1.273 | 0.108 |
| | **Delta-JEPA** | **0.688** | **0.226** | **1.001** | **0.159** |
| Δ Block Yaw | PLDM | 1.668 | -0.055 | 34.418 | -0.085 |
| | Sub-JEPA | 1.562 | -0.012 | 1.651 | -0.025 |
| | LeWM | 1.638 | -0.041 | 3.424 | -0.013 |
| | **Delta-JEPA** | **1.574** | **0.050** | **1.592** | **0.043** |

**关键发现**：OGB-Cube 上 Delta-JEPA 在绝大多数状态差分属性（关节位置、关节速度、末端执行器位置/Yaw、方块位置）的 MLP 探测全面领先，尤其 Δ 末端位置（r=0.999）和 Δ End-Effector Yaw（r=0.910）远超其他方法。

---

## 实验

### 数据集 / 环境

| 环境 | 类型 | 关键特点 |
|------|------|---------|
| Two-Room | 2D 导航 | 智能体需通过门洞移动 |
| Reacher (DMC) | 机械臂到达 | DeepMind Control Suite 连续控制 |
| Push-T | 2D 非抓取操作 | 推动 T 形物体到目标位置 |
| OGB-Cube | 3D 夹爪操作 | 高维视觉观测，3D 机器人操作 |

验证集：每个任务 50 条轨迹；测试集：每个任务 500 条轨迹，3 个随机种子评测。

### 实现细节

- **视觉编码器**：[[ViT]]-Tiny（最小 ViT 变体）
- **动力学预测器**：6 层因果 [[Transformer]]，16 头，64 维，MLP 宽度 2048
- **动作解码器（LDAD）**：3 层非因果 [[Transformer]]，8 头，64 维，FFN 512，$N=5$ 可学习 Action Query，[[AdaLN]] 注入位移
- **训练**：50 轮次从头训练，学习率 $5 \times 10^{-5}$，默认 $\lambda = 10.0$
- **规划**：[[Cross-Entropy Method|交叉熵方法（CEM）]] 在学习到的潜空间中进行无模型式规划

### 可视化结果

- **PCA 潜空间结构**（Figure 4）：训练过程中潜空间持续扩展，避免坍缩
- **轨迹对比**（Figure 5）：Delta-JEPA 潜在轨迹体现组合性，不同动作历史产生不同分叉
- **注意力可视化**（Figures 7-8）：编码器注意力聚焦于任务相关可控区域，非背景

---

## 批判性思考

### 优点

1. **极简架构**：仅两项损失，无冻结编码器、无 EMA、无多项正则化，实现防坍缩 + 动作敏感，工程简洁。
2. **强分析性**：提供物理状态探测和状态差分探测两类分析，全面量化表示质量，不止步于规划成功率。
3. **直觉清晰**：位移监督的几何直觉（Figure 2）可解释性强，LDAD 的三重防坍缩机制论证充分。

### 局限性

1. **离线规划设置有限**：所有实验基于离线数据 + [[Cross-Entropy Method|CEM]] 规划，未验证在在线学习或策略蒸馏场景下的效果。
2. **旋转量编码弱**：多个环境上 Yaw / Quaternion 类旋转属性的探测分数整体不佳（包括 Delta-JEPA），说明 SO(3) 几何在潜变量差分框架中难以线性化。
3. **物体间接操作短板**：Push-T 中方块位移的状态差分探测不如 LeWM，说明 LDAD 对"间接被动物体"的转移建模仍有改进空间。
4. **规模未验证**：仅用 ViT-Tiny 编码器，未验证方法随编码器规模扩展的行为。

### 潜在改进方向

1. 将 LDAD 扩展到**多物体位移监督**，为每个可辨别物体单独监督位移向量。
2. 与 Dreamer 类在线 RL 框架结合，探索 LDAD 在奖励驱动设置中的效果。
3. 用 SO(3)-equivariant 编码器替换 ViT，改善旋转量的建模能力。

### 可复现性评估

- [ ] 代码开源（未提供链接）
- [ ] 预训练模型（未提供）
- [x] 训练细节完整（超参数、架构、轮次全部在论文中给出）
- [x] 数据集可获取（所有环境均为公开 benchmark）

---

## 关联笔记

### 基于

- [[JEPA]]: 联合嵌入预测架构基础框架
- [[I-JEPA]]: 图像域 JEPA，证明无像素重建自监督可行
- [[V-JEPA]]: 视频域 JEPA 扩展

### 对比

- [[LeWM]]: 使用 SIGReg 高斯正则化的 JEPA 世界模型，是最强基线之一
- [[PLDM]]: VICReg + 逆动力学（拼接输入）的 JEPA 世界模型
- [[Sub-JEPA]]: 子空间高斯正则化方案
- [[DINO-WM]]: 冻结 DINOv2 特征的稳定化方案

### 方法相关

- [[LDAD]]: 本文核心贡献，潜在差分动作解码器
- [[Representation Collapse]]: 本文解决的核心问题
- [[AdaLN]]: LDAD 解码器中注入位移信息的机制
- [[VICReg]]: PLDM 使用的正则化方法（本文避免使用）
- [[Cross-Entropy Method]]: 规划阶段使用的优化算法

### 硬件/数据相关

- [[ViT]]: 编码器骨干
- [[PlaNet]]: 早期像素重建世界模型，作为方法演化起点
- [[DreamerV3]]: 重建式世界模型代表

---

## 速查卡片

> [!summary] Delta-JEPA (2026)
> - **核心**: 从潜变量差分 $\Delta z_t$ 而非拼接状态解码动作，驱动 JEPA 世界模型学习动作敏感表示
> - **方法**: LDAD（位移解码）+ 前向预测损失，仅两项目标端到端训练
> - **结果**: 四个视觉连续控制任务规划成功率全面领先，OGB-Cube 最高提升 +15.14 pp
> - **代码**: 未开源

---

*笔记创建时间: 2026-07-02*
