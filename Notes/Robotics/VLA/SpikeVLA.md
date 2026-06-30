---
title: "SpikeVLA: Vision-Language-Action Models with Spiking Neural Networks"
method_name: "SpikeVLA"
authors: [Ruiqi Song, Dujun Nie, Siyu Teng, Baiyong Ding, Xiaotong Zhang, Dong Li, Chenming Zhang, Yuchen Li, Hangbin Wu, Long Chen]
year: 2026
venue: ICML 2026
tags: [vla, spiking-neural-network, energy-efficient, vision-language-navigation, robot-control]
zotero_collection: Robotics/VLA
image_source: online
arxiv_html: https://arxiv.org/html/2606.27807
created: 2026-06-30
---

# 论文笔记：SpikeVLA: Vision-Language-Action Models with Spiking Neural Networks

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | 未在摘要中明确列出 |
| 日期 | June 2026 |
| 项目主页 | — |
| 对比基线 | [[NaVILA]] |
| 链接 | [arXiv](https://arxiv.org/abs/2606.27807) / Code: — |

---

## 一句话总结

> SpikeVLA 首次将[[脉冲神经网络|SNN]]引入 VLA 框架，以约 35% 的能耗实现与 ANN 基线相当的导航与机器人控制性能。

---

## 核心贡献

1. **首个全 SNN VLA 框架**: 将[[脉冲神经网络|Spiking Neural Network (SNN)]] 应用于视觉-语言-动作模型的完整 pipeline，实现事件驱动的稀疏计算。
2. **三模块脉冲架构 (Spike-V/L/A)**: 分别设计脉冲视觉编码器、脉冲多模态语言模型和脉冲动作策略网络，每个模块都针对 SNN 特性定制转换方案。
3. **能效-性能帕累托最优**: 在 R2R-CE、RxR-CE、VLN-CE-Isaac 三个 benchmark 上能耗约为 NaVILA 的 34%，同时在成功率上持平甚至超越 ANN 量化基线。

---

## 问题背景

### 要解决的问题

传统 [[VLA|Vision-Language-Action]] 模型（如 NaVILA）基于 [[Transformer]] 架构，在推理时需要大量矩阵乘法累加操作（MAC），能耗高（>100 J），GPU 显存需求大（>16 GB），难以部署在资源受限的机器人平台（微型机器人、太空探索机器人）上。

### 现有方法的局限

- **ANN 量化**（INT4）：NaVILA INT4 能耗降至 72.49 J，但导航成功率 SR 从 53.9% 降至 48.2%，牺牲了性能。
- **传统 SNN 转换**：直接从 ANN 转 SNN 会在非线性操作（归一化、注意力、激活函数）处精度损失严重，尤其在大型语言模型中。
- **VLN 专用方法**（ETPNav, GridMM）：依赖全景相机、深度传感器和里程计，硬件门槛高；SpikeVLA 仅用 RGB 图像。

### 本文的动机

[[脉冲神经网络]] 以 0/1 脉冲替代连续激活，乘法变加法（[[突触操作|Synaptic Operation, SOP]]），能耗从 4.6 pJ/MAC 降至 0.9 pJ/AC（45nm 工艺），理论上可实现数量级节能。作者设计了专门的**微分编码机制**解决非线性操作转换难题，使大规模 LLM 也能被 SNN 化。

---

## 方法详解

### 模型架构

SpikeVLA 采用**三模块流水线** 架构：
- **输入**: RGB 图像序列（历史帧 $V^h$ + 当前帧 $V^c$）+ 语言指令 $T$
- **Spike-V**: 脉冲视觉编码器，基于 [[SigLIPv2]] 改造
- **Spike-L**: 脉冲多模态语言模型，基于 [[LLaMA]]-8B
- **Spike-A**: 脉冲动作策略网络，输出速度指令
- **训练**: Spike-A 使用 [[PPO]] 算法 + [[代理梯度|Surrogate Gradient]] 时空反向传播

### 核心模块

#### 模块1: Spike-V（脉冲视觉编码器）

**设计动机**: 将 [[SigLIPv2]] 的连续激活转换为脉冲表示，同时保持图像特征质量。

**具体实现**:
- 引入**微分脉冲神经元**（Differential Spiking Neuron），通过时间步驱动计算支持稳定动态
- 使用**微分编码**将连续激活表示为增量更新（见公式1）
- 线性层直接转换为 SNN 等价形式（公式3）
- 非线性操作（LayerNorm、GELU、Attention）通过**微分梯度单元**（Differential Graded Unit）转换（公式4）

#### 模块2: Spike-L（脉冲多模态 LLM）

**设计动机**: 将 LLaMA-8B 中的 [[Transformer]] 注意力和 FFN 全部 SNN 化，实现稀疏 token 处理。

**具体实现**:
- 将视觉历史帧、当前帧、文本 token 统一为序列 $I_t = [V^h, V^c, T] \in \mathbb{R}^{(196 \times t + 196 + N_\text{text}) \times d}$（公式6）
- 通过脉冲神经元动态将 token 转换为稀疏脉冲 token（公式7-8）
- 实现**微分时序稀疏分配**（Differential Temporal Sparsity Allocation）：为信息丰富的通道分配更长的脉冲时域，提升表达力
- 使用**二阶优化**（Second-Order Optimization）进行精确权重调整

#### 模块3: Spike-A（脉冲动作策略网络）

**设计动机**: 将连续观测空间（速度、位姿）离散化为脉冲，驱动四足机器人运动控制。

**具体实现**:
- 使用**拉普拉斯高斯核种群编码**（Laplacian-kernel Population Encoding）将连续观测编码为脉冲（公式11）
- 全连接 SNN 层 + 迭代神经元动态
- 使用 [[PPO]] + [[代理梯度]] 时空反向传播（STBP）训练
- 在 Unitree Go2 四足平台上验证

---

## 关键公式

### 公式1: [[微分编码|微分编码（Differential Coding）]]

$$
\begin{aligned}
\delta^l[t] &= \bar{a}^l[t-1] + \theta^l z^l[t] \\
\bar{a}^l[t] &= \frac{1}{t} \sum_{i=1}^{t} \delta^l[i]
\end{aligned}
$$

**含义**: 微分编码将连续激活分解为离散时间步上的增量更新，使稀疏脉冲能够逼近连续激活的累积平均。

**符号说明**:
- $\delta^l[t]$: 第 $l$ 层在时间步 $t$ 的微分增量
- $\bar{a}^l[t]$: 第 $l$ 层到时间步 $t$ 的累积平均激活
- $\theta^l$: 第 $l$ 层的脉冲发射阈值
- $z^l[t] \in \{0, 1\}$: 时间步 $t$ 的脉冲输出（二值）

### 公式2: [[微分脉冲神经元|微分脉冲神经元输入电流]]

$$
\begin{aligned}
I^l[t] &= cr^l[t] + x^{l-1}[t] \\
cr^l[t+1] &= cr^l[t] + \frac{x^{l-1}[t]}{t} - \frac{\theta^l z^l[t]}{t}
\end{aligned}
$$

**含义**: 微分脉冲神经元在每个时间步维护一个残差状态 $cr^l[t]$，将历史输入的累积信息注入当前时间步，实现稳定的动态特性。

**符号说明**:
- $I^l[t]$: 第 $l$ 层时间步 $t$ 的输入电流
- $cr^l[t]$: 残差电流状态
- $x^{l-1}[t]$: 上一层在时间步 $t$ 的输出

### 公式3: [[SNN 线性层转换|线性层 SNN 等价转换]]

$$
x^l[t] = W^l x^{l-1}[t]
$$

**含义**: 线性层在 SNN 中可以直接等价转换——对稀疏脉冲输入应用权重矩阵，乘法变为加法（仅对发射脉冲的神经元执行），实现 [[突触操作|SOP]] 替代 MAC。

**符号说明**:
- $W^l$: 第 $l$ 层权重矩阵
- $x^{l-1}[t]$: 稀疏脉冲输入（含 0 的向量）

### 公式4: [[非线性算子转换|非线性层 SNN 转换（微分梯度单元）]]

$$
\begin{aligned}
c^l[t] &= c^l[t-1] + \frac{x^{l-1}[t]}{t} \\
\Delta F^l[t] &= F^l(c^l[t]) - F^l(c^l[t-1]) \\
x^l[t] &= t \cdot \Delta F^l[t]
\end{aligned}
$$

**含义**: 对于非线性函数 $F(\cdot)$（如 GELU、LayerNorm），通过跟踪累积均值 $c^l[t]$ 并计算其差分来近似非线性变换，避免直接将非线性作用于脉冲导致精度损失。

**符号说明**:
- $c^l[t]$: 第 $l$ 层输入的累积均值
- $F^l(\cdot)$: 第 $l$ 层的非线性激活函数（GELU、LayerNorm 等）
- $\Delta F^l[t]$: 非线性输出的差分（相当于线性化近似）

### 公式5: [[统一 Token 序列|多模态统一 Token 序列]]

$$
I_t = [V^h, V^c, T] \in \mathbb{R}^{(196 \times t + 196 + N_\text{text}) \times d}
$$

**含义**: Spike-L 将历史视觉帧、当前视觉帧和文本指令统一拼接为序列，共同输入语言模型，实现多模态融合。

**符号说明**:
- $V^h \in \mathbb{R}^{196 \times t \times d}$: 历史帧视觉 token（$t$ 帧，每帧 196 个 patch）
- $V^c \in \mathbb{R}^{196 \times d}$: 当前帧视觉 token
- $T \in \mathbb{R}^{N_\text{text} \times d}$: 文本语言指令 token
- $d$: 特征维度

### 公式6: [[种群编码|拉普拉斯高斯核种群编码（Population Encoding）]]

$$
A_E(s) = \Phi_\text{LoG}(s;\, \mu, \sigma)
$$

**含义**: 将连续观测值 $s$（速度、角速度等）通过拉普拉斯高斯核映射为种群响应向量，使 SNN 能接受连续控制信号作为输入。拉普拉斯核对中间距离敏感度高，适合足式运动控制。

**符号说明**:
- $s$: 连续标量观测（如线速度 $v_x$）
- $\Phi_\text{LoG}$: 拉普拉斯高斯（Laplacian of Gaussian）核函数
- $\mu, \sigma$: 核中心和宽度参数（均匀分布在值域范围内）

### 公式7: [[能量消耗模型|SNN 能量消耗计算]]

$$
\begin{aligned}
E_\text{SpikeVLA} &= E_\text{MAC} \cdot \text{FLOPs}_1 + E_\text{AC} \cdot \sum_{l=2}^{L} \text{SOPs}_l \\
\text{SOPs}_l &= r_l \cdot T \cdot \text{FLOPs}_l
\end{aligned}
$$

**含义**: SpikeVLA 的总能耗由第一层（输入层，仍用 MAC）和后续 SNN 层（用 AC，即加法操作）两部分构成。SOPs 数量与平均发射率 $r_l$ 和时间步数 $T$ 成正比。

**符号说明**:
- $E_\text{MAC} = 4.6\,\text{pJ}$: 单次乘加操作能耗（45nm 工艺）
- $E_\text{AC} = 0.9\,\text{pJ}$: 单次加法操作能耗（45nm 工艺）
- $r_l$: 第 $l$ 层平均脉冲发射率
- $T$: 时间步数（超参数，导航任务取 $T=16$）
- $\text{FLOPs}_l$: 第 $l$ 层的稠密浮点运算量

---

## 关键图表

### Figure 1: 论文整体概览

![[SpikeVLA_fig1_page1.png]]

**说明**: SpikeVLA 系统概览，展示从 RGB 输入经过三个脉冲模块（Spike-V、Spike-L、Spike-A）到机器人导航控制动作的完整流程，以及与 ANN 基线的能耗对比定性示意。

### Figure 2: SpikeVLA 系统架构

![[SpikeVLA_fig2_page2.png]]

**说明**: 完整系统架构图。左侧 Spike-V 将 RGB 图像通过微分脉冲神经元转换为视觉 token；中间 Spike-L 将多模态 token 统一处理并生成高层语义特征；右侧 Spike-A 通过种群编码和 PPO 训练实现低层速度控制。所有模块均以[[脉冲神经网络|SNN]]实现。

### Figure 3: 资源消耗与性能对比

![[SpikeVLA_fig3_page3.png]]

**说明**: 对比 SpikeVLA 与 ANN 基线（NaVid、UniNaVid、NaVILA）在 GPU 显存、能耗和导航成功率上的权衡。SpikeVLA 在能耗和显存上均具有显著优势，成功率与 NaVILA 相当。

### Figure 4: SNN 动作策略网络消融

![[SpikeVLA_fig4_page4.png]]

**说明**: Spike-A 不同组件的消融结果，比较不同种群编码核函数（Laplacian、Gaussian RBF、Triangular、Inverse Multiquadric）和时间步数对控制性能的影响。Laplacian 核取得最高奖励（26.72）。

### Figure 5: 不同地形下的控制性能

![[SpikeVLA_fig5_page5.png]]

**说明**: SpikeVLA 在崎岖地形、斜坡和障碍物环境中的速度跟踪可视化。展示 Spike-A 对线速度和角速度的跟踪曲线，验证在不同难度地形下的鲁棒性。

### Figure 6: 时间步超参数消融

![[SpikeVLA_fig6_page8.png]]

**说明**: 不同时间步 $T$（T=2,3,5）对动作策略控制误差和能耗的影响。T=3 时线速度误差 0.35 m/s、能耗 0.10 μJ，为精度-效率最优点。

### Figure 7: 种群编码尺寸超参数消融

![[SpikeVLA_fig7_page9.png]]

**说明**: 不同种群编码尺寸 $P$（P=2,5,10,20,30）对线速度误差和能耗的影响。P=5 时误差 0.35 m/s、能耗 0.10 μJ，之后收益递减。

### Figure 8: Actor 网络维度消融

![[SpikeVLA_fig8_page10.png]]

**说明**: Spike-A 中 Actor 网络不同隐层维度对控制性能（奖励、MEL）和资源消耗（能耗、显存）的影响。

### Figure 9: Critic 网络维度消融

![[SpikeVLA_fig9_page11.png]]

**说明**: Spike-A 中 Critic 网络不同隐层维度对训练稳定性和最终策略性能的影响。

### Table 1: R2R-CE Val-Unseen 导航性能对比

| Method | Observation | Waypoint | NE↓ | OS↑ | SR↑ | SPL↑ | Mem (MB)↓ | Eng (J)↓ | ACEs (10¹²)↓ |
|--------|-------------|----------|-----|-----|-----|------|-----------|---------|-------------|
| CM2 | RGB, Depth, Odom | ✗ | 7.02 | 41.0 | 34.0 | 27.0 | — | — | — |
| WS-MGMap | RGB, Depth, Odom | ✗ | 6.28 | 47.0 | 38.0 | 34.0 | — | — | — |
| CMA | Pano, Depth, Odom | ✓ | 6.20 | 52.0 | 41.0 | 36.0 | — | — | — |
| Sim2Sim | Pano, Depth, Odom | ✓ | 6.07 | 52.0 | 43.0 | 36.0 | — | — | — |
| GridMM | Pano, Depth, Odom | ✓ | 5.11 | 61.0 | 49.0 | 41.0 | — | — | — |
| Ego2-Map | Pano, Depth, Odom | ✓ | 5.54 | 56.0 | 47.0 | 41.0 | — | — | — |
| DreamWalker | Pano, Depth, Odom | ✓ | 5.53 | 59.0 | 49.0 | 44.0 | — | — | — |
| HAMT+ScaleVLN | Pano, Depth, Odom | ✓ | 4.80 | — | 55.0 | 51.0 | — | — | — |
| ETPNav | Pano, Depth, Odom | ✓ | 4.71 | 65.0 | 57.0 | 49.0 | — | — | — |
| HNR | Pano, Depth, Odom | ✓ | 4.42 | 67.0 | 61.0 | 51.0 | — | — | — |
| AO-Planner | Pano, Depth | ✗ | 5.55 | 59.0 | 47.0 | 33.0 | — | — | — |
| NaVid | RGB | ✗ | 5.47 | 49.0 | 37.0 | 35.0 | 14231.96 | 157.29 | 4376.68 |
| UniNaVid | RGB | ✗ | 5.58 | 53.3 | 47.0 | 42.7 | 14231.96 | 157.29 | 4376.68 |
| NaVILA | RGB | ✗ | 5.28 | 61.5 | 53.9 | 49.3 | 16119.98 | 141.25 | 3930.21 |
| MapNav | RGB | ✗ | 4.93 | 53.0 | 39.7 | 37.2 | — | — | — |
| **SpikeVLA** | **RGB** | **✗** | **5.38** | **63.4** | **53.3** | **47.9** | **6249.18** | **49.09** | **1196.16** |

**关键发现**: SpikeVLA 仅用 RGB 输入，在 OS 上超越 NaVILA（63.4 vs 61.5），SR 基本持平（53.3 vs 53.9），但能耗仅为 NaVILA 的 34.7%（49.09 J vs 141.25 J），显存需求减少 61%。

### Table 2: VLN-CE-Isaac 真实动力学测试（1,077 episodes）

| Method | NE↓ | OS↑ | SR↑ | SPL↑ | Mem (MB)↓ | Eng (J)↓ | ACEs (10¹²)↓ |
|--------|-----|-----|-----|------|-----------|---------|-------------|
| NaVILA-R | 6.29 | 52.1 | 36.5 | 29.5 | 16126.18 | 141.25 | 3930.21 |
| **SpikeVLA** | **6.02** | **53.6** | **32.7** | **28.5** | **6251.53** | **44.23** | **1109.61** |

**关键发现**: 在引入真实物理动力学的仿真环境中，SpikeVLA 在 NE 和 OS 上优于 NaVILA-R，SR 略低（32.7 vs 36.5），能耗降低 68.7%。

### Table 3: RxR-CE Val-Unseen 长视距导航对比

| Method | Observation | Waypoint | NE↓ | SR↑ | SPL↑ | nDTW↑ | Mem (MB)↓ | Eng (J)↓ | ACEs (10¹²)↓ |
|--------|-------------|----------|-----|-----|------|-------|-----------|---------|-------------|
| CMA | Pano, Depth, Odom | ✓ | 8.76 | 26.5 | 22.1 | 47.0 | — | — | — |
| ETPNav | Pano, Depth, Odom | ✓ | 5.64 | 54.7 | 44.8 | 61.9 | — | — | — |
| HNR | Pano, Depth, Odom | ✓ | 5.50 | 56.3 | 46.7 | 63.5 | — | — | — |
| AO-Planner | Pano, Depth | ✗ | 7.06 | 43.3 | 30.5 | 50.1 | — | — | — |
| UniNaVid | RGB | ✗ | 6.24 | 48.7 | — | — | 14231.96 | 157.29 | 4376.68 |
| NaVILA | RGB | ✗ | 6.12 | 52.3 | 46.1 | 61.0 | 16119.98 | 141.25 | 3930.21 |
| **SpikeVLA** | **RGB** | **✗** | **6.20** | **51.9** | **45.3** | **60.4** | **6249.18** | **49.09** | **1196.16** |

**关键发现**: 在更长路径的 RxR-CE benchmark 上，SpikeVLA 与 NaVILA 性能基本持平（SR 差 0.4%，nDTW 差 0.6%），但能耗仍为其 34.7%。

### Table 4: 低层控制策略性能

| Method | Linear Vel. Error↓ (m/s) | Angular Vel. Error↓ (rad/s) | Mem (MB)↓ | Eng (μJ)↓ | ACEs (10⁶)↓ |
|--------|--------------------------|------------------------------|-----------|---------|------------|
| NaVILA | 0.23 | 0.38 | 1.20 | 5.80 | 161.48 |
| **SpikeVLA** | **0.42** | **0.29** | **2.35** | **0.31** | **5.53** |

**关键发现**: Spike-A 在角速度控制上优于 ANN（0.29 vs 0.38 rad/s），线速度略差（0.42 vs 0.23 m/s）。能耗降低 94.7%（0.31 vs 5.80 μJ），ACE 降低 96.6%。

### Table 5: ANN 量化基线对比（SpikeVLA vs INT4）

| Method | NE↓ | OS↑ | SR↑ | SPL↑ | Mem (GB)↓ | Eng (J)↓ | ACEs (10¹²)↓ |
|--------|-----|-----|-----|------|-----------|---------|-------------|
| NaVILA (FP16) | 5.28 | 61.5 | 53.9 | 49.3 | 15.7 | 141.25 | 3930.21 |
| NaVILA (INT4) | 5.66 | 56.8 | 48.2 | 43.6 | 8.6 | 72.49 | 982.55 |
| **SpikeVLA** | **5.38** | **63.4** | **53.3** | **47.9** | **6.1** | **49.09** | **1196.16** |

**关键发现**: INT4 量化使 NaVILA 的 SR 从 53.9% 降至 48.2%，而 SpikeVLA 保持 53.3% 且能耗（49.09 J）低于 INT4（72.49 J）。SpikeVLA 实现了优于量化方法的精度-效率权衡。

### Table 6: 种群编码核函数消融

| Kernel Type | Rewards | MEL↑ | Mem (MB)↓ | Eng (μJ)↓ | ACEs (10⁶)↓ |
|-------------|---------|------|-----------|---------|------------|
| ANN | 33.45 | 976.81 | 1.20 | 5.80 | 161.48 |
| Gaussian RBF | 23.10 | 973.11 | 2.35 | 0.41 | 7.34 |
| Inverse Multiquadric | 22.73 | 939.35 | 2.35 | 0.68 | 12.06 |
| Triangular | 25.15 | 966.29 | 2.35 | 0.25 | 4.42 |
| **Laplacian** | **26.72** | **983.94** | **2.35** | **0.31** | **5.53** |

**关键发现**: Laplacian 核在奖励（26.72）和 MEL（983.94）上均优于其他 SNN 核，仅比 ANN 基线（33.45）低约 20%，但能耗仅 0.31 μJ（ANN 为 5.80 μJ）。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| R2R-CE | Val-Unseen: 783 episodes | 真实室内环境导航，仅 RGB | 主要评测 |
| RxR-CE | Val-Unseen: 长视距 | 更长路径、多语言指令 | 泛化性验证 |
| VLN-CE-Isaac | 1,077 episodes | 引入真实物理动力学 | 真实部署评测 |
| Unitree Go2 仿真 | 多种地形 | 崎岖、斜坡、障碍物 | 低层控制评测 |

### 实现细节

- **Spike-V 基础模型**: SigLIPv2（视觉编码器）
- **Spike-L 基础模型**: LLaMA-8B
- **Spike-A 训练算法**: PPO + 代理梯度时空反向传播（STBP）
- **导航时间步**: T=16
- **动作控制时间步**: T=3（平衡精度与能耗）
- **种群编码尺寸**: P=5
- **能耗计算工艺节点**: 45nm

### 可视化结果

- **导航任务**: SpikeVLA 能够在无地图、无路径点前提下仅凭 RGB 图像完成长视距室内导航
- **地形适应**: 在崎岖、斜坡、障碍物三类环境中 Spike-A 均能稳定跟踪速度指令
- **能效**: 约 1/3 的能耗、约 1/2.5 的显存

---

## 批判性思考

### 优点
1. **首创性**: 首个将 SNN 完整应用于 VLA pipeline 的工作，覆盖视觉-语言-动作三个模态，系统完整
2. **能效领先**: 34% 能耗即达到 ANN 相当性能，优于 INT4 量化（效率-精度权衡更优）
3. **微分编码解决难点**: 非线性层（LayerNorm、Attention）的 SNN 转换是此前工作的痛点，本文提出微分梯度单元系统性地解决了这一问题

### 局限性
1. **神经形态硬件未验证**: 能耗节省基于理论模型（45nm MAC/AC 估算），实际神经形态芯片（如 Intel Loihi、BrainScaleS）上的验证缺失
2. **低层控制精度略差**: Spike-A 线速度误差（0.42 m/s）高于 ANN 基线（0.23 m/s），对精密操作任务可能不足
3. **SR 略低于最优 ANN**: VLN-CE-Isaac 上 SR 32.7% vs NaVILA-R 36.5%，差距约 4%

### 潜在改进方向
1. 在 Intel Loihi 2、BrainScaleS-2 等神经形态芯片上实测能耗，验证理论估算
2. 引入连续值脉冲编码（如率编码 vs 时间编码的混合）提升低层控制精度
3. 扩展到双手灵巧操作等需要更高精度控制的机器人任务

### 可复现性评估
- [ ] 代码开源（论文未提供 GitHub 链接）
- [ ] 预训练模型（未公开）
- [x] 训练细节完整（hyperparameter 和消融实验详细）
- [x] 数据集可获取（R2R-CE、RxR-CE 等公开 benchmark）

---

## 关联笔记

### 基于
- [[NaVILA]]: 主要对比基线，ANN-based VLA 导航系统
- [[LLaMA]]: Spike-L 的基础语言模型（8B 参数）
- [[SigLIPv2]]: Spike-V 的基础视觉编码器
- [[PPO]]: Spike-A 的训练算法

### 对比
- [[NaVid]]: RGB-only 导航方法，SpikeVLA 的对比基线之一
- [[UniNaVid]]: 统一导航基线
- [[ETPNav]]: 全景传感器导航方法，代表上界性能

### 方法相关
- [[脉冲神经网络|Spiking Neural Network (SNN)]]: 核心计算范式
- [[种群编码|Population Encoding]]: Spike-A 的连续-脉冲转换机制
- [[代理梯度|Surrogate Gradient]]: SNN 反向传播的关键技术
- [[突触操作|Synaptic Operation (SOP)]]: 替代 MAC 的能效计算单元
- [[微分编码|Differential Coding]]: 本文核心创新之一

### 硬件/数据相关
- [[Unitree Go2]]: 验证 Spike-A 的四足机器人平台
- [[VLN-CE]]: 视觉语言导航的连续环境 benchmark

---

## 速查卡片

> [!summary] SpikeVLA (ICML 2026)
> - **核心**: 首个全 SNN VLA 框架，覆盖视觉-语言-动作三模块
> - **方法**: 微分编码解决非线性 SNN 转换；Laplacian 种群编码实现连续控制
> - **结果**: R2R-CE SR=53.3%，能耗仅为 NaVILA 的 34.7%（49.09 J vs 141.25 J）
> - **代码**: 未公开

---

*笔记创建时间: 2026-06-30*
