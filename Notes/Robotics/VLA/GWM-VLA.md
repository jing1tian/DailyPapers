---
title: "GWM-VLA: Geometry-Aware Latent World Modeling for Vision-Language-Action Learning"
method_name: "GWM-VLA"
authors: [Yanping Zhao, Hang Yu, Yiwei Wang, Chen Ye, Siyu Tian, Di Zhang, Qingjun Wang, Qian Chen, Junqiao Zhao, Guang Chen]
year: 2026
venue: arXiv
tags: [vla, world-model, geometry-aware, multi-view-encoding, flow-matching, robot-manipulation, robustness]
zotero_collection: Robotics/VLA
image_source: online
arxiv_html: https://arxiv.org/html/2608.07619v1/
created: 2026-08-12
---

# 论文笔记：GWM-VLA

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | 未标注（推测国内机构） |
| 日期 | August 2026 |
| 项目主页 | 未提供 |
| 对比基线 | [[VLA-JEPA]] |
| 链接 | [arXiv](https://arxiv.org/abs/2608.07619) |

---

## 一句话总结

> GWM-VLA 将 [[VGGT]] 的几何感知多视角编码与 [[Latent Predictive World Model|潜在世界模型]] 结合，通过共享的潜在动作表示同时驱动目标视角预测和流匹配动作生成，显著提升 VLA 在分布偏移下的鲁棒性。

---

## 核心贡献

1. **几何感知多视角状态编码**: 使用冻结的 [[VGGT-Ω]] 在每个时间步联合聚合多视角观测，保留跨视角几何关系（视觉对应、相机相对场景几何），而非独立编码后拼接。
2. **全局上下文条件目标视角预测**: 潜在世界模型仅预测目标视角（腕部摄像头）下一步的 patch tokens，同时以 register tokens 作为全局几何上下文条件；专注末端执行器运动和抓取交互，减少噪声。
3. **共享潜在动作表示**: 统一的潜在动作 tokens 同时条件化世界模型预测头和流匹配动作头，将预测学习与连续机器人控制耦合，确保两个目标联合塑造相同的表示空间。

---

## 问题背景

### 要解决的问题

现有 [[VLA-JEPA]] 等基于潜在世界建模的 VLA 方法在视觉和环境分布偏移下（相机视角变化、光照变化、背景变化等）鲁棒性不足，导致实际部署性能显著下降。

### 现有方法的局限

- **[[VLA-JEPA]]**: 独立编码每个摄像头视角，再通过拼接融合特征，跨视角几何关系是隐式的，缺乏显式几何先验。
- **WorldVLA 等完整状态预测方法**: 需要预测完整多视角状态，计算开销大，且预测目标包含过多与机器人控制无关的视觉变化（背景运动、相机运动等）。
- **独立视角潜在动作学习**: 过渡目标可能捕获主导视觉变化（相机运动、背景变化），而非机器人控制相关的变化。

### 本文的动机

几何感知基础模型（如 [[VGGT]] 系列）能够联合聚合多视角观测、建模跨视角关系。若将其引入潜在世界建模，可以显式地将几何约束注入状态表示，使世界模型预测更关注机器人控制相关的场景变化，从而提升跨分布鲁棒性。

---

## 方法详解

### 模型架构

GWM-VLA 采用 **多模块联合训练** 架构：
- **输入**: 语言指令 $\ell$ + 多视角观测 $O_t = \{I_t^v\}_{v=1}^{V}$ + 本体感知状态 $s_t$
- **Backbone**: Qwen3-VL-2B（视觉语言主干，可训练）+ [[VGGT-Ω]]（几何感知多视角编码器，冻结）
- **核心模块**: [[VGGT-Ω]] 用于联合多视角编码；[[Latent Predictive World Model|潜在世界模型]] 用于目标视角预测；[[Conditional Flow Matching|流匹配]]动作头用于连续动作生成
- **输出**: [[Action Chunking|动作块]] $u_{0:T-1}$，预测未来 $T$ 步机器人动作
- **硬件**: 8× NVIDIA A800 GPU（主实验）/ RTX 6000D（消融）

### 核心模块

#### 模块1: 几何感知多视角状态编码（VGGT-Ω Encoder）

**设计动机**: 利用 [[VGGT-Ω]] 在预训练中获得的跨视角几何理解能力，保留多视角之间的几何关系（相机相对位置、视觉对应），避免独立编码后信息损失。

**具体实现**:
- 在每个时间步 $t$，将所有视角图像 $\{I_t^v\}_{v=1}^V$ 输入冻结的 [[VGGT-Ω]] 进行联合处理
- 输出两类 token：**register tokens** $R_t$（全局几何上下文）和 **patch tokens** $P_t$（局部视角特征）
- 不同时间步独立处理（时间维度不进入 VGGT-Ω）
- 编码器在全程训练中保持冻结（❄️），保留几何基础模型的预训练能力

$$
\{R_t, P_t\} = E_\Omega\left(\{I_t^v\}_{v=1}^V\right)
$$

#### 模块2: 全局上下文条件目标视角预测（Latent World Model）

**设计动机**: 仅预测目标视角（腕部摄像头 $v^*$）的 patch tokens，而不预测完整多视角状态；register tokens 作为全局几何上下文条件，在降低预测难度的同时保留几何关系。

**具体实现**:
- 提取目标视角特征：$r_t = R_t^{v^*}, p_t = P_t^{v^*}$
- 潜在世界模型 $F_\phi$ 以时间因果方式处理序列：

$$
\hat{p}_{1:T} = F_\phi(p_{0:T-1},\ r_{0:T-1},\ A_{0:T-1})
$$

- **时间因果注意力掩码（[[Time-Causal Attention]]）**: 同一时间步内的 tokens 可以双向交互；时间步 $t$ 的 tokens 只能 attend 到不晚于 $t$ 的时间步 tokens

$$
M_{ij} = \begin{cases} 0 & \text{if } \tau(j) \leq \tau(i) \\ -\infty & \text{if } \tau(j) > \tau(i) \end{cases}
$$

- 训练时使用 [[Teacher Forcing]]：监督目标由未来多视角观测经 VGGT-Ω 编码得到

$$
p_{t+1} = E_\Omega^{(P, v^*)}(O_{t+1})
$$

#### 模块3: 统一潜在动作表示与流匹配动作头

**设计动机**: 共享的潜在动作表示同时驱动世界模型预测和动作生成，使两个目标联合塑造同一组表示，鼓励捕获机器人控制相关的动态变化。

**具体实现**:
- 在 Qwen3-VL-2B 中插入可学习的时间步分组查询 tokens $Q^A = [Q_0^A, Q_1^A, \ldots, Q_{T-1}^A]$，其中每个 $Q_t^A = [q_{t,1}^A, \ldots, q_{t,K}^A]$
- 所有表示从**初始观测**产生（而非各时间步独立输入）：

$$
A_{0:T-1} = Q_\theta(O_0,\ \ell,\ Q^A)
$$

- [[Conditional Flow Matching|条件流匹配]]动作头以共享潜在动作序列和当前本体感知状态为条件，生成 $T$ 步动作块：

$$
\mathcal{L}_\text{action} = \mathbb{E}\left[\left\|v_\psi(u_\gamma, \gamma \mid A_{0:T-1}, s_0) - (u_{0:T-1} - \varepsilon)\right\|_2^2\right]
$$

其中 $u_\gamma = \gamma \cdot u_{0:T-1} + (1-\gamma)\varepsilon$，$\gamma \in [0,1]$ 为流时间，$\varepsilon$ 为噪声。

---

## 关键公式

### 公式1: [[VGGT-Ω|几何感知多视角编码]]

$$
\{R_t, P_t\} = E_\Omega\left(\{I_t^v\}_{v=1}^V\right)
$$

**含义**: 在时间步 $t$，用冻结的 VGGT-Ω 联合聚合所有视角图像，输出 register tokens（全局几何上下文）和 patch tokens（局部视觉特征）。

**符号说明**:
- $I_t^v$: 时间步 $t$、视角 $v$ 的 RGB 图像
- $V$: 摄像头视角总数
- $R_t$: 全局 register tokens（跨视角几何汇总）
- $P_t$: 各视角 patch tokens（局部感知特征）

### 公式2: [[Latent Predictive World Model|目标视角潜在预测]]

$$
\hat{p}_{1:T} = F_\phi\left(p_{0:T-1},\ r_{0:T-1},\ A_{0:T-1}\right)
$$

**含义**: 潜在世界模型以过去的目标视角 patch tokens、register tokens 和潜在动作为条件，预测未来目标视角的 patch tokens 序列。

**符号说明**:
- $p_t = P_t^{v^*}$: 目标视角（腕部摄像头）的 patch tokens
- $r_t = R_t^{v^*}$: 目标视角对应的 register tokens（全局几何上下文）
- $A_{0:T-1}$: 共享潜在动作 tokens
- $\hat{p}_{t+1}$: 对下一时间步目标视角 patch tokens 的预测

### 公式3: [[Time-Causal Attention|时间因果注意力掩码]]

$$
M_{ij} = \begin{cases} 0 & \text{if } \tau(j) \leq \tau(i) \\ -\infty & \text{if } \tau(j) > \tau(i) \end{cases}
$$

**含义**: 允许同一时间步内 tokens 双向交互，同时禁止访问未来时间步的信息，实现因果自回归特性。

**符号说明**:
- $\tau(i)$: token $i$ 所属的时间步
- $M_{ij}=0$: token $i$ 可以 attend 到 token $j$
- $M_{ij}=-\infty$: token $i$ 不可 attend 到 token $j$（future masking）

### 公式4: [[Latent Predictive World Model|世界模型 L1 损失]]

$$
\mathcal{L}_\text{wm} = \frac{1}{T} \sum_{t=0}^{T-1} \left\|\hat{p}_{t+1} - p_{t+1}\right\|_1
$$

**含义**: 以 L1 范数衡量预测的目标视角 patch tokens 与真实未来编码之间的差异，强迫世界模型捕获机器人控制相关的视觉动态。

**符号说明**:
- $\hat{p}_{t+1}$: 世界模型预测的下一步 patch tokens
- $p_{t+1} = E_\Omega^{(P,v^*)}(O_{t+1})$: 真实未来多视角观测经 VGGT-Ω 编码的监督目标（[[Teacher Forcing]]）
- $T$: 轨迹时间步数

### 公式5: [[Conditional Flow Matching|条件流匹配动作损失]]

$$
\mathcal{L}_\text{action} = \mathbb{E}\left[\left\|v_\psi(u_\gamma, \gamma \mid A_{0:T-1}, s_0) - (u_{0:T-1} - \varepsilon)\right\|_2^2\right]
$$

**含义**: 训练向量场网络 $v_\psi$ 将噪声动作引导到真实动作，条件为共享潜在动作序列和初始本体感知状态。

**符号说明**:
- $v_\psi$: 流匹配向量场网络（动作头）
- $u_\gamma = \gamma \cdot u_{0:T-1} + (1-\gamma)\varepsilon$: 噪声-真值插值（$\gamma \in [0,1]$ 为流时间）
- $\varepsilon \sim \mathcal{N}(0, I)$: 高斯噪声
- $s_0$: 初始本体感知状态
- $A_{0:T-1}$: 条件潜在动作 tokens

### 公式6: [[World Model|联合优化目标]]

$$
\mathcal{L} = \mathcal{L}_\text{action} + \lambda \mathcal{L}_\text{wm}
$$

**含义**: 以动作生成为主目标、潜在预测为辅助目标，联合训练整个框架。

**符号说明**:
- $\lambda = 0.1$: 世界模型损失权重（默认值，辅助目标）
- $\mathcal{L}_\text{action}$: 流匹配动作损失（主损失）
- $\mathcal{L}_\text{wm}$: 世界模型 L1 预测损失（辅助损失）

---

## 关键图表

### Figure 1: 与 VLA-JEPA 的机制对比

![Figure 1 - GWM-VLA vs VLA-JEPA Overview](https://arxiv.org/html/2608.07619v1/overview.png)

**说明**: 左侧为 [[VLA-JEPA]] 的独立视角编码与多视角潜在预测方式；右侧为 GWM-VLA 的三个关键改进：[[VGGT-Ω]] 几何感知多视角编码、全局上下文条件目标视角预测、共享潜在动作条件化。

### Figure 2: GWM-VLA 整体架构

![Figure 2 - GWM-VLA Architecture](https://arxiv.org/html/2608.07619v1/architecture.png)

**说明**: 冻结的 [[VGGT-Ω]] 编码器（❄️）在每个时间步联合聚合多视角观测；共享潜在动作 tokens 同时条件化潜在世界模型预测头和流匹配动作头（🔥为可训练模块）。Qwen3-VL-2B 作为视觉语言主干，接受初始观测和语言指令，产生贯穿全序列的潜在动作表示。

### Figure 3: 真实世界实验结果

**说明**: SO-101 机器人在 ID（同分布）、OOD-task（跨任务）、OOD-layout（新布局）三种设定下的成功率柱状图。GWM-VLA 在有限示教数据（100 条）条件下，在 OOD-layout 场景中取得最高平均成功率，展示了几何感知表示对真实世界泛化的贡献。

### Figure 4: 目标视角选择与动作条件化消融

**说明**: 在 LIBERO-Spatial 套件上对比腕部视角预测（92.8%）与第三人称视角预测（87.8%），以及直接潜在动作条件化与独立具身动作查询（差 4.8 pp），验证两个设计决策的有效性。

### Figure 5: VGGT-Ω 表示深度影响

**说明**: 对比 VGGT-Ω 不同层深度（浅层 vs 深层）的表示用于下游 VLA 训练的效果，验证更深层特征包含更丰富几何与语义信息。

### Figure 6: 预测潜在状态的定性可视化

![Figure 6 - Latent State Visualization](https://arxiv.org/html/2608.07619v1/visualization.png)

**说明**: 使用单独训练的冻结深度解码器和相机坐标点云，可视化世界模型预测的下一步潜在状态。GWM-VLA 预测的潜在状态能够重建出合理的深度图和三维点云，验证潜在世界模型捕获了有意义的几何动态信息。

### Table 1: LIBERO 标准 Benchmark 成功率 (%)

| 方法 | Spatial | Object | Goal | LIBERO-10 | Avg. |
|------|---------|--------|------|-----------|------|
| LAPA | 73.8 | 74.6 | 58.8 | 55.4 | 65.7 |
| UniVLA | 96.5 | 96.8 | 95.6 | 92.0 | 95.2 |
| OpenVLA-OFT | 97.6 | 98.4 | 97.9 | 94.5 | 97.1 |
| π₀ | 96.8 | 98.8 | 95.8 | 85.2 | 94.2 |
| π₀-FAST | 96.4 | 96.8 | 88.6 | 60.2 | 85.5 |
| CoT-VLA | 87.5 | 91.6 | 87.6 | 69.0 | 81.1 |
| WorldVLA | 87.6 | 96.2 | 83.4 | 60.0 | 81.8 |
| villa-X | 97.5 | 97.0 | 91.5 | 74.5 | 90.1 |
| GR00T N1 | 94.4 | 97.6 | 93.0 | 90.6 | 93.9 |
| π₀.₅ | 98.8 | 98.2 | 98.0 | 92.4 | 96.9 |
| VLA-JEPA (无人类视频) | 94.8 | 99.6 | 95.8 | 94.0 | 96.1 |
| **GWM-VLA** | **96.8** | **99.0** | **98.0** | **94.4** | **97.1** |

**说明**: GWM-VLA 在仅使用机器人演示（无人类视频预训练）的条件下，平均成功率 97.1%，与顶尖方法持平。

### Table 2: LIBERO-Plus 鲁棒性 Benchmark 成功率 (%)

| 方法 | Camera | Robot | Language | Light | Background | Noise | Layout | Avg. |
|------|--------|-------|----------|-------|------------|-------|--------|------|
| UniVLA | 1.8 | 46.2 | 69.6 | 69.0 | 81.0 | 21.2 | 31.9 | 42.9 |
| OpenVLA-OFT | 56.4 | 31.9 | 79.5 | 88.7 | 93.3 | 75.8 | 74.2 | 69.6 |
| π₀ | 13.8 | 6.0 | 58.8 | 85.0 | 81.4 | 79.0 | 68.9 | 53.6 |
| π₀-FAST | 65.1 | 21.6 | 61.0 | 73.2 | 73.2 | 74.4 | 68.8 | 61.6 |
| WorldVLA | 0.1 | 27.9 | 41.6 | 43.7 | 17.1 | 10.9 | 38.0 | 25.0 |
| VLA-JEPA (无人类视频) | 40.3 | 55.7 | 72.9 | 88.2 | 70.5 | 38.2 | 74.6 | 62.9 |
| **GWM-VLA** | **57.9** | **54.7** | **89.8** | **95.4** | **90.8** | **72.5** | **77.1** | **76.9** |

**关键发现**: GWM-VLA 以 76.9% 平均成功率超越机器人演示版 VLA-JEPA 基线 **14.0 个百分点**，在 Camera 和 Language 扰动维度提升最为显著。WorldVLA 在 Camera（0.1%）和 Background（17.1%）下崩溃，说明完整状态预测对视角变化极其敏感。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| DROID | ~76,000 条轨迹 | 多机构、多操作员、多摄像头，多样任务/背景/工作空间 | 预训练 |
| LIBERO | 标准演示集（4 套任务） | Franka Emika Panda，Spatial/Object/Goal/LIBERO-10 | 仿真微调+评测 |
| LIBERO-Plus | LIBERO 扩展（7 类扰动） | Camera/Robot/Language/Light/Background/Noise/Layout | 鲁棒性评测 |
| SO-101 Real | 100 条遥操作演示 | 5 个拾放任务，ID/OOD-task/OOD-layout | 真实机器人评测 |

### 实现细节

- **视觉语言主干**: Qwen3-VL-2B（全程可训练 🔥）
- **几何编码器**: [[VGGT-Ω]]（全程冻结 ❄️，使用较深层特征）
- **动作头**: [[Conditional Flow Matching|条件流匹配]]，预测 $T$ 步动作块
- **损失权重**: $\lambda = 0.1$（世界模型辅助损失）
- **目标视角**: 腕部摄像头（$v^* =$ wrist cam）
- **硬件（主实验）**: 8× NVIDIA A800 GPU
- **硬件（消融）**: 单卡 RTX 6000D GPU
- **部署框架**: LeRobot（真实机器人实验）

### 可视化结果

定性分析（Figure 6）表明，GWM-VLA 预测的潜在状态通过冻结的深度解码器可重建出有意义的深度图和相机坐标系三维点云，验证 [[VGGT-Ω]] 的几何感知表示被成功整合进世界模型的时序预测中。腕部摄像头作为目标视角，使模型优先关注末端执行器与目标物体的交互区域。

---

## 批判性思考

### 优点

1. **几何先验注入自然**: 复用冻结的 [[VGGT-Ω]] 无需额外几何监督，几何感知以零额外标注代价嵌入 VLA 训练。
2. **鲁棒性提升显著**: 在 LIBERO-Plus 7 类扰动中均超越 VLA-JEPA，尤其 Camera 扰动从 40.3% 提升到 57.9%，证明几何约束对视角鲁棒性有直接贡献。
3. **计算高效的预测目标**: 仅预测单视角 patch tokens（而非完整多视角状态），在性能与计算开销之间取得更好平衡。

### 局限性

1. **强多视角依赖**: 需要每个时间步的同步多视角观测，无法直接扩展利用大规模单视角互联网视频数据进行预训练。
2. **固定目标视角**: 腕部摄像头作为固定目标视角，在腕部摄像头缺失或不同摄像头配置的机器人平台上泛化性有限。
3. **编码器冻结假设**: VGGT-Ω 固定参数的假设可能限制端到端微调带来的进一步性能提升。

### 潜在改进方向

1. **动态目标视角选择**: 根据任务语义自适应选择最相关视角作为预测目标（参考 Action-Entropy View Selection 思路）。
2. **单视角兼容**: 探索在无多视角输入时利用单视角 VGGT 特征维持基本功能，提升部署灵活性。
3. **端到端几何编码器微调**: 在保持几何先验的同时，探索 VGGT-Ω 的轻量级适配（如 LoRA），可能进一步提升任务专化性能。

### 可复现性评估

- [ ] 代码开源
- [ ] 预训练模型
- [x] 训练细节完整（论文中有详细实现说明）
- [x] 数据集可获取（DROID、LIBERO 均为公开数据集）

---

## 关联笔记

### 基于

- [[VLA-JEPA]]: 主要对比基线，GWM-VLA 在其潜在世界建模框架基础上引入几何感知改进
- [[VGGT]]: 几何感知多视角编码基础模型，VGGT-Ω 是其扩展版本

### 对比

- [[VLA-JEPA]]: 独立视角编码 vs GWM-VLA 的联合几何感知编码；LIBERO-Plus 差 14 pp
- [[WorldVLA]]: 完整多视角状态预测（鲁棒性差，Camera 0.1%）vs GWM-VLA 的目标视角潜在预测（57.9%）

### 方法相关

- [[VGGT-Ω]]: 核心几何感知编码器
- [[Conditional Flow Matching]]: 动作头采用的生成方式
- [[Latent Predictive World Model]]: 潜在世界模型范式
- [[Time-Causal Attention]]: 世界模型内部使用的时序注意力机制
- [[Latent Action]]: 共享潜在动作表示的概念基础
- [[Action Chunking]]: 动作预测的输出形式

### 硬件/数据相关

- [[DROID]]: 预训练数据集
- [[LIBERO]]: 仿真评测 Benchmark

---

## 速查卡片

> [!summary] GWM-VLA
> - **核心**: 几何感知 VGGT-Ω 编码 + 目标视角潜在预测 + 共享潜在动作条件化
> - **方法**: 冻结 VGGT-Ω 多视角联合编码，仅预测腕部摄像头 patch tokens，共享潜在动作驱动流匹配动作头
> - **结果**: LIBERO 97.1% (与 SOTA 持平)；LIBERO-Plus **76.9%**（+14pp vs VLA-JEPA）
> - **代码**: 未开源

---

*笔记创建时间: 2026-08-12*
