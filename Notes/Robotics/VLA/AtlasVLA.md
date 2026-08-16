---
title: "AtlasVLA: Persistent World-Ego State Modeling for Vision-Language-Action Models"
method_name: "AtlasVLA"
authors: [Guiyu Zhao, Longteng Guo, Yanghong Mei, Zilin Zhu, Yu Zhang, Bin Cao, Mingming Yu, Xingjian He, Jie Jiang, Jing Liu]
year: 2026
venue: arXiv
tags: [vla, memory-augmented, long-horizon-manipulation, world-model, diffusion-policy, wrist-camera, manipulation]
zotero_collection: Robotics/VLA
image_source: online
arxiv_html: https://arxiv.org/html/2608.06729v1
created: 2026-08-11
---

# 论文笔记：AtlasVLA: Persistent World-Ego State Modeling for Vision-Language-Action Models

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Zhejiang University; University of Cambridge; University of Bristol |
| 日期 | August 2026 |
| 项目主页 | 未公开 |
| 对比基线 | [[MemoryVLA]] |
| 链接 | [arXiv](https://arxiv.org/abs/2608.06729) |

---

## 一句话总结

> AtlasVLA 通过 4D 持久空间状态记忆与自我工作状态记忆的双记忆机制，让仅用腕部相机的 VLA 模型在长时序操作任务中超越多视角基线，取得 LIBERO-Long 9.4% 和真实世界 17.5% 的成功率提升。

---

## 核心贡献

1. **[[Persistent World State Memory|持久世界状态记忆]]**: 将 2D 腕部相机观测通过深度估计和空间反投影提升到 4D 潜空间，用体素哈希策略维护全局空间地图，解决物体离开视野后的"感知遗忘"问题。
2. **[[Ego-Working State Memory|自我工作状态记忆]]**: 使用意图感知可学习查询追踪任务进度，通过冗余感知潜在整合防止语义重复、界定记忆增长边界，解决多步任务中的"任务进度遗忘"问题。
3. **世界-自我引导的动作生成**: 采用双路径检索机制——自我工作记忆检索提供时序上下文，自我引导世界检索提供空间锚定——结合解耦步骤条件 [[Diffusion Transformer (DiT)|扩散变换器]] 生成连续 7-DoF 末端执行器动作。

---

## 问题背景

### 要解决的问题

当前 [[VLA（视觉-语言-动作模型）]] 在仅腕部相机设置下存在两大瓶颈：
1. **感知遗忘（Perception Forgetting）**: 腕部相机视野有限，物体移出画面后模型失去对其位置的感知。
2. **任务进度遗忘（Task-Progress Forgetting）**: 反应式模型缺乏时序上下文，无法感知已完成的操作步骤。

### 现有方法的局限

- 大多数 VLA 模型依赖第三视角相机或多视角配置，在仅腕部相机场景下性能严重下降。
- [[MemoryVLA]] 等现有记忆方法以时序缓存为主，缺乏显式空间建模，无法根本解决部分可观测性问题。
- 循环神经网络（RNN）等传统时序建模方式难以有效编码空间几何信息。

### 本文的动机

作者认为，机器人需要同时维护两类持久状态：**对外部世界的空间感知**（世界状态）与**对自身任务进度的意图感知**（自我状态）。通过将 2D 观测提升到 4D 表征并持久化维护，可从根本上解决部分可观测性带来的感知瓶颈；通过意图感知查询机制可解决长时序任务的进度追踪问题。

---

## 方法详解

### 模型架构

AtlasVLA 采用 **双记忆 + 扩散策略** 架构：

- **输入**: 语言指令 $L$ + 腕部相机 RGB 观测 $O^w_t$ + 本体感觉状态 $S_t$
- **视觉编码器**: [[DINOv2]] 和 [[SigLIP]] 处理 RGB；[[Depth Anything 3]] 估计深度
- **语言骨干**: LLaMA-2 7B 进行语义推理
- **记忆模块**: [[Persistent World State Memory|持久世界状态记忆]] + [[Ego-Working State Memory|自我工作状态记忆]]
- **动作头**: 约 300M 参数的 [[Diffusion Transformer (DiT)|扩散变换器]] 生成动作块
- **输出**: [[Action Chunking|动作块]] $A_t = [a_t, a_{t+1}, \ldots, a_{t+k-1}]$（chunk size = 16）

### 核心模块

#### 模块1：持久世界状态记忆（Persistent World State Memory）

**设计动机**: 将腕部相机的局部 2D 观测转化为全局 3D 空间表征，使模型能"记住"已扫视过的空间区域。

**具体实现**:

1. **瞬时世界状态构建**：
   - 提取腕部 RGB 的视觉 token $X^w_t$
   - 使用 [[Depth Anything 3]] 估计深度图 $D^w_t$
   - 获取相机内参 $\mathbf{T}^{in}$ 和外参 $\mathbf{T}^{ex}_t$
   - 通过[[深度引导反投影|空间反投影]]将 2D token 提升到 3D 潜空间，获得 3D 位置 $P_t$ 和 3D 特征 $m_t$

2. **时空位置编码**：
   - 注入可学习空间位置编码 $\mathcal{E}_{spatial}(P_t)$ 提供精确几何感知
   - 注入可学习时序位置编码 $\mathcal{E}_{temporal}(t)$ 防止空间混叠

3. **世界状态时空更新**：
   - 采用[[体素哈希]]策略维护持久全局地图
   - 对局部体素内的潜特征进行加权聚合（灵感来源于 [[TSDF]] 映射技术）
   - "永久初始化"规则：第一帧状态被锚定以提供持久全局上下文
   - 时序滑窗 + 指数衰减权重限制记忆容量（最大 2048 token）

#### 模块2：自我工作状态记忆（Ego-Working State Memory）

**设计动机**: 追踪任务进度和意图状态，防止多步任务中的意图遗忘。

**具体实现**:

1. **意图感知查询（Intent-aware Query）**：
   - 4 个可学习查询 $Q^{ego}$ 通过交叉注意力从视觉-语言 token 中聚合目标导向信息
   - 输出当前时刻的自我工作 token $Z^{ego}_t$

2. **自我工作记忆库（Ego-Working Memory Bank）**：
   - 维护时序意图记忆：$\mathcal{M}^{ego}_t = \text{Cons}(\mathcal{M}^{ego}_{t-1} \cup \{Z^{ego}_t + \mathcal{E}_{temporal}(t)\})$
   - 通过冗余感知整合（Redundancy-aware Consolidation）合并时序相邻、语义相似的意图 token
   - 记忆长度默认设为 16

#### 模块3：世界-自我引导动作生成（World-Ego-Guided Action Generation）

**设计动机**: 将持久世界状态和自我工作状态注入扩散过程，实现空间锚定与时序一致的动作生成。

**具体实现**:

1. **自我工作记忆检索**：当前自我工作 token 通过交叉注意力检索历史意图记忆，确保时序一致的动作解码

2. **自我引导世界检索**：以历史自我工作上下文 $C^{ego}_t$ 为查询，对全局世界状态记忆 $\mathcal{M}_t$ 进行意图注意力（IntentAttn），选择性提取与当前任务相关的空间状态

3. **解耦步骤条件扩散变换器**：
   - 标准扩散模型的条件在每步去噪中保持不变
   - AtlasVLA 采用解耦机制：在每个扩散步骤中，带噪动作 token 先关注自我工作上下文（任务进度），再关注世界表征（几何锚定）
   - 两种注意力机制使用独立的 attention head

---

## 关键公式

### 公式1：[[Action Chunking|动作块生成策略]]

$$
A_t = [a_t, a_{t+1}, \ldots, a_{t+k-1}] \sim \pi_\theta(\cdot \mid O^w_t, S_t, L)
$$

**含义**: AtlasVLA 的策略函数 $\pi_\theta$ 在给定腕部观测、本体状态和语言指令的条件下生成连续 $k$ 步动作块。

**符号说明**:
- $A_t$: $t$ 时刻起的动作块（chunk size $k=16$）
- $O^w_t$: 腕部相机 RGB 观测
- $S_t$: 机器人本体感觉状态（关节角、末端位姿等）
- $L$: 自然语言任务指令
- $\pi_\theta$: AtlasVLA 策略网络（参数 $\theta$）

### 公式2：[[深度引导反投影|3D 空间反投影]]

$$
m_t, P_t = \text{Back-Projection}(X_w^t, D^w_t, \mathbf{T}^{in}, \mathbf{T}^{ex}_t)
$$

**含义**: 将 2D 视觉 token 和深度图通过相机内外参反投影到 3D 潜空间，得到带 3D 坐标的特征表征。

**符号说明**:
- $X_w^t$: 腕部 RGB 提取的视觉 token
- $D^w_t$: 深度图（[[Depth Anything 3]] 估计）
- $\mathbf{T}^{in}$: 相机内参矩阵（焦距、主点）
- $\mathbf{T}^{ex}_t$: 相机外参矩阵（机器人当前姿态决定）
- $m_t$: 3D 潜特征
- $P_t$: 对应的 3D 坐标

### 公式3：[[Persistent World State Memory|时空嵌入融合]]

$$
\widehat{m}_t = m_t + \mathcal{E}_{spatial}(P_t) + \mathcal{E}_{temporal}(t)
$$

**含义**: 将 3D 潜特征叠加空间位置编码和时序位置编码，赋予特征精确的时空感知能力，防止空间混叠。

**符号说明**:
- $\widehat{m}_t$: 时空增强后的特征
- $\mathcal{E}_{spatial}(P_t)$: 可学习空间位置编码（编码 3D 坐标）
- $\mathcal{E}_{temporal}(t)$: 可学习时序位置编码（编码时间步）

### 公式4：[[TSDF|体素加权聚合更新]]

$$
\mathcal{M}_t(v) = \frac{\mathcal{W}_{t-1}(v)\mathcal{M}_{t-1}(v) + w_t m_t(v)}{\mathcal{W}_{t-1}(v) + w_t}
$$

$$
w_t(v) = c_t(v), \quad \mathcal{W}_t(v) = \lambda\mathcal{W}_{t-1}(v) + w_t(v)
$$

**含义**: 对体素 $v$ 中的历史特征和新观测特征进行加权平均融合，权重由观测置信度 $c_t(v)$ 决定，历史权重以衰减因子 $\lambda$ 指数衰减。灵感来源于 TSDF 映射中的加权融合技术。

**符号说明**:
- $\mathcal{M}_t(v)$: $t$ 时刻体素 $v$ 的聚合特征
- $\mathcal{W}_t(v)$: $t$ 时刻体素 $v$ 的累积权重
- $w_t(v) = c_t(v)$: 当前观测置信度（作为融合权重）
- $\lambda$: 历史权重衰减因子（指数衰减以限制记忆容量）

### 公式5：[[Ego-Working State Memory|意图感知查询注意力]]

$$
Z^{ego} = \operatorname{Softmax}\!\left(\frac{Q^{ego}K^T}{\sqrt{d}}\right)V
$$

**含义**: 可学习意图查询 $Q^{ego}$ 通过缩放点积注意力从视觉-语言 token（键 $K$、值 $V$）中聚合目标导向信息，得到当前时刻的自我工作 token。

**符号说明**:
- $Q^{ego}$: 可学习意图查询（4 个 token）
- $K, V$: 来自视觉-语言骨干的键和值（语义上下文）
- $d$: 注意力头的维度（用于缩放）
- $Z^{ego}$: 输出的自我工作 token

### 公式6：[[Ego-Working State Memory|自我工作记忆库更新]]

$$
\mathcal{M}^{ego}_t = \text{Cons}\!\left(\mathcal{M}^{ego}_{t-1} \cup \{Z^{ego}_t + \mathcal{E}_{temporal}(t)\}\right)
$$

**含义**: 将当前时刻的时序增强自我工作 token 加入历史记忆库，并通过冗余感知整合（Cons）压缩相邻时间步中语义相似的 token，以绑定记忆增长。

**符号说明**:
- $\mathcal{M}^{ego}_t$: 更新后的自我工作记忆库
- $\text{Cons}(\cdot)$: 冗余感知整合函数（合并语义相似 token）
- $Z^{ego}_t + \mathcal{E}_{temporal}(t)$: 加时序位置编码的当前意图 token

### 公式7：自我工作记忆检索

$$
C^{ego}_t = \text{CrossAttn}(Z^{ego}_t, \mathcal{M}^{ego}_t, \mathcal{M}^{ego}_t)
$$

**含义**: 以当前自我工作 token 为查询，对历史意图记忆库进行交叉注意力检索，获取时序一致的任务进度上下文。

**符号说明**:
- $C^{ego}_t$: 检索得到的自我工作上下文
- $Z^{ego}_t$: 查询（当前意图 token）
- $\mathcal{M}^{ego}_t$: 键和值（历史意图记忆库，最大长度 16）

### 公式8：自我引导世界检索

$$
C^{world}_t = \text{AddNorm}(\text{FFN}(\text{IntentAttn}(C^{ego}_t, \mathcal{M}_t, \mathcal{M}_t)))
$$

**含义**: 以任务进度上下文 $C^{ego}_t$ 为查询，通过意图注意力对全局 4D 世界状态记忆 $\mathcal{M}_t$ 进行选择性检索，提取当前任务相关的空间状态。

**符号说明**:
- $C^{world}_t$: 检索得到的世界状态上下文
- $\text{IntentAttn}$: 意图注意力（以自我工作上下文为查询导向世界状态检索）
- $\mathcal{M}_t$: 全局 4D 世界状态记忆（最大 2048 token）
- $\text{FFN}$: 前馈网络；$\text{AddNorm}$: 残差归一化

### 公式9：[[Diffusion Transformer (DiT)|前向扩散过程]]

$$
a_k = \sqrt{\bar{\alpha}_k}A_t + \sqrt{1-\bar{\alpha}_k}\epsilon
$$

**含义**: 扩散过程的正向加噪公式，将干净动作 $A_t$ 在扩散步 $k$ 处混入高斯噪声。

**符号说明**:
- $a_k$: 扩散步 $k$ 的带噪动作
- $\bar{\alpha}_k$: 扩散步 $k$ 的信噪比累积系数
- $\epsilon \sim \mathcal{N}(0, I)$: 标准高斯噪声
- $A_t$: 干净目标动作块

### 公式10：[[Diffusion Transformer (DiT)|去噪扩散损失]]

$$
\mathcal{L}_{act} = \mathbb{E}_{A_t, \epsilon, k}\!\left[\|\epsilon - \epsilon_\theta(a_k, k, C^{ego}_t, C^{world}_t)\|_2^2\right]
$$

**含义**: 扩散模型训练目标：让去噪网络 $\epsilon_\theta$ 在给定自我工作上下文 $C^{ego}_t$ 和世界状态上下文 $C^{world}_t$ 条件下准确预测加入的噪声。

**符号说明**:
- $\epsilon_\theta$: 参数化去噪网络（~300M 参数扩散变换器）
- $k$: 扩散时间步（推理时使用 10 步 DDIM）
- $C^{ego}_t$: 自我工作上下文（任务进度条件）
- $C^{world}_t$: 世界状态上下文（空间几何条件）

---

## 关键图表

### Figure 1：双瓶颈问题与 AtlasVLA 优势对比

![Figure 1](https://arxiv.org/html/2608.06729v1/x1.png)

**说明**: 左侧展示当前反应式 VLA 的两大瓶颈——感知遗忘（物体移出腕部相机视野导致空间盲区）和任务进度遗忘（多步任务中无法追踪已完成动作）；右侧展示 AtlasVLA 通过双记忆机制如何解决这两个问题。

### Figure 2：AtlasVLA 整体架构

![Figure 2](https://arxiv.org/html/2608.06729v1/x2.png)

**说明**: 展示 AtlasVLA 的双记忆系统架构。下方为[[Persistent World State Memory|持久世界状态记忆]]模块（2D → 4D 提升 + 体素哈希更新），上方为[[Ego-Working State Memory|自我工作状态记忆]]模块（意图感知查询 + 冗余整合），右侧为世界-自我引导的扩散变换器动作头，三个模块通过双路径检索机制连接。

### Figure 3：真实世界长时序任务定性结果（一）

![Figure 3](https://arxiv.org/html/2608.06729v1/x3.png)

**说明**: AtlasVLA 在真实世界长时序操作任务上的定性可视化，展示机器人利用腕部相机在多步骤操作（如换方块位置、按序堆叠）中的执行过程。

### Figure 4：真实世界机器人平台

![Figure 4](https://arxiv.org/html/2608.06729v1/x4.png)

**说明**: AtlasVLA 使用的真实世界机器人平台，展示腕部相机的安装位置和硬件配置。整个系统仅使用腕部单目相机而非多视角或第三视角相机。

### Figure 5：真实世界长时序任务定性结果（二）

![Figure 5](https://arxiv.org/html/2608.06729v1/x5.png)

**说明**: 更多长时序任务的定性可视化，展示 AtlasVLA 在不同任务场景（清理桌面、按顺序拾取放置）中的执行效果。

### Figure 6：真实世界通用操作任务定性结果

![Figure 6](https://arxiv.org/html/2608.06729v1/x6.png)

**说明**: AtlasVLA 在通用操作任务（放置胡椒、堆叠积木、取放罐子等）上的定性可视化，展示模型在多种物体和任务类型上的泛化能力。

### Figure 7：LIBERO Benchmark 定性结果

![Figure 7](https://arxiv.org/html/2608.06729v1/x7.png)

**说明**: AtlasVLA 在 [[LIBERO]] 仿真 benchmark 上的任务执行可视化，覆盖 LIBERO-Spatial、Object、Goal、Long 四个子集的代表性任务。

### Figure 8：LIBERO-10 Long Horizon 更多定性结果

![Figure 8](https://arxiv.org/html/2608.06729v1/x8.png)

**说明**: AtlasVLA 在 LIBERO-10 长时序任务上的更多可视化，进一步验证持久记忆机制在复杂多步操作场景中的有效性。

### Table 1：LIBERO Benchmark 对比结果

| 方法 | 相机 | Spatial | Object | Goal | Long | 90 | Average |
|------|------|---------|--------|------|------|-----|---------|
| OpenVLA | 3rd | 84.7 | 88.4 | 79.2 | 53.7 | 73.5 | 75.9 |
| π₀ | 3rd | 90.8 | 91.8 | 89.6 | 80.2 | — | 88.1 |
| 4D-VLA | 3rd | 93.8 | 92.8 | 95.6 | 86.5 | — | 92.2 |
| CogACT | 3rd | 97.2 | 98.0 | 90.2 | 88.8 | 92.1 | 93.2 |
| MemoryVLA | 3rd | 98.4 | 98.4 | 96.4 | 93.4 | 95.6 | 96.5 |
| π₀ | 3rd + wrist | 96.8 | 98.8 | 95.8 | 85.2 | — | 94.2 |
| OpenVLA-OFT | 3rd + wrist | 97.6 | 98.4 | 97.9 | 94.5 | — | 97.1 |
| GE-ACT | 3rd + wrist | 98.2 | 97.6 | 95.8 | 94.4 | — | 96.5 |
| CogACT | wrist | 96.4 | 95.8 | 88.6 | 86.2 | 87.4 | 90.9 |
| π₀ | wrist | 94.4 | 96.6 | 90.8 | 80.8 | — | 90.7 |
| MemoryVLA | wrist | 96.2 | 99.2 | 96.4 | 87.6 | 90.7 | 94.0 |
| **AtlasVLA** | **wrist** | **99.4** | **99.8** | **98.2** | **94.6** | **95.8** | **97.6** |

**表格说明**: AtlasVLA（仅腕部）以 97.6% 平均成功率超越所有腕部基线，并超越多视角基线 MemoryVLA（3rd，96.5%）3.4 个百分点。在最具挑战性的 Long 子集上，从 MemoryVLA 腕部版的 87.6% 提升至 94.6%（+7.0%）。

### Table 2：RLBench Benchmark 对比结果

| 方法 | 相机 | Sweep | Phone | Umbrella | Frame | Wine | Water | Avg. |
|------|------|-------|-------|----------|-------|------|-------|------|
| OpenVLA | 3rd | 50.0 | 20.0 | 35.0 | 15.0 | 10.0 | 10.0 | 23.3 |
| CogACT | 3rd | 50.0 | 50.0 | 55.0 | 45.0 | 30.0 | 25.0 | 42.5 |
| FiS-VLA | 3rd | 55.0 | 50.0 | 50.0 | 70.0 | 55.0 | 20.0 | 50.0 |
| MemoryVLA | 3rd | 50.0 | 60.0 | 75.0 | 60.0 | 80.0 | 55.0 | 63.3 |
| π₀ | 3rd + wrist | 30.0 | 30.0 | 30.0 | 70.0 | 10.0 | 30.0 | 33.3 |
| GE-ACT | 3rd + wrist | 10.0 | 15.0 | 40.0 | 35.0 | 40.0 | 45.0 | 30.8 |
| CogACT | wrist | 40.0 | 35.0 | 50.0 | 35.0 | 20.0 | 25.0 | 34.2 |
| MemoryVLA | wrist | 40.0 | 55.0 | 65.0 | 60.0 | 60.0 | 50.0 | 55.0 |
| **AtlasVLA** | **wrist** | **70.0** | **70.0** | **80.0** | **65.0** | **75.0** | **65.0** | **70.8** |

**表格说明**: AtlasVLA 在 [[RLBench]] 6 项任务平均成功率 70.8%，超越 MemoryVLA 腕部版 15.8 个百分点，并超越所有第三视角基线。

### Table 3：真实世界通用操作任务结果

| 方法 | 相机 | Pepper on Plate | Pepper in Box | Stack Cubes | Carrot on Plate | Cube in Drawer | Can in Drawer | Avg. |
|------|------|-----------------|---------------|-------------|-----------------|----------------|---------------|------|
| π₀ | 3rd + wrist | 68.0 | 60.0 | 62.0 | 74.0 | 66.0 | 70.0 | 66.7 |
| MemoryVLA | 3rd | 72.0 | 64.0 | 66.0 | 78.0 | 70.0 | 74.0 | 70.7 |
| MemoryVLA | wrist | 62.0 | 56.0 | 58.0 | 70.0 | 62.0 | 66.0 | 62.3 |
| **AtlasVLA** | **wrist** | **78.0** | **72.0** | **76.0** | **84.0** | **82.0** | **80.0** | **78.7** |

**表格说明**: AtlasVLA（仅腕部）在真实世界通用任务上以 78.7% 超越 MemoryVLA 多视角版（70.7%），相对腕部基线 MemoryVLA 提升 16.4 个百分点。

### Table 4：真实世界长时序任务结果

| 方法 | Change Cubes | Stack Cubes Order | Clean Desk | Pick Place Order | Avg. |
|------|--------------|-------------------|------------|-----------------|------|
| π₀ | 54.0 | 50.0 | 52.0 | 52.0 | 52.0 |
| MemoryVLA | 62.0 | 58.0 | 60.0 | 62.0 | 60.5 |
| **AtlasVLA** | **74.0** | **66.0** | **68.0** | **70.0** | **69.5** |

**表格说明**: AtlasVLA 在真实世界长时序 4 项任务平均成功率 69.5%，超越 MemoryVLA 9.0 个百分点，验证双记忆机制在复杂长链任务中的优势。

### Table 5：消融实验（LIBERO + 真实世界长时序）

| 序号 | 配置 | LIBERO Avg. | Real-world Long |
|------|------|-------------|-----------------|
| 1 | w/o 世界状态记忆 | 93.5 | 54.0 |
| 2 | w/o 自我工作记忆 | 95.0 | 56.5 |
| 3 | **AtlasVLA（完整）** | **97.6** | **69.5** |
| 4 | w/o 世界状态更新 | 94.6 | 58.0 |
| 5 | w/ 世界状态更新 | 97.6 | 69.5 |
| 6 | w/o 空间位置编码 | 96.4 | 67.5 |
| 7 | w/o 时序位置编码 | 96.8 | 65.0 |
| 8 | 时空位置编码（完整）| 97.6 | 69.5 |
| 9 | w/o 世界状态条件 | 95.2 | 61.5 |
| 10 | w/ 世界状态条件 | 97.6 | 69.5 |

**关键发现**: 移除世界状态记忆导致真实世界长时序任务下降 15.5%（69.5% → 54.0%），移除自我工作记忆下降 13.0%（69.5% → 56.5%），两个模块缺一不可。时序位置编码（贡献 +4.5%）比空间位置编码（+2.0%）更重要。

### Table 6：超参数设置

| 超参数 | 值 |
|--------|-----|
| 全局 Batch Size | 256（32 × 8 GPU）|
| 学习率 | 2×10⁻⁵ |
| Action Chunk Size | 16 |
| 推理去噪步数 | 10（DDIM）|
| CFG Scale | 1.5 |
| Condition Dropout | 0.1 |
| 世界状态记忆容量 | 2048 token |
| 体素大小 | 0.025m |
| 自我工作记忆检索层数 | 2 |
| 自我引导世界检索层数 | 4 |
| 自我工作 Token 数 | 4 |

### Table 7：推理效率分析

| 方法 | 延迟 (s/step) | 吞吐量 (Hz) | GPU 内存 (GB) | 成功率 (%) |
|------|-------------|------------|--------------|-----------|
| MemoryVLA | 0.146 | 109.5 | 16.7 | 62.3 |
| **AtlasVLA** | **0.158** | **101.3** | **18.1** | **78.7** |

**表格说明**: AtlasVLA 仅比 MemoryVLA 多消耗 1.4GB GPU 内存，延迟增加 0.012s（8%），但成功率提升 16.4 个百分点，效率-性能权衡合理。

### Table 8：自我工作记忆长度消融

| 记忆长度 | LIBERO | Real-world Long |
|----------|--------|-----------------|
| 8 | 97.3 | 66.4 |
| **16（默认）** | **97.6** | **69.5** |
| 32 | 97.2 | 69.8 |

**关键发现**: 记忆长度 16 在性能和计算成本间取得最佳平衡（32 略优但差异不显著，8 有轻微损失）。

### Table 9：体素大小消融

| 体素大小 | LIBERO | Real-world Long |
|----------|--------|-----------------|
| 0.01m | 96.3 | 65.7 |
| **0.025m（默认）** | **97.6** | **69.5** |
| 0.05m | 97.2 | 64.0 |
| 0.1m | 95.9 | 58.5 |

**关键发现**: 体素过小（0.01m）计算开销大且特征稀疏，过大（0.1m）丢失精细几何信息，0.025m（约 2.5cm 分辨率）最优。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| [[LIBERO]] | 5 子集（Spatial/Object/Goal/Long/90）| 仿真 tabletop 操作，Long 子集含 10 步任务 | 仿真 benchmark 测试 |
| [[RLBench]] | 6 代表性任务 | 仿真，任务难度高，涉及精细操作 | 仿真 benchmark 测试 |
| 真实世界自采集 | 6 通用 + 4 长时序任务 | 真实机器人，6 类物体，腕部相机 | 真实场景验证 |

### 实现细节

- **视觉编码器**: [[DINOv2]] + [[SigLIP]]（RGB）；[[Depth Anything 3]]（深度估计）
- **语言骨干**: LLaMA-2 7B
- **动作头**: ~300M 参数扩散变换器
- **优化器**: AdamW，学习率 2×10⁻⁵
- **Batch Size**: 256（32 per GPU × 8 GPU）
- **体素大小**: 0.025m；**记忆容量**: 2048 token（世界状态）+ 16 token（自我工作）
- **推理**: 10 步 DDIM 去噪，CFG Scale 1.5
- **推理延迟**: 0.158s/step（101.3 Hz）

### 可视化结果

- 在 LIBERO 仿真中，AtlasVLA 能在物体被遮挡或移出视野后准确恢复操作。
- 在真实长时序任务（如 Clean Desk、Stack Cubes Order）中，模型能根据已完成步骤调整后续动作策略，而 MemoryVLA 常在步骤 3-4 后出现位置混乱或步骤遗漏。

---

## 批判性思考

### 优点

1. **从根本上解决部分可观测性**: 4D 空间记忆实现了超出即时视野的持久空间感知，比时序缓存更有原则性。
2. **双记忆正交互补**: 世界状态记忆（空间/外部）与自我工作记忆（任务进度/内部）分工明确，消融实验验证两者均不可或缺。
3. **仅腕部相机超越多视角基线**: 在实际部署中，单腕部相机配置更简单、成本更低，这一结果具有重要实用价值。
4. **效率合理**: 相比 MemoryVLA 仅增加 8% 延迟和 1.4GB 内存，性能提升 16.4%，权衡合理。

### 局限性

1. **深度估计误差传播**: 系统依赖 Depth Anything v3 的单目深度估计，在高反射、透明物体或低纹理场景下可能引入系统误差，影响 3D 空间建模精度。
2. **外参估计依赖**: 需要机器人提供精确的实时相机外参（末端位姿），对机器人标定和运动学精度有较高要求。
3. **体素分辨率与内存权衡**: 固定体素大小（0.025m）在超大空间场景或需要毫米级精度的精细操作中可能不足。
4. **未在双臂/移动机器人上验证**: 当前实验仅在单臂固定底座机器人上进行，泛化性有待验证。

### 潜在改进方向

1. 引入语义/实例分割提升体素特征的语义区分能力（如分别追踪不同物体）。
2. 使用立体相机或 RGB-D 传感器替代单目深度估计，提升空间精度。
3. 探索自适应体素大小策略，在任务相关区域使用更细粒度的体素。
4. 将记忆机制与主动感知（active perception）结合，让机器人主动扫视以填补记忆盲区。

### 可复现性评估

- [ ] 代码开源（论文中未提及代码链接）
- [ ] 预训练模型（论文中未提及模型发布）
- [x] 训练细节完整（超参数、架构细节较完善）
- [x] 数据集可获取（LIBERO、RLBench 均为公开 benchmark）

---

## 关联笔记

### 基于

- [[MemoryVLA]]: 主要对比基线和先验工作，AtlasVLA 在其时序缓存思路上增加显式空间建模
- [[Diffusion Transformer (DiT)]]: 动作生成头的骨干架构
- [[DINOv2]]: RGB 视觉编码器
- [[SigLIP]]: 视觉-语言对齐编码器
- [[Depth Anything 3]]: 单目深度估计模块

### 对比

- [[MemoryVLA]]: 主要对比对象，AtlasVLA 腕部版超越 MemoryVLA 多视角版
- [[VLA（视觉-语言-动作模型）]]: 通用 VLA 范式，AtlasVLA 是其记忆增强变体

### 方法相关

- [[Persistent World State Memory]]: 4D 持久空间状态记忆（本文核心创新 1）
- [[Ego-Working State Memory]]: 自我工作状态记忆（本文核心创新 2）
- [[Action Chunking]]: 动作块生成策略（chunk size = 16）
- [[TSDF]]: 体素加权聚合的灵感来源
- [[体素哈希]]: 世界状态记忆的空间索引结构
- [[深度引导反投影|空间反投影]]: 2D → 3D 提升的核心操作

### 硬件/数据相关

- [[LIBERO]]: 仿真 benchmark
- [[RLBench]]: 仿真 benchmark

---

## 速查卡片

> [!summary] AtlasVLA
> - **核心**: 双记忆机制（4D 世界状态 + 自我工作状态）使腕部相机 VLA 超越多视角基线
> - **方法**: 空间反投影 + 体素哈希持久记忆 + 意图感知查询 + 解耦步骤条件 DiT
> - **结果**: LIBERO 97.6%（+3.4% vs 多视角 MemoryVLA）；真实长时序 69.5%（+9.0% vs MemoryVLA）
> - **代码**: 暂未开源

---

*笔记创建时间: 2026-08-11*
