---
title: "GaussVLA: Geometry-Aware Spatial Reasoning for Vision-Language-Action Model"
method_name: "GaussVLA"
authors: [Md Selim Sarowar, Md Tanvir Islam, Sungho Kim, Sangtae Ahn]
year: 2026
venue: BMVC
tags: [vla, 3d-gaussian, geometry-aware, flow-matching, depth-estimation, spatial-reasoning, mamba, chain-of-thought]
zotero_collection: Robotics/VLA
image_source: online
arxiv_html: https://arxiv.org/html/2608.24959v1
created: 2026-08-28
---

# 论文笔记：GaussVLA: Geometry-Aware Spatial Reasoning for Vision-Language-Action Model

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | 未明确（来自 cs.RO/cs.CV） |
| 日期 | August 2026 |
| 项目主页 | N/A |
| 对比基线 | [[SpatialVLA]], [[CoT-VLA]], [[π₀]] |
| 链接 | [arXiv](https://arxiv.org/abs/2608.24959) / Code N/A |

---

## 一句话总结

> GaussVLA 将语义和深度特征提升为 3D Gaussian token，结合深度感知思维链模块，以 200M 参数在 LIBERO 上达到 93.5% 成功率，超越 7B 级别的基线。

---

## 核心贡献

1. **[[GST|Gaussian Spatial Tokenizer (GST)]]**: 将 2D patch 特征 + 单目深度提升为参数化的 3D Gaussian token（位置 $\mu$、协方差 $\sigma$、置信度 $\alpha$），通过置信度偏置的空间注意力池化得到几何感知表示。
2. **[[DA-CoT|Depth-Aware Chain-of-Thought (DA-CoT)]]**: 非自回归几何推理模块，使用 $N_r=4$ 个可学习 queries 在语言和 flow-time 条件下跨 GST token 做结构化空间推理。
3. **参数高效的完整框架**: 冻结双流编码器（[[SigLIP]] + [[Depth Anything|Depth-Anything-V2]]），可训练部分仅 179M 参数，总推理延迟 12.97 ms，在 LIBERO Spatial 子集达到 100% 成功率。

---

## 问题背景

### 要解决的问题

现有 [[VLA]]（Vision-Language-Action model）使用扁平 2D patch token 作为视觉表示，缺乏几何结构信息，导致在需要精确空间推理（距离、朝向、深度关系）的操作任务上表现不足。

### 现有方法的局限

- **[[SpatialVLA]]**（4B 参数）等方法依赖更大规模模型来隐式学习空间信息，参数效率低。
- **[[CoT-VLA]]**（7B 参数）虽引入链式推理，但仍基于 2D token，无几何先验。
- 直接拼接深度特征（depth concat）无法保留表面法向、置信度等几何结构（见消融 Table 10）。

### 本文的动机

单目深度估计器（[[Depth Anything|Depth-Anything-V2]]）已能可靠提供每 patch 的深度值；通过可微的 3D Gaussian 参数化，可以将 2D 证据提升为包含方向性和置信度的几何令牌，再用非自回归 CoT 做结构化几何推理，从而以极低参数量实现高水平的空间理解。

---

## 方法详解

### 模型架构

GaussVLA 采用**双流编码 + Mamba 主干**架构：

- **输入**: 语言指令 $l$ + 多视角 RGB 观测 $\{x_t^{(m)}\}$ + 动作 $A$
- **语义流**: 冻结 [[SigLIP]] (SigLIP-SO400M/14) 提取 patch 级特征 $F_t^{(m)} \in \mathbb{R}^{P \times d_s}$
- **几何流**: 冻结 [[Depth Anything|Depth-Anything-V2]] (ViT-L) 提取单目深度 $d_t^{(m)} \in \mathbb{R}^P$
- **核心模块**: [[GST]] 将双流融合为 3D Gaussian token，[[DA-CoT]] 做几何推理
- **Backbone**: [[Mamba]]（5 块，$d_b=512$）线性时间序列建模
- **输出**: [[Action Chunking|动作块]] $a_{t:t+H}$，$H=10$，通过 [[Flow-Matching]] 解码
- **总参数**: 200M（可训练 179M），冻结编码器不计入

### 核心模块

#### 模块 1: GST（Gaussian Spatial Tokenizer）

**设计动机**: 利用 [[3DGS|3D Gaussian]] 的紧凑参数化，将每个 patch 的深度和语义特征提升为带空间位置、尺度、置信度的几何令牌，从而为下游动作预测提供几何感知输入。

**具体实现**:
1. 对原始深度做可学习仿射校正（Eq.3）
2. 通过相机内参 $K^{(m)}$ 反投影到 3D 相机坐标（Eq.4）
3. 网络预测每 patch 的残差均值偏移 $\Delta\mu$、log-scale $\eta$、置信度 $\alpha$（Eq.5）
4. 用 3D Fourier 位置编码 $\gamma(\mu)$ 构造几何感知 patch token（Eq.8, 9）
5. 用 $N_g=128$ 个可学习 Gaussian queries 做置信度偏置的空间注意力池化（Eq.10）

#### 模块 2: DA-CoT（Depth-Aware Chain-of-Thought）

**设计动机**: 用结构化推理令牌在执行动作前做一次几何思考，提升 long-horizon 和空间任务的成功率，同时避免自回归 CoT 的串行推理开销。

**具体实现**:
- $N_r=4$ 个可学习推理 queries 初始化 $Y^{(0)}$
- 通过 Self-Attention → Cross-Attention（cross GST token）→ FFN 计算推理令牌 $R_\tau$（Eq.14-16）
- Context 由 GST token、语言嵌入 $l_{\text{cot}}$、flow-time 嵌入 $\phi_\tau$ 构成（Eq.13）
- 推理令牌 $\bar{R}_\tau$ 拼接到 Mamba 输入序列（Eq.17）

---

## 关键公式

### 公式 1: [[Depth Anything|深度仿射校正]] (Eq.3)

$$
\tilde{d}_{t,r}^{(m)} = \lambda_d \cdot d_{t,r}^{(m)} + \beta_d
$$

**含义**: 通过可学习标量 $\lambda_d, \beta_d$ 对单目深度估计做仿射修正，消除尺度-偏移模糊性。

**符号说明**:
- $d_{t,r}^{(m)}$: 第 $m$ 个相机第 $r$ 个 patch 在时刻 $t$ 的原始深度估计
- $\lambda_d, \beta_d \in \mathbb{R}$: 全局可学习标量

---

### 公式 2: 反投影到 3D 坐标 (Eq.4)

$$
c_{t,r}^{(m)} = \tilde{d}_{t,r}^{(m)} \cdot (K^{(m)})^{-1} \cdot u_r \in \mathbb{R}^3
$$

**含义**: 利用相机内参将校正后的深度和像素坐标反投影为 3D 相机坐标系下的点。

**符号说明**:
- $K^{(m)} \in \mathbb{R}^{3\times3}$: 第 $m$ 个相机的内参矩阵
- $u_r$: 第 $r$ 个 patch 中心的归一化像素坐标（齐次形式）

---

### 公式 3: Gaussian 均值定义 (Eq.6)

$$
\mu_{t,r}^{(m)} = c_{t,r}^{(m)} + \rho \cdot \tanh(\Delta\mu_{t,r}^{(m)})
$$

**含义**: 以反投影点为基础，加上受工作空间半径 $\rho$ 约束的残差偏移，得到 Gaussian 均值（3D 位置）。

**符号说明**:
- $\rho = 0.8\text{ m}$: 固定工作空间半径，约束 Gaussian 中心不越界
- $\Delta\mu_{t,r}^{(m)}$: 由 $f_{\text{par}}$ 预测的残差偏移

---

### 公式 4: 3D Fourier 位置编码 (Eq.8)

$$
\gamma(\mu_{t,r}^{(m)}) = \bigl[\sin(2^0\pi\mu),\, \cos(2^0\pi\mu),\, \ldots,\, \sin(2^{L-1}\pi\mu),\, \cos(2^{L-1}\pi\mu)\bigr] \in \mathbb{R}^{6L}
$$

**含义**: 将 3D Gaussian 均值编码为多频正弦位置特征，使模型能感知精细的空间结构。

**符号说明**:
- $L=10$: Fourier 频率带数，编码维度 $d_\gamma = 6L = 60$
- $\mu_{t,r}^{(m)} \in \mathbb{R}^3$: Gaussian 均值的 x/y/z 分量各编码 $2L$ 维

---

### 公式 5: 几何感知 Patch Token 构造 (Eq.9)

$$
g_{t,r}^{(m)} = \bigl[f_{t,r}^{(m)} \oplus \gamma(\mu_{t,r}^{(m)}) \oplus \eta_{t,r}^{(m)}\bigr] \in \mathbb{R}^{d_g}
$$

**含义**: 将语义特征、3D 位置编码、log-scale 拼接，得到融合几何信息的 patch-level token。

**符号说明**:
- $f_{t,r}^{(m)} \in \mathbb{R}^{d_s}$: SigLIP 语义特征（$d_s$）
- $\eta_{t,r}^{(m)} \in \mathbb{R}^3$: 预测的 log-scale 向量（协方差对角线）
- $d_g = d_s + 6L + 3$: 几何感知 token 总维度

---

### 公式 6: [[GST|置信度偏置的空间注意力池化]] (Eq.10)

$$
\hat{Z}_t^{(m)} = \text{softmax}\!\left(\frac{(\text{LN}_1(Q)W_Q)\,(\tilde{G}_t^{(m)}W_K)^\top}{\sqrt{d_p}} + \mathbf{1}_{N_g} \cdot \log(\alpha_t^{(m)})^\top\right) \cdot (\tilde{G}_t^{(m)}W_V)
$$

**含义**: 以 $N_g=128$ 个可学习 queries 对几何感知 patch token 做交叉注意力，置信度 $\alpha$ 作为 logit 偏置使几何可靠区域获得更高注意力权重。

**符号说明**:
- $Q \in \mathbb{R}^{N_g \times d_p}$: 可学习 Gaussian queries
- $\alpha_t^{(m)} \in \mathbb{R}^P$: 每 patch 置信度（不透明度），高置信度区域权重更大
- $d_p$: 池化维度

---

### 公式 7: [[Flow-Matching|Flow-Matching 动作损失]] (Eq.20)

$$
\mathcal{L}_{\text{FM}} = \mathbb{E}_{(O,s,A)\sim\mathcal{D},\, \tau\sim p(\tau),\, \varepsilon\sim\mathcal{N}(0,I)}\!\left[\|v_\theta(Z_\tau, O, l, \tau) - U\|_2^2\right]
$$

**含义**: 训练速度场网络 $v_\theta$ 匹配从噪声到真实动作的线性插值轨迹的目标速度。

**符号说明**:
- $\tau \sim \text{Beta}(1.5, 1.0)$: flow 时间采样分布
- $U = \varepsilon - A$: 目标速度场（线性轨迹斜率）
- $Z_\tau = \tau\varepsilon + (1-\tau)A$: 插值动作状态

---

### 公式 8: GST 深度一致性损失 (Eq.21)

$$
\mathcal{L}_{\text{GST}} = \frac{\sum_{t,m,r} \alpha_{t,r}^{(m)} \cdot \bigl|\mu_{t,r,z}^{(m)} - \tilde{d}_{t,r}^{(m)}\bigr|}{\sum_{t,m,r} \alpha_{t,r}^{(m)}}
$$

**含义**: 以置信度 $\alpha$ 加权的 Gaussian 深度与校正深度的 L1 一致性损失，约束学到的几何令牌与深度估计保持一致。

**符号说明**:
- $\mu_{t,r,z}^{(m)}$: Gaussian 均值的 z（深度）分量
- 高置信度 patch 的深度误差被更强地惩罚

---

### 公式 9: DA-CoT 监督损失 (Eq.24)

$$
\mathcal{L}_{\text{CoT}} = \frac{1}{H \cdot d_a}\left\|\hat{U}_{\text{CoT}} - U_\tau\right\|_F^2
$$

**含义**: 用推理令牌的辅助速度预测做监督，强迫 DA-CoT 令牌编码几何相关的动作信息。

**符号说明**:
- $\hat{U}_{\text{CoT}}$: 由 DA-CoT 摘要均值 $\bar{r}_\tau$ + 动作令牌计算的辅助速度
- $U_\tau$: 当前 flow-time 下的目标速度

---

### 公式 10: 联合训练目标 (Eq.25)

$$
\mathcal{L} = \mathcal{L}_{\text{FM}} + \lambda_{\text{GST}} \mathcal{L}_{\text{GST}} + \lambda_{\text{CoT}} \mathcal{L}_{\text{CoT}}
$$

**含义**: 三项损失联合优化：动作预测 + 几何一致性 + 推理监督。

**符号说明**:
- $\lambda_{\text{GST}} = 0.05$: GST 深度损失权重
- $\lambda_{\text{CoT}} = 0.10$: DA-CoT 辅助损失权重

---

## 关键图表

### Figure 1: Overview

![Figure 1 - GaussVLA Overview](https://arxiv.org/html/2608.24959v1/teaser.jpg)

**说明**: GaussVLA 整体概览。左侧冻结双流编码器提取语义+深度特征，经 [[GST]] 提升为 3D Gaussian token，[[DA-CoT]] 做几何推理后送入 [[Mamba]] 主干，最终 [[Flow-Matching]] 解码为动作序列。

---

### Figure 2: Framework Architecture

![Figure 2 - GaussVLA Framework](https://arxiv.org/html/2608.24959v1/framework.jpg)

**说明**: 完整框架图。冻结双流编码器（[[SigLIP]] + [[Depth Anything|Depth-Anything-V2]]）分别提取语义 patch 特征和密集深度图；[[GST]] 通过 Gaussian 参数化（$\mu, \sigma, \alpha$）将其提升为几何感知 patch token，再经空间注意力池化聚合；语言、CoT、动作、时间嵌入拼接后经 [[Mamba]] 主干，[[DA-CoT]] 精化空间推理，最后 flow-matching 动作头输出动作。

---

### Figure 3: LIBERO Rollouts

![Figure 3 - LIBERO Rollouts](https://arxiv.org/html/2608.24959v1/libero_tasks.jpg)

**说明**: LIBERO 基准的代表性 rollout，展示 GaussVLA 从初始状态到完成任务的完整执行轨迹，涵盖 Spatial/Object/Goal/Long 四个子集。

---

### Figure 4: Real-world SO-101 Evaluation

![Figure 4 - Real-world Evaluation](https://arxiv.org/html/2608.24959v1/visual_visual.jpg)

**说明**: 真实 SO-101 机器人评估。对比 ACT 和 [[SpatialVLA]] 在多任务、Pick-Place ID/OOD 场景的性能，以及 GaussVLA 执行物理堆叠任务的 rollout 帧。GaussVLA 多任务成功率 58.8%，ID 81.0%，OOD 46.7%。

---

### Table 1: LIBERO Benchmark 性能对比

| Method | Venue | Spatial | Object | Goal | Long | Average | Parameters |
|--------|-------|---------|--------|------|------|---------|------------|
| DP | RSS'23 | 78.3 | 92.5 | 68.3 | 50.5 | 72.4 | 80M |
| MaIL | CoRL'24 | 53.8 | 81.5 | 56.3 | 41.7 | 58.3 | 24M |
| QueST | NeurIPS'24 | 89.0 | 90.0 | 88.4 | 87.0 | 88.6 | 152M |
| OpenVLA | CoRL'24 | 84.7 | 88.4 | 79.2 | 53.7 | 76.5 | 7B |
| SpatialVLA | RSS'25 | 88.2 | 89.9 | 78.6 | 55.5 | 78.1 | 4B |
| ThinkAct | NeurIPS'25 | 88.3 | 91.4 | 87.1 | 70.9 | 84.4 | 7B |
| CoT-VLA | CVPR'25 | 81.5 | 91.6 | 87.6 | 69.0 | 82.43 | 7B |
| π₀ | RSS'26 | 90.0 | 86.0 | 95.0 | 73.0 | 86.0 | 3.3B |
| SUREFlow | IROS'26 | 94.8 | 91.0 | 93.8 | 90.2 | 92.5 | 179.1M |
| **GaussVLA** | **BMVC'26** | **100.0** | **95.8** | **95.3** | **83.0** | **93.5** | **200M** |

**关键发现**: GaussVLA 以 200M 参数超越所有 7B/4B/3B 级基线，Spatial 子集达到 100%，对 [[SpatialVLA]] 实现 19.7% 相对提升。

---

### Table 2: Meta-World 和 CALVIN 结果

**Meta-World:**

| Method | Venue | Easy | Mid | Hard | V.Hard | Average |
|--------|-------|------|-----|------|--------|---------|
| BC-RNN | CoRL'21 | 4.5 | 3.8 | 3.2 | 3.0 | 3.6 |
| [[Diffusion Policy\|DP]] | IJRR'25 | 83.6 | 31.1 | 9.0 | 26.6 | 37.6 |
| TinyVLA | ICRA'25 | 77.6 | 21.5 | 11.4 | 15.8 | 31.6 |
| π₀ | RSS'26 | 71.8 | 48.2 | 41.7 | 30.0 | 47.9 |
| **GaussVLA** | **BMVC'26** | **92.7** | **47.9** | **37.3** | **41.6** | **54.9** |

**[[CALVIN]] (平均序列长度):**

| Method | Venue | SR@1 | SR@2 | SR@3 | SR@4 | SR@5 | Avg. Length |
|--------|-------|------|------|------|------|------|-------------|
| MCIL | RSS'21 | 0.370 | 0.027 | 0.002 | 0.000 | 0.000 | 0.40 |
| GR-1 | ICLR'24 | 0.778 | 0.533 | 0.332 | 0.218 | 0.139 | 2.0 |
| **GaussVLA** | **BMVC'26** | **0.637** | **0.367** | **0.291** | **0.179** | **0.000** | **1.474** |

**关键发现**: [[Meta-World]] 平均 54.9%，排名最高；CALVIN 平均序列长度 1.474，说明在长链任务上有时序一致性。

---

### Table 3: LIBERO-PRO 鲁棒性评估

| Model | LIBERO-Goal | LIBERO-Spatial | LIBERO-10 | LIBERO-Object | Avg SR |
|-------|-------------|----------------|-----------|---------------|--------|
| OpenVLA | 0.96/0.00/0.98/0.00/0.98 | 0.97/0.00/0.97/0.00/0.89 | 0.81/0.00/0.96/0.00/0.85 | 0.98/0.00/0.98/0.00/0.00 | 0.52 |
| π₀.₅ | 0.97/0.38/0.97/0.00/0.46 | 0.97/0.20/0.97/0.01/0.46 | 0.92/0.08/0.93/0.01/0.46 | 0.98/0.17/0.96/0.01/0.73 | 0.53 |
| π₀ | 0.94/0.00/0.93/0.00/0.39 | 0.95/0.00/0.97/0.00/0.60 | 0.79/0.00/0.82/0.00/0.27 | 0.94/0.00/0.90/0.00/0.29 | 0.44 |
| **GaussVLA** | **0.81/0.00/0.70/0.00/0.00** | **0.93/0.00/0.73/0.00/0.27** | **0.83/0.00/0.67/0.00/0.00** | **0.92/0.00/0.63/0.00/0.17** | **0.33** |

格式：Obj/Pos/Sem/Task/Env。**关键发现**: 所有模型在 Position（Pos）扰动下均接近 0，说明位置分布偏移是当前 [[LIBERO-PRO]] 的主要瓶颈；GaussVLA avg 33.3%，弱于 π₀.₅ 的 53%。

---

### Table 4: 消融实验（核心模块贡献 + 效率）

| 变体 | 2D Tokens | GST | DA-CoT | LIBERO ↑ | LIBERO-PRO ↑ | Params(M) | Trainable(M) | GFLOPs | Latency(ms) |
|------|-----------|-----|--------|----------|--------------|-----------|--------------|--------|-------------|
| Vanilla | ✓ | ✗ | ✗ | 78.1 | 11.2 | 179 | 158 | 3.50 | 10.85 |
| + GST only | ✗ | ✓ | ✗ | 90.5 | 29.0 | 190.2 | 169.2 | 4.33 | 12.27 |
| + DA-CoT only | ✓ | ✗ | ✓ | 82.1 | 16.7 | 188.8 | 167.8 | 4.00 | 11.55 |
| **Full GaussVLA** | ✗ | ✓ | ✓ | **93.5** | **33.3** | **200** | **179** | 4.83 | 12.97 |

**关键发现**: [[GST]] 贡献最大（+12.4%），[[DA-CoT]] 额外贡献 +3.0%；GST 对鲁棒性（LIBERO-PRO）提升 17.8 个点最显著。延迟增加仅 2.1ms。

---

### Table 5: 超参数敏感性

**GST Queries ($N_g$):**

| $N_g$ | 8 | 32 | 64 | 128 | 256 |
|--------|---|----|----|-----|-----|
| LIBERO (%) | 86.4 | 90.7 | 92.4 | **93.5** | 93.6 |
| Latency (ms) | 11.6 | 12.0 | 12.4 | 12.97 | 14.1 |

**DA-CoT Queries ($N_r$):**

| $N_r$ | 1 | 2 | 4 | 8 | 16 |
|--------|---|---|---|---|-----|
| LIBERO (%) | 91.8 | 92.8 | **93.5** | 93.4 | 92.9 |
| Latency (ms) | 12.7 | 12.8 | 12.97 | 13.2 | 13.7 |

**关键发现**: $N_g=128$、$N_r=4$ 是精度-延迟的最优权衡点。

---

### Table 6: 架构与训练超参数

| 组件 | 参数 | 值 |
|------|------|---|
| GST | Gaussian queries $N_g$ | 128 |
| GST | 推理 queries $N_r$ | 4 |
| GST | Fourier 频带 $L$ | 10 |
| DA-CoT | 推理维度 $d_c$ | 512 |
| GST | 工作空间半径 $\rho$ | 0.8 m |
| Backbone | Mamba 块数 $L_b$ | 5 |
| Backbone | 动作 horizon $H$ | 10 |
| Backbone | 骨干维度 $d_b$ | 512 |
| 优化器 | Optimizer | AdamW |
| 优化器 | Learning Rate | 2.5×10⁻⁵ |
| 优化器 | Weight Decay | 1×10⁻⁴ |
| 优化器 | Flow 时间 $p(\tau)$ | Beta(1.5, 1.0) |
| 优化器 | Batch Size | 256 |
| 优化器 | 精度 | BF16 |

---

### Table 7: 各组件延迟分解

| 组件 | 可训练 | 延迟 (ms) |
|------|--------|---------|
| 冻结 SigLIP-SO400M/14 | ✗ | 5.20 |
| 冻结 Depth-Anything-V2 (ViT-L) | ✗ | 3.30 |
| GST 模块（lift + pool, $N_g=128$） | ✓ | 1.42 |
| Mamba backbone | ✓ | 1.50 |
| DA-CoT conditioner | ✓ | 0.70 |
| Action decoder | ✓ | 0.25 |
| Flow-matching ODE（10 步 Euler） | ✓ | 0.60 |
| **Total** | – | **12.97** |

**关键发现**: 冻结编码器占总延迟 65%（8.50 ms），可训练模块仅 4.47 ms，GST+Mamba+DA-CoT+解码器非常轻量。

---

### Table 8: LIBERO 线性探针回归（几何信息分析）

| 探针目标 | 平坦 2D patches | 池化 GST | DA-CoT $R_\tau$ |
|----------|-----------------|---------|----------------|
| 物体中心（3D 位置）| R²=0.34 | R²=0.66 | **R²=0.78** |
| 夹爪-物体距离 | 0.41 | 0.71 | **0.82** |
| 物体朝向（主轴）| 0.28 | 0.49 | **0.61** |
| 表面法向（从 $\sigma$ 线性推导）| — | 0.67 | — |

**关键发现**: [[DA-CoT]] 令牌比平坦 2D patches 的 3D 物体中心预测 R² 提升从 0.34 到 0.78，说明推理令牌确实编码了丰富的几何信息。

---

### Table 10: GST 设计消融

| 视觉表示 | 深度 | 3D 提升 | Gaussian 参数化 | 空间池化 | LIBERO | Spatial | PRO |
|----------|------|---------|----------------|---------|--------|---------|-----|
| 平坦 2D patch | ✗ | ✗ | ✗ | ✗ | 78.1 | 71.2 | 11.2 |
| 2D patch + 深度拼接 | ✓ | ✗ | ✗ | ✗ | 73.3 | 78.0 | 14.9 |
| **GST token** | ✓ | ✓ | ✓ | ✓ | **90.5** | **100.0** | **29.0** |

**关键发现**: 简单深度拼接反而比平坦 2D tokens 更差（73.3 < 78.1），说明几何提升（3D lifting + Gaussian 参数化）是关键，不是深度本身。

---

### Table 11: DA-CoT 消融

| 推理变体 | 深度感知 | 结构化令牌 | CoT 监督 | LIBERO | Long-Horizon | Avg |
|----------|----------|-----------|---------|--------|--------------|-----|
| 无推理模块 | ✗ | ✗ | ✗ | 78.1 | 75.2 | 76.7 |
| CoT 无深度 | ✗ | ✓ | ✓ | 80.5 | 87.3 | 83.9 |
| DA-CoT 无监督 | ✓ | ✓ | ✗ | 81.5 | 90.0 | 85.8 |
| **DA-CoT（完整）** | ✓ | ✓ | ✓ | **82.1** | **91.6** | **87.0** |

**关键发现**: 深度感知条件化和辅助监督都有独立贡献，完整 DA-CoT 在 long-horizon 任务上提升最显著。

---

### Table 12: 深度不确定性代理相关性

| 代理 | Pearson ρ(α, -u) ↑ | p-value |
|------|-------------------|---------|
| 深度梯度幅值 $u_\nabla$ | +0.61 | <10⁻⁶ |
| MC-dropout 方差 $u_{\sigma_d}$ | +0.54 | <10⁻⁶ |
| 光度不一致性 $u_{pc}$ | +0.47 | <10⁻⁶ |

**关键发现**: 学到的置信度 $\alpha$ 与深度不确定性代理显著负相关，说明 GST 自发学会了在不确定区域降低权重。

---

## 实验

### 数据集

| 数据集 | 特点 | 用途 |
|--------|------|------|
| [[LIBERO]] (4 子集) | 标准桌面操作，含 Spatial/Object/Goal/Long | 主要评估 |
| [[LIBERO-PRO]] | 五类分布偏移（物体/位置/语义/任务/环境） | 鲁棒性评估 |
| [[Meta-World]] | 50 个任务，4 难度级别 | 泛化能力评估 |
| [[CALVIN]] | 长链多步骤操作 | Long-horizon 评估 |
| SO-101 真实机器人 | 7 自由度桌面机械臂，多任务 + Pick-Place | 真实场景验证 |

### 实现细节

- **语义编码器**: 冻结 SigLIP-SO400M/14（patch 特征 $d_s$）
- **深度编码器**: 冻结 Depth-Anything-V2 ViT-L（单目深度估计）
- **语言编码器**: 冻结 CLIP（$E_{\text{lang}}$）
- **骨干网络**: Mamba（5 块，$d_b=512$），线性时间复杂度
- **优化器**: AdamW，lr=2.5×10⁻⁵，weight decay=1×10⁻⁴，cosine schedule，warm-up 5 epochs
- **Batch Size**: 256，BF16 精度
- **Flow 时间**: Beta(1.5, 1.0) 分布采样，ODE 10 步 Euler 推理

### 可视化结果

- **LIBERO Spatial**: 100% 成功率（完美），GaussVLA 能准确判断物体的相对空间关系（左/右/前/后）
- **真实机器人**: 物理堆叠任务 rollout 展示，OOD 46.7% 表明有一定泛化能力但仍有提升空间
- **线性探针**: DA-CoT 令牌上的 3D 物体中心预测 R²=0.78，可视化验证几何表示质量

---

## 批判性思考

### 优点

1. **几何建模精准**: 3D Gaussian 参数化（位置 + 协方差 + 置信度）比简单深度拼接更完整地表达局部几何结构，消融证明了这一点（Table 10）
2. **参数效率极高**: 200M 总参数（179M 可训练）超越 7B/4B 基线，inference 延迟仅 12.97ms
3. **设计解释性强**: 置信度-不确定性相关性（Table 12）提供了可解释的几何学习证据，而非黑盒提升

### 局限性

1. **LIBERO-PRO 鲁棒性不足**: Avg 33.3%，低于 π₀.₅（53%）；Position 扰动下几乎为 0，说明当前几何表示对视角/位置分布偏移脆弱
2. **需要相机内参**: 反投影到 3D 坐标（Eq.4）依赖已知 $K^{(m)}$，在相机参数未知或变化的场景下无法直接应用
3. **CALVIN 不是最优**: 平均序列长度 1.474，未超过 GR-1 的 2.0，long-horizon 能力有提升空间
4. **实际机器人数据有限**: 仅在 SO-101 单一机型上验证，跨机型泛化未探讨

### 潜在改进方向

1. 引入相机自标定或无标定的 3D 提升方法（如 [[DUSt3R]] 风格的稠密匹配），消除内参依赖
2. 针对 LIBERO-PRO Position 的 0% 问题，引入数据增强（相机随机化、域随机化）或 equivariant 表示
3. 扩展至多机器人形态（双臂、足式机器人），验证 GST 的通用几何建模能力

### 可复现性评估

- [ ] 代码开源（未公开）
- [ ] 预训练模型（未公开）
- [x] 训练细节完整（Table 6 提供了完整超参数）
- [x] 数据集可获取（LIBERO/Meta-World/CALVIN 均公开）

---

## 关联笔记

### 基于

- [[SigLIP]]: 语义流骨干编码器（冻结）
- [[Depth Anything|Depth Anything]]: 几何流深度估计编码器（冻结，ViT-L）
- [[Mamba]]: 线性时间序列 backbone
- [[Flow-Matching]]: 动作解码范式
- [[3DGS|3D Gaussian Splatting]]: Gaussian 参数化的理论基础

### 对比

- [[SpatialVLA]]: 同为空间感知 VLA，但用 4B 参数；GaussVLA 超越 19.7%
- [[CoT-VLA]]: 同引入推理机制，但基于 7B 平坦 2D token；GaussVLA 以 200M 超越
- [[π₀]]: 3.3B VLA 基线；LIBERO avg 86.0% vs GaussVLA 93.5%

### 方法相关

- [[GST]]: 核心几何令牌化模块
- [[DA-CoT]]: 深度感知推理模块
- [[Chain-of-Thought Reasoning]]: DA-CoT 的推理范式来源
- [[Action Chunking]]: 动作预测输出形式

### 数据集相关

- [[LIBERO]]: 主要评估基准
- [[LIBERO-PRO]]: 鲁棒性评估
- [[CALVIN]]: Long-horizon 评估
- [[Meta-World]]: 多任务泛化评估

---

## 速查卡片

> [!summary] GaussVLA (BMVC 2026)
> - **核心**: 用 3D Gaussian token 替代扁平 2D patch，实现几何感知 VLA
> - **方法**: GST（Gaussian Spatial Tokenizer）+ DA-CoT + Mamba + Flow-Matching
> - **结果**: LIBERO avg 93.5%（Spatial 100%），仅 200M 参数超越 7B 基线
> - **代码**: 未公开

---

*笔记创建时间: 2026-08-28*
