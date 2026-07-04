---
title: "VLAFlow: A Unified Training Framework for Vision-Language-Action Models via Co-training and Future Latent Alignment"
method_name: "VLAFlow"
authors: [Guoyang Xia, Fengfa Li, Hongjin Ji, Lei Ren, Fangxiang Feng, Kun Zhan, Yan Xie]
year: 2026
venue: arXiv
tags: [vla, flow-matching, co-training, future-latent-alignment, robot-manipulation, pre-training, transfer-learning]
zotero_collection: Robotics/VLA
image_source: online
arxiv_html: https://arxiv.org/html/2607.01586v1/
created: 2026-07-04
---

# 论文笔记：VLAFlow

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | 未明确列出（推测国内机构）|
| 日期 | July 2026 |
| 项目主页 | 未知 |
| 对比基线 | [[π0]], [[OpenVLA-OFT]], [[WorldVLA]], [[UniVLA]] |
| 链接 | [arXiv](https://arxiv.org/abs/2607.01586) |

---

## 一句话总结

> VLAFlow 在统一 [[Flow Matching]] 框架下对比四种 VLA 预训练范式，发现语言监督与未来潜变量对齐的联合方案 MindLWPI 在异构机器人数据上迁移最稳定。

---

## 核心贡献

1. **统一受控对比框架**: 在相同架构、数据集、评估协议下系统对比四种预训练范式（MindPI/MindLPI/MindWPI/MindLWPI），排除混淆变量
2. **OXEMix 数据集**: 整合 DROID、OpenX-Embodiment、OpenX-Augmented、RoboCOIN 约 5000 小时的异构机器人数据
3. **元动作空间视角**: 提出"元动作空间"解释框架，说明语言监督和[[Future Latent Alignment|未来潜变量对齐]]如何从互补角度约束动作学习

---

## 问题背景

### 要解决的问题

不同 VLA 模型在预训练阶段采用截然不同的监督信号（纯动作、语言联合、世界模型预测等），但由于架构、数据、评估标准各异，难以判断哪种预训练范式在异构机器人数据上具备最佳的迁移能力。

### 现有方法的局限

- **π0 / OpenVLA-OFT**：依赖纯动作预训练，在异构数据上容易产生负迁移
- **WorldVLA / UniVLA**：引入未来状态或语言监督，但训练框架差异导致无法公平对比
- 缺乏受控实验：现有工作难以区分"架构优势"与"预训练范式优势"

### 本文的动机

通过固定 [[Qwen3-VL]]（4B）骨干网络和 [[DiT]] 动作专家架构，在相同数据集上切换监督信号，以量化不同预训练范式的贡献。

---

## 方法详解

### 模型架构

VLAFlow 采用 **解耦双流** 架构：

- **输入**: 语言指令 $l$ + 多视角视觉观测 $o_t$（静态相机 + 腕部相机）
- **Backbone**: [[Qwen3-VL]]（Qwen3-VL-4B-Instruct，36 层，2048 隐藏维度）
- **动作专家**: [[DiT]] 解码器（36 块，通过 [[KV 缓存]] 共享机制接收 VLM 上下文）
- **核心机制**: [[KV-Cache Conditioning|KV-Cache 共享]]——动作专家复用 VLM 各层的 K/V 缓存，无需额外跨注意力模块
- **输出**: [[Action Chunking|动作块]] $\mathbf{a}_{t:t+16}$（默认块长 16）
- **动作维度**: 14 维（6-DoF 末端执行器位姿 + 夹爪）
- **总参数**: 约 4B（VLM 主体）+ DiT 动作专家

### 核心模块

#### 模块 1：KV-Cache 共享

**设计动机**: 避免为动作专家单独引入跨注意力层，利用 [[KV 缓存]] 实现 VLM 语义与动作生成的零额外开销融合。

**具体实现**:
- VLM 第 $l$ 层输出缓存 $\mathbf{K}^l_\text{vlm},\ \mathbf{V}^l_\text{vlm} \in \mathbb{R}^{B \times H_{kv} \times S_\text{vlm} \times d_h}$
- 动作专家第 $i$ 块将 VLM 缓存与自身 K/V 拼接：$\tilde{\mathbf{K}}_i = [\mathbf{K}^i_\text{vlm};\ \mathbf{K}_i]$
- 潜变量 token 可以 attend 到 VLM 缓存和潜变量 token，但屏蔽动作 token（防止 shortcut）
- 动作 token 可以 attend 到 VLM 缓存、潜变量 token 和动作 token

#### 模块 2：四种预训练范式

| 范式 | 辅助监督 | 预训练损失 | 微调损失 |
|------|---------|-----------|---------|
| MindPI | 无 | $\mathcal{L}_\text{act}$ | $\mathcal{L}_\text{act}$ |
| MindLPI | 语言 | $\mathcal{L}_\text{act} + \mathcal{L}_\text{lang}$ | $\mathcal{L}_\text{act}$ |
| MindWPI | 未来潜变量 | $\mathcal{L}_\text{act} + \mathcal{L}_\text{lat}$ | $\mathcal{L}_\text{act} + \mathcal{L}_\text{lat}$ |
| MindLWPI | 语言 + 未来潜变量 | $\mathcal{L}_\text{act} + \mathcal{L}_\text{lat} + \mathcal{L}_\text{lang}$ | $\mathcal{L}_\text{act} + \mathcal{L}_\text{lat}$ |

#### 模块 3：未来潜变量对齐（MindWPI/MindLWPI）

**设计动机**: 利用 [[V-JEPA2|V-JEPA 2]] 提取的视觉潜变量表示未来帧的状态转移，用预测损失约束模型学习动作的物理后果。

**具体实现**:
- 使用冻结的 [[V-JEPA2|V-JEPA 2]] 分别提取当前帧和未来帧的潜变量特征（256 个 token）
- 动作专家预测 $\hat{z}_\text{fut}$，与冻结特征 $z_\text{fut}$ 计算 L2 损失
- 注意力 mask 确保潜变量 query 不能 attend 动作 token（防止走捷径）
- 使用 AvgPool-k4 压缩将 256 token 降至 64 token，减少计算开销

#### 模块 4：语言动作描述监督（MindLPI/MindLWPI）

**设计动机**: 将低维动作标签转化为高层次动作意图描述，缓解异构数据的分布差异。

**具体实现**:
- 将每步动作转化为描述性文本模板（如 "move left 2cm, rotate wrist 5 degrees"）
- 以自回归交叉熵损失监督，权重 $\lambda_\text{lang} = 0.1$
- 需移除 stop-gradient（消融实验验证：有 stop-gradient 时 LIBERO-Plus 仅 45.8%，去掉后恢复到 72.3%）

---

## 关键公式

### 公式 1：[[Flow Matching|Flow Matching 噪声插值]]

$$
\mathbf{x}_t = (1 - t)\boldsymbol{\epsilon} + t\mathbf{a}
$$

**含义**: 在 $t \in [0,1]$ 上将高斯噪声 $\boldsymbol{\epsilon}$ 线性插值到目标动作 $\mathbf{a}$，构建连续时间噪声轨迹。

**符号说明**:
- $\mathbf{x}_t$：时刻 $t$ 的噪声动作
- $t$：连续时间参数，$t \sim \mathcal{U}(0,1)$
- $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$：标准高斯噪声
- $\mathbf{a}$：真实动作块

### 公式 2：[[Flow Matching|动作流匹配损失]]（$\mathcal{L}_\text{act}$）

$$
\mathcal{L}_\text{act} = \mathbb{E}_{t,\boldsymbol{\epsilon}}\Big[\|\mathbf{v}_\theta(\mathbf{x}_t, t, \mathbf{c}) - (\mathbf{a} - \boldsymbol{\epsilon})\|^2\Big]
$$

**含义**: 让模型预测速度场 $\mathbf{v}_\theta$，使其与从噪声到真实动作的方向 $(\mathbf{a} - \boldsymbol{\epsilon})$ 对齐。

**符号说明**:
- $\mathbf{v}_\theta$：以 VLM 上下文 $\mathbf{c}$ 为条件的学习速度场
- $\mathbf{c}$：VLM 编码的多模态上下文
- $\|\cdot\|^2$：L2 范数

### 公式 3：[[Future Latent Alignment|未来潜变量损失]]（$\mathcal{L}_\text{lat}$）

$$
\mathcal{L}_\text{lat} = \|\hat{z}_\text{fut} - z_\text{fut}\|^2
$$

**含义**: 预测 [[V-JEPA2|V-JEPA 2]] 提取的未来帧潜变量特征，约束模型理解动作的物理后果。

**符号说明**:
- $\hat{z}_\text{fut}$：动作专家预测的未来潜变量
- $z_\text{fut}$：冻结 V-JEPA 2 编码的真实未来帧特征

### 公式 4：MindLWPI 联合损失

$$
\mathcal{L}_\text{MindLWPI} = \mathcal{L}_\text{act} + \lambda_\text{lat} \cdot \mathcal{L}_\text{lat} + \lambda_\text{lang} \cdot \mathcal{L}_\text{lang}
$$

**含义**: 三路监督联合训练——动作流匹配、未来状态预测、语言动作描述。

**符号说明**:
- $\lambda_\text{lat}$：潜变量损失权重（预训练 1:1，微调 0.1:1）
- $\lambda_\text{lang} = 0.1$：语言损失权重
- $\mathcal{L}_\text{lang}$：自回归语言交叉熵损失

### 公式 5：KV-Cache 拼接

$$
\tilde{\mathbf{K}}_i = [\mathbf{K}^i_\text{vlm};\ \mathbf{K}_i], \quad \tilde{\mathbf{V}}_i = [\mathbf{V}^i_\text{vlm};\ \mathbf{V}_i]
$$

**含义**: 动作专家第 $i$ 块将 VLM 的 K/V 缓存与自身 K/V 拼接，使动作生成直接使用 VLM 上下文。

### 公式 6：带 Mask 的注意力计算

$$
\operatorname{Attn}(\mathbf{Q}_i, \tilde{\mathbf{K}}_i, \tilde{\mathbf{V}}_i) = \operatorname{softmax}\!\left(\frac{\mathbf{Q}_i\tilde{\mathbf{K}}_i^\top}{\sqrt{d_h}} + \mathbf{M}\right)\tilde{\mathbf{V}}_i
$$

**含义**: 结合注意力 Mask $\mathbf{M}$ 控制潜变量 token 不能 attend 动作 token，防止走捷径。

**符号说明**:
- $\mathbf{M}$：注意力 mask，潜变量 query 屏蔽动作 token
- $d_h = 128$：每头维度

### 公式 7：物理动作还原

$$
\mathbf{a}_\text{phys} = \tfrac{1}{2}(\mathbf{a}_\text{norm} + 1) \odot (q_{99} - q_{01}) + q_{01}
$$

**含义**: 将归一化动作 $\mathbf{a}_\text{norm} \in [-1,1]$ 还原为物理单位。

**符号说明**:
- $q_{01}, q_{99}$：全局动作分布的第 1、99 百分位数
- $\odot$：逐元素乘法

---

## 关键图表

### Figure 1：OXEMix 数据集构成

![Figure 1](https://arxiv.org/html/2607.01586v1/x1.png)

**说明**: OXEMix 预训练语料的持续时间和轨迹数量分布。数据来自 DROID、OpenX-Embodiment、OpenX-Augmented、RoboCOIN 四个来源，共约 5000 小时。各数据源的机器人平台、采样频率、动作定义均不同，形成典型的异构场景。

### Figure 2：VLAFlow 总体架构

![Figure 2](https://arxiv.org/html/2607.01586v1/x2.png)

**说明**: VLAFlow 的统一架构与三步流水线。A. 输入与共享骨干：多视角视觉观测 + 语言指令经 [[Qwen3-VL]] 编码后通过 [[KV 缓存]] 共享传递给动作专家。B. 预训练范式：展示 MindPI、MindLPI、MindWPI 三种基础范式及其组合 MindLWPI。C. 微调与评估：所有预训练模型使用相同下游协议微调后在 LIBERO、LIBERO-Plus、SimplerEnv 评估。

### Figure 3：LIBERO 上的 LoRA Rank 缩放

![Figure 3](https://arxiv.org/html/2607.01586v1/x3.png)

**说明**: 横轴为可训练参数量（M），纵轴为 LIBERO 四个子集的平均成功率。VLM-LoRA（r=128，47.2M 参数）以 96.2% 接近全参数微调水平，而动作专家 [[LoRA]] 即使增加到 r=512（103.8M 参数）也只能达到 78.6%，表明 VLM 侧的表示更新对下游迁移更关键。

### Figure 4：元动作空间视角对比

![Figure 4](https://arxiv.org/html/2607.01586v1/x4.png)

**说明**: 从[[Meta-Action Space|元动作空间]]视角比较四种范式。MindPI 直接拟合异构动作标签，容易受到不同机器人平台动作分布差异的干扰；MindLPI 通过语言空间提供高层次动作意图约束；MindWPI 通过未来视觉潜变量空间提供状态转移约束；MindLWPI 同时使用两者，形成更平滑的元动作空间。

### Figure 5：MindWPI/MindLWPI 的注意力 Mask

![Figure 5](https://arxiv.org/html/2607.01586v1/x5.png)

**说明**: 行为 query token，列为 key/value token。潜变量 query 可以 attend VLM KV 缓存和潜变量 token，但被屏蔽于动作 token（防止 shortcut 预测）；动作 query 可以 attend VLM KV 缓存、潜变量 token 和动作 token，利用预测性潜变量上下文生成动作。

### Table 1：四种 VLA 预训练范式的受控对比汇总

| 范式 | 辅助监督 | 预训练损失 | 微调损失 | 主要角色 |
|------|---------|-----------|---------|---------|
| MindPI | 无 | $\mathcal{L}_\text{act}$ | $\mathcal{L}_\text{act}$ | 纯动作迁移基线 |
| MindLPI | 语言 | $\mathcal{L}_\text{act}$, $\mathcal{L}_\text{lang}$ | $\mathcal{L}_\text{act}$ | 通过语言空间注入高层次动作意图 |
| MindWPI | 未来潜变量 | $\mathcal{L}_\text{act}$, $\mathcal{L}_\text{lat}$ | $\mathcal{L}_\text{act}$, $\mathcal{L}_\text{lat}$ | 通过未来状态预测正则化学习 |
| MindLWPI | 语言 + 未来潜变量 | 三路 | $\mathcal{L}_\text{act}$, $\mathcal{L}_\text{lat}$ | 同时约束意图和状态转移 |

### Table 2：主实验结果（综合对比）

| 方法 | 机器人预训练 | 辅助监督 | LIBERO Avg | LIBERO-Plus Total | WidowX Avg | RT-1 VM | RT-1 VA |
|------|-----------|---------|-----------|------------------|-----------|---------|---------|
| MindPI w/o PT | 无 | 无 | 97.0 | 59.9 | 59.6 | 75.7 | 60.4 |
| MindWPI w/o PT | 无 | 未来潜变量 | 97.4 | 66.1 | 71.9 | 75.2 | 51.6 |
| MindPI (Frozen VLM) | 有 | 无 | 97.2 | 74.9 | 54.4 | 72.7 | 66.0 |
| MindPI (Full PT) | 有 | 无 | 97.5 | 68.8 | 65.9 | 68.2 | 55.5 |
| MindLPI | 有 | 语言 | 97.2 | 72.3 | 65.6 | 74.6 | 59.2 |
| MindWPI | 有 | 未来潜变量 | 98.5 | 72.6 | 74.5 | **86.7** | 71.1 |
| **MindLWPI** | 有 | 语言+潜变量 | **99.1** | **74.8** | **75.5** | 84.4 | **69.8** |

**表格说明**: MindLWPI 在 LIBERO 和 WidowX 上全面领先；MindWPI 在 RT-1 Visual Matching 任务上达到最高 86.7%，单独的未来潜变量对齐对跨平台视觉匹配贡献突出。

### Table 3：LIBERO Benchmark 详细结果

| 方法 | L-Spatial | L-Object | L-Goal | L-Long | Avg |
|------|-----------|----------|--------|--------|-----|
| OpenVLA | 84.7 | 88.4 | 79.2 | 53.7 | 76.5 |
| π0-Fast | 96.4 | 96.8 | 88.6 | 60.2 | 85.5 |
| GROOT-N1 | 94.4 | 97.6 | 93.0 | 90.6 | 93.9 |
| π0 | 98.0 | 96.8 | 94.4 | 88.4 | 94.4 |
| π0.5 | 98.8 | 98.2 | 98.0 | 92.4 | 96.9 |
| OpenVLA-OFT | 97.6 | 98.4 | 97.9 | 94.5 | 97.1 |
| MindPI w/o PT | 97.8 | 99.0 | 97.0 | 94.0 | 97.0 |
| MindPI (Frozen VLM) | 98.4 | 99.2 | 96.8 | 94.6 | 97.2 |
| MindPI (Full PT) | 98.2 | 98.4 | 98.0 | 95.2 | 97.5 |
| MindLPI | 97.8 | 98.4 | 98.0 | 94.8 | 97.2 |
| MindWPI w/o PT | 98.0 | 99.6 | 98.2 | 93.6 | 97.4 |
| MindWPI | 99.0 | 99.6 | 98.6 | 96.8 | 98.5 |
| **MindLWPI** | **99.2** | **99.8** | **99.2** | **98.2** | **99.1** |

**表格说明**: LIBERO 已接近饱和，MindLWPI 以 99.1% 平均成功率领先，各子集均最高，说明联合监督有利于长程任务（L-Long 提升最明显）。

### Table 4：LIBERO-Plus 零样本鲁棒性测试

| 方法 | Camera | Robot | Language | Light | Background | Noise | Layout | Total |
|------|--------|-------|----------|-------|------------|-------|--------|-------|
| OpenVLA | 0.8 | 3.5 | 23.0 | 8.1 | 34.8 | 15.2 | 28.5 | 15.6 |
| OpenVLA-OFT | 56.4 | 31.9 | 79.5 | 88.7 | 93.3 | 75.8 | 74.2 | 69.6 |
| NORA | 2.2 | 37.0 | 65.1 | 45.7 | 58.6 | 12.8 | 62.1 | 39.0 |
| WorldVLA | 0.1 | 27.9 | 41.6 | 43.7 | 17.1 | 10.9 | 38.0 | 25.0 |
| UniVLA | 1.8 | 46.2 | 69.6 | 69.0 | 81.0 | 21.2 | 31.9 | 42.9 |
| π0 | 13.8 | 6.0 | 58.8 | 85.0 | 81.4 | 79.0 | 68.9 | 53.6 |
| π0-Fast | 65.1 | 21.6 | 61.0 | 73.2 | 73.2 | 74.4 | 68.8 | 61.6 |
| RIPT-VLA | 55.2 | 31.2 | 77.6 | 88.4 | 91.6 | 73.5 | 74.2 | 68.4 |
| MindPI w/o PT | 26.1 | 32.0 | 71.0 | 88.0 | 94.4 | 58.3 | 70.3 | 59.9 |
| MindWPI w/o PT | 32.7 | 48.7 | 77.0 | 90.7 | 93.2 | 64.0 | 73.6 | 66.1 |
| MindPI (Frozen VLM) | 42.3 | 58.6 | 86.6 | 94.7 | 96.3 | 81.9 | 78.0 | 74.9 |
| MindPI (Full PT) | 42.0 | 57.8 | 75.4 | 91.5 | 93.2 | 57.4 | 86.7 | 68.8 |
| MindLPI | 46.2 | 61.7 | 84.6 | 93.4 | 95.2 | 58.4 | 82.2 | 72.3 |
| MindWPI | 36.7 | **73.8** | 86.2 | 96.5 | 91.6 | 54.2 | **84.8** | 72.6 |
| **MindLWPI** | 35.1 | **74.2** | **93.0** | **97.1** | **96.7** | 58.9 | 84.5 | **74.8** |

**表格说明**: MindWPI 在机器人状态（73.8%）和物体布局（84.8%）扰动上表现突出，体现未来潜变量对状态转移的建模优势；MindLWPI 在语言扰动（93.0%）和光照（97.1%）上最优，综合总分 74.8% 最高。

### Table 5：SimplerEnv 跨平台迁移基准

| 方法 | 参数量 | RT-1 VM | RT-1 VA | WidowX |
|------|--------|---------|---------|--------|
| SpatialVLA | 4B | 75.1 | 70.7 | 42.7 |
| FPC-VLA | 7B | 78.0 | 65.8 | 64.6 |
| MemoryVLA | 7B | 77.7 | 72.7 | 71.9 |
| π0 | 3B | 58.8 | 56.8 | 27.8 |
| π0+FAST | 3B | 61.9 | 60.5 | 39.5 |
| OpenVLA-OFT | 7B | 63.0 | 54.3 | 31.3 |
| DD-VLA | 7B | 71.2 | 64.1 | 49.3 |
| MindPI w/o PT | 4B | 75.7 | 60.4 | 59.6 |
| MindWPI w/o PT | 4B | 75.2 | 51.6 | 71.9 |
| MindPI (Frozen VLM) | 4B | 72.7 | 66.0 | 54.4 |
| MindPI (Full PT) | 4B | 68.2 | 55.5 | 65.9 |
| MindLPI | 4B | 74.6 | 59.2 | 65.6 |
| MindWPI | 4B | **86.7** | 71.1 | 74.5 |
| **MindLWPI** | 4B | 84.4 | **69.8** | **75.5** |

**表格说明**: MindWPI 在 RT-1 Visual Matching 上达到 86.7%，远超同参数量方法；MindLWPI 在 WidowX（75.5%）上最高，说明联合监督有利于 WidowX 的操作泛化。

### Table 6：WidowX 混合微调 vs 纯 Bridge 微调

| 方法 | 混合 SimplerEnv FT | 纯 Bridge-only FT | 变化 |
|------|------------------|-----------------|------|
| MindPI w/o PT | 63.0 | 59.6 | −3.4 |
| MindWPI w/o PT | 73.4 | 71.9 | −1.5 |
| MindPI | 55.5 | 65.9 | **+10.4** |
| MindWPI | 71.9 | 74.5 | +2.6 |

**表格说明**: MindPI 全参预训练后使用纯 Bridge 数据微调提升显著（+10.4%），而 MindWPI 更稳定，两种微调策略差异小（+2.6%）。

### Table 7：MindLPI 梯度截断消融

| 方法 | LIBERO Avg | LIBERO-Plus Total |
|------|-----------|------------------|
| MindLPI w/ stop-gradient | 96.3 | 45.8 |
| MindLPI w/o stop-gradient | 97.2 | 72.3 |

**关键发现**: 移除 stop-gradient 后 LIBERO-Plus 从 45.8% 大幅提升到 72.3%，说明动作损失提供了有效的对齐信号。

### Table 8：MindWPI 损失比例消融

| 方法 | 预训练目标 | $\lambda_\text{lat}^\text{ft}$ | $\lambda_\text{act}^\text{ft}$ | WidowX Avg |
|------|-----------|-----|-----|-----------|
| MindWPI (Latent-only PT) | 仅潜变量 (0:1) | 0.1 | 1.0 | 64.6 |
| MindWPI (1:1 FT) | act+lat (1:1) | 1.0 | 1.0 | 61.5 |
| MindWPI (Action-only FT) | act+lat (1:1) | 0 | 1.0 | 70.8 |
| MindWPI (0.01:1 FT) | act+lat (1:1) | 0.01 | 1.0 | 67.2 |
| MindWPI (1:0.1 PT) | act+lat (1:0.1) | 0.1 | 1.0 | 67.7 |
| **MindWPI (1:1 PT)** | act+lat (1:1) | 0.1 | 1.0 | **71.9** |

**关键发现**: 预训练时 1:1 的动作-潜变量比例（71.9%）显著优于 1:0.1（67.7%），说明潜变量监督在预训练阶段需要足够强度。微调时弱化潜变量正则（0.1:1）最佳。

### Table 9：MindWPI w/o PT 设计消融

| 方法 | 损失设置 | 未来帧偏移 | WidowX Avg |
|------|---------|-----------|-----------|
| MindWPI w/o PT (1:1 FT) | 1:1 | 8 | 64.3 |
| MindWPI w/o PT (Action-only FT) | 仅动作 | — | 69.0 |
| MindWPI w/o PT | 0.1:1 | 8 | **73.4** |
| MindWPI w/o PT (0.1:1, Offset 32) | 0.1:1 | 32 | 59.9 |

**关键发现**: 偏移 8 帧（0.1:1 损失）效果最优；偏移 32 帧性能下降明显，过长的未来帧偏移反而有害。

### Table 10：潜变量压缩方法消融

| 设置 | 压缩方法 | 训练阶段 | WidowX Avg |
|------|---------|---------|-----------|
| MindWPI w/o PT | 无压缩 | 直接下游微调 | 73.4 |
| MindWPI compressed | AvgPool-k4 | 直接下游微调 | **74.0** |
| MindWPI compressed | AvgPool-k16 | 直接下游微调 | 67.2 |
| MindWPI compressed | MLP-k4 | 直接下游微调 | 70.8 |
| MindWPI compressed | MLP-k16 | 直接下游微调 | 60.7 |
| **MindLWPI** | AvgPool-k4 | 全预训练+Bridge FT | **75.5** |

**关键发现**: AvgPool-k4 将 256 token 压缩至 64 的同时略微提升了性能（74.0 > 73.4），是计算效率与性能的最优平衡点；MLP 压缩效果更差。

### Table 11：MindPI 预训练数据来源消融

| 预训练数据 | Stack | Carrot | Spoon | Eggplant | WidowX Avg |
|---------|-------|--------|-------|----------|-----------|
| 无预训练 | 32.3 | 58.3 | 67.7 | 93.8 | 63.0 |
| RoboCOIN subset | 0.05 | 31.3 | 38.5 | 91.7 | 40.4 |
| OXE-Augmented | 13.5 | 59.4 | 84.4 | 81.3 | 59.7 |
| DROID only | 46.9 | 52.1 | 81.2 | 68.8 | 62.2 |
| OXE raw subset | 37.5 | 57.3 | 87.5 | 78.1 | **65.1** |

**关键发现**: 不同数据来源带来截然不同的迁移行为（40.4% 到 65.1%），证明纯动作预训练对异构数据极度敏感。RoboCOIN 单独使用反而带来灾难性遗忘（Stack 任务仅 0.05%）。

### Table 12：LIBERO 上的 LoRA 适配结果

| 方法 | 可训练参数 | L-Spatial | L-Object | L-Goal | L-Long | Avg |
|------|-----------|-----------|----------|--------|--------|-----|
| Action LoRA r=64 | 13.0M | 71.8 | 85.2 | 55.2 | 40.6 | 63.2 |
| Action LoRA r=128 | 25.9M | 83.0 | 88.8 | 64.8 | 49.4 | 71.5 |
| Action LoRA r=256 | 51.9M | 88.4 | 93.6 | 69.2 | 56.0 | 76.8 |
| Action LoRA r=512 | 103.8M | 87.8 | 95.0 | 68.8 | 63.0 | 78.6 |
| **VLM LoRA r=128** | **47.2M** | 97.4 | 95.8 | 99.2 | 92.2 | **96.2** |
| VLM LoRA r=256 | 94.4M | 98.2 | 99.2 | 98.8 | 92.2 | 97.1 |
| Both LoRA r=128 | 73.1M | 96.8 | 99.0 | 97.8 | 89.2 | 95.7 |
| Both LoRA r=256 | 146.3M | 98.0 | 99.6 | 98.8 | 94.8 | 97.8 |

**关键发现**: VLM-LoRA（r=128，47.2M）以 96.2% 接近全参微调水平，而动作专家 LoRA 即使参数量翻倍也难以超过 80%，说明 VLM 表示的质量是下游性能的瓶颈。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| [[DROID]] | ~大规模 | 高质量机器人操作，多场景 | 预训练 |
| OpenX-Embodiment | 多机器人平台 | 异构机器人数据 | 预训练 |
| OpenX-Augmented | OXE 数据增广版 | 扩展数据多样性 | 预训练 |
| RoboCOIN | 专用操作数据 | 精细操作 | 预训练 |
| [[LIBERO]] | 4 个子集 | 长程任务、多物体 | 下游测试 |
| [[LIBERO-Plus]] | 7 种扰动类型 | 零样本鲁棒性 | 下游测试 |
| [[SimplerEnv]] | RT-1 + WidowX | 跨平台迁移 | 下游测试 |

### 实现细节

- **VLM Backbone**: Qwen3-VL-4B-Instruct（36 层，2048 隐藏维度，8 KV 头，128 头维度）
- **动作专家**: DiT 解码器（36 块）
- **动作块长度**: 默认 16 步
- **动作维度**: 14（6-DoF + 夹爪）
- **潜变量压缩**: AvgPool-k4（256→64 token）
- **预训练数据**: OXEMix（约 5000 小时）
- **语言损失权重**: $\lambda_\text{lang} = 0.1$
- **潜变量损失权重**: 预训练 1:1（act:lat），微调 1:0.1

---

## 批判性思考

### 优点

1. **受控对比设计严格**: 固定架构、数据、超参，是目前 VLA 预训练范式比较中少见的高质量受控实验
2. **互补监督信号有效**: 语言监督和未来潜变量从不同层次约束动作学习，MindLWPI 的组合提升幅度超过单一信号
3. **LoRA 分析实用**: 明确指出 VLM 侧 LoRA 的效率优于动作专家 LoRA，有直接工程价值

### 局限性

1. **架构固定**: 只在 Qwen3-VL + DiT 组合上验证，结论是否适用于其他骨干（如 Llama3-VL、phi-3）未验证
2. **数据规模受限**: 5000 小时远小于 π0 等工作的数据规模，预训练充分度有待验证
3. **Camera 扰动鲁棒性弱**: MindLWPI 在 Camera 扰动（35.1%）和传感器噪声（58.9%）上明显弱于光照/背景，单一感知扰动依然是瓶颈

### 潜在改进方向

1. 探索更强大的未来帧预测器替代 V-JEPA 2（如视频 World Model）
2. 在数据更多样化的场景（如双臂、移动操作）下验证结论
3. 研究更动态的 $\lambda_\text{lat}$ 调度策略（curriculum 式损失权重）

### 可复现性评估

- [ ] 代码开源（论文未提及）
- [ ] 预训练模型（论文未提及）
- [x] 训练细节完整（超参数表述清晰）
- [x] 数据集可获取（DROID、OXE 均为公开数据集）

---

## 关联笔记

### 基于

- [[V-JEPA2|V-JEPA 2]]: 提供冻结视觉编码器用于未来潜变量提取
- [[Qwen3-VL]]: VLM 骨干网络
- [[Flow Matching]]: 核心动作生成框架

### 对比

- [[π0]]: 经典 VLA，同在 LIBERO 评估
- [[OpenVLA-OFT]]: 同在 LIBERO 和 LIBERO-Plus 评估
- [[WorldVLA]]: 有世界模型预训练，同类研究
- [[UniVLA]]: 语言监督 VLA，同类研究

### 方法相关

- [[Flow Matching]]: 核心生成框架
- [[DiT]]: 动作专家架构
- [[KV 缓存]]: KV-Cache 共享机制
- [[LoRA]]: 参数高效微调
- [[Future Latent Alignment|未来潜变量对齐]]: 核心辅助监督
- [[Meta-Action Space|元动作空间]]: 解释框架

### 硬件/数据相关

- [[DROID]]: 预训练数据来源
- [[LIBERO]]: 下游评估基准
- [[LIBERO-Plus]]: 零样本鲁棒性评估
- [[SimplerEnv]]: 跨平台迁移评估

---

## 速查卡片

> [!summary] VLAFlow
> - **核心**: 统一框架受控对比四种 VLA 预训练范式（纯动作/语言/未来潜变量/联合）
> - **方法**: Flow Matching 动作专家 + KV-Cache 共享 + V-JEPA 2 未来潜变量对齐 + 语言动作描述联合训练
> - **结果**: MindLWPI 在 LIBERO(99.1%)、LIBERO-Plus(74.8%)、WidowX(75.5%) 上综合最优；纯动作预训练对异构数据极度敏感
> - **代码**: 未公开

---

*笔记创建时间: 2026-07-04*
