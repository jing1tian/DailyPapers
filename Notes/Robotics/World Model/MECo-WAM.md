---
title: "Learning 4D Geometric Priors for Inference-Efficient World Action Models"
method_name: "MECo-WAM"
authors: [Jianjun Zhang, Jian Zhu, Taiyi Su, Chong Ma, Zitai Huang, Yi Xu, Hanli Wang]
year: 2026
venue: arXiv
tags: [world-action-model, robot-manipulation, geometric-learning, 4d-reconstruction, knowledge-distillation, diffusion-transformer, multi-expert-training]
zotero_collection: Robotics/World Model
image_source: online
arxiv_html: https://arxiv.org/html/2607.05468v1
created: 2026-07-09
---

# 论文笔记：Learning 4D Geometric Priors for Inference-Efficient World Action Models

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | 未明确标注（第一作者 Jianjun Zhang 等） |
| 日期 | July 2026 |
| 项目主页 | 暂未公开 |
| 对比基线 | [[Fast-WAM]] |
| 链接 | [arXiv](https://arxiv.org/abs/2607.05468) / Code: 暂未公开 |

---

## 一句话总结

> MECo-WAM 通过多专家协同训练，在训练阶段将冻结 [[VGGT]] 编码器提供的 4D 几何先验注入 [[World Action Model|WAM]] 的表示中，并通过渐进衰减的读取掩码在训练结束时完全移除几何分支，推理时无额外开销。

---

## 核心贡献

1. **多专家协同训练框架（MECo）**: 引入与视频专家、动作专家并行的 4D 专家，以冻结 [[VGGT]] 编码器的几何 token 为监督目标，将 4D 几何先验注入共享表示，推理时 4D 分支完全移除
2. **衰减 4D 读取掩码注意力**: 在训练早期允许视频/动作 query 临时读取当前帧几何 token，概率随训练步数线性衰减至零，使模型主动吸收几何知识后脱钩辅助依赖
3. **动作感知时序几何蒸馏**: 以 token 间 L2 关系距离而非绝对特征为对齐目标，通过动作感知权重强调与机器人运动耦合的视觉 token，并跨关键帧监督几何关系的时序变化

---

## 问题背景

### 要解决的问题

[[World Action Model|WAM]] 通过视频-动作联合预测学习丰富的运动先验，但其视觉表示仍以外观为主导——像素预测不要求模型保留空间关系，导致对"抓取是否可达"等几何敏感决策能力不足。

### 现有方法的局限

- 纯外观预测（如 [[Fast-WAM]]）：视频潜变量足以支持合理的未来合成，但不必然保留决定抓取是否可行的空间关系
- 显式 4D 重建：直接引入深度预测或密集几何输出会增加推理开销，且优化目标与动作生成弱耦合
- [[GeoSem-WAM]] 等方案仍以辅助预测头形式存在，改变了推理计算图

### 本文的动机

4D 几何应作为**训练时的表示约束**而非推理时的输入/输出。若能在训练期间让模型主动学习几何先验，然后在推理时完全脱离几何分支，就可以同时获得几何感知带来的表示增益和轻量级推理的部署效率。

---

## 方法详解

### 模型架构

![Figure 2: MECo-WAM 整体架构](https://arxiv.org/html/2607.05468v1/x2.png)

**说明**: 左侧为训练过程（三路专家并行），右侧为推理过程（仅保留视频专家与动作专家）。4D 专家、冻结 [[VGGT]] 编码器和衰减读取掩码在推理时完全移除。

MECo-WAM 采用 [[Mixture-of-Experts|多专家协同]] 架构，以 [[Wan 2.2-5B|Wan2.2-TI2V-5B]] [[Video Diffusion Transformer]] 为主干：

- **输入**: 语言指令 $\ell$ + 当前观测 $o_0$ + 机器人上下文 $a_0$
- **视频专家** $E_v$: 对未来 [[VAE]] token 去噪，以当前帧 VAE token $f_0$ 为干净锚点
- **动作专家** $E_a$: 以 $a_0$ 为干净锚点，预测动作序列 $a_{1:H}$，条件化于视觉特征
- **4D 专家** $E_{4d}$（仅训练）: token 来自冻结 [[VGGT]] 编码器，提供几何监督
- **输出**: [[Action Chunking|动作块]] $a_{1:H}$（H=32 步）
- **总参数**: 动作专家 ~1B，4D 专家 ~0.45B，视频主干 ~5B

### 核心模块

#### 模块 1: 混合注意力多专家序列

三路专家 token 拼接后共享 [[Mixed Attention|混合注意力]]：
$$X = [X_v, X_a, X_{4d}]$$

每个专家 $e \in \{v, a, 4d\}$ 具有独立的 Query/Key/Value 投影，通过掩码混合注意力避免信息捷径（例如，未来视频 token 不直接读取动作 token）。

#### 模块 2: 衰减 4D 读取掩码注意力

![Figure 3: 衰减读取掩码注意力可视化](https://arxiv.org/html/2607.05468v1/x3.png)

**说明**: 行/列分别为 query/key slot。阴影格为临时读取边（未来视频/动作 query → 当前帧几何 $g_0$）；白格为掩码区域。随训练进行，阴影格概率线性衰减至零。

**设计动机**: 将当前帧几何 token $g_0$（由当前帧编码，无未来信息泄露）开放给未来视频/动作 query，以[[Knowledge Distillation|隐式蒸馏]]方式传递几何知识。读取边以伯努利采样控制，概率随步数线性衰减。

**注意力掩码规则**:
- 当前锚点（$f_0, a_0, g_0$）：仅自注意力
- 未来视频 token：读取视频分支（不读动作 token）
- 未来动作 token：读取干净视觉上下文 + 动作分支
- 临时读取边：未来视频/动作 query → 当前帧几何 $g_0$（概率衰减）

#### 模块 3: 动作感知时序几何蒸馏

**VGGT 对齐几何**: 以 token 对之间的 L2 关系距离而非绝对特征为对齐目标，避免假设共享绝对特征坐标系：

$$R_{4d}^k(i,j) = \|Z_i^k - Z_j^k\|_2$$
$$R_T^k(i,j) = \|G_{T,i}^k - G_{T,j}^k\|_2$$

其中 $Z^k$ 为 4D 专家预测的 token，$G_T^k$ 为 [[VGGT]] 编码的目标几何 token。

**动作感知权重**: 计算视觉 token 与汇聚动作表示的余弦相似度，识别与机器人运动耦合的视觉区域：

$$s_i^k = \frac{(W_v v_i^k)^\top (W_a \bar{a}^k)}{\tau \|W_v v_i^k\|_2 \|W_a \bar{a}^k\|_2}$$

混合均匀先验防止权重坍塌：$w_i^k = N\left((1-\eta)r_i^k + \frac{\eta}{N}\right)$

**时序几何**: 监督关键帧之间的几何关系如何变化（接近、接触、搬运、释放）：

$$\Delta(R^k, R^{k+}) = \frac{R^{k+} - R^k}{|R^{k+}| + |R^k| + \epsilon}$$

---

## 关键公式

### 公式 1: [[Flow Matching|流匹配目标函数]]

$$
\mathcal{L}_{FM}(y) = \mathbb{E}_{y, \epsilon, r}\left[\|u_\theta(y_r, r, o_0, a_0, \ell) - (\epsilon - y)\|_2^2\right]
$$

**含义**: 模型学习预测从噪声 $y_r = (1-r)y + r\epsilon$ 到目标 $y$ 的速度场，$r \in [0,1]$ 为噪声水平

**符号说明**:
- $y$: 目标（视频 token 或动作序列）
- $\epsilon$: 高斯噪声
- $r$: 噪声比例（0 = 干净，1 = 纯噪声）
- $u_\theta$: 神经网络预测的速度场

### 公式 2: [[Bernoulli Sampling|衰减 4D 读取概率]]

$$
p_{4d}(s) = \begin{cases}
p_{\text{start}} + (p_{\text{end}} - p_{\text{start}}) \dfrac{s}{S_{\text{decay}}}, & s < S_{\text{decay}} \\
p_{\text{end}}, & s \geq S_{\text{decay}}
\end{cases}
$$

$$\gamma_s \sim \text{Bernoulli}(p_{4d}(s))$$

**含义**: 训练早期（$s=0$）读取概率为 $p_{\text{start}}=1.0$，在前半段训练中线性衰减至 $p_{\text{end}}=0$；推理时 $p_{4d}=0$，4D 分支完全移除

**符号说明**:
- $s$: 当前训练步数
- $S_{\text{decay}}$: 衰减截止步数（设为训练总步数的一半）
- $\gamma_s$: 当前步是否激活读取边的伯努利随机变量

### 公式 3: [[Relational Distillation|帧内几何蒸馏损失]]

$$
\mathcal{L}_{\text{geo}}^{\text{act}} = \frac{\sum_{k \in \mathcal{K}, i, j} w_{ij}^k \left|\hat{R}_{4d}^k(i,j) - \hat{R}_T^k(i,j)\right|}{\sum_{k \in \mathcal{K}, i, j} w_{ij}^k}
$$

**含义**: 以动作感知权重 $w_{ij}^k = \sqrt{w_i^k w_j^k}$ 加权的归一化关系距离 L1 损失，强调与机器人运动耦合的 token 对

**符号说明**:
- $\mathcal{K}$: 关键帧集合
- $\hat{R}$: 归一化关系距离矩阵
- $w_{ij}^k$: token 对 $(i,j)$ 在关键帧 $k$ 的动作感知权重

### 公式 4: [[Temporal Geometry|时序几何蒸馏损失]]

$$
\mathcal{L}_{\text{tem}}^{\text{act}} = \frac{\sum_{(k, k^+) \in \mathcal{A}_{\mathcal{K}}, i, j} w_{ij}^{k,k^+} D_{ij}^{k,k^+}}{\sum_{(k, k^+) \in \mathcal{A}_{\mathcal{K}}, i, j} w_{ij}^{k,k^+}}
$$

**含义**: 监督相邻关键帧对 $(k, k^+)$ 之间几何关系变化量的预测误差，捕获接近/接触/搬运/释放等阶段的动态几何演变

**符号说明**:
- $\mathcal{A}_{\mathcal{K}}$: 相邻关键帧对的集合
- $D_{ij}^{k,k^+}$: token 对 $(i,j)$ 在帧对 $(k,k^+)$ 上的时序变化预测误差
- $w_{ij}^{k,k^+}$: 时序帧对的动作感知权重

### 公式 5: [[Multi-Expert Training|总训练损失]]

$$
\mathcal{L}_{4d} = \alpha_{\text{geo}} \mathcal{L}_{\text{geo}}^{\text{act}} + \alpha_{\text{tem}} \mathcal{L}_{\text{tem}}^{\text{act}}
$$

$$
\mathcal{L}_{\text{total}} = \lambda_{\text{video}} \mathcal{L}_{\text{video}} + \lambda_{\text{action}} \mathcal{L}_{\text{action}} + \lambda_{4d} \mathcal{L}_{4d}
$$

**含义**: 三路专家的流匹配损失加权求和，$\lambda_{4d}$ 控制几何蒸馏的贡献权重

**符号说明**:
- $\mathcal{L}_{\text{video}}$: 视频专家流匹配损失
- $\mathcal{L}_{\text{action}}$: 动作专家流匹配损失
- $\alpha_{\text{geo}}, \alpha_{\text{tem}}$: 帧内/时序几何损失权重

---

## 关键图表

### Figure 1: 推理延迟与成功率对比

![Figure 1: MECo-WAM 与基线在 RoboTwin 上的推理延迟与任务成功率对比](https://arxiv.org/html/2607.05468v1/x1.png)

**说明**: MECo-WAM 在保持与 [[Fast-WAM]] 相同推理延迟（单次前向传播）的前提下，在 RoboTwin 2.0 上达到更高任务成功率，优于需要视频展开的方法。

### Figure 2: 整体架构

![Figure 2: MECo-WAM 训练（左）与推理（右）流程](https://arxiv.org/html/2607.05468v1/x2.png)

**说明**: 训练时三路专家并行，4D 专家以冻结 [[VGGT]] token 为输入提供几何监督；推理时移除 4D 专家、VGGT 编码器和衰减掩码，仅保留视频-动作路径。

### Figure 3: 衰减读取掩码注意力

![Figure 3: 衰减读取掩码注意力矩阵可视化](https://arxiv.org/html/2607.05468v1/x3.png)

**说明**: 展示 [[Mixed Attention|混合注意力]] 掩码设计。阴影格为临时读取边（未来 query → 当前帧几何 $g_0$），随训练步数线性衰减至完全掩码。

### Figure 4: 深度探针对比

![Figure 4: 共享视频-动作表示的深度探针对比](https://arxiv.org/html/2607.05468v1/x4.png)

**说明**: 对共享表示训练 [[DPT|DPT 风格探针]]头，MECo-WAM 的表示展现出更清晰的深度结构，说明 4D 几何先验已被有效编码到视频-动作表示中。

### Figure 5: 真实机器人实验

![Figure 5: 真实机器人桌面实验（堆叠方块与尺寸分拣）](https://arxiv.org/html/2607.05468v1/x5.png)

**说明**: ARX-R5 机械臂上的堆叠方块（Stack Cubes）和按尺寸分拣方块（Sort Cubes by Size）任务，MECo-WAM 成功率和效率均优于 [[Fast-WAM]] 基线。

### Figure 6: 3D 位置与姿态敏感性分析

![Figure 6: 真实机器人抓取过程中的 3D 位置与姿态敏感性](https://arxiv.org/html/2607.05468v1/x6.png)

**说明**: 分析模型对方块 3D 位置、抓取姿态和接触几何的保留程度，MECo-WAM 更好地维持了与动作相关的空间几何信息。

### Table 1: LIBERO 成功率（%）

| Method | Spatial | Object | Goal | Long | Average |
|--------|---------|--------|------|------|---------|
| π₀ | 96.8 | 98.8 | 95.8 | 85.2 | 94.1 |
| OpenVLA-OFT | 97.6 | 98.4 | 97.9 | 94.5 | 97.1 |
| DD-VLA | 97.2 | 96.6 | 97.4 | 92.0 | 96.3 |
| X-VLA | 98.2 | 98.6 | 97.8 | 97.6 | 98.1 |
| Fast-WAM | 98.2 | 100.0 | 97.0 | 95.2 | 97.6 |
| **MECo-WAM** | **98.8** | **100.0** | **98.2** | **95.8** | **98.2** |

**关键发现**: MECo-WAM 在全部四个 LIBERO 子集上均达到最优或并列最优，平均成功率 98.2%，超越 [[Fast-WAM]] (+0.6 点)，与有预训练的 LingBot-VA (98.5%) 接近。

### Table 2: RoboTwin 2.0 成功率（%）

| Method | 预训练 | Clean | Randomized | Average |
|--------|--------|-------|-----------|---------|
| π₀.₅ | ✓ | 82.74 | 76.76 | 79.75 |
| Motus | ✓ | 88.66 | 87.02 | 87.84 |
| Fast-WAM | ✗ | 91.88 | 91.78 | 91.83 |
| **MECo-WAM** | **✗** | **93.26** | **91.98** | **92.62** |

**关键发现**: MECo-WAM 无需预训练，以 92.62% 平均成功率超越所有基线（含有预训练的 Motus），超越 [[Fast-WAM]] +0.79 点，也超越有预训练的 LingBot-VA (92.20%)。

### Table 3: 真实世界桌面实验结果

**堆叠方块（Stack Cubes）:**

| Method | 成功率 | 进度率 | 纠正次数 | 完成时间 |
|--------|--------|--------|---------|--------|
| Fast-WAM | 60.0% | 75.0% | 1.67 | 27.06s |
| **MECo-WAM** | **60.0%** | **75.0%** | **0.83** | **25.71s** |

**按尺寸分拣方块（Sort Cubes by Size）:**

| Method | 成功率 | 进度率 | 纠正次数 | 完成时间 |
|--------|--------|--------|---------|--------|
| Fast-WAM | 60.0% | 75.0% | 1.33 | 38.49s |
| **MECo-WAM** | **70.0%** | **80.0%** | **1.00** | **31.96s** |

**关键发现**: 在需要精确空间感知的真实任务中，MECo-WAM 平均提升 5.0 SR / 2.5 PR，纠正次数减少 39%，完成时间缩短 12%。

### Table 4: 消融实验（RoboTwin 2.0）

| 配置 | PSNR↑ | SSIM↑ | LPIPS↓ | MSE(×10)↓ | Avg SR↑ |
|------|-------|-------|--------|-----------|---------|
| Fast-WAM（基线） | 29.55 | 0.936 | 0.038 | 0.032 | 91.83 |
| + 4D 专家 | 29.81 | 0.935 | 0.039 | 0.034 | 91.87 |
| + 衰减读取 | 30.06 | 0.939 | 0.038 | 0.026 | 92.14 |
| + $\mathcal{L}_{\text{geo}}^{\text{act}}$ | 29.97 | 0.938 | 0.037 | 0.022 | 92.22 |
| + $\mathcal{L}_{\text{tem}}^{\text{act}}$ | 30.42 | 0.940 | 0.039 | 0.019 | 92.25 |
| Full（无动作感知权重） | 30.31 | 0.942 | 0.038 | 0.017 | 92.38 |
| **MECo-WAM（Full）** | **30.72** | **0.942** | **0.037** | **0.013** | **92.62** |

**关键发现**: 单独添加 4D 专家收益极小（+0.04 SR）；衰减读取掩码贡献最大单步增益（+0.31 SR）；动作感知权重为最终 +0.24 SR 的关键差异来源。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| [[LIBERO]] | 4 子集 × 10 任务 × 500 演示 | 桌面操作，含空间/目标/长视野任务 | 测试 |
| [[RoboTwin]] 2.0 | 双臂操作，Clean + Randomized | 场景随机化测试泛化 | 测试 |
| 真实世界（ARX-R5） | 两个桌面任务，各 10 次试验 | 物理真实，含堆叠与分拣 | 测试 |

### 实现细节

- **视频主干**: [[Wan 2.2-5B|Wan2.2-TI2V-5B]]
- **动作专家**: 30 个 DiT blocks，24 注意力头，$d_a = 1024$，~1B 参数
- **4D 专家**: $d_{4d} = 512$，~0.45B 参数
- **衰减概率**: 线性从 1.0 衰减至 0，训练前半段完成
- **训练分块**: 33 步机器人动作（H=32），9 帧视频，动作:视频 = 4:1
- **优化器**: AdamW，lr = $1 \times 10^{-4}$，bfloat16 精度
- **硬件（训练）**: 64 × NVIDIA H20 GPU
- **硬件（推理）**: 单张 RTX 5090
- **推理设置**: 10 步去噪，CFG scale = 1.0

---

## 批判性思考

### 优点

1. **推理零开销**: 几何分支在训练结束后完全移除，不改变推理计算图，是"免费"的表示提升
2. **衰减机制设计精巧**: 通过渐进衰减而非硬切换，让模型有足够时间内化几何知识后再脱离辅助信号，避免突然移除导致的表示退化
3. **动作感知权重**: 相比均匀蒸馏，聚焦于与机器人动作耦合的视觉区域，提升几何先验的任务相关性

### 局限性

1. **训练成本翻倍**: 4D 专家 + VGGT 编码器仅在训练时存在，但显著增加训练显存和计算量（64 × H20 GPU）
2. **消融增益有限**: 在 RoboTwin 上最终提升仅 +0.79 SR，且逐步消融显示各组件单独收益较小，方法复杂度与增益的比例值得商榷
3. **依赖 VGGT 质量**: 几何先验的质量完全取决于冻结 [[VGGT]] 编码器，若 VGGT 对特定场景泛化差，蒸馏目标本身就存在噪声

### 潜在改进方向

1. 探索更轻量的几何监督来源（如单目深度估计器）以降低训练成本
2. 将衰减调度自适应化（基于表示对齐度动态调整）而非固定线性衰减
3. 扩展到双目/多视角场景，利用真实 3D 点云作为几何目标

### 可复现性评估

- [ ] 代码开源
- [ ] 预训练模型
- [x] 训练细节完整（优化器、学习率、GPU 配置等均已提供）
- [x] 数据集可获取（LIBERO、RoboTwin 2.0 均公开）

---

## 关联笔记

### 基于

- [[Fast-WAM]]: 直接基线，MECo-WAM 在其基础上添加 4D 几何专家分支
- [[VGGT]]: 冻结几何编码器，提供蒸馏目标

### 对比

- [[GeoSem-WAM]]: 同样将几何引入 WAM 训练，但通过 DPT 辅助预测头而非关系蒸馏
- [[WAM4D]]: 在推理时显式使用 4D 输入，MECo-WAM 则在推理时移除几何依赖

### 方法相关

- [[Flow Matching]]: 核心去噪框架
- [[Knowledge Distillation]]: 衰减读取掩码的本质是几何知识的渐进式隐式蒸馏
- [[Mixed Attention]]: 多专家之间的注意力混合机制
- [[Action Chunking]]: 动作专家的预测目标格式

### 数据集相关

- [[LIBERO]]: 主要评测基准之一（桌面操作）
- [[RoboTwin]]: 主要评测基准之一（双臂操作）

---

## 速查卡片

> [!summary] MECo-WAM (2607.05468)
> - **核心**: 训练时引入冻结 VGGT 作为 4D 几何监督，推理时完全移除，零额外开销
> - **方法**: 多专家协同训练 + 衰减读取掩码 + 动作感知时序几何蒸馏
> - **结果**: LIBERO 98.2%、RoboTwin 2.0 92.62%，真实任务纠正次数 -39%
> - **代码**: 暂未公开

---

*笔记创建时间: 2026-07-09*
