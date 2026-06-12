---
title: "Dreaming when Necessary: Advancing World Action Models with Adaptive Multi-Modal Reasoning"
method_name: "AdaWAM"
authors: [Yinzhou Tang, Jingbo Xu, Yu Shang, Zihao Song, Chen Gao, Wei Wu, Yong Li]
year: 2026
venue: arXiv
tags: [world-action-model, adaptive-reasoning, multimodal-reasoning, robot-manipulation, flow-matching, diffusion-policy, embodied-ai]
zotero_collection: Robotics/World Model
image_source: online
arxiv_html: https://arxiv.org/html/2606.07089v1
created: 2026-06-09
---

# 论文笔记：Dreaming when Necessary: Advancing World Action Models with Adaptive Multi-Modal Reasoning

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Tsinghua University & Manifold AI |
| 日期 | June 2026 |
| 项目主页 | [adawam.github.io](https://adawam.github.io/) |
| 对比基线 | [[Fast-WAM]] |
| 链接 | [arXiv](https://arxiv.org/abs/2606.07089) / Code: adawam.github.io |

---

## 一句话总结

> AdaWAM 为[[World Action Model|世界动作模型]]引入轻量级动态路由器，按需自适应地激活文本推理或视觉推理模块，在保持推理效率的同时显著提升精细操作任务性能。

---

## 核心贡献

1. **自适应多模态推理框架**: 提出动态路由机制，根据任务执行阶段自主决定是否启用文本推理（高层指令更新）或视觉推理（未来状态预见），而非统一执行所有推理模式
2. **双阶段数据标注管线**: 设计基于轨迹线索的子任务标注与基于运动模式的精细操作标注，为路由器提供高质量监督信号
3. **兼顾效率与性能**: 在 LIBERO-Long（99.1%）和 RoboTwin 2.0 Hard（88.43%）上均达到 SOTA，同时推理延迟接近纯动作预测基线

---

## 问题背景

### 要解决的问题

现有[[World Action Model|世界动作模型（WAM）]]存在两种范式的固有矛盾：
- **视频-动作联合预测**（如 [[Flash-WAM]]）：提供物理预见能力，但每步需生成未来帧，计算延迟大
- **纯动作预测**（如 [[Fast-WAM]]）：推理效率高，但缺乏视觉预见，在需要精细操作的阶段表现不足

### 现有方法的局限

1. [[World Action Model|WAM]] 中的视觉推理（future frame generation）在每个时间步都被强制触发，即使在简单的轨迹跟踪阶段也产生大量冗余计算
2. 纯动作模型在接触密集型操作（如挂杯、叠碗）时缺乏物理约束，精度不足
3. 文本推理（子任务指令更新）如果过于频繁会引入噪声；如果固定间隔触发则无法适应任务动态

### 本文的动机

不同执行阶段对推理模式的需求本质上是不同的：
- **任务过渡阶段**（子任务切换时）：需要文本推理更新高层指令
- **精细操作阶段**（抓取、插入、对齐时）：需要视觉推理提供精确的物理约束
- **常规执行阶段**：直接动作解码即可

因此应当"**按需做梦**"（dreaming when necessary）——让模型自主学习何时需要激活哪种推理模式。

---

## 方法详解

### 模型架构

AdaWAM 采用**双扩散变换器 + 动态路由**架构：

- **输入**: 语言指令 $l$ + 当前观测序列 $z_{\leq t}$ + 子任务指令 $c_t$
- **世界模型主干**: [[Wan 2.2-5B]]（Video[[Diffusion Transformer (DiT)|DiT]]）用于生成未来视觉状态
- **动作策略主干**: [[Wan 2.2-5B]] 压缩 1B 变体（[[Action Chunking|动作块]]预测）
- **核心模块**: [[动态路由器|Dynamic Router]] 预测 `<TR>` 和 `<VR>` 令牌，控制推理模式
- **文本模块**: [[Qwen3-VL]] 4B 作为轻量级 [[VLA（视觉-语言-动作模型）|VLM]] 更新子任务指令
- **总参数**: 约 6B+（VideoDiT 5B + ActionDiT 1B + Qwen3-VL 4B + 路由器）

### 核心模块

#### 模块 1: Video-Action DiT（双扩散变换器）

**设计动机**: 继承 [[World Action Model|WAM]] 的视频-动作联合生成能力，通过[[Flow Matching|流匹配]]在潜空间中同时建模未来视觉状态和动作序列

**具体实现**:
- **VideoDiT** $\mathcal{M}_\theta$：[[Wan 2.2-5B]] 完整版，预测未来视觉状态 $\tilde{z}_f$
- **ActionDiT** $\pi_\phi$：[[Wan 2.2-5B]] 压缩至 1B（隐藏维度 1024），根据动态条件集 $\mathcal{S}_t = \{z_{\leq t}, \tilde{c}_t, \tilde{z}_f\}$ 生成动作块 $a_{t:t+H}$
- 两者均使用[[Flow Matching|流匹配]]目标函数训练，支持联合优化

#### 模块 2: Dynamic Router（动态路由器）$\mathcal{R}_\psi$

**设计动机**: 以极低的计算代价决策当前时间步应激活哪种推理模式，避免不必要的计算浪费

**具体实现**:
- 输入上下文 $C_t = [v_t \| e_l \| e_{c_t}]$，其中：
  - $v_t$：当前潜变量 $z_t$ 的**池化嵌入**
  - $e_l$：全局任务文本嵌入
  - $e_{c_t}$：当前子任务文本嵌入
- 独立预测两个[[二元交叉熵损失|二值路由令牌]]：
  - **`<TR>` 令牌**：1 → 激活[[Qwen3-VL|文本推理模块]]更新子任务指令
  - **`<VR>` 令牌**：1 → 激活 [[Diffusion Transformer (DiT)|VideoDiT]] 生成未来视觉预见
- 使用[[二元交叉熵损失|BCE 损失]]进行有监督路由学习

#### 模块 3: Text Reasoning Module（文本推理模块）$\mathcal{V}_\omega$

**设计动机**: 在任务阶段切换时动态更新子任务指令 $c_t$，提供更精准的语义条件

**具体实现**:
- 使用[[Qwen3-VL]] 4B 作为紧凑型 VLM
- 仅在 `<TR>=1` 时触发，根据当前观测历史 $z_{\leq t}$ 和全局任务指令 $l$ 预测新子任务
- 两阶段训练：Stage 2 冻结其他模块，专门微调该模块（10,000 步）

### 数据标注管线

**两阶段自动标注**（Figure 2）：

**阶段 1 — 轨迹引导的子任务标注**：
1. 解析末端执行器运动轨迹、夹爪状态转换等物理线索
2. 识别候选子任务切换窗口（轨迹折点、停顿等）
3. 调用 [[Qwen3-VL]] 8B 作为语义验证器，确认子任务完成帧
4. 生成 `<TR>` 令牌的监督标签（子任务边界处为 1）

**阶段 2 — 基于运动的精细操作标注**：
1. 分析末端执行器位移、方向调整、局部运动变化和夹爪活动
2. 识别三类精细操作阶段：抓取（grasping）、插入（insertion）、对齐（alignment）
3. 生成 `<VR>` 令牌的监督标签（精细操作阶段为 1）

---

## 关键公式

### 公式 1: [[Flow Matching|流匹配损失]]

$$
\mathcal{L}_{\text{FM}} = \mathbb{E}\left[\|v_\theta(z_\tau, \cdots) - (z_1 - z_0)\|^2 + \|v_\phi(a_\tau, \cdots) - (a_1 - a_0)\|^2\right]
$$

其中 $z_\tau = \tau z_1 + (1-\tau)z_0$，$a_\tau = \tau a_1 + (1-\tau)a_0$

**含义**: 同时优化 VideoDiT 和 ActionDiT，使速度场预测分别逼近视觉潜变量和动作序列的真实流方向

**符号说明**:
- $\tau \sim \mathcal{U}(0,1)$: 流匹配时间步（插值系数）
- $z_0, z_1$: 视觉潜空间中的噪声态和真实态
- $a_0, a_1$: 动作空间中的噪声态和真实态
- $v_\theta$: VideoDiT 的速度场预测网络
- $v_\phi$: ActionDiT 的速度场预测网络

### 公式 2: [[二元交叉熵损失|联合训练目标]]

$$
\mathcal{L} = \mathcal{L}_{\text{FM}} + \lambda \sum_{k \in \{\text{text}, \text{video}\}} \mathcal{L}_{\text{BCE}}(y_k, \hat{y}_k)
$$

**含义**: 在流匹配损失基础上，加入路由器的二值分类损失，联合优化世界模型/策略网络和动态路由器

**符号说明**:
- $\lambda$: 路由损失权重系数
- $y_k$: 第 $k$ 个路由令牌的真实标签（由标注管线生成）
- $\hat{y}_k$: 路由器 $\mathcal{R}_\psi$ 的预测值
- $\mathcal{L}_{\text{BCE}}$: [[二元交叉熵损失|二元交叉熵损失]]

### 公式 3: [[动态路由器|文本推理条件更新]]

$$
\tilde{c}_t = \begin{cases} \mathcal{V}_\omega(z_{\leq t}, l) & \text{if } \langle\text{TR}\rangle = 1 \\ c_t & \text{if } \langle\text{TR}\rangle = 0 \end{cases}
$$

**含义**: 当路由器判断需要文本推理时，调用 VLM 更新当前子任务指令；否则沿用上一步的子任务

**符号说明**:
- $\tilde{c}_t$: 更新后的子任务指令
- $\mathcal{V}_\omega$: 文本推理模块（Qwen3-VL 4B）
- $l$: 全局任务语言指令

### 公式 4: [[动态路由器|视觉推理激活]]

$$
\tilde{z}_f = \begin{cases} \mathcal{M}_\theta(z_{\leq t}, \tilde{c}_t) & \text{if } \langle\text{VR}\rangle = 1 \\ \emptyset & \text{if } \langle\text{VR}\rangle = 0 \end{cases}
$$

**含义**: 当路由器判断需要视觉推理时，调用 VideoDiT 生成未来视觉预见帧；否则该条件置空，直接由历史观测驱动动作

**符号说明**:
- $\tilde{z}_f$: 预测的未来视觉潜变量
- $\mathcal{M}_\theta$: VideoDiT（Wan2.2-5B）

### 公式 5: [[Action Chunking|动作采样]]

$$
a_{t:t+H} \sim \pi_\phi(\cdot | \mathcal{S}_t), \quad \mathcal{S}_t = \{z_{\leq t}, \tilde{c}_t, \tilde{z}_f\}
$$

**含义**: ActionDiT 根据动态条件集（历史观测 + 更新后子任务指令 + 可选视觉预见）生成长度为 $H$ 的动作块

**符号说明**:
- $a_{t:t+H}$: 从时刻 $t$ 开始长度为 $H$ 的[[Action Chunking|动作块]]
- $\mathcal{S}_t$: 动态条件集，随路由决策而变化
- $H$: 动作块长度（chunk size）

---

## 关键图表

### Figure 1: 三种 WAM 推理范式对比

![Figure 1](https://arxiv.org/html/2606.07089v1/x1.png)

**说明**: 对比三种范式——（左）视频-动作联合预测每步生成未来帧，延迟高；（中）纯动作预测高效但缺乏物理预见；（右）AdaWAM 的自适应多模态推理，按需激活文本或视觉推理模块。

### Figure 2: 数据标注管线概览

![Figure 2](https://arxiv.org/html/2606.07089v1/x2.png)

**说明**: 轨迹线索（末端执行器运动、夹爪状态）定位精细操作区间和候选子任务窗口，[[Qwen3-VL]] 8B VLM 进行语义验证，生成 `<TR>` 和 `<VR>` 路由标签。

### Figure 3: AdaWAM 整体架构

![Figure 3](https://arxiv.org/html/2606.07089v1/x3.png)

**说明**: [[动态路由器]] $\mathcal{R}_\psi$ 处理视觉嵌入和文本嵌入，独立预测 `<TR>`/`<VR>` 令牌。`<TR>=1` 时文本模块 $\mathcal{V}_\omega$ 更新子任务指令；`<VR>=1` 时 VideoDiT $\mathcal{M}_\theta$ 生成未来预见；ActionDiT $\pi_\phi$ 最终根据动态条件集 $\mathcal{S}_t$ 输出动作块。

### Figure 4: 真实世界任务可视化

![Figure 4](https://arxiv.org/html/2606.07089v1/x4.png)

**说明**: 在 [[ALOHA]] 平台上的真实世界任务演示——"Clean Up Trash"（清理垃圾）和"Wipe Table"（擦桌子），均为长视野任务，包含多个子阶段转换。

### Figure 5: 纯动作 WAM vs AdaWAM 精细操作对比

![Figure 5](https://arxiv.org/html/2606.07089v1/x5.png)

**说明**: 在精细操作任务"挂杯（HangingMug）"中，纯动作 WAM（[[Fast-WAM]]）因缺乏视觉预见而操作失败；AdaWAM 在精细操作阶段激活视觉推理，成功完成挂杯。

### Figure 6: 推理时间与成功率气泡图

![Figure 6a](https://arxiv.org/html/2606.07089v1/x6.png)

![Figure 6b](https://arxiv.org/html/2606.07089v1/x7.png)

**说明**: 以"叠三只碗（StackBowlsThree）"和"放物体入柜（PutObjectCabinet）"为例，对比各方法在推理时间/步骤（x 轴）、成功率（y 轴）和任务完成时间（气泡大小）上的 trade-off。AdaWAM 推理延迟与 [[Fast-WAM]] 相当，但成功率更高，任务完成时间更短。

### Table 1: LIBERO 基准测试结果

| 方法 | Spatial | Object | Goal | **Long** | **Overall** |
|------|---------|--------|------|----------|-------------|
| OpenVLA | 84.7 | 88.4 | 79.2 | 53.7 | 76.5 |
| π₀ | 96.8 | 98.8 | 95.8 | 85.2 | 94.2 |
| CoT-VLA | 87.5 | 91.6 | 87.6 | 69.0 | 81.1 |
| ACoT-VLA | 98.6 | 99.0 | 99.4 | 97.0 | 98.5 |
| MM-ACT | 97.8 | 99.4 | 94.8 | 88.0 | 95.0 |
| X-VLA | 98.2 | 98.6 | 97.8 | 97.6 | 98.1 |
| LingBot-VA | 98.5 | 99.6 | 97.2 | 98.5 | 98.5 |
| Motus | 96.8 | 99.8 | 96.6 | 97.6 | 97.7 |
| Fast-WAM | 98.2 | **100.0** | 97.0 | 95.2 | 97.6 |
| AdaWAM w/o V.R. | 97.5 | 99.4 | 96.8 | 96.6 | 97.6 |
| AdaWAM w/o T.R. | — | — | — | 97.4 | — |
| **AdaWAM** | **98.0** | 99.6 | **97.1** | **99.1** | **98.5** |

**关键发现**: AdaWAM 在 LIBERO-Long 上以 99.1% 达到所有方法最高分，说明文本推理对长视野任务至关重要；Overall 98.5% 与 LingBot-VA 并列最优。

### Table 2: RoboTwin 2.0 基准测试结果（Clean 环境）

| 任务 | GO-1 | π₀ | π₀.₅ | X-VLA | LingBot-VA | Motus | Fast-WAM | AdaWAM w/o V.R. | AdaWAM w/o T.R. | **AdaWAM** |
|------|------|-----|------|-------|-----------|-------|---------|---------------|---------------|-----------|
| HangingMug | 0 | 14 | 18 | 23 | 40 | 38 | 58 | 56 | 56 | **59** |
| PickDiverseBottles | 61 | 69 | 81 | 58 | 89 | 90 | 80 | 79 | 81 | **87** |
| PutObjectCabinet | 60 | 85 | 80 | 46 | 80 | 88 | 94 | 92 | 91 | **96** |
| RotateQRCode | 22 | 74 | 89 | 34 | **96** | 89 | 93 | 95 | **96** | 94 |
| ScanObject | 1 | 55 | 72 | 14 | **96** | 67 | 89 | 91 | 88 | 92 |
| StackBowlsThree | 4 | 77 | 77 | 76 | **100** | 79 | 80 | 77 | 82 | **100** |
| StampSeal | 19 | 46 | 79 | 76 | **96** | 93 | 90 | 93 | 92 | 91 |
| **Hard SR** | 23.86 | 60.00 | 70.86 | 46.71 | 85.29 | 77.71 | 83.43 | 83.29 | 83.71 | **88.43** |
| **Overall SR** | 37.80 | 65.92 | 82.74 | 72.80 | 92.90 | 88.66 | 91.88 | 91.31 | 91.88 | **93.11** |

**RoboTwin 2.0（Random 环境）**:

| 任务 | GO-1 | π₀ | π₀.₅ | X-VLA | LingBot-VA | Motus | Fast-WAM | AdaWAM w/o V.R. | AdaWAM w/o T.R. | **AdaWAM** |
|------|------|-----|------|-------|-----------|-------|---------|---------------|---------------|-----------|
| HangingMug | 0 | 11 | 17 | 27 | 28 | 38 | 62 | 54 | 53 | **60** |
| PickDiverseBottles | 56 | 31 | 71 | 36 | 82 | 91 | 85 | 83 | 82 | **86** |
| PutObjectCabinet | 43 | 87 | 79 | 48 | 79 | 71 | 89 | 87 | 88 | **91** |
| RotateQRCode | 9 | 70 | 87 | 33 | 91 | 73 | 89 | 89 | **94** | **94** |
| ScanObject | 2 | 42 | 65 | 36 | 91 | 66 | 92 | **93** | 90 | 91 |
| StackBowlsThree | 7 | 75 | 71 | 86 | **98** | 87 | 81 | 78 | 77 | **98** |
| StampSeal | 13 | 33 | 55 | 82 | 97 | 92 | **94** | 91 | 89 | 86 |
| **Hard SR** | 18.57 | 49.86 | 63.57 | 49.71 | 80.86 | 74.00 | 84.57 | 82.14 | 81.86 | **86.57** |
| **Overall SR** | 36.24 | 58.40 | 76.76 | 72.84 | 91.50 | 87.02 | **91.78** | 90.63 | 90.41 | 91.35 |

**关键发现**: AdaWAM 在精细操作任务（HangingMug、StackBowlsThree）上优势最显著，其中 StackBowlsThree 达到满分 100/100；Hard SR 在 Clean/Random 环境下均排名第一。

### Table 3: 真实世界任务结果

| 任务 | π₀.₅ | X-VLA | Motus | GigaBrain-0 | FastWAM | AdaWAM w/o V.R. | AdaWAM w/o T.R. | **AdaWAM** |
|------|------|-------|-------|-------------|---------|-----------------|-----------------|-----------|
| Clean Up Trash | 60 | 60 | 50 | 10 | 30 | 30 | 50 | **70** |
| Wipe Table | 50 | 20 | 20 | 10 | 50 | **60** | **60** | **60** |

**关键发现**: AdaWAM 在真实世界两项任务上均领先，Clean Up Trash 以 70% 显著超越所有基线。

### Table 4: 泛化到未见任务组合

| 环境 | 任务 | Fast-WAM | MM-ACT | AdaWAM w/o T.R. | AdaWAM w/o V.R. | AdaWAM |
|------|------|---------|--------|-----------------|-----------------|--------|
| 已见 | soup & cheese | 98 | 84 | 96 | 95 | **98** |
| 已见 | cheese & butter | 93 | 80 | 94 | 92 | **96** |
| **未见** | soup & butter | 0 | 24 | 3 | 38 | **61** |

**关键发现**: 在未见任务组合上，AdaWAM 以 61% 大幅超越所有基线（第二名仅 38%），说明文本推理对零样本组合泛化的关键作用。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| [[LIBERO]] | 130 个操作序列套件 | 长视野多子任务，4个子集（Spatial/Object/Goal/Long） | 训练/测试 |
| [[RoboTwin 2.0]] | 50 个场景（含 Hard 子集） | 双臂操作，含叠碗/扫描/邮戳等精细任务，Clean & Random 环境 | 测试 |
| 真实世界 | 2 个任务 | ALOHA 平台，Clean Up Trash & Wipe Table | 测试 |

### 实现细节

- **视觉主干**: [[Wan 2.2-5B]]（完整版用于 VideoDiT）+ [[Wan 2.2-5B]] 压缩至 1B（ActionDiT，隐藏维度 1024）
- **文本模块**: [[Qwen3-VL]] 4B
- **路由标注 VLM**: [[Qwen3-VL]] 8B
- **优化器 & 学习率**:
  - Stage 1（50,000 步）：lr = 3×10⁻⁵，weight decay = 0.005
  - Stage 2（10,000 步）：lr = 1×10⁻⁵，weight decay = 0.01
- **硬件**: 8× NVIDIA A100 (80GB)
- **真实机器人平台**: [[ALOHA]] 双臂系统

### 可视化结果

- Figure 5 展示 AdaWAM 在挂杯任务中的视觉推理激活时机：在末端执行器接近挂钩时 `<VR>` 令牌被触发，生成精确的未来对齐预见，引导精细操作成功完成
- Figure 6 气泡图显示 AdaWAM 推理时间/步骤与 Fast-WAM 相当，但任务完成时间（气泡大小）更小，说明自适应推理提高了整体执行效率

---

## 批判性思考

### 优点

1. **优雅的设计哲学**: "按需推理"的思路本质上是对计算资源的智能分配，完美契合实时机器人控制的延迟约束
2. **强泛化能力**: Table 4 中未见任务组合（soup & butter）从 0% 提升至 61%，文本推理的组合泛化能力远超预期
3. **数据标注创新**: 利用轨迹物理线索+VLM 验证的自动标注管线，无需人工逐帧标注路由标签，可扩展性强

### 局限性

1. **视觉空间局限**: 观测空间仅限于 RGB 图像，无法利用深度信息或触觉传感，限制了真正接触密集操作的上限
2. **启发式标注依赖**: 路由监督信号来自启发式规则（轨迹折点、运动模式），若轨迹本身不规则，标注质量可能下降
3. **模块耦合较深**: 两阶段训练中 Stage 2 需要冻结 Stage 1 的参数，端到端训练的潜力尚未充分挖掘

### 潜在改进方向

1. 引入无监督或自监督的路由学习，减少对启发式标注的依赖
2. 扩展观测空间至多模态（深度、触觉），增强精细操作的感知质量
3. 将路由决策与在线强化学习结合，使路由器能够自适应地调整推理频率

### 可复现性评估

- [ ] 代码开源（项目页仅有演示）
- [ ] 预训练模型（未公开）
- [x] 训练细节完整（学习率、步数、GPU 配置均有描述）
- [x] 数据集可获取（LIBERO、RoboTwin 2.0 均为公开 benchmark）

---

## 关联笔记

### 基于

- [[World Action Model]]: AdaWAM 的框架基础，继承 WAM 的视频-动作联合预测能力
- [[Flow Matching]]: 核心训练目标，VideoDiT 和 ActionDiT 均使用流匹配
- [[Diffusion Transformer (DiT)]]: 视频生成主干架构
- [[Wan 2.2-5B]]: 视觉生成基座模型
- [[Qwen3-VL]]: 文本推理模块基座模型

### 对比

- [[Fast-WAM]]: 纯动作 WAM 基线，AdaWAM 以接近其延迟获得更高成功率
- [[Flash-WAM]]: 每步都生成视觉预见的 WAM，AdaWAM 通过自适应推理降低该范式的计算开销

### 方法相关

- [[动态路由器]]: AdaWAM 的核心创新模块
- [[Action Chunking]]: 动作块生成机制
- [[Conditional Flow Matching]]: 条件化流匹配推理
- [[Chain-of-Thought Reasoning]]: 文本推理模块的高层指令链
- [[二元交叉熵损失]]: 路由器训练损失

### 硬件/数据相关

- [[ALOHA]]: 真实世界实验平台（双臂系统）
- [[LIBERO]]: 长视野操作基准
- [[RoboTwin 2.0]]: 双臂精细操作基准

---

## 速查卡片

> [!summary] AdaWAM
> - **核心**: 轻量级动态路由器，按需激活文本推理（子任务更新）或视觉推理（未来预见）
> - **方法**: Video-Action DiT + Dynamic Router + Text Reasoning Module（Qwen3-VL 4B）
> - **结果**: LIBERO-Long 99.1%、RoboTwin 2.0 Hard SR 88.43%，均为 SOTA；真实世界 Clean Trash 70%
> - **代码**: https://adawam.github.io/

---

*笔记创建时间: 2026-06-09*
