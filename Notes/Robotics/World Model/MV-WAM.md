---
title: "MV-WAM: Manifold-Aware World Action Model with Value Augmentation"
method_name: "MV-WAM"
authors: [Jintao Chen, Peidong Jia, Qingpo Wu, Jiaming Liu, Mengfei Du, Chun-Kai Fan, Xiaowei Chi, Hao Chen, Chengyu Bai, Zezhong Qian, Hao Wang, Jiajun Cao, Weishi Mi, Xiaozhu Ju, Jian Tang, Shanghang Zhang]
year: 2026
venue: arXiv
tags: [world-action-model, robot-manipulation, manifold-learning, flow-matching, mixture-of-transformers, value-function, dual-arm-manipulation]
zotero_collection: Robotics/World Model
image_source: online
arxiv_html: https://arxiv.org/html/2606.21088v1
created: 2026-06-24
---

# 论文笔记：MV-WAM: Manifold-Aware World Action Model with Value Augmentation

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | State Key Laboratory of Multimedia Information Processing, School of Computer Science, Peking University; Beijing Innovation Center of Humanoid Robotics |
| 日期 | June 2026 |
| 项目主页 | 暂未公开 |
| 对比基线 | [[Fast-WAM]] |
| 链接 | [arXiv](https://arxiv.org/abs/2606.21088) / Code: 暂未公开 |

---

## 一句话总结

> MV-WAM 指出视觉与动作模态的内在几何曲率不同，用非对称专家 + 分模态预测目标 + 价值引导回滚解决联合优化损害动作鲁棒性的问题。

---

## 核心贡献

1. **结构错配的几何归因**: 首次从流形曲率角度定量证明视觉（高曲率）与动作（低曲率）模态存在结构性差异，解释了为何联合优化会不成比例地损害动作分支的分布外鲁棒性
2. **流形感知双预测目标**: 视觉分支用速度场（flow/velocity）预测匹配其高曲率流形，动作分支用直接状态回归（$x_0$-prediction）匹配其低曲率流形，而非对两个模态使用统一的预测参数化
3. **跨模态因果掩码 + 价值引导回滚**: 用非对称注意力掩码让动作以预测视觉为条件但不反向干扰视觉生成，并引入 Progress-Value token 在线检测执行偏差、自动回滚重采样，无需人工干预

---

## 问题背景

### 要解决的问题

[[VLA]] 模型在语义泛化上表现良好，但在视觉/环境泛化（未见过的光照、纹理、物体摆放）上仍然脆弱。[[World Action Model|World Action Model (WAM)]] 试图通过引入无动作标注的视频数据来扩大视觉覆盖范围，但实测发现其域内（in-domain）与分布外（OOD）性能之间存在巨大且有时会扩大的泛化差距（generalization gap）。

### 现有方法的局限

- 现有 WAM 架构无论是解耦式（cascaded）、完全统一式（fully unified），还是基于 [[Mixture-of-Transformers|MoT]] 的混合式，都表现出相同的泛化差距问题
- 根本原因在于：视觉观测是高维、densely structured，受空间和光度规律支配；机器人动作是低维、时序结构化、对精度高度敏感——二者是**内在异质的流形（heterogeneous manifolds）**
- 对两个异质流形做联合优化，会不成比例地损害动作分支的学习，即"joint optimization over incompatible manifolds inevitably compromises action learning"

### 本文的动机

如果视觉和动作模态确实分布在曲率不同的流形上，那么用同一套预测目标（如统一的噪声预测或统一的 flow 预测）去训练两者，本质上是用不适配的几何假设去拟合数据。MV-WAM 的核心动机是：让每个模态的预测目标与其自身流形的几何属性相匹配，同时通过跨模态注意力让动作生成仍能利用视觉分支学到的世界动态先验，并额外引入在线价值信号让模型具备自我纠偏能力。

---

## 方法详解

### 模型架构

![Figure 1: MV-WAM Overview](https://arxiv.org/html/2606.21088v1/x1.png)

**说明**: MV-WAM 引入非对称的 video-action 专家，通过因果掩码让动作以预测的视觉动态为条件，同时保留各模态独立的预测目标。

MV-WAM 采用 **[[Mixture-of-Transformers|MoT]]（Mixture-of-Transformers）** 骨干，两个模态专属专家共享全局注意力但保持独立参数：

- **输入**: 语言指令通过 [[T5|umT5]] 编码 + 视觉观测通过 [[SigLIP2]] / [[VAE|spatiotemporal VAE]] 压缩为视觉 token + 机器人状态 $s_t$
- **Video Expert**（30 个 Transformer block）：从 WoW-1.3B 预训练模型初始化，处理经时空 VAE 压缩的视觉 token
- **Action-Value Expert**（30 个 block）：处理动作 token 和 [[Progress-Value Regulation|价值 token]]，隐藏维度较低（768），仅在共享自注意力中投影到视觉 token 空间（1536 维）
- **核心模块**: [[Cross-Modality Causal Masking|跨模态因果掩码]] 用于 [[Mixture-of-Transformers|非对称专家间信息流动]]
- **输出**: [[Action Chunking|动作块]] $a_{t:t+k}$ + 执行进度价值估计 $\hat{c}(s_t, a_t)$
- **总参数**: 1.9B（不含 umT5 与 SigLIP2 编码器）

### 核心模块

#### 模块1: 跨模态因果掩码 (Cross-Modality Causal Masking)

**设计动机**: 利用 [[Mixture-of-Transformers|MoT]] 共享注意力机制，在不破坏视觉生成保真度的前提下，让动作分支充分利用视觉分支预测的未来动态。

**具体实现**:
- 视觉 token 仅限于自注意力（self-attention），保持高保真度生成不被动作分支干扰
- 动作 token 可同时attend视觉上下文与其他动作 token
- 价值 token 联合 attend 视觉、动作与状态 token
- [[RoPE|Rotary Positional Embedding]] 缩放因子：Action-Value Expert 使用 Video Expert 缩放因子的 1/4，以实现时间对齐

#### 模块2: 流形感知双预测目标 (Manifold-Aware Dual Prediction)

**设计动机**: 利用流形曲率分析（视觉曲率 $\kappa_v = 3.8\pm0.6$ 远高于动作曲率 $\kappa_a = 1.3\pm0.2$，$p<0.001$）证明两模态几何属性显著不同，因此分别采用与各自流形曲率匹配的预测参数化。

**具体实现**:
- 视觉分支（高曲率流形）：使用 [[Flow Matching|速度场预测]]（velocity/flow-matching），适合捕捉弯曲流形上的非线性变化
- 动作分支（低曲率流形）：使用直接状态回归（$x_0$-prediction），在低曲率、接近线性的流形上回归终点比预测速度更稳定
- 消融证实：统一用噪声预测的 baseline 在 Clean/Random 上仅 15.0%/6.0%，统一用 flow 预测的 baseline 甚至跌到 6.0%（完全失败），印证了流形错配假设

#### 模块3: Progress-Value 调节与回滚

**设计动机**: 利用 [[Progress-Value Regulation|蒙特卡洛价值估计]]在执行过程中自动检测偏离预期轨迹的情况，无需人工监督即可纠偏。

**具体实现**:
- 从已执行轨迹中通过蒙特卡洛方法估计价值目标 $\hat{c}(s_t, a_t)$
- 价值引导回滚：当预测价值低于学习到的阈值 $\delta$ 时，系统回退到缓存的高价值状态并重新采样动作
- 阈值 $\delta=0.20$ 时取得最佳表现（84.0% Clean / 55.7% Random），且对阈值变化鲁棒

---

## 关键公式

### 公式1: [[Flow Matching|动作流匹配损失]]

$$
\mathcal{L}_{\text{FM}} = \mathbb{E}_{\tau, a_0, a_1} \left\| v_\theta(a_\tau, \tau, \mathcal{C}) - (a_1 - a_0) \right\|_2^2
$$

**含义**: 动作生成的通用 flow matching 损失形式，作为视觉/动作两条具体分支损失（公式2、3）的统一表达。

**符号说明**:
- $\tau \sim \mathcal{U}(0,1)$: 插值时间步
- $a_0, a_1$: 噪声起点与真实动作终点
- $\mathcal{C}$: 条件信息（语言指令 + 视觉上下文）
- $v_\theta$: 待学习的速度场网络

### 公式2: 视觉专家速度预测损失

$$
\mathcal{L}_v = \mathbb{E}_{\tau, z_0, z_1} \left\| \pi_\theta^v(z_\tau, \tau, c) - (z_1 - z_0) \right\|_2^2
$$

**含义**: 视觉分支（高曲率流形）采用速度场预测目标，让模型学习视觉潜变量插值路径上的瞬时速度。

**符号说明**:
- $z_\tau = (1-\tau)z_0 + \tau z_1$: 视觉 VAE 潜变量的插值
- $\pi_\theta^v$: Video Expert 的速度预测头
- $c$: 条件（语言 + 状态）

### 公式3: 动作专家 $x_0$-prediction 损失

$$
\mathcal{L}_a = \mathbb{E}_{\tau, a_0, a_1} \left\| \pi_\theta^a(a_\tau, \tau, c) - a_0 \right\|_2^2
$$

**含义**: 动作分支（低曲率流形）不预测速度，而是直接回归带噪动作 $a_\tau$ 对应的真实终点 $a_0$，与流形低曲率特性匹配，训练更稳定。

**符号说明**:
- $a_\tau$: 插值路径上的带噪动作
- $\pi_\theta^a$: Action-Value Expert 的 $x_0$ 预测头
- $a_0$: 真实动作（回归目标，注意与公式1中作为"噪声起点"的记号 $a_0$ 含义在此处对齐为真实终点，遵循原论文符号约定）

### 公式4: [[Progress-Value Regulation|蒙特卡洛价值目标]]

$$
\hat{c}(s_t, a_t) = \sum_{i=0}^{H-t} \gamma^i \, r(s_{t+i}, a_{t+i})
$$

**含义**: 用蒙特卡洛回报估计当前状态-动作对的执行进度价值，作为 Progress-Value token 的训练监督信号。

**符号说明**:
- $\gamma = 0.99$: 折扣因子
- $H$: 任务总时长（horizon）
- $r(s_{t+i}, a_{t+i})$: 第 $t+i$ 步的即时奖励/进度信号

### 公式5: CFM 线性插值路径

$$
x_t = (1-t)x_0 + t x_1, \quad t \in [0,1]
$$

**含义**: Conditional Flow Matching 的标准直线插值路径，是公式2、3 中 $z_\tau$、$a_\tau$ 构造方式的通用形式。

**符号说明**:
- $x_0$: 噪声采样点
- $x_1$: 真实数据点
- $t$: 插值系数，对应正文中的 $\tau$

### 公式6: 端点预测重加权

$$
\left\| \frac{x_t - \hat{x}_{0,\theta}}{t} - \frac{x_t - x_0}{t} \right\|_2^2 = \frac{1}{t^2} \left\| \hat{x}_{0,\theta} - x_0 \right\|_2^2
$$

**含义**: 揭示 $x_0$-prediction 与 velocity-prediction 在数学上等价，但 $1/t^2$ 的重加权因子会在 $t \to 0$ 时显著放大梯度方差，这正是低曲率（动作）流形上直接用 $x_0$-prediction 而非 velocity-prediction 更稳定的理论依据之一。

**符号说明**:
- $\hat{x}_{0,\theta}$: 网络预测的终点
- $t$: 插值时间步，$t\to 0$ 时重加权系数 $1/t^2$ 趋于无穷

---

## 关键图表

### Figure 1: Overview / 总览

![Figure 1: MV-WAM Overview](https://arxiv.org/html/2606.21088v1/x1.png)

**说明**: MV-WAM 引入非对称 video-action 专家，通过因果掩码使动作以预测的视觉动态为条件。

### Figure 2: 详细架构与训练流程

![Figure 2: Detailed Architecture and Training](https://arxiv.org/html/2606.21088v1/x2.png)

**说明**: (a) 详细架构由 Video Expert 与 Action-Value Expert 构成；(b) 流形感知训练（视觉用速度预测，动作用 $x_0$ 预测）；(c) 两阶段训练（视频专家预训练 → 视频-动作联合训练）；(d) 在线执行中的价值引导回滚机制。

### Figure 3: 消融实验

![Figure 3: Ablation Studies](https://arxiv.org/html/2606.21088v1/x3.png)

**说明**: (a) 流形感知预测目标的消融——统一噪声预测/统一 flow 预测均大幅劣于分模态目标；(b) 去噪步数的影响——3 步即可恢复 5 步性能的 96%；(c) 回滚阈值 $\delta$ 的敏感性分析——$\delta=0.20$ 最优且整体鲁棒。

### Figure 4: 真实世界执行可视化

![Figure 4: Real-World Execution Visualization](https://arxiv.org/html/2606.21088v1/x4.png)

**说明**: 展示四个真实世界操作任务（背包抓取、布料投放、布料拾取、布料折叠）的完整执行进度序列。

### Figure 5: 真实机器人实验平台

![Figure 5: Real-World Robot Setup](https://arxiv.org/html/2606.21088v1/x5.png)

**说明**: [[Dual-Arm Robot|TienKung 双臂机器人]]平台，配备两个手腕安装摄像头，构成三视角相机配置用于真实数据采集。

### Figure 6: 泛化差距对比

![Figure 6: Generalization Gap](https://arxiv.org/html/2606.21088v1/x6.png)

**说明**: 对比各类 [[VLA]] 与 [[World Action Model|WAM]] 模型从 Clean 到 Random 设置下的性能跌幅，MV-WAM 的相对差距（33.7%）远小于最强基线 HALO（67.2%）。

### Figure 7: t-SNE 表示可视化

![Figure 7: t-SNE Visualization](https://arxiv.org/html/2606.21088v1/x7.png)

**说明**: 对视觉专家与动作专家表示做 per-task 均值中心化后的 t-SNE 可视化，揭示分布外条件下两个专家表示空间呈现的不对称漂移，佐证了模态流形异质性的假设。

### Table 1: RoboTwin 2.0 基准测试结果（50 任务）

| 方法 | Clean (%) | Random (%) |
|------|-----------|-------------|
| DP | — | — |
| RDT | — | — |
| $\pi_0$ | — | — |
| UP-VLA | — | — |
| BagelVLA | — | — |
| HALO | 80.5 | 26.4 |
| Fast-WAM | — | — |
| **MV-WAM** | **84.0** | **55.7** |

**说明**: MV-WAM 相对最强基线 HALO 在 Clean 设置提升 +3.5pp，在 Random（OOD）设置提升 +29.3pp；HALO 从 Clean 到 Random 跌幅达 67.2%，而 MV-WAM 仅跌 33.7%，体现更小的泛化差距。原文 Table 1 含完整逐任务分解（对应原文 Table 7），此处保留汇总数值；逐方法的 DP/RDT/$\pi_0$/UP-VLA/BagelVLA/Fast-WAM 具体数值见原文表格。

### Table 2: 零样本与少样本泛化（10 个未见 RoboTwin 任务）

| 设置 | Clean (%) | Random (%) |
|------|-----------|-------------|
| 0-shot | 55.6 | 54.0 |
| 10-shot | 77.1 | 74.6 |

**说明**: 即使在完全未见过的任务上零样本部署，MV-WAM 仍取得过半成功率，10-shot 微调后进一步逼近域内水平。

### Table 3: 真实世界双臂机器人结果

| 方法 | Pick Backbag | Drop Cloth | Pick Cloth | Fold Cloth | Mean SR |
|------|--------------|------------|------------|------------|---------|
| $\pi_0$ | 50% | 60% | 50% | 10% | 42.5% |
| RDT | 30% | 40% | 60% | 0% | 32.5% |
| **MV-WAM** | **90%** | **100%** | **100%** | **20%** | **77.5%** |

**关键发现**: MV-WAM 在三项任务上接近或达到满分，在难度最高的 Fold Cloth 上虽仅 20% 但仍领先或持平基线，整体均值显著超过两个 VLA 基线。

### Table 4: 预训练数据集构成

| 数据来源 | 时长 |
|----------|------|
| [[EgoDex]] | 800h |
| [[AgiBot World\|Agibot]] | 2,500h |
| [[RoboMIND\|RoboMind]] | 1,000h |
| [[RoboCOIN\|RoboCoin]] | 1,000h |
| [[Open X-Embodiment]] | 2,000h |
| Internal | 200h |
| **总计** | **7,500h** |

**说明**: 预训练语料以第三方大规模具身数据为主（Agibot 占比最高），并补充少量内部数据。

### Table 5: 流形曲率估计

| 模态 | 曲率 $\kappa$ |
|------|----------------|
| 视觉 | $3.8 \pm 0.6$（$p<0.001$） |
| 动作 | $1.3 \pm 0.2$ |

**关键发现**: 视觉流形曲率显著高于动作流形（约 2.9 倍），是本文"流形异质性导致联合优化损害动作鲁棒性"假设的核心定量证据。

### Table 6: CFM 预测参数化对比

| 参数化 | 损失形式 | 推理速度场 |
|--------|----------|------------|
| v-prediction | 预测速度差 | 直接使用网络输出 |
| $x_0$-prediction | 预测终点 | 由终点反推速度 |
| noise-prediction | 预测噪声 | 由噪声反推速度 |

**关键发现**: 三种参数化在数学上等价（见公式6），但梯度尺度随插值时间步 $t$ 呈现不同的重加权效应，在不同曲率流形上的实际训练稳定性差异显著，这是选择分模态参数化的理论依据。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| 预训练语料 | ~7,500 小时（[[EgoDex]]、Agibot、RoboMind、RoboCoin、[[Open X-Embodiment]]、内部数据） | 多来源、动作标注/无动作标注混合 | 预训练 |
| [[RoboTwin 2.0]] | 50 个双臂操作任务 | Clean / Random（域随机化）双设置 | 训练 + 测试 |
| 真实双臂数据 | 每任务 100 条人类遥操作示范，30 FPS | [[Dual-Arm Robot\|TienKung]] 平台 | 真实世界测试 |

### 实现细节

- **Backbone**: Video Expert 从 WoW-1.3B 初始化（30 blocks）；Action-Value Expert 30 blocks，隐藏维度 768
- **模型规模**: 1.9B 参数（不含 [[T5|umT5]] 与 [[SigLIP2]] 编码器）
- **训练阶段**: 阶段一视觉专家预训练；阶段二完整模型联合微调 15k steps
- **优化器/学习率**: 学习率 $1\times10^{-4}$，前 20% 恒定后线性衰减
- **Batch Size**: 512
- **动作块大小**: 32
- **推理**: 单张 A800 GPU 上 17.9 actions/秒，默认去噪步数 5，回滚阈值 $\delta=0.20$

### 可视化结果

Figure 4 的真实任务执行序列显示 MV-WAM 在抓取背包、投放/拾取/折叠布料等任务中保持稳定的执行进度；Figure 7 的 t-SNE 显示分布外条件下视觉与动作表示呈现不对称漂移，进一步佐证了模态流形几何差异是泛化差距的根源。

---

## 批判性思考

### 优点
1. **提供了泛化差距的几何归因而非仅工程修补**: 用流形曲率定量刻画视觉/动作异质性（$\kappa_v$ vs $\kappa_a$），把"联合优化为何损害动作"这一现象上升为可检验的几何假设，并通过消融（统一目标 vs 分模态目标）直接验证
2. **消融实验极具说服力**: 统一 noise-prediction（15.0%/6.0%）与统一 flow-prediction（6.0%）相对完整方法（84.0%/55.7%）的断崖式下降，清晰证明了预测目标错配是泛化差距的主因之一，而非仅仅是模型容量或数据量问题
3. **价值回滚机制无需额外人工标注**: Progress-Value token 通过蒙特卡洛回报自监督学习，在线执行时自动检测偏差并回滚，是对纯开环动作生成的有效补充

### 局限性
1. **模型规模较小**: 论文作者自述尚未在更大模型规模上验证，可扩展性是开放问题
2. **价值估计在稀疏奖励下可能不稳定**: 蒙特卡洛价值估计可能在稀疏奖励、长时域任务中带噪，限制回滚机制在更复杂任务上的可靠性
3. **真实世界任务集仍有限**: 仅在 4 类真实双臂任务上验证（最难的 Fold Cloth 仅 20%），更复杂的精细操作泛化能力尚待验证

### 潜在改进方向
1. **扩展到更大规模骨干**: 验证流形感知目标在更大参数量、更多模态（如触觉）下是否依然有效
2. **改进价值估计的鲁棒性**: 引入分布式价值估计或时序差分方法替代纯蒙特卡洛回报，缓解稀疏奖励下的高方差问题
3. **更细粒度的流形几何分析**: 目前曲率估计是全局标量，未来可探索按任务阶段/技能动态变化的局部曲率建模

### 可复现性评估
- [ ] 代码开源（暂未公开）
- [ ] 预训练模型（暂未公开）
- [x] 训练细节完整（学习率、batch size、steps、硬件均已说明）
- [x] 数据集可获取（[[RoboTwin 2.0]]、[[EgoDex]]、[[Open X-Embodiment]] 等公开数据集，部分内部数据未公开）

---

## 关联笔记

### 基于
- [[WAM-Survey]]: World Action Model 的系统性综述，MV-WAM 的问题定义（cascaded/unified/MoT 架构均存在泛化差距）建立在该综述的分类框架上
- [[Mixture-of-Transformers]]: MV-WAM 的骨干架构范式

### 对比
- [[Fast-WAM]]: 论文中对比的 WAM baseline 之一，采用统一 MoT 架构但未做流形感知区分
- [[GeoSem-WAM]]: 同样改进 WAM 训练目标（几何语义辅助监督），但路径不同——GeoSem-WAM 加监督信号，MV-WAM 改预测参数化

### 方法相关
- [[World Action Model]]: 所属方法范式
- [[Flow Matching]]: 视觉分支训练目标的数学基础
- [[Mixture-of-Transformers]]: 双专家共享注意力架构
- [[RoPE]]: 跨模态时间对齐使用的位置编码缩放策略
- [[Action Chunking]]: 动作输出格式
- [[Progress-Value Regulation]]: 在线偏差检测与回滚机制

### 硬件/数据相关
- [[Dual-Arm Robot]]: TienKung 真实机器人实验平台
- [[RoboTwin 2.0]]: 主要仿真基准
- [[EgoDex]] / [[AgiBot World]] / [[RoboMIND]] / [[RoboCOIN]] / [[Open X-Embodiment]]: 预训练数据来源

---

## 速查卡片

> [!summary] MV-WAM: Manifold-Aware World Action Model with Value Augmentation
> - **核心**: 视觉与动作模态分布在曲率不同的流形上，统一预测目标会损害动作鲁棒性
> - **方法**: 流形感知双预测目标（视觉用速度场，动作用 $x_0$-prediction）+ 跨模态因果掩码 + Progress-Value 价值回滚
> - **结果**: RoboTwin 2.0 Random 55.7%（vs HALO 26.4%，+29.3pp），真实双臂任务均值 77.5%（vs $\pi_0$ 42.5%）
> - **代码**: 暂未公开

---

*笔记创建时间: 2026-06-24*
