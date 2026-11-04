---
title: "CausalWAM: Learning the Causes of Actions in World Action Models"
method_name: "CausalWAM"
authors: [Juntao Cheng, Jinzhou Tang, Fan Feng, Henrik I. Christensen, Yujia Zheng]
year: 2026
venue: arXiv
tags: [world-action-model, causal-identification, disentanglement, robot-manipulation, sparse-routing, identifiability-theory, rectified-flow]
zotero_collection: Robotics/World Model
image_source: online
arxiv_html: https://arxiv.org/html/2610.42096v1
created: 2026-11-03
---

# 论文笔记：CausalWAM: Learning the Causes of Actions in World Action Models

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | UIUC（Cheng, Zheng）、UCSD（Tang, Feng, Christensen） |
| 日期 | October 2026 |
| 项目主页 | 无 |
| 对比基线 | [[Cosmos3]]（Cosmos 3 Nano） |
| 链接 | [arXiv](https://arxiv.org/abs/2610.42096) / Code 未开源 |

---

## 一句话总结

> 在无需干预信号的条件下，通过稀疏路由约束理论证明并实现了从 WAM 潜变量中可辨识地分离驱动动作的因果状态，RoboCasa365 平均成功率从 24% 提升至 43%。

---

## 核心贡献

1. **可辨识性理论（Theorem 1 & 2）**: 在[[支撑稀疏性|支撑稀疏（support sparsity）]]约束下，严格证明估计的动作支撑集与真实集一致（置换等价），并且压缩后的动作相关状态是真实动作相关块的可逆函数——无需干预、无需时序依赖、无需潜变量动力学假设。
2. **CWAM 估计方法**: 在预训练 [[WAM]] 动作头之前加入稀疏路由路径，仅训练路由器（世界建模目标 $\mathcal{L}_{WAM}$ 冻结），通过 top-k 注意力门控自动定位因果相关的视觉 token。
3. **跨域实验验证**: 合成数据（理论验证）+ RoboCasa365（17 任务）+ LIBERO（40 任务）+ 真实 Franka 机器人（50 次试验），全面支持理论结论并展示了实际性能提升。

---

## 问题背景

### 要解决的问题

[[WAM|World Action Model]] 学习从观测 $o_t$ 预测场景演化并生成动作 $a_t$，但场景中只有**部分**潜变量（action-relevant 块）真正驱动动作。标准 WAM 训练不保证动作头只读取因果相关的潜变量，导致动作预测与噪声/无关变量（nuisance）发生**纠缠（entanglement）**。

### 现有方法的局限

- 标准 [[Causal VAE]] 等方法需要干预信号（interventional data）才能实现[[因果可辨识性|因果辨识]]
- 密集读出（dense readout）虽然预测精度高，但不区分因果和非因果潜变量（Δnuis 高达 0.646）
- 随机子集选取（random matched coords）则因子集选取不稳定导致 Relevant R² 仅 0.549

### 本文的动机

仅用**演示数据**（无干预标注），通过[[稀疏路由|稀疏路由约束]]在动作生成路径上施加稀疏性，即可在信息理论上保证所选 token 集等价于真实的 action-relevant 潜变量块。

---

## 方法详解

### 模型架构

CausalWAM（**CWAM**）以 [[Cosmos3|Cosmos 3 Nano]] 为基础模型，采用**两阶段训练 + 稀疏路由头**架构：

- **输入**: 语言指令 $l$ + 观测 $o_t$
- **Backbone**: [[Cosmos3|Cosmos 3 Nano]]（[[Rectified Flow]] 视频生成 + 动作解码）
- **核心模块**: [[稀疏路由|稀疏路由器（Sparse Router）]]，将干净上下文 token $\mathcal{J}_t$ 中具有因果相关性的子集 $C_t$ 路由到动作头
- **输出**: 动作块 $\hat{a}_{t:t+H-1}$（[[Action Chunking]]）
- **路由超参**: $d_r = 512$，8 heads，top-k 支撑 $k = 32$，残差门控初始化 0.01

### 核心模块

#### 模块 1：稀疏路由器（Sparse Router）

**设计动机**: 利用[[支撑稀疏性]]约束，强制动作生成只依赖少量视觉 token，从而实现[[因果可辨识性|因果辨识]]。

**具体实现**:
- 在动作头前增加独立的多头注意力层，查询来自动作 token，键值来自干净上下文 token $\mathcal{J}_t$
- 使用[[稀疏路由|Top-k 稀疏化]]：只保留每行注意力权重前 $k$ 大的位置
- 通过[[支撑稀疏损失]]（$\mathcal{L}_{sparse}$）惩罚注意力质量落在 top-k 以外的部分
- $\lambda = 0.01$，从步骤 1000 开始线性预热 4000 步

#### 模块 2：两阶段训练（Stage 1 + Stage 2）

**Stage 1**（10,000 步）: 联合微调世界建模头和动作头，建立基础的动作预测能力；无稀疏损失。

**Stage 2**（50,000 步）: 冻结世界建模目标，只训练稀疏路由器，学习率从 Stage 1 末尾以 0.3871 倍率重置。$\mathcal{L}_{sparse}$ 在此阶段生效。

**实现细节**:
- 优化器：FusedAdamW（$\beta_1=0.9, \beta_2=0.99, \varepsilon=10^{-8}$）
- 基础学习率：$5 \times 10^{-5}$；动作适配器学习率倍数：5.0
- 权重衰减：0.05；梯度裁剪：1.0；精度：bf16；FSDP over 8 GPUs
- EMA 速率：0.1；[[Rectified Flow]] 损失尺度：10.0；动作损失权重：10.0

---

## 关键公式

### 公式 1：[[WAM|WAM 观测与动作生成模型]]

$$
o_t = f(s_t)
$$

$$
a_t = \pi(s_t, \eta_t)
$$

**含义**: 观测 $o_t$ 由真实状态 $s_t$ 通过确定性映射 $f$ 生成；动作 $a_t$ 由状态 $s_t$ 和随机噪声 $\eta_t$ 生成，其中只有 $s_t$ 的一个子块（action-relevant block）真正影响 $a_t$。

**符号说明**:
- $s_t$: 真实潜在状态（包含多个块，只有 $s_{t,\mathcal{I}_A}$ 驱动动作）
- $f$: 可逆确定性映射（观测生成函数）
- $\eta_t$: 随机噪声（动作随机性来源）
- $\pi$: 真实策略函数

### 公式 2：[[WAM|WAM 训练目标一致性]]

$$
\mathcal{L}_{f,\pi} = \mathcal{L}_{\hat{f},\hat{\pi}}
$$

**含义**: 训练后估计模型的损失等于真实模型的损失（最优拟合条件），是后续可辨识性定理的前提。

### 公式 3：[[因果可辨识性|无稀疏约束时的 Jacobian 不等式]]（Proposition 1）

$$
\|\mathcal{I}(J_{\hat{\pi}})\|_0 \leq \|\mathcal{I}(J_{\pi})\|_0
$$

**含义**: 不加稀疏约束时，估计策略的 Jacobian 支撑集**不超过**真实策略的支撑集大小，即估计支撑可能小于真实支撑（信息丢失），无法保证辨识。

**符号说明**:
- $\mathcal{I}(\cdot)$: 矩阵中非零元素的索引集（支撑算子）
- $J_{\pi}$: 策略 $\pi$ 关于潜变量的 Jacobian 矩阵
- $\|\cdot\|_0$: 非零元素计数（$\ell_0$ 范数）

### 公式 4：[[CWAM|CWAM 动作预测与因果状态提取]]

$$
\hat{a}_{t:t+H-1} = \hat{\pi}(o_t, \epsilon_t; C_t), \quad C_t = \mathcal{S}(\hat{f}^{-1}(o_t))
$$

**含义**: CWAM 用稀疏路由选出的紧凑状态 $C_t$（而非全部潜变量）驱动动作预测，$\mathcal{S}$ 为稀疏路由操作。

**符号说明**:
- $\epsilon_t$: 动作扩散噪声
- $C_t$: 稀疏路由选出的因果相关 token 集
- $\mathcal{S}$: 稀疏路由函数
- $H$: 动作预测时域长度（[[Action Chunking]]）

### 公式 5：[[支撑稀疏性|稀疏训练目标]]

$$
\min_{\hat{f},\hat{\pi}} \mathcal{D}(\mathcal{L}_{f,\pi}, \mathcal{L}_{\hat{f},\hat{\pi}}) + \mathcal{R}_{\mathrm{sparse}}(\hat{\pi})
$$

**含义**: 在保证预测质量（$\mathcal{D}$ 最小化）的同时，通过正则项 $\mathcal{R}_{sparse}$ 施加稀疏约束，驱动动作头只依赖少量 token。

### 公式 6：[[稀疏路由|稀疏路由器注意力计算]]

$$
w_{ij} = \mathrm{softmax}_{j \in \mathcal{J}_t}\!\left(\frac{q_i^\top k_j}{\sqrt{d_r}} + b_{ij}(\text{relations})\right), \quad \tilde{h}_i = \sum_{j \in \mathcal{J}_t} w_{ij} v_j
$$

**含义**: 动作 token $i$ 对上下文 token $j$ 的注意力权重，包含可学习的关系偏置 $b_{ij}$；加权求和得到路由后的隐状态 $\tilde{h}_i$。

**符号说明**:
- $q_i$: 动作 token $i$ 的查询向量
- $k_j, v_j$: 上下文 token $j$ 的键、值向量
- $d_r = 512$: 路由维度
- $b_{ij}$: 关系感知偏置（捕获 token 间空间/语义关系）
- $\mathcal{J}_t$: 干净上下文 token 全集

### 公式 7：[[支撑稀疏损失|稀疏性损失]]

$$
\mathcal{L}_{\mathrm{sparse}} = \lambda \, \mathbb{E}_i\!\left[1 - \sum_{j \in T_k(i)} w_{ij}\right]
$$

**含义**: 惩罚注意力质量（probability mass）落在 top-k 以外的部分，强制每个动作 token 只关注最相关的 $k$ 个上下文 token。

**符号说明**:
- $\lambda = 0.01$: 稀疏权重（线性预热）
- $T_k(i)$: 动作 token $i$ 的注意力权重前 $k$ 大的上下文 token 索引集
- $k = 32$: top-k 支撑大小

### 公式 8：[[CWAM|CWAM 总训练损失]]

$$
\mathcal{L}_{\mathrm{total}} = \mathcal{L}_{\mathrm{WAM}} + \mathcal{L}_{\mathrm{sparse}}
$$

**含义**: 总损失由冻结的世界建模损失（Stage 2 中不更新 WAM 参数）和稀疏路由损失组成，确保路由器学习因果性的同时不破坏世界建模质量。

### 公式 9：[[稀疏路由|因果状态集合]]

$$
C_t = \{ j \in \mathcal{J}_t \mid \exists i : j \in T_k(i) \}
$$

**含义**: 因果状态 $C_t$ 是所有被至少一个动作 token 路由到的上下文 token 的并集，是 CWAM 从 WAM 潜变量中提取的"动作原因"紧凑表示。

---

## 关键图表

### Figure 1: 方法概览

*（arXiv HTML 未提取到 Figure 1 图片链接，请访问 [arXiv PDF](https://arxiv.org/pdf/2610.42096) 查看原图）*

**说明**:
- **Panel (a)**: 展示世界动作过程中的因果结构——完整状态 $s_t$ 分为 action-relevant 块 $s_{t,\mathcal{I}_A}$ 和 nuisance 块，动作 $a_t$ 只由前者决定；
- **Panel (b)**: CWAM 架构——在 [[Cosmos3|Cosmos 3 Nano]] 的联合 [[Rectified Flow]] 训练框架中，动作头通过[[稀疏路由]]器从干净上下文 token 中选出因果子集 $C_t$，世界建模目标 $\mathcal{L}_{WAM}$ 保持冻结。

### Figure 2: RoboCasa365 + LIBERO + Real Robot 结果

![Figure 2](https://arxiv.org/html/2610.42096v1/fig_robocasa_libero_real_radar_v110.png)

**说明**: CWAM 在三个评估维度全面超越 [[Cosmos3|Cosmos 3 Nano]] 基线：
- **Panel (a)** [[RoboCasa365]]: CWAM 在 17 个任务中的 13 个取得更高成功率（宏平均 0.43 vs 0.24）
- **Panel (b)** [[LIBERO]]: CWAM 在全部四个 LIBERO 套件（Spatial/Object/Goal/Long）上超越基线，LIBERO-Goal 提升最大
- **Panel (c)** 真实 Franka 叠方块（50 次）：CWAM 成功率超过 50%，基线不足 50%

### Figure 3: 路由质量分析（Routing Mass 可视化）

![Figure 3](https://arxiv.org/html/2610.42096v1/fig_router_mass_combined_stack_v7.png)

**说明**: [[稀疏路由]] 的注意力质量（暖色 = 高路由权重）自动集中在被操纵的物体及其目标位置：
- **上排**: RoboCasa365 rollout，路由器在执行取放任务时追踪目标物体和目标容器
- **下排**: 真实机器人叠方块，路由器追踪要抓取的方块和放置目标位置
- 这一可视化结果与理论预测一致：稀疏路由自动找到驱动动作的因果相关视觉区域

### Table 1: 合成实验结果（$n=20$ 潜变量，$|\mathcal{I}_A|=4$，5 个随机种子均值 ± 标准差）

| 方法 | \|C_t\| | Relevant R² ↑ | Δnuis ↓ | Selected/full ↓ | Complement/full ↑ |
|------|---------|--------------|---------|----------------|-------------------|
| Dense（20D）| 20 | 0.9961 ± 0.0015 | 0.646 ± 0.115 | n/a | n/a |
| Random matched（4D）| 4 | 0.5493 ± 0.2964 | 0.392 ± 0.144 | 48.067 ± 31.855 | 6.36 ± 6.42 |
| CWAM w/o sparsity | 4 | 0.9876 ± 0.0164 | -0.008 ± 0.006 | 3.456 ± 3.180 | 50.58 ± 28.81 |
| MLP VAE | 4 | 0.9440 ± 0.0448 | 0.316 ± 0.186 | 3.259 ± 2.211 | 51.13 ± 28.54 |
| **CWAM（sparsity）** | 4 | **0.9957 ± 0.0014** | **0.039 ± 0.093** | **1.704 ± 1.398** | **54.65 ± 25.78** |

**关键发现**:
- Dense 读出虽然 Relevant R² 最高，但 Δnuis = 0.646 说明严重纠缠；
- CWAM w/o sparsity 已能消除纠缠（Δnuis ≈ −0.008），但 Selected/full 仍较高；
- CWAM with sparsity 在消除纠缠（Δnuis = 0.039）的同时保持高 Relevant R²（0.9957），验证了 Theorem 1。

### Table 2: RoboCasa365 最优四任务对比（100 次 rollout）

| 任务 | CWAM | C-only | Random matched |
|------|------|--------|----------------|
| PickPlaceSinkToCtr | **74%** | 63% | 28% |
| PickPlaceToasterToCtr | **72%** | 61% | 21% |
| OpenDrawer | **70%** | 59% | 33% |
| OpenStandMixerHead | **69%** | 58% | 16% |
| **平均** | **71.3%** | 60.3% | 24.5% |

**关键发现**: CWAM 比 C-only（仅使用 $C_t$ 无稀疏约束）高约 11%，说明**稀疏约束本身**（而非子集选取）是性能提升的关键。

### Table 3: 完整 RoboCasa365 per-task 成功率（100 次/任务）

| 技能族 | 任务 | Baseline | CWAM |
|--------|------|----------|------|
| Pick & place | PickPlaceCtrToCab | .31 | .43 |
| Pick & place | PickPlaceCtrToStove | .22 | .45 |
| Pick & place | PickPlaceDrawerToCtr | .15 | .07 |
| Pick & place | PickPlaceSinkToCtr | .24 | **.74** |
| Pick & place | PickPlaceToasterToCtr | .28 | **.72** |
| Opening/closing doors | CloseFridge | .14 | **.61** |
| Opening/closing doors | CloseToasterOvenDoor | .51 | .37 |
| Opening/closing doors | OpenCabinet | .31 | .24 |
| Closing/opening lids | CloseBlenderLid | .02 | .03 |
| Closing/opening lids | OpenStandMixerHead | .30 | **.69** |
| Pressing buttons | TurnOnMicrowave | .27 | .26 |
| Pressing buttons | TurnOnElectricKettle | .35 | **.45** |
| Opening/closing drawers | OpenDrawer | .25 | **.70** |
| Sliding racks | SlideDishwasherRack | .32 | **.61** |
| Twisting knobs | TurnOffStove | .05 | **.18** |
| Turning levers | TurnOnSinkFaucet | .25 | **.37** |
| Insertion | CoffeeSetupMug | .10 | **.37** |
| **Macro avg（17 tasks）** | | **.24** | **.43** |

**关键发现**: CWAM 在 17 个任务中的 13 个取得提升；取放（Sink/Toaster）、开关（Fridge/Drawer）和插入（Mug）类任务增益最大；推按钮和推盖类任务几乎持平或略降。

### Table 4: 隐藏环境（Hidden Environments）泛化测试（5 个种子）

| 环境 E | Dense Relevant R² | Dense Δnuis | CWAM Relevant R² | CWAM Δnuis |
|--------|-------------------|-------------|------------------|------------|
| 1 | 0.9961 ± 0.0015 | 0.646 ± 0.115 | **0.9957 ± 0.0014** | **0.039 ± 0.093** |
| 2 | 0.9961 ± 0.0016 | 0.640 ± 0.110 | **0.9961 ± 0.0016** | **−0.004 ± 0.003** |
| 3 | 0.9961 ± 0.0017 | 0.677 ± 0.109 | **0.9961 ± 0.0017** | **−0.004 ± 0.003** |
| 5 | 0.9961 ± 0.0017 | 0.683 ± 0.114 | **0.9961 ± 0.0017** | **−0.003 ± 0.002** |

**关键发现**: CWAM 在从未见过的环境（E=2,3,5）中 Δnuis ≈ 0，因果辨识完全成功，验证了泛化能力。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| 合成数据 | $n=20$ 潜变量，$|\mathcal{I}_A|=4$ | 可控因果结构 | 理论验证（Table 1） |
| [[RoboCasa365]] | 17 tasks × 100 rollouts | 真实厨房操纵，多技能族 | 闭环成功率 |
| [[LIBERO]] | 40 tasks（4 suites）| 标准桌面操纵 benchmark | 闭环成功率 |
| 真实 Franka | 叠方块，50 次/policy | 真实机器人，sim2real | 闭环成功率 |

### 实现细节

- **Backbone**: [[Cosmos3|Cosmos 3 Nano]]（NVIDIA）
- **路由器**: $d_r = 512$，8 头，$k = 32$，残差门控初始化 0.01
- **稀疏正则**: $\lambda = 0.01$，步骤 1000 启动，线性预热 4000 步
- **训练**: Stage 1（10K 步）+ Stage 2（50K 步）
- **硬件**: 8 GPUs（FSDP），bf16，梯度累积 2
- **探针**: Ridge 回归（$\alpha=1$），250 校准回合拟合

### 可视化结果

路由器的注意力质量（Figure 3）自动聚焦到被操纵物体和目标位置，表现出清晰的**空间因果归因**能力，无需任何显式空间监督。

---

## 批判性思考

### 优点
1. **理论严格**: 在弱假设下（无干预、无时序、无动力学先验）给出了可辨识性的充分条件，理论贡献明确
2. **即插即用**: 只需在预训练 WAM 动作头前加稀疏路由器，不改变世界建模目标，计算开销极小
3. **可解释性**: Figure 3 路由可视化直接展示了"哪些视觉区域驱动动作"，提供了良好的可解释性

### 局限性
1. **非单调任务下降**: PickPlaceDrawerToCtr（.15→.07）、CloseToasterOvenDoor（.51→.37）等任务性能退步，说明稀疏约束可能过度裁剪某些需要全局上下文的任务
2. **依赖预训练 WAM 质量**: 方法完全依赖 WAM 的潜变量质量，若底层 WAM 表征较弱，因果辨识效果未知
3. **$k$ 的选取敏感性**: $k=32$ 是固定超参，不同任务可能需要不同稀疏度，缺乏自适应机制
4. **代码未开源**: 可复现性受限

### 潜在改进方向
1. 自适应稀疏度：根据任务复杂度动态调整 $k$
2. 层级因果辨识：对不同时间尺度（粗粒度目标 vs. 精细动作）分别施加不同稀疏约束
3. 探索是否可在 RLHF 微调阶段结合 CWAM，提升因果稀疏路由的下游鲁棒性

### 可复现性评估
- [ ] 代码开源（未开源）
- [ ] 预训练模型（未发布）
- [x] 训练细节完整（超参数详细记录于附录）
- [x] 数据集可获取（RoboCasa365 / LIBERO 公开）

---

## 关联笔记

### 基于
- [[Cosmos3]]: 基础模型（Cosmos 3 Nano），提供 WAM 预训练骨干
- [[Rectified Flow]]: 联合世界建模与动作生成的扩散框架
- [[WAM]]: 方法所在的系统范式

### 对比
- [[ABot-M0.5]]: RoboCasa365 SOTA 46.6%，高于 CWAM 43%，但方法侧重动作表示而非因果辨识
- [[Causal VAE]]: 同属因果解耦方向，但需要干预数据

### 方法相关
- [[稀疏路由]]: 核心机制，通过 top-k 注意力实现[[因果可辨识性]]
- [[因果可辨识性]]: 理论贡献的核心概念
- [[支撑稀疏性]]: Theorem 1 的关键假设
- [[Action Chunking]]: 动作预测形式

### 硬件/数据相关
- [[RoboCasa365]]: 主要评估基准
- [[LIBERO]]: 次要评估基准

---

## 速查卡片

> [!summary] CausalWAM
> - **核心**: 稀疏路由约束 + 可辨识性理论，从 WAM 潜变量中分离驱动动作的因果状态
> - **方法**: 在预训练 Cosmos 3 Nano 动作头前加 top-k 稀疏路由器，$\mathcal{L}_{total} = \mathcal{L}_{WAM} + \mathcal{L}_{sparse}$
> - **结果**: RoboCasa365 宏平均 24% → 43%，LIBERO 全套件超越基线，真实 Franka >50% 成功率
> - **代码**: 未开源（arXiv:2610.42096）

---

*笔记创建时间: 2026-11-03*
