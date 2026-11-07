---
title: "VTAM: Efficient Visuo-Tactile World-Action Models for Contact-Rich Manipulation without Tactile Pretraining"
method_name: "VTAM"
authors: [Haoran Yuan, Weigang Yi, Zhenyu Zhang, Wendi Chen, Yuchen Mo, Jiashi Yin, Xinzhuo Li, Yue Luo, Jialin Li, Xiangyu Zeng, Chuan Wen, Cewu Lu, Katherine Driggs-Campbell, Ismini Lourentzou]
year: 2026
venue: arXiv
tags: [visuo-tactile, world-action-model, contact-rich-manipulation, flow-matching, tactile-sensing, robot-policy]
zotero_collection: Robotics/World Model
image_source: online
arxiv_html: https://arxiv.org/html/2603.23481
created: 2026-11-07
---

# 论文笔记：VTAM: Efficient Visuo-Tactile World-Action Models for Contact-Rich Manipulation without Tactile Pretraining

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | UIUC (PLAN Lab), Shanghai Jiao Tong University, Virginia Tech |
| 日期 | March 2026 (v2: November 2026) |
| 项目主页 | [plan-lab.github.io/projects/vtam](https://plan-lab.github.io/projects/vtam/) |
| 对比基线 | [[Fast-WAM]], [[Tactile-WAM]], [[π0.5]], [[RDP]] |
| 链接 | [arXiv](https://arxiv.org/abs/2603.23481) |

---

## 一句话总结

> VTAM 将预训练视频主干网络扩展为视觉-触觉联合世界-动作模型，无需触觉预训练即可在接触丰富操作任务中达到与大规模触觉预训练模型相当的性能，同时推理速度提升 20-70 倍。

---

## 核心贡献

1. **无触觉预训练的视觉-触觉 WAM**: 通过两阶段训练将 [[Genie Envisioner]] 预训练视频主干扩展为 [[Visuo-Tactile World Models|视觉-触觉世界-动作模型]]，仅使用任务示范数据，不依赖大规模触觉预训练
2. **Flow-Force Proxy（流力代理）**: 从触觉光学流中导出轻量 3D 力代理信号，无需力传感器即可引导模型主动利用触觉接触信息
3. **高效推理**: 推理时跳过视频生成去噪，仅一次主干前向传播 + 少量 Euler 步，编译后延迟 145ms，比全视频去噪模型快 20-70 倍；且触觉编码器可零样本迁移到异构触觉传感器

---

## 问题背景

### 要解决的问题

接触丰富的机器人操作（如插孔、剥皮、擦拭）需要精细的触觉感知——机器人可能执行了正确的运动轨迹，却因接触不充分而失败。如何让 [[世界动作模型（WAM）]] 有效融合触觉信息，在不依赖大规模触觉数据集的前提下实现精准的接触感知操作，是核心挑战。

### 现有方法的局限

- **触觉仅作为附加输入**：现有 Tactile-VLA / 触觉-WAM 方法将触觉视为额外的输入通道，但视觉特征主导使触觉信息被忽视
- **依赖大规模触觉预训练**：N0-TWAM、FTP-1 等模型需要 30K+ 小时触觉预训练数据，获取代价高昂
- **推理速度慢**：[[Fast-WAM]]、LingBot-VA 等全视频去噪 WAM 在推理时需迭代生成视频帧，每步 3-10 秒

### 本文的动机

预训练视频模型已学到丰富的时空特征，触觉图像与 RGB 图像同为像素级时序信号，因此可以共享同一冻结视频 VAE 和 Transformer 权重。通过在两阶段训练中加入触觉流的世界建模目标和 [[Flow-Force Proxy]] 监督，可在不预训练的情况下让模型主动利用触觉接触信息。推理时省略视频去噪步骤可大幅提速。

---

## 方法详解

### 模型架构

VTAM 采用 **两阶段训练 + 解耦推理** 架构：

- **输入**: 2 路 RGB 图像流（$v^1, v^2$）+ 1 路触觉图像流（$v^3$）+ 本体感知状态 $s_k$
- **Backbone**: [[Genie Envisioner]] GE-base checkpoint（基于 LTX-Video，28 个 Transformer 块），共享冻结视频 [[VAE]] 和 Transformer 权重
- **核心模块**: [[Intra-View Attention|视图内自注意力]] + [[Cross-View Attention|跨视图交叉注意力]] 编码多模态时序特征；[[Action Expert]] 通过交叉注意力解码动作
- **输出**: 动作块 $a$（7D）、[[Flow-Force Proxy]]（3D）、本体感知状态 $s$（16D）
- **总参数**: 7.29B

### 核心模块

#### Stage I：视觉-触觉世界建模微调

**设计动机**: 让 Backbone 学会预测 RGB 和触觉的未来状态，建立对接触动态的内部表示。

**具体实现**:
- 使用 [[Flow Matching]] 目标在 2 路 RGB + 1 路触觉的潜变量空间中进行联合去噪
- 插值轨迹：$x_\tau = (1-\tau)x_0 + \tau\varepsilon$
- 以上一时刻的观测 $c_\text{obs}$ 为条件，预测未来帧的潜变量
- 训练 50,000 步，学习率 $3\times10^{-4}$，Backbone 参数可训练

#### Stage II：动作专家联合微调

**设计动机**: 在 Stage I 的视觉-触觉表示基础上，训练 [[Action Expert]] 预测动作，同时端对端微调 Backbone。

**具体实现**:
- 随机初始化 28 层 [[Action Expert]]，通过 [[Cross-View Attention|交叉注意力]] 访问 Backbone 特征
- [[Action Expert]] 联合去噪动作 $a$（7D）、[[Flow-Force Proxy]] $f$（3D）和本体感知状态 $s$（16D）
- 联合目标向量：$y_0 = [a;\, f;\, s]$，条件 token：$c_k = [\mathbf{0}_{10};\, s_k]$
- 训练 20,000 步，学习率 $5\times10^{-5}$，Backbone 端对端微调

#### Flow-Force Proxy（流力代理）

**设计动机**: 力传感器难以安装在并联夹爪上，但触觉图像（如 [[GelSight]]）通过光学流可以间接反映切向力和法向力。

**具体实现**:
- 计算当前触觉帧与无接触参考帧之间的 [[Optical Flow]] 场 $u_k(\mathbf{p})$
- 对像素位置 $\mathbf{p}$ 求空间平均，得到 3D 代理向量
- 切向分量：$f_{x,k} = \mathbb{E}_{\mathbf{p}}[u_{x,k}(\mathbf{p})]$，$f_{y,k} = \mathbb{E}_{\mathbf{p}}[u_{y,k}(\mathbf{p})]$
- 法向分量（散度）：$f_{z,k} = \mathbb{E}_{\mathbf{p}}[\nabla \cdot u_k(\mathbf{p})]$
- 无需力传感器或学习式力估计器

#### 推理加速

推理时跳过 Stage I 的视频去噪过程，仅执行一次 Backbone 前向传播，再对 [[Action Expert]] 执行少量 Euler 步（[[Flow Matching]] 求解器），大幅降低延迟。

---

## 关键公式

### 公式 1: [[Flow Matching|流匹配插值轨迹]]

$$
x_\tau = (1-\tau)x_0 + \tau\varepsilon, \quad v^* = \varepsilon - x_0
$$

**含义**: 在干净样本 $x_0$ 和噪声 $\varepsilon \sim \mathcal{N}(0,I)$ 之间线性插值，目标速度场为常数 $v^*$

**符号说明**:
- $\tau \sim \mathcal{U}(0,1)$: 插值时间步
- $x_0$: 干净的视觉/触觉潜变量
- $\varepsilon$: 高斯噪声
- $v^*$: 目标速度场（常数）

### 公式 2: [[Flow Matching|Stage I 世界建模损失]]

$$
\mathcal{L}_{\text{stage1}} = \mathbb{E}_{\tau, \varepsilon}\left[\|v_\theta(x_\tau, \tau \mid c_{\text{obs}}) - v^*\|^2\right]
$$

**含义**: 以上一时刻的观测 $c_\text{obs}$ 为条件，训练 Backbone 在 2 路 RGB + 1 路触觉的潜变量空间中预测未来帧

**符号说明**:
- $v_\theta$: 参数为 $\theta$ 的 Backbone 速度场预测
- $c_\text{obs}$: 上一时刻的观测条件（RGB + 触觉帧）
- $x_\tau$: 噪声插值后的潜变量

### 公式 3: [[Cross-View Attention|跨视图注意力]]

$$
\widetilde{Z}_b^v = \text{SelfAttention}(Z_{b-1}^v), \quad v \in \{1, 2, 3\}
$$

$$
Z_b = \text{CrossViewAttention}\!\left(\text{Concat}(\widetilde{Z}_b^1,\, \widetilde{Z}_b^2,\, \widetilde{Z}_b^3)\right)
$$

**含义**: 第 $b$ 个 Transformer 块先对每路视图做视图内自注意力，再拼接三路特征做跨视图交叉注意力，实现 RGB 与触觉的特征融合

**符号说明**:
- $Z_b^v$: 第 $b$ 块第 $v$ 路视图的 token 序列（$v=3$ 为触觉）
- $\widetilde{Z}_b^v$: 视图内自注意力后的特征

### 公式 4: [[Flow-Force Proxy|流力代理]]

$$
f_{x,k} = \mathbb{E}_{\mathbf{p}}[u_{x,k}(\mathbf{p})], \quad
f_{y,k} = \mathbb{E}_{\mathbf{p}}[u_{y,k}(\mathbf{p})], \quad
f_{z,k} = \mathbb{E}_{\mathbf{p}}[\nabla \cdot u_k(\mathbf{p})]
$$

$$
F_k^v = [f_{x,k},\; f_{y,k},\; f_{z,k}]^\top \in \mathbb{R}^3
$$

**含义**: 从触觉帧与无接触参考帧的 [[Optical Flow]] 场 $u_k$ 中导出切向力代理（$f_x, f_y$）和法向力代理（$f_z$，以散度表示），无需力传感器

**符号说明**:
- $u_k(\mathbf{p})$: 时刻 $k$ 像素 $\mathbf{p}$ 处的光学流向量
- $f_{x,k}, f_{y,k}$: 触觉图像切向位移均值（对应切向力）
- $f_{z,k}$: 光流散度均值（对应法向力/压力）
- $F_k^v \in \mathbb{R}^3$: 3D 流力代理向量

### 公式 5: [[Flow Matching|Stage II 联合训练损失]]

$$
\mathcal{L}_{\text{stage2}} = \mathcal{L}_{\text{action}} + \lambda_1 \mathcal{L}_{\text{state}} + \lambda_2 \mathcal{L}_{\text{force}}, \quad \lambda_1 = \lambda_2 = 1
$$

其中每个分量均为流匹配损失：

$$
\mathcal{L}_{(\cdot)} = \mathbb{E}_{\tau, \varepsilon}\!\left[\left\|v^{(\cdot)}_\phi(y_\tau, \tau \mid h, c_k) - v^{*(\cdot)}\right\|^2\right]
$$

**含义**: [[Action Expert]] 以 Backbone 特征 $h$ 和当前条件 $c_k$ 为输入，联合去噪动作 $a$（7D）、流力代理 $f$（3D）、本体感知状态 $s$（16D）

**符号说明**:
- $y_0 = [a;\, f;\, s]$: 联合目标向量（$7+3+16=26$ 维）
- $c_k = [\mathbf{0}_{10};\, s_k]$: 条件 token（本体感知状态）
- $h$: Backbone 中间层特征（供 [[Action Expert]] 交叉注意力使用）
- $\phi$: [[Action Expert]] 参数
- $\lambda_1 = \lambda_2 = 1$: 状态和力代理损失权重

---

## 关键图表

### Figure 1: VTAM 总览

![Figure 1 Overview](https://plan-lab.github.io/projects/vtam/static/images/overview_v2.png)

**说明**: 左图为 VTAM 两阶段训练框架概览（Stage I 视觉-触觉世界建模 + Stage II 动作专家联合微调）。右侧雷达图对比了性能、推理效率和数据效率三个维度，VTAM 在无触觉预训练的前提下综合表现最佳。

### Figure 1b: 性能-效率-数据效率雷达图

![Radar Chart](https://plan-lab.github.io/projects/vtam/static/images/radar_v2.png)

**说明**: VTAM（83.4% UniVTAC 均值，145ms 推理延迟）在三维综合权衡上优于所有基线，包括使用 30K+ 小时触觉预训练数据的 N0-TWAM（84.5%，5136ms）。

### Figure 1 (arXiv): 系统架构概览

![Figure 1 arXiv](https://arxiv.org/html/2603.23481v2/overview_diagram.png)

**说明**: VTAM 的 Stage I（Backbone 世界建模微调）和 Stage II（[[Action Expert]] 联合微调）的完整数据流。推理时跳过 Stage I 的视频去噪，仅执行一次 Backbone 前向传播。

### Figure 3: UniVTAC 仿真任务

![Figure 3 UniVTAC Tasks](https://arxiv.org/html/2603.23481v2/assets/figures/experiments/setup/univtac_tasks.jpg)

**说明**: 8 个 [[UniVTAC]] 仿真任务，每个展示关键接触时刻的头部相机帧（内嵌 [[GelSight]] 触觉图像）。任务从左到右：Lift Bottle、Pull-out Key、Lift Can、Put Bottle、Insert Hole、Insert Tube、Insert HDMI、Grasp Classify。

### Figure 8: 注意力调制对比（有无流力代理监督）

| 无流力代理监督 | 有流力代理监督 |
|---|---|
| ![Fig8a](https://arxiv.org/html/2603.23481v2/assets/figures/experiments/ablation/attention-modulation-unreg.png) | ![Fig8b](https://arxiv.org/html/2603.23481v2/assets/figures/experiments/ablation/attention-modulation-reg.png) |

**说明**: [[Action Expert]] 第 25 层对三路相机视图的交叉注意力权重随时间变化。**有** [[Flow-Force Proxy]] 监督时，接触阶段触觉注意力占比从 0.22 升至 0.26（+18%）；**无**监督时，接触/非接触阶段触觉注意力几乎无差异（0.20 vs 0.21），说明流力代理有效引导模型主动利用触觉。

### Table 1: UniVTAC 仿真基准（8 任务，100 seeds）

| 方法 | LiftB | Key | LiftC | PutB | InsH | InsT | HDMI | GraspC | **均值** |
|------|-------|-----|-------|------|------|------|------|--------|---------|
| ACT + UniVTAC | - | - | - | - | - | - | - | - | 48.0% |
| Fast-WAM | - | - | - | - | - | - | - | - | 48.0% |
| FTP-1 (预训练) | - | - | - | - | - | - | - | - | 64.8% |
| N0-VTLA (预训练) | - | - | - | - | - | - | - | - | 83.1% |
| **N0-TWAM (预训练)** | - | - | - | - | - | - | - | - | **84.5%** |
| **VTAM（本文）** | 91% | 92% | 75% | 100% | 67% | 97% | 45% | 100% | **83.4%** |

**说明**: VTAM 在无触觉预训练的情况下达到 83.4%，接近需要 30K+ 小时触觉预训练数据的 N0-TWAM（84.5%），远超其他无预训练基线（≤48%）。

### Table 2: 真机实验结果

**xArm6 + GelSight Mini**（各任务 20/20/40 次试验）：

| 方法 | Chip Pick | Peel | Wipe | 说明 |
|------|-----------|------|------|------|
| π0.5 (vision only) | 10% | - | - | 视觉策略 |
| Tactile-VLA | - | 20% | - | 触觉 VLA |
| RDP | - | - | 50% | 触觉 |
| **VTAM（本文）** | **90%** | **85%** | **95%** | 触觉 WAM |

**Flexiv Rizon 4 + Xense Photon**（20 次试验）：

| 方法 | Hanoi Disk Insertion |
|------|----------------------|
| Tactile-VLA | 0% |
| Genie Envisioner | 5% |
| [[π0.5]] | 25% |
| FTP-1 (预训练) | 55% |
| **VTAM（本文）** | **75%** |

**说明**: VTAM 在真机测试中大幅超越所有基线，包括 FTP-1（依赖大规模触觉预训练）。Wipe 任务稳定接触率 0.945 vs RDP 的 0.750，定量验证触觉反馈的有效性。

### Table 3: 推理效率对比（L40S, BF16）

| 模型 | 参数量 | Eager 延迟 (ms) | Compiled 延迟 (ms) |
|------|--------|----------------|-------------------|
| Fast-WAM | 12.41B | 548 | **127** |
| **VTAM（本文）** | **7.29B** | **588** | **145** |
| N0-TWAM | 13.59B | 3,393 | 5,136 |
| LingBot-VA | 11.47B | 9,380 | 10,404 |

**说明**: VTAM（145ms）比全视频去噪的 N0-TWAM（5136ms）快约 **35 倍**，比 LingBot-VA（10404ms）快约 **72 倍**，参数量也最少（7.29B）。

### Table 4: 消融实验（真机 + UniVTAC 均值）

| 配置 | Chip | Wipe | Hanoi | UniVTAC 均值 |
|------|------|------|-------|-------------|
| 冻结 VAE 触觉特征直接送 Expert | 0% | 0% | 0% | - |
| 无 Flow-Force Proxy | 10% | 10% | 55% | - |
| 无 Stage I | - | - | - | 65.8% |
| Stage II 冻结 Backbone（有 Stage I）| 18%\* | 10%\* | - | - |
| **完整 VTAM** | **90%** | **95%** | **75%** | **83.4%** |

\* Lift Bottle / Put Bottle 任务数值

**关键发现**: (1) 触觉特征必须经过 Backbone 端对端微调才能有效使用，仅冻结 VAE 特征导致 0% 成功率；(2) Stage I 世界建模是关键，去除后均值下降 17.6%；(3) Flow-Force Proxy 在接触敏感任务（Chip, Wipe）中影响尤为显著。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| [[UniVTAC]] | 8 任务，100 seeds | 仿真触觉操作 benchmark，含 GelSight 仿真 | 仿真训练与评估 |
| 自采真机数据 | 任务示范（少量） | xArm6/Flexiv + GelSight Mini/Xense Photon | 真机训练与评估 |

### 实现细节

- **Backbone**: [[Genie Envisioner]] GE-base（LTX-Video, 28 blocks），共享冻结视频 [[VAE]]
- **Stage I 优化器**: AdamW，学习率 $3\times10^{-4}$，步数 50,000
- **Stage II 优化器**: AdamW，学习率 $5\times10^{-5}$，步数 20,000
- **[[Action Expert]]**: 随机初始化 28 层 Transformer，与 Backbone 层数相同
- **动作维度**: 7D 末端执行器控制
- **[[Action Chunking]]**: 动作块预测，含本体感知状态（16D）联合去噪
- **推理**: 单次 Backbone 前向传播 + 少量 Euler 步（无视频去噪）
- **硬件**: L40S GPU，BF16 精度

### 触觉零样本迁移

- 冻结视频 [[VAE]] 编码器在两种传感器（GelSight Mini / Xense Photon）上的线性探测平衡精度达 90%/91%（有接触 vs 无接触帧）
- 超过直接用原始像素的基线（88%/89%）
- t-SNE 可视化显示触觉潜变量按任务聚类并跟踪 episode 进展

---

## 批判性思考

### 优点

1. **无需触觉预训练**: 通过 Stage I 世界建模微调和 [[Flow-Force Proxy]] 监督，从任务示范中自动学习触觉表示
2. **高效推理**: 推理时省略视频去噪，145ms 延迟接近实时，实用性强
3. **跨传感器零样本迁移**: 相同架构在 GelSight Mini 和 Xense Photon 上均有效，无需传感器特定适配

### 局限性

1. **触觉数据多样性有限**: 目前仅在两种视觉触觉传感器上验证，对更多传感器类型（如压力阵列、电子皮肤）的泛化未知
2. **流力代理精度有限**: 光学流导出的 3D 代理是力的粗略估计，对需要精确力控制的任务可能不足
3. **Insert Hole 任务的异常**: 去除 Stage I 后该任务成功率反升（95% vs 67%），表明世界建模目标在某些任务上可能引入噪声

### 潜在改进方向

1. 在更大规模、更多样的触觉数据（不同传感器、不同接触模态）上训练以研究涌现能力
2. 将流力代理替换为更精确的力估计方法（如学习式力估计器），进一步提升精确力控任务的性能
3. 探索与 [[π0.5]] 等大型视觉-语言-动作模型的结合，实现语言条件的接触丰富操作

### 可复现性评估

- [ ] 代码开源（项目页面未提供代码链接）
- [ ] 预训练模型（未提供）
- [x] 训练细节完整（详细记录了两阶段超参数、损失权重等）
- [x] 数据集可获取（[[UniVTAC]] 仿真 benchmark 公开）

---

## 关联笔记

### 基于

- [[Genie Envisioner]]: 使用其 GE-base checkpoint 作为 Backbone 初始化
- [[Flow Matching]]: Stage I 和 Stage II 均采用流匹配目标训练
- [[Optical Flow]]: 触觉图像 [[Flow-Force Proxy]] 由光学流导出

### 对比

- [[Fast-WAM]]: 无触觉预训练的 WAM 基线（UniVTAC 均值 48.0%，VTAM 83.4%）
- [[Tactile-WAM]]: 触觉 WAM 对比，依赖大规模触觉预训练
- [[π0.5]]: 视觉-语言-动作模型基线（真机 Chip 仅 10%）
- [[RDP]]: 触觉策略基线（Wipe 50%，VTAM 95%）

### 方法相关

- [[Visuo-Tactile World Models]]: VTAM 属于该类方法
- [[世界动作模型（WAM）]]: VTAM 的核心范式
- [[Action Expert]]: VTAM 的动作解码器模块
- [[Flow-Force Proxy]]: VTAM 提出的核心创新组件
- [[Action Chunking]]: VTAM 预测动作块的方式

### 硬件/数据相关

- [[GelSight]]: 真机实验使用的视觉触觉传感器
- [[UniVTAC]]: 主要仿真 benchmark

---

## 速查卡片

> [!summary] VTAM (2026)
> - **核心**: 无触觉预训练的视觉-触觉世界-动作模型
> - **方法**: 两阶段训练（视觉-触觉世界建模 → 动作专家联合微调）+ Flow-Force Proxy 监督
> - **结果**: UniVTAC 83.4%（≈有预训练的 N0-TWAM 84.5%），推理 145ms（快 35-70×），真机 Chip 90%/Peel 85%/Wipe 95%/Hanoi 75%
> - **代码**: 未开源（项目页: plan-lab.github.io/projects/vtam/）

---

*笔记创建时间: 2026-11-07*
