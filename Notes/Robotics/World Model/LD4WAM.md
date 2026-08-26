---
title: "LD4WAM: Learning Latent Dynamics from Human Videos for World Action Models"
method_name: "LD4WAM"
authors: [Zhenhao Shen, Jiaqi Liang, Jasper Lu, Feng Jiang, Yuran Wang, Chuanbo Wei, Jiayi Liu, Jianchun Yang, Qize Yu, Jiadi You, Ce Hao, Guanqi He, Chen Xie, Ruihai Wu]
year: 2026
venue: arXiv
tags: [world-action-model, latent-dynamics, human-video, robot-manipulation, mixture-of-transformers, vector-quantization, embodiment-agnostic, dexterous-hand]
zotero_collection: Robotics/World Model
image_source: online
arxiv_html: https://arxiv.org/html/2608.22403
created: 2026-08-26
---

# 论文笔记：LD4WAM: Learning Latent Dynamics from Human Videos for World Action Models

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | 未公开具体机构信息 |
| 日期 | August 2026 |
| 项目主页 | 暂无 |
| 对比基线 | [[Pi05\|π₀.₅]]、[[Fast-WAM]]、[[LingBot-Video\|Lingbot-VA]] |
| 链接 | [arXiv](https://arxiv.org/abs/2608.22403) / Code 暂未开放 |

---

## 一句话总结

> LD4WAM 从 5000+ 小时人类视频中学习 **运动对齐的隐空间动力学**，作为视频先验与机器人动作之间的桥梁，通过 [[Mixture of Transformers|MoT]] 架构的世界动态动作模型（WDAM）在仿真和真实双臂灵巧手系统上实现了 93.4%（仿真）/ 70.5%（实物）的操作成功率，超越 π₀.₅ 和 Fast-WAM。

---

## 核心贡献

1. **运动对齐隐空间动力学（Motion-Aligned Latent Dynamics）**: 提出一种与具身形态无关（embodiment-agnostic）的中间表征，同时接受语义重建与末端执行器运动监督，将人类视频中的运动先验迁移至机器人控制
2. **双模型协同框架（LDM + WDAM）**: [[Latent Dynamics Model|LDM]] 用冻结 [[DINOv3]] 特征 + [[Soft Vector Quantization|软向量量化]] 提取隐动力学；[[World Dynamics Action Model|WDAM]] 用三专家 [[Mixture of Transformers]] 架构，通过可学习查询蒸馏动力学并预测动作
3. **大规模多源数据统一训练**: 整合 5 个人类第一视角数据源和 3 个多具身机器人数据源，共 274.66M 帧（5000+ 小时），统一为 [[LeRobot]] 格式的末端执行器增量动作

---

## 问题背景

### 要解决的问题

[[WAM|World Action Model（WAM）]] 依赖机器人示范数据进行训练，而机器人数据的规模和多样性远不及人类视频数据。如何有效利用大量人类视频（无动作标注）来增强 WAM 的泛化能力，是核心挑战。

### 现有方法的局限

- **直接视频预训练**：视频特征包含大量外观信息，与机器人动作空间存在具身差距（embodiment gap），难以直接转化为可执行动作
- **像素级重建**：用像素重建作为监督信号会保留与运动无关的纹理细节，导致学到的表征对机器人控制无实质帮助
- **单阶段端到端**：缺乏专门针对运动信息提炼的模块，视频先验无法有效流入动作预测头

### 本文的动机

通过引入专门的**运动对齐**监督信号，迫使隐空间表征捕获帧间变化（而非静态外观），同时使用语义特征空间（而非像素空间）抑制与运动无关的外观变化，从而让隐动力学在跨具身场景中保持通用性。

---

## 方法详解

### 整体架构

LD4WAM 由两个核心模型组成：

- **[[Latent Dynamics Model]]（LDM）**：从视频帧对中提取运动对齐的隐动力学 token
- **[[World Dynamics Action Model]]（WDAM）**：以三专家 [[Mixture of Transformers]] 结构生成未来帧、蒸馏隐动力学并预测机器人动作

推理时，LDM 将观测帧编码为隐动力学；WDAM 以此为中间表征输出离散化动作序列。

### 核心模块 1：Latent Dynamics Model（LDM）

**设计动机**：利用冻结的 [[DINOv3]] 语义特征捕获帧间有意义的变化，同时通过末端执行器运动监督（motion alignment）确保隐空间具备可操作性。

**具体实现**：

- 输入：当前帧 $o_t$ 与下一帧 $o_{t+1}$ 的 DINOv3 特征 patch token
- 通过 [[Soft Vector Quantization|Soft VQ]] 对隐向量进行量化，得到离散化隐动力学 token $z_t$
- **语义重建损失** $\mathcal{L}_{\text{sem}}$：预测下一帧 DINOv3 特征（而非像素），抑制外观噪声
- **运动对齐损失** $\mathcal{L}_{\text{mot}}$：用 $z_t$ 预测末端执行器增量 $\Delta p$（腕部坐标系下的 6-DoF delta），确保隐空间对运动敏感

**多步长采样（Multi-stride Sampling）**：在训练时以不同帧间步长（stride）采样帧对，覆盖不同速度尺度的运动，提升对快速和慢速动作的表征能力。

### 核心模块 2：World Dynamics Action Model（WDAM）

**设计动机**：以 [[Mixture of Transformers]]（MoT）的三专家结构解耦视频生成、动力学提炼与动作预测，通过非对称注意力模式建立单向信息流 video → latent dynamics → action。

**三专家结构**：

| 专家 | 职责 | 注意力范围 |
|------|------|-----------|
| Video Expert | 生成未来帧（[[Flow Matching]] 去噪） | 全帧序列自注意力 |
| Latent Dynamics Expert | 用可学习查询蒸馏隐动力学 | 访问视频 token，输出动力学 query |
| Action Expert | 以 [[Flow Matching]] 预测动作序列 | 访问动力学 query（不直接访问视频 token） |

**非对称注意力（Asymmetric Attention）**：动作专家仅能访问隐动力学专家的输出，而无法直接访问原始视频 token，从而强制动作预测必须经过隐动力学瓶颈，增强了可迁移性。

### 三阶段训练流程

| 阶段 | 训练目标 | 数据 | 冻结/解冻 |
|------|----------|------|----------|
| Stage I 预训练 | 视频生成 + 隐动力学提炼 | 全部人类视频 + 机器人视频 | Action Expert 冻结 |
| Stage II 对齐训练 | 全专家联合训练 | 通用机器人数据 | 全部解冻 |
| Stage III 后训练 | 任务特化微调 | 任务特定机器人数据 | 全部解冻 |

---

## 关键公式

### 公式 1：[[Latent Dynamics Model|LDM 总损失]]

$$
\mathcal{L}_{\text{LDM}} = \mathcal{L}_{\text{sem}} + \lambda_m \mathcal{L}_{\text{mot}} + \mathcal{L}_{\text{vq}}
$$

**含义**：LDM 同时优化语义重建、运动对齐和向量量化三项损失，$\lambda_m$ 控制运动监督的权重。

**符号说明**：
- $\mathcal{L}_{\text{sem}}$：语义重建损失，预测下一帧 DINOv3 特征
- $\mathcal{L}_{\text{mot}}$：运动对齐损失，从隐动力学 token 回归末端执行器增量
- $\mathcal{L}_{\text{vq}}$：向量量化正则项（见下式）
- $\lambda_m$：运动损失权重超参数

### 公式 2：[[Soft Vector Quantization|软向量量化正则项]]

$$
\mathcal{L}_{\text{vq}} = \lambda_u \mathcal{L}_{\text{use}} + \lambda_e \mathcal{L}_{\text{ent}}
$$

**含义**：Soft VQ 通过码本使用率损失 $\mathcal{L}_{\text{use}}$ 和熵损失 $\mathcal{L}_{\text{ent}}$ 防止码本坍塌（codebook collapse），$\lambda_u, \lambda_e$ 分别控制各项权重。

**符号说明**：
- $\mathcal{L}_{\text{use}}$：码本使用率损失，鼓励均匀使用码本条目
- $\mathcal{L}_{\text{ent}}$：熵正则项，最大化分配分布的熵以提升码本利用率
- $\lambda_u, \lambda_e$：各项权重系数

### 公式 3：[[World Dynamics Action Model|WDAM 总损失]]

$$
\mathcal{L}_{\text{WDAM}} = \lambda_v \mathcal{L}_{\text{video}}^{\text{fm}} + \lambda_d \mathcal{L}_{\text{dyn}}^{\text{mse}} + \lambda_a \mathcal{L}_{\text{act}}^{\text{fm}}
$$

**含义**：WDAM 的三专家分别贡献视频生成损失（[[Flow Matching]]）、隐动力学回归损失（MSE）和动作预测损失（[[Flow Matching]]），$\lambda_v, \lambda_d, \lambda_a$ 为权重系数。

**符号说明**：
- $\mathcal{L}_{\text{video}}^{\text{fm}}$：视频专家的 Flow Matching 去噪损失
- $\mathcal{L}_{\text{dyn}}^{\text{mse}}$：动力学专家的 MSE 回归损失（预测 LDM 产出的隐 token）
- $\mathcal{L}_{\text{act}}^{\text{fm}}$：动作专家的 Flow Matching 损失（预测末端执行器增量序列）

### 公式 4：腕部坐标系定义（Wrist Frame）

$$
\hat{z} = \widehat{p_m - p_w}, \quad \hat{x} = \widehat{\hat{z} \times (p_l - p_i)}, \quad \hat{y} = \hat{z} \times \hat{x}
$$

**含义**：统一的腕部坐标系定义，使末端执行器增量动作 $\Delta p$ 在不同具身形态间保持一致的语义，是实现跨具身迁移的关键设计。

**符号说明**：
- $p_w$：腕关节位置（坐标系原点）
- $p_m$：中指指尖位置（定义 z 轴方向）
- $p_l, p_i$：食指与小指指尖（定义 x 轴方向）

---

## 关键图表

### Figure 1：系统总览

![LD4WAM Overview](https://arxiv.org/html/2608.22403v1/LD4WAM_Teaser_final.png)

**说明**：LD4WAM 的整体概览。左侧为统一预训练数据集（5000+ 小时人类 + 机器人视频）；中间为 LDM 通过语义重建和运动对齐监督提取隐动力学，WDAM 以可学习查询融合隐动力学并预测动作；右侧为 RoboTwin 仿真和真实机器人上的定性与定量结果。

### Figure 2：人类数据处理流程

![Human Data Process](https://arxiv.org/html/2608.22403v1/dataprocess_v2.png)

**说明**：长视频分割为子任务片段，下采样到 15 FPS，经过内容过滤后得到高质量人类示范片段。关键步骤包括场景边界检测、手部可见性过滤，以及腕部坐标系下的末端执行器轨迹估计。

### Figure 3：LD4WAM 方法流程图

![LD4WAM Method Pipeline](https://arxiv.org/html/2608.22403v1/LD4WAM_Method.png)

**说明**：完整的 LD4WAM 方法流程，展示 [[Latent Dynamics Model]]（LDM）、[[World Dynamics Action Model]]（WDAM）以及三阶段训练过程的具体结构。LDM 以 DINOv3 特征为输入，经 Soft VQ 量化为隐 token；WDAM 三个专家通过非对称注意力形成 video→dynamics→action 的信息流。

### Figure 4：泛化能力评估设置

![Generalization Setting](https://arxiv.org/html/2608.22403v1/Genelization_setting.png)

**说明**：真实场景泛化测试设置，分别针对**物体外观迁移**（Unseen Objects）和**背景变化**（Background Shift）两个维度，在 Tidy Desk、Fold Shirt、Handover Mug 三个任务上评估。

### Figure 5：跨域隐动力学检索

![Cross-domain Latent Dynamics Retrieval](https://arxiv.org/html/2608.22403v1/Reterival.png)

**说明**：通过余弦相似度检索最近邻，展示 LD4WAM 学到的隐动力学在人类视频和机器人视频之间的跨域对齐能力——相似动作（如"拾取杯子"）无论由人类还是机器人执行，都映射到相近的隐空间位置。

### Figure 6：真实机器人执行可视化

![Real-world Deployment](https://arxiv.org/html/2608.22403v1/RealDeployment-new.png)

**说明**：7 个真实机器人任务（Sorting、Shift Test Tube、Tidy Desk、Fold Shirt、Handover Mug、Place Rubik's Cube、Spray Water）的执行轨迹可视化，涵盖夹爪和灵巧手两个硬件平台。

### Figure 7：真实机器人硬件平台

![Real Robot Setup](https://arxiv.org/html/2608.22403v1/LD4WAM_RealSetup.png)

**说明**：两个真实机器人平台。左：双臂夹爪系统（AgileX PIPER 臂 + 平行夹爪）；右：双臂灵巧手系统（Tianji 臂 + Wuji 灵巧手）。灵巧手具备高自由度手指控制，代表更高难度的操作场景。

### Figure 8：真实场景泛化可视化

![Generalization Rollouts](https://arxiv.org/html/2608.22403v1/more_gene_vis.png)

**说明**：Tidy Desk、Fold Shirt、Handover Mug 三个任务在物体外观和背景变化条件下的执行轨迹，展示 LD4WAM 在分布外场景中保持较高操作成功率的能力。

### Table 1：RoboTwin 仿真基准结果

| 方法 | Clean | Random | **Average** |
|------|-------|--------|------------|
| Fast-WAM | 91.88 | 91.78 | 91.8 |
| Lingbot-VA | 92.90 | 91.50 | 92.2 |
| **LD4WAM** | **93.96** | **92.78** | **93.4** |

**说明**：LD4WAM 在 RoboTwin 50 个任务的仿真基准上以 93.4% 平均成功率领先，在 Clean 和 Random 环境配置下均优于 Fast-WAM（+1.6pp）和 Lingbot-VA（+1.2pp）。

### Table 2：真实机器人任务成功率（%）

| 任务 | LD4WAM | π₀.₅ | Lingbot-VA | Fast-WAM |
|------|--------|-------|------------|---------|
| Sorting | **80.0** | 63.3 | **80.0** | — |
| Shift Test Tube | **50.0** | 43.3 | 40.0 | — |
| Tidy Desk | 70.0 | **73.3** | 63.3 | — |
| Fold Shirt | **80.0** | 76.7 | 73.3 | — |
| Handover Mug | **83.3** | 70.0 | 76.7 | — |
| Place Rubik's Cube | — | — | — | — |
| Spray Water | — | — | — | — |
| **Average** | **70.5** | 63.3 | 64.3 | 47.1 |

**说明**：LD4WAM 真实机器人平均成功率 70.5%，分别超越 π₀.₅（+7.2pp）、Lingbot-VA（+6.2pp）和 Fast-WAM（+23.4pp）。灵巧手任务（Place Rubik's Cube、Spray Water）展示了系统在高自由度操作中的能力。

### Table 3：LDM 消融实验（隐动力学回归误差，越低越好）

| 配置 | High-rate MSE | Low-rate MSE |
|------|--------------|-------------|
| Full LDM（完整） | **0.21** | **2.13** |
| w/o motion alignment | 0.78 | 10.59 |
| w/o multi-stride sampling | 0.24 | 2.54 |
| w/ pixel reconstruction | 0.25 | 2.69 |
| w/ hard VQ | 0.37 | 4.00 |

**关键发现**：去除运动对齐监督后误差暴增 3.7–5.0×（0.78 vs 0.21；10.59 vs 2.13），证明运动对齐是 LDM 最核心的设计；Soft VQ 相对 Hard VQ 将低频误差从 4.00 降至 2.13（-47%），体现更好的码本利用率。

### Table 4：WDAM 架构消融实验（真实机器人成功率，%）

| 配置 | Tidy Desk | Fold Shirt | Handover Mug | Average |
|------|-----------|-----------|-------------|---------|
| Video-Action Dual-Expert（基线） | 56.7 | 63.3 | 66.7 | 40.0 |
| + Latent Dynamic Expert | 63.3 | 73.3 | 73.3 | 48.1 |
| + Pre-training | 63.3 | 76.7 | 73.3 | 58.9 |
| + Align Training（完整 LD4WAM） | **70.0** | **80.0** | **83.3** | **63.7** |

**关键发现**：隐动力学专家贡献 +8.1pp；预训练贡献 +10.8pp；对齐训练再增 +4.8pp；三阶段完整流程相对双专家基线提升 23.7pp，各组件缺一不可。

---

## 实验

### 数据集

| 数据集 | 类型 | 规模 | 比例 |
|--------|------|------|------|
| 人类第一视角视频（5 个来源）| 人类视频 | ~210M 帧 | 76.4% |
| 多具身机器人示范（3 个来源）| 机器人数据 | ~65M 帧 | 23.6% |
| **合计** | — | **274.66M 帧 / 5000+ 小时** | 100% |

所有数据统一转换为 [[LeRobot]] 格式，以 15 FPS 采样，动作表示为腕部坐标系下的末端执行器增量（6-DoF delta）。

### 实现细节

- **视频骨干**: 冻结 [[DINOv3]] ViT 特征提取器（LDM 阶段）
- **量化**: [[Soft Vector Quantization|Soft VQ]] 码本（码本坍塌防护）
- **动作表示**: [[Action Chunking|动作块]] —— 腕部坐标系下末端执行器增量序列
- **动作生成**: [[Flow Matching]] 去噪（WDAM 动作专家）
- **视频生成**: [[Flow Matching]] 去噪（WDAM 视频专家）
- **训练**: 三阶段（预训练 → 对齐训练 → 后训练微调）
- **数据格式**: [[LeRobot]] 统一格式

### 仿真评估

在 [[RoboTwin]] 基准（50 个操作任务，Clean/Random 两种配置）上达到 93.4% 平均成功率，超越所有对比方法。

### 真实机器人评估

在 7 个真实世界任务上评估，覆盖：
- **夹爪平台**：Sorting、Shift Test Tube、Tidy Desk（AgileX PIPER + 平行夹爪）
- **灵巧手平台**：Fold Shirt、Handover Mug、Place Rubik's Cube、Spray Water（Tianji 臂 + Wuji 灵巧手）

### 泛化评估

- **未见物体**：保留分布内性能的 88.6%（各方法最优）
- **背景迁移**：达到 63.7% 平均成功率（优于 Fast-WAM 的 31.1%，接近 π₀.₅ 的 62.6%）

---

## 批判性思考

### 优点

1. **跨具身迁移设计精巧**：腕部坐标系统一表示 + 语义特征空间的组合，使动力学表征在人类手和机器人末端之间保持语义一致，是该框架最具创新性的设计
2. **隐瓶颈强制信息流**：非对称注意力确保动作预测必须经过隐动力学瓶颈，从架构上保证了视频先验对动作输出的影响路径
3. **大规模数据验证**：5000+ 小时数据 + 真实灵巧手测试，实验规模和实用价值较为充分

### 局限性

1. **DINOv3 依赖**：LDM 的语义重建依赖冻结的 DINOv3 特征，DINOv3 的语义质量直接决定隐动力学的上限，对视觉特征基础模型有较强绑定
2. **三阶段训练复杂度**：预训练 → 对齐训练 → 后训练的流水线增加了工程复杂度，Stage II 对齐训练的数据需求和超参敏感性未作详细分析
3. **灵巧手任务数据有限**：部分灵巧手任务（Place Rubik's Cube、Spray Water）仅有定性展示，缺乏与基线方法的定量比较

### 潜在改进方向

1. **端到端 LDM + WDAM 联合训练**：目前 LDM 与 WDAM 分阶段训练，端到端优化可能进一步提升隐动力学对动作预测的适配性
2. **时序隐动力学**：当前隐动力学基于帧对（单步），扩展至多步时序动力学（如 RSSM 风格）可能捕获更长程的运动规律
3. **码本结构化**：Soft VQ 的码本目前无结构约束，引入层次化码本（如 [[Residual VQ]]）可能提升表征效率

### 可复现性评估

- [ ] 代码开源
- [ ] 预训练模型
- [x] 训练细节描述（三阶段流程有文字说明）
- [ ] 数据集可获取（自建多源混合数据集）

---

## 关联笔记

### 基于

- [[Fast-WAM]]：WAM 基线方法，LD4WAM 在其基础上引入隐动力学桥接
- [[Pi05|π₀.₅]]：真实机器人对比基线，LD4WAM 在平均任务成功率上超越 π₀.₅
- [[LeRobot]]：数据格式标准，所有训练数据统一为 LeRobot 格式

### 对比

- [[LingBot-Video|Lingbot-VA]]：同类 WAM 方法，LD4WAM 在真实机器人任务上以 70.5% vs 64.3% 领先
- [[Fast-WAM]]：对比基线，LD4WAM 在仿真（+1.6pp）和真实（+23.4pp）均优
- [[Pi05|π₀.₅]]：扩散策略基线，LD4WAM 真实机器人平均成功率高出 7.2pp

### 方法相关

- [[Latent Dynamics Model]]：LDM 核心概念，运动对齐隐表征提取
- [[Mixture of Transformers]]：WDAM 的三专家架构设计
- [[Soft Vector Quantization]]：Soft VQ 量化方法，防止码本坍塌
- [[DINOv3]]：冻结视觉骨干，提供语义特征用于重建监督
- [[Flow Matching]]：WDAM 视频专家和动作专家的去噪生成框架
- [[Action Chunking]]：动作序列预测方式

### 硬件/数据相关

- [[RoboTwin]]：仿真评估基准（50 个操作任务）
- [[LeRobot]]：统一数据格式

---

## 速查卡片

> [!summary] LD4WAM
> - **核心**: 从人类视频学习运动对齐隐动力学，桥接视频先验与机器人动作
> - **方法**: LDM（DINOv3 + Soft VQ + 运动对齐）+ WDAM（三专家 MoT + 非对称注意力）
> - **结果**: 仿真 93.4%（RoboTwin 50 任务）/ 真实 70.5%（7 任务，含灵巧手），超越 π₀.₅ 和 Fast-WAM
> - **代码**: 暂未开放

---

*笔记创建时间: 2026-08-26*
