---
title: "G³VLA: Geometric Inductive Bias for Vision-Language-Action Models"
method_name: "G3VLA"
authors: [Yue Peng, Yongzhe Zhao, Artur Habuda, Khuyen Pham, Yanheng Zhu, Tran Nguyen Le, Fares Abu-Dakka, Li Guo]
year: 2026
venue: arXiv (submitted to CoRL 2026)
tags: [vla, geometric-inductive-bias, multi-camera, camera-calibration, knowledge-distillation, robot-manipulation]
zotero_collection: Robotics/VLA
image_source: online
arxiv_html: https://arxiv.org/html/2606.24472v1
created: 2026-06-25
---

# 论文笔记：G³VLA: Geometric Inductive Bias for Vision-Language-Action Models

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | — （CoRL 2026 双盲评审，论文与项目主页均匿名） |
| 日期 | June 2026 |
| 项目主页 | [sites.google.com/view/g3vla](https://sites.google.com/view/g3vla)（匿名版） |
| 对比基线 | [[π0]]、[[π0.5]]、[[GR00T N1.5|GR00T 1.5]]、[[RVT]]、[[Act3D]] |
| 链接 | [arXiv](https://arxiv.org/abs/2606.24472) / Code（未公开） |

---

## 一句话总结

> G³VLA 把标定的多相机几何（内参光线嵌入 + 投影位置编码 + 跨视角融合）作为轻量插件注入预训练 VLA 的视觉 token 通路，在不改动动作空间和模仿学习目标的前提下提升空间敏感任务的成功率。

---

## 核心贡献

1. **几何感知视觉 token 模块**: 提出由内参条件光线嵌入（Ray Embeddings）、[[PRoPE|投影位置编码]]、双向跨视角融合三部分组成的 $F_\psi$ 模块，将相机内参 $K^v$ 与外参 $T^v$ 显式注入视觉 token，而无需改变预训练 backbone、动作空间或模仿学习目标
2. **几何蒸馏 + 两阶段训练**: 利用几何教师模型（[[VGGT|π³]] 或仿真器真值深度）提供稠密点图监督，通过置信度门控蒸馏损失预训练几何模块，再解冻整个策略网络做联合微调
3. **跨 backbone 通用性验证**: 在 [[π0]]、[[π0.5]]、[[GR00T N1.5|GR00T 1.5]] 三种不同架构的 VLA 上验证方法兼容性，发现单塔架构（π0/π0.5）收益明显大于双塔架构（GR00T），揭示了几何信号能否直达动作生成路径是关键因素

---

## 问题背景

### 要解决的问题

标准 [[VLA]] 模型把多相机观测当作独立的 2D 图像处理，**丢弃了系统中本就已知的相机内参和外参信息**。即便机器人系统通常是经过标定的多相机设置，VLA 的视觉 token 仍然只编码 2D 图像坐标，相机几何只能依靠动作监督间接、隐式地学习。

### 现有方法的局限

- [[RT-2]]、[[OpenVLA]]、[[π0]]、[[π0.5]]、[[GR00T N1.5|GR00T]] 等方法复用了预训练视觉-语言 backbone 的语义先验，但继承了纯 2D 的视觉接口，无法显式利用标定几何
- [[PerAct]]、[[RVT]]、[[Act3D]]、PolarNet 等结构化 3D 表征方法证明了几何结构的价值，但通常需要针对任务重新设计体素化/点云表征，难以直接复用预训练 VLM 权重
- 3D-VLA、SpatialVLM、[[SpatialVLA]] 等空间感知 VLA 要求显式 3D 输入或修改动作表示，增加了系统复杂度

### 本文的动机

论文提出的核心研究问题是："标定的相机几何能否作为一条轻量级的视觉 token 通路，在不修改 backbone、动作空间、模仿学习目标的情况下提升预训练 VLA 的空间泛化能力？" 作者认为可以借助 [[PRoPE]] 等相机感知表征技术与 [[VGGT|DUSt3R/VGGT/π³]] 等几何教师模型的预测能力，把几何结构以"旁路注入"的方式嫁接到现有 VLA 之上。

---

## 方法详解

### 模型架构

G³VLA 采用**几何旁路注入**架构，可插入到任意基于 transformer 的 VLA 视觉编码通路：
- **输入**: 语言指令 $l$ + 机器人状态 $s_t$ + 多视角 RGB 观测 $\{I_t^v, K^v, T^v\}_{v=1}^V$（$K^v$ 为内参，$T^v$ 为外参）
- **Backbone**: [[SigLIP]] 视觉编码器（224×224 输入产生 16×16 patch token 网格），与 [[π0]]/[[π0.5]]/[[GR00T N1.5|GR00T 1.5]] 等不同 VLA 的预训练权重保持冻结或可微调
- **核心模块**: [[PRoPE|内参条件光线嵌入]] 与 [[PRoPE|投影位置编码]] 联合驱动的 [[Cross-View Attention|双向跨视角融合]]
- **输出**: 与原 VLA 完全一致的 [[Action Chunking|动作块]] $a_{t:t+H-1}$（不改变动作空间和损失形式）
- **训练范式**: 零初始化保证插入模块初始时不改变预训练行为；两阶段训练逐步引入几何信号

### 核心模块

#### 模块 1：内参条件光线嵌入（Intrinsic-Conditioned Ray Embeddings）

**设计动机**: 利用相机内参 $K^v$ 把每个 patch token 标记上其对应的反投影观察方向，使 token 携带"这是从哪条光线看到的"这一几何先验。

**具体实现**:
- 对齐次像素坐标 $u=(x,y,1)^T$，通过逆内参 $(K^v)^{-1}$ 计算归一化投影光线，取前两维作为图像平面光线坐标 $R^v(x,y)$
- 用可学习函数 $G_\varphi$ 将光线坐标图嵌入后，逐 patch 加到 [[SigLIP]] 编码器输出 token 上
- 零初始化使该模块在训练初期不扰动预训练视觉特征

#### 模块 2：投影位置编码（[[PRoPE]]）

**设计动机**: 仅靠光线嵌入只能编码"单视角内"的几何信息，跨视角之间的相对投影关系需要专门机制。[[PRoPE]] 通过相机模型（而非单纯学习到的外观相关性）显式捕捉跨视角的投影关系。

**具体实现**:
- 从每个视角的内参 $K^v$、相机到世界变换（源自 $T^v$）以及 patch 位置出发，推导出固定的投影变换，分别作用于跨视角注意力的 Query / Key / Value
- 使跨视角注意力的 attention bias 直接来自相机模型几何，而不是单纯依赖外观相似度学习得到的隐式对应关系

#### 模块 3：双向跨视角融合（Bidirectional Cross-View Fusion）

**设计动机**: 多相机信息需要先在视角内部聚合上下文，再做跨视角的几何对齐融合，分两步走能保留视角局部结构的同时实现全局几何一致性。

**具体实现**:
- **第一步（Frame Attention）**: 在每个相机视角内部独立做自注意力，保留视角局部结构
- **第二步（Cross-View Attention）**: 展平视角和 patch 维度，让所有视角的 token 在 [[PRoPE]] 提供的位置信号下做双向跨视角注意力交互，得到融合后的几何感知 token $H$

### 几何蒸馏与两阶段训练

为了让几何模块学到真正有用的几何先验（而非依赖动作监督隐式学习），论文引入一个辅助点头（point head）和两阶段训练策略：

- **辅助监督目标**: 点头从融合 token 预测每个 patch 的光线坐标 $\hat{q}_u^v$ 和对数深度 $\hat{d}_u^v$
- **监督来源**: 仿真器中可用渲染深度得到的真值（G3VLA (GT)）；真实场景则用几何教师模型 [[VGGT|π³]] 的预测作为蒸馏目标（G3VLA (π³X)），并用置信度门控掩码过滤不可靠区域
- **Stage 1（几何预训练）**: 仅训练几何模块、跨视角融合层和辅助点头，backbone 与原策略网络冻结；蒸馏损失主导（5k steps，$\lambda_{act}=0.1$，$\lambda_{distill}=1.0$）
- **Stage 2（联合微调）**: 解冻完整策略网络，动作损失主导，蒸馏损失退化为正则项（30k steps，$\lambda_{act}=1.0$，$\lambda_{distill}=0.05$）

---

## 关键公式

### 公式 1：[[VLA|VLA 策略formulation]]

$$
\pi_\theta\big(a_{t:t+H-1} \mid l, s_t, \{I_t^v, K^v, T^v\}_{v=1}^{V}\big)
$$

**含义**: 策略网络在给定语言指令、机器人状态和多视角标定相机观测（图像 + 内参 + 外参）的条件下，预测未来 $H$ 步的动作块

**符号说明**:
- $\pi_\theta$: 待学习的策略网络
- $a_{t:t+H-1}$: 时刻 $t$ 起长度为 $H$ 的[[Action Chunking|动作块]]
- $l$: 语言指令；$s_t$: 机器人本体状态
- $I_t^v, K^v, T^v$: 第 $v$ 个视角在 $t$ 时刻的图像、内参矩阵、外参（相机位姿）
- $V$: 相机视角总数

### 公式 2：几何感知 token 转换

$$
h_{1:P}^{1:V} = F_\psi\big(z_{1:P}^{1:V}, \{K^v, T^v\}_{v=1}^{V}\big)
$$

**含义**: 几何模块 $F_\psi$ 以原始视觉 patch token 和相机标定参数为输入，输出注入几何结构后的融合 token

**符号说明**:
- $z_p^v \in \mathbb{R}^d$: 第 $v$ 视角第 $p$ 个 patch 的视觉编码器输出 token（共 $P$ 个 patch / 视角）
- $F_\psi$: 可学习的标定条件几何变换模块，参数为 $\psi$
- $h_{1:P}^{1:V}$: 输出的几何感知融合 token

### 公式 3：[[PRoPE|光线坐标提取]]

$$
R^v(x,y) = \big[\tilde{r}^v(x,y,1)\big]_{1:2} \in \mathbb{R}^2, \qquad \tilde{r}^v(u) = (K^v)^{-1} u
$$

**含义**: 把齐次像素坐标通过逆内参映射为归一化投影光线，并取其图像平面（前两维）分量作为该像素的几何坐标表示

**符号说明**:
- $u = (x,y,1)^T$: 齐次像素坐标
- $K^v$: 第 $v$ 个视角的相机内参矩阵
- $\tilde{r}^v(u)$: 归一化投影光线（3 维）
- $R^v(x,y)$: 取前两维后的图像平面光线坐标

### 公式 4：光线嵌入注入

$$
z_{0,p}^v = z_p^v + G_\varphi\big(R^v\big)_p
$$

**含义**: 把光线坐标图经可学习嵌入函数 $G_\varphi$ 编码后，逐 patch 加到原始视觉 token 上，作为跨视角融合前的几何增强 token

**符号说明**:
- $G_\varphi$: 可学习的光线嵌入函数，参数为 $\varphi$
- $z_{0,p}^v$: 注入光线嵌入后、送入跨视角融合层之前的 token

### 公式 5：[[PRoPE|双向跨视角融合]]

$$
H = \mathrm{Fusion}_\psi\big(Z; \{K^v, T^v\}_{v=1}^{V}\big)
$$

**含义**: 跨视角融合模块先做视角内 frame attention，再以 [[PRoPE]] 为位置信号做跨视角双向注意力，输出最终融合 token 供下游动作生成模块使用

**符号说明**:
- $Z$: 所有视角光线增强 token 的集合
- $\mathrm{Fusion}_\psi$: 两步式（frame attention → cross-view attention）融合算子
- $H$: 输出给动作生成路径的几何感知 token

### 公式 6：[[Knowledge Distillation|置信度门控掩码]]

$$
m_u^v = \mathbb{1}\big[\sigma(c_u^v) > \tau\big], \qquad \tau = 0.1
$$

**含义**: 将几何教师模型（[[VGGT|π³]]）输出的置信度 logit 经 sigmoid 映射后与阈值 $\tau$ 比较，生成硬门控掩码，过滤掉教师预测不可靠的像素位置

**符号说明**:
- $c_u^v$: 教师模型在视角 $v$、位置 $u$ 处的置信度 logit
- $\sigma(\cdot)$: sigmoid 函数
- $\mathbb{1}[\cdot]$: 指示函数；$\tau=0.1$: 置信度门控阈值

### 公式 7：[[Knowledge Distillation|几何蒸馏损失]]

$$
\mathcal{L}_{distill} = \frac{\displaystyle\sum_{v,u} m_u^v\Big(\tfrac{1}{2}\|\hat{q}_u^v - q_u^v\|_2^2 + (\hat{d}_u^v - d_u^v)^2\Big)}{\displaystyle\sum_{v,u} m_u^v + \epsilon}
$$

**含义**: 在置信度掩码筛选后的有效像素上，同时监督光线坐标预测和对数深度预测，两类损失统一在一个归一化的蒸馏目标中

**符号说明**:
- $\hat{q}_u^v, \hat{d}_u^v$: 辅助点头预测的光线坐标与对数深度
- $q_u^v, d_u^v$: 监督目标（仿真器真值或 [[VGGT|π³]] 教师预测）
- $\epsilon$: 防止分母为零的小常数

### 公式 8：[[Knowledge Distillation|两阶段联合训练损失]]

$$
\mathcal{L} = \lambda_{act}\,\mathcal{L}_{act} + \lambda_{distill}\,\mathcal{L}_{distill}
$$

**含义**: 总损失由原 VLA 的动作目标（[[Flow Matching|流匹配]]损失，针对 π0/π0.5）与几何蒸馏损失加权组合；两阶段训练通过调整 $\lambda_{act}, \lambda_{distill}$ 的相对权重，从"几何预训练"过渡到"动作主导微调"

**符号说明**:
- $\mathcal{L}_{act}$: 原始 VLA 模仿学习损失（保持不变）
- $\lambda_{act}, \lambda_{distill}$: 阶段相关权重 —— Stage 1: $(0.1, 1.0)$；Stage 2: $(1.0, 0.05)$

---

## 关键图表

### Figure 1：G³VLA 整体架构与两阶段训练流程

![Figure 1](https://arxiv.org/html/2606.24472v1/x1.png)

**说明**: (A) 几何归纳偏置通过内参条件光线嵌入（$K^{-1}$）与配合 [[PRoPE]] 的双向跨视角融合注入 VLA 视觉 token，预训练 backbone 和动作目标保持不变；(B) Stage 1 从 [[VGGT|π³]] 蒸馏稠密点图以预训练几何模块，Stage 2 在动作损失与蒸馏损失联合监督下微调完整策略。

### Figure 2：真实世界实验设置（双臂 UR5 工作台）

![Figure 2](https://arxiv.org/html/2606.24472v1/imgs/npbc_nyu_paper_2.png)

**说明**: 双臂 UR5 机械臂工作台上的两项评测任务——Pick and Place Test Tube（蓝色标注）与 Pouring Nut（绿色标注）。数据集分布表给出了每项任务相关物体位置组合，每种位置组合在不同上下文相机位置下重复 10 次采集。

### Figure 3：LIBERO 消融实验

![Figure 3](https://arxiv.org/html/2606.24472v1/x2.png)

**说明**: 沿三个轴的消融——几何组件（光线嵌入、[[PRoPE]]）、训练范式（两阶段 vs. 单阶段）、监督来源（仿真器 GT vs. [[VGGT|π³]] 蒸馏）。虚线标出 baseline（84.6%）与 G3VLA (GT)（88.1%）。移除光线嵌入掉 2.0 点（87.0%→85.0%），移除 PRoPE 掉 1.1 点（87.0%→85.9%），单阶段训练掉 0.7 点（87.0%→86.3%）。

### Figure 4：π³ 教师与仿真器真值深度对比诊断（RoboTwin handover_block）

![Figure 4](https://arxiv.org/html/2606.24472v1/imgs/handover_block_pi3x_gt_diagnostic_ep000024_frame0227.png)

**说明**: 在 RoboTwin2.0 的 handover_block 任务中逐帧对比 [[VGGT|π³]] 教师预测深度与仿真器渲染真值深度，用于诊断蒸馏失败的原因——干净合成场景下教师预测存在尺度不一致问题。

### Figure 5：π³ 与真值深度的尺度误差分布直方图

![Figure 5](https://arxiv.org/html/2606.24472v1/imgs/handover_block_pi3x_gt_error_histogram.png)

**说明**: handover_block 任务缓存数据上，[[VGGT|π³]] 教师预测深度相对仿真器真值的中位数尺度差异分布，解释了为什么 π³X 蒸馏在 RoboTwin2.0（干净合成场景）上效果不如 GT 监督（49.0% vs. 41.0%）。

### Figure 6：失败恢复案例 — Realign

![Figure 6](https://arxiv.org/html/2606.24472v1/imgs/failure_recovery/realign.png)

**说明**: 真实机器人实验中的重新对齐（realign）恢复行为示例，展示策略在抓取位置偏差后主动调整的能力。

### Figure 7：失败恢复案例 — Regrasp

![Figure 7](https://arxiv.org/html/2606.24472v1/imgs/failure_recovery/regrasp.png)

**说明**: 重新抓取（regrasp）恢复行为案例，物体首次抓取失败后策略触发二次抓取尝试。

### Figure 8：失败恢复案例 — Recovery

![Figure 8](https://arxiv.org/html/2606.24472v1/imgs/failure_recovery/recovery.png)

**说明**: 通用恢复行为案例，展示策略在执行过程中从异常状态恢复到任务轨迹的能力。

### Figure 9：失败案例 — Overfit Position

![Figure 9](https://arxiv.org/html/2606.24472v1/imgs/failure_recovery/overfit-pos.png)

**说明**: 失败案例分析——策略对训练时见过的特定物体位置出现过拟合迹象，在分布外位置上表现下降。

### Figure 10：失败案例 — Grasp Wrong Side

![Figure 10](https://arxiv.org/html/2606.24472v1/imgs/failure_recovery/grasp_wrong_side.png)

**说明**: 失败案例分析——策略从物体错误的一侧尝试抓取，反映了空间几何理解仍有局限。

### Table 1：LIBERO 上 π0 的主结果

| Suite | Baseline | G3VLA (π³X) | G3VLA (GT) | Gain |
|-------|----------|------------|-----------|------|
| Goal | 87.4 | 88.4 | 88.4 | +1.0 |
| Spatial | 85.2 | 88.6 | 89.2 | +4.0 |
| Object | 89.4 | 93.4 | 94.4 | +5.0 |
| L-10 | 76.5 | 77.6 | 80.4 | +3.9 |
| **Average** | **84.6** | **87.0** | **88.1** | **+3.5** |

**表格说明**: G3VLA 在 Spatial 和 Object 两类空间敏感任务上提升最大（+4.0 / +5.0），整体平均成功率从 84.6% 提升到 88.1%（GT 监督），即便用蒸馏教师 π³X 也能拿到 87.0%。

### Table 2：RoboCasa24 与 RoboTwin2.0 上 π0 的结果

| Benchmark | Method | Success |
|-----------|--------|---------|
| RoboCasa24 | Baseline | 34.2% |
| RoboCasa24 | G3VLA (π³X) | 36.5% |
| RoboCasa24 | G3VLA (GT) | 37.1% |
| RoboTwin2.0 | Baseline | 44.0% |
| RoboTwin2.0 | G3VLA (π³X) | 41.0% |
| RoboTwin2.0 | G3VLA (GT) | 49.0% |

**表格说明**: RoboCasa24 上两种监督方式都带来一致提升；RoboTwin2.0 上 π³X 蒸馏反而**低于** baseline（41.0% vs. 44.0%），原因是干净合成场景中 π³ 教师预测存在尺度不一致（见 Figure 4/5），而 GT 监督则带来最大单项提升（+5.0）。

### Table 3：π0.5 在 LIBERO 上的验证

| Suite | Baseline | G3VLA (π³X) |
|-------|----------|------------|
| Spatial | 98.8 | 98.2 |
| Object | 98.4 | 99.0 |
| Goal | 94.4 | 98.0 |
| L-10 | 91.8 | 92.8 |
| **Average** | **95.9** | **97.0** |

**表格说明**: π0.5 baseline 已接近饱和（95.9%），G3VLA 仍带来 +1.1 点提升，证明方法在更强 backbone 上依然有效，只是增益空间受限于已有高基线。

### Table 4：GR00T 1.5 在 LIBERO 上的验证

| Suite | Baseline | G3VLA (GT) | G3VLA (π³X) |
|-------|----------|-----------|------------|
| Spatial | 96.6 | 94.2 | 96.6 |
| Object | 97.0 | 97.0 | 99.0 |
| Goal | 95.4 | 94.2 | 95.8 |
| L-10 | 90.6 | 92.6 | 89.6 |
| **Average** | **94.90** | **94.50** | **95.25** |

**表格说明**: GR00T 1.5 上结果呈现混合趋势（GT 监督反而略低于 baseline），归因于其双塔架构——几何 token 需要先跨过额外的注意力瓶颈才能到达动作生成模块，导致几何信号被削弱。这是论文强调"架构依赖性"的关键证据。

### Table 5：真实机器人成功率（双臂 UR5）

**(a) Pick-and-Place Test Tube**

| Checkpoint | π0 ID/OOD | G3VLA(π0+π³X) ID/OOD | π0.5 ID/OOD | G3VLA(π0.5+π³X) ID/OOD |
|------------|-----------|---------------------|-----------|----------------------|
| 20K | 75.0/50.0 | 75.0/33.3 | 50.0/41.7 | 62.5/33.3 |
| 25K | 62.5/41.7 | 75.0/41.7 | 37.5/25.0 | 50.0/50.0 |
| 30K | 75.0/58.3 | 75.0/58.3 | 37.5/41.7 | 62.5/58.3 |

**(b) Pouring Nut Task**

| Checkpoint | π0 ID/OOD | G3VLA(π0+GT) ID/OOD | π0.5 ID/OOD | G3VLA(π0.5+GT) ID/OOD |
|------------|-----------|------------------|-----------|-------------------|
| 20K | 100/70.8 | 100/83.3 | 93.75/66.7 | 93.75/70.8 |
| 25K | 100/75.0 | 100/87.5 | 100/66.7 | 87.50/79.2 |

**表格说明**: 增益主要体现在分布外（OOD）相机视角上——Pouring 任务 OOD 成功率从 70.8%~75.0% 提升到 83.3%~87.5%（π0），从 66.7% 提升到 79.2%（π0.5），说明几何归纳偏置主要帮助模型应对训练时未见过的相机视角，而非在分布内场景上"锦上添花"。

---

## 实验结果

### 数据集

| 数据集 | 规模/特点 | 用途 |
|--------|------|------|
| [[LIBERO]] | Goal / Spatial / Object / L-10 四套任务 | 仿真训练与评测，侧重空间敏感任务 |
| [[RoboCasa|RoboCasa24]] | 家庭场景操作任务子集 | 仿真泛化评测 |
| [[RoboTwin 2.0]] | 双臂协作任务（diagnostic: handover_block） | 仿真评测 + 蒸馏失败诊断 |
| 真实世界数据 | 双臂 [[UR5]] 工作台，10×10 cm 工作区网格，2 项任务，10 个训练视角 | ID（视角 1、3）/ OOD（视角 11-13）评测 |

### 实现细节

- **视觉 Backbone**: [[SigLIP]]，224×224 输入 → 16×16 patch / 视角
- **基础策略**: [[π0]]（flow-matching）、[[π0.5]]（flow-matching 变体）、[[GR00T N1.5|GR00T 1.5]]（双塔 diffusion）
- **几何教师**: [[VGGT|π³]]（confidence-gated 蒸馏）/ 仿真器渲染深度（GT 监督）
- **优化器**: AdamW（$\beta_1=0.9, \beta_2=0.95$），bfloat16 精度，cosine 学习率衰减 + warmup
- **Batch Size**: 全局 32
- **训练步数**: Stage 1 5k steps，Stage 2 30k steps
- **置信度门控阈值**: $\tau = 0.1$

### 可视化结果

Figure 6-10 的失败案例分析显示，G³VLA 策略具备一定的重新对齐（realign）、重新抓取（regrasp）等恢复行为，但仍存在对训练位置过拟合（overfit-pos）以及从错误一侧抓取（grasp wrong side）等空间理解局限，说明几何归纳偏置改善的是视觉 token 层面的空间表征，但无法完全弥补动作空间或语言对齐层面的不足。

---

## 批判性思考

### 优点
1. **架构无侵入性**: 不改变预训练 backbone、动作空间和模仿学习目标，作为"视觉 token 旁路"即可插入现有 VLA，迁移成本低
2. **几何信号来源灵活**: 同时支持仿真器真值深度和无标注真实数据下的几何教师（[[VGGT|π³]]）蒸馏两种监督模式，兼顾仿真训练效果与真实部署可行性
3. **诚实的失败案例分析**: 论文专门用 Figure 4/5 诊断了 π³X 蒸馏在 RoboTwin2.0 上失败的具体原因（尺度不一致），而非简单回避负面结果
4. **跨 backbone 的系统性验证**: 在 π0、π0.5、GR00T 1.5 三种架构上做对照实验，揭示出"单塔 vs. 双塔"架构差异对几何信号有效性的影响，提供了有价值的架构设计启示

### 局限性
1. **依赖精确标定**: 方法本质上假设相机标定准确，对标定漂移和多相机时间同步误差敏感，论文明确承认这一点
2. **教师预测质量瓶颈**: 在遮挡和高光（specularity）场景下，几何教师（π³）目标本身不准确，蒸馏效果受限于教师能力上限
3. **架构依赖性强**: 双塔架构（如 GR00T）下几何 token 需跨越额外注意力瓶颈，效果被显著削弱，方法的普适性打了折扣
4. **改进范围局限于视觉 token**: 仅修改视觉表征通路，无法解决动作空间或语言对齐层面的失败模式（如 Figure 9/10 所示的过拟合和错误抓取侧问题）
5. **真实世界提升幅度不稳定**: Table 5 中部分 checkpoint（如 20K/25K 的 ID 场景）出现 G3VLA 不及甚至低于 baseline 的情况，说明方法收益并非在所有训练阶段都稳定体现

### 潜在改进方向
1. 引入标定不确定性建模（如对 $K^v, T^v$ 加入噪声鲁棒性训练），缓解真实部署中的标定漂移问题
2. 针对双塔架构设计专门的几何 token 直达路径，绕开额外注意力瓶颈
3. 探索在线自适应教师选择机制，在遮挡/高光场景下动态降权不可靠的几何监督信号
4. 将几何归纳偏置进一步扩展到动作空间（如末端执行器位姿的几何先验），而非仅限于视觉 token

### 可复现性评估
- [ ] 代码开源（论文中未提供 GitHub 链接，CoRL 双盲评审阶段）
- [ ] 预训练模型（未发布）
- [x] 训练细节完整（优化器、batch size、两阶段步数和损失权重均已说明）
- [x] 数据集可获取（LIBERO、RoboCasa24、RoboTwin2.0 均为公开 benchmark；真实机器人数据未公开）

---

## 关联笔记

### 基于
- [[SigLIP]]: 视觉编码器骨干，提供初始 patch token
- [[PRoPE]]: 跨视角投影位置编码的核心技术基础
- [[VGGT]]: 几何教师模型 π³ 所属技术谱系（DUSt3R / VGGT / π³ 系列稠密几何预测模型）
- [[Flow Matching]]: π0 / π0.5 的动作生成范式，G3VLA 不改变此目标

### 对比
- [[π0]]: 单塔 flow-matching VLA，G3VLA 收益最明显的 backbone（LIBERO +3.5，真实世界 OOD 显著提升）
- [[π0.5]]: π0 增强版，baseline 已接近饱和（95.9%），G3VLA 仍有 +1.1 点提升
- [[GR00T N1.5|GR00T 1.5]]: 双塔 diffusion 架构，几何 token 需跨注意力瓶颈，收益被削弱甚至出现负向结果
- [[RVT]] / [[Act3D]]: 结构化 3D 表征代表工作，确认几何结构价值但需任务专用设计，与 G3VLA 的"轻量旁路注入"思路形成对比
- [[SpatialVLA]]: 空间感知 VLA 的另一条路线（修改动作表示/3D 输入），与 G3VLA 仅修改视觉 token 通路的设计不同

### 方法相关
- [[Action Chunking]]: VLA 动作输出形式，G3VLA 保持不变
- [[Knowledge Distillation|几何蒸馏]]: Stage 1 预训练几何模块的核心机制
- [[Cross-View Attention]]: 双向跨视角融合的实现基础

### 硬件/数据相关
- [[LIBERO]]: 主要仿真评测 benchmark（四套任务）
- [[RoboCasa|RoboCasa24]]: 家庭场景仿真评测
- [[RoboTwin 2.0]]: 双臂仿真评测平台，也用于诊断蒸馏失败原因
- [[UR5]]: 真实世界双臂实验机械臂平台

---

## 速查卡片

> [!summary] G³VLA: Geometric Inductive Bias for VLA
> - **核心**: 把标定相机几何（光线嵌入 + PRoPE + 跨视角融合）作为轻量插件注入预训练 VLA 视觉 token，不改动作空间
> - **方法**: 内参条件光线嵌入 + 投影位置编码 + 双向跨视角融合 + 两阶段几何蒸馏训练
> - **结果**: LIBERO 上 π0 平均 84.6%→88.1%（GT）/ 87.0%（π³X 蒸馏）；真实世界 OOD 视角下 Pouring 任务从 70.8-75.0% 提升到 83.3-87.5%
> - **代码**: 未公开（CoRL 2026 投稿中）

---

*笔记创建时间: 2026-06-25*
