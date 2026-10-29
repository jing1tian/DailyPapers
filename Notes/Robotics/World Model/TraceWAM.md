---
title: "Trace-WAM: Learning to Act via 3D Motion Prediction for World Action Models"
method_name: "TraceWAM"
authors: [Xu Hu, Changjie Wu, Linzhe Shi, Hang Zhang, Chen Yang, Lingjun Zhang, Guowen Zhang, Mu Xu, Junran Peng, Qing Li, Zhaoxiang Zhang]
year: 2026
venue: arXiv
tags: [world-action-model, 3d-motion-prediction, auxiliary-supervision, flow-matching, mixture-of-transformers, out-of-distribution, robot-manipulation]
zotero_collection: Robotics/World Model
image_source: online
arxiv_html: https://arxiv.org/html/2610.36840
created: 2026-10-29
---

# 论文笔记：Trace-WAM: Learning to Act via 3D Motion Prediction for World Action Models

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Chinese Academy of Sciences / CASIA |
| 日期 | October 2026 |
| 项目主页 | [trace-wam.github.io](https://trace-wam.github.io) |
| 对比基线 | [[Fast-WAM]]、[[GeoSem-WAM]]、[[LAWA]] |
| 链接 | [arXiv](https://arxiv.org/abs/2610.36840) / [Project](https://trace-wam.github.io) |

---

## 一句话总结

> Trace-WAM 以未来三维运动轨迹（图像位移 + 深度变化）作为辅助训练目标，通过 [[Tri-MoT|三分支 Mixture-of-Transformer]] 共享表征，在不增加推理开销的前提下大幅提升操作策略的泛化性和少样本能力。

---

## 核心贡献

1. **3D 运动预测辅助监督**: 提出将未来三维点运动（图像平面位移 + 深度变化）作为[[World Action Model|WAM]] 的辅助训练目标，聚焦机器人与任务相关物体的运动，而非重建整个场景
2. **Tri-MoT 架构**: 设计[[Tri-MoT|三分支 Mixture-of-Transformer]]，包含视频预测专家、运动预测专家、动作生成专家，通过结构化注意力机制让运动监督塑造动作表征
3. **推理零开销**: 推理时运动分支完全移除，仅需单次视觉前向传播 + 动作去噪，与 [[Fast-WAM]] 保持相同推理结构

---

## 问题背景

### 要解决的问题

[[World Action Model|WAMs]] 通过联合学习未来视频预测与动作生成来建立机器人控制表征。然而，未来视频预测需要重建整个场景，大量计算资源浪费在与操作无关的背景区域（如桌面纹理、墙壁等）。

### 现有方法的局限

- **Joint-WAM / [[Fast-WAM]]**: 视频预测的监督信号包含大量无关背景信息，对任务关键运动的表征学习效率低
- **视频预测为主的 WAM**: 分辨率高、计算代价大，但与操作相关的有效信息密度低
- **4D/几何辅助方法**（如 [[GeoSem-WAM]]）: 主要利用静态几何信息（深度、分割），缺乏时序运动信息

### 本文的动机

操作任务的成功与否本质上取决于**机器人末端执行器**和**任务相关物体**如何随时间移动。若能以三维点运动轨迹（密集、时序一致的对应关系）作为辅助监督，可以更高效地引导动作表征学习——这一假设由图 2 中深度通道的存活率曲线（约 30% 的点发生显著深度变化）所验证。

---

## 方法详解

### 模型架构

Trace-WAM 以 [[Tri-MoT|三分支 Mixture-of-Transformer (Tri-MoT)]] 为核心架构：
- **输入**: 语言指令 $\ell$ + 当前 RGB 观测 $I_0$ + 本体感知状态 $s_0$
- **Backbone**: 冻结的因果视频 VAE（编码 RGB 为潜变量 token）+ 冻结文本编码器
- **核心模块**: [[Tri-MoT]] 三专家分支（视频 / 运动 / 动作）并行训练，通过结构化注意力解耦推理
- **输出**: [[Action Chunking|动作块]] $A \in \mathbb{R}^{H_A \times D_A}$
- **隐藏维度**: 1,024；动作/运动 horizon $H_A = H_M = 32$

### 核心模块

#### 模块 1: 三维运动表征

**设计动机**: 利用[[点追踪|持续点对应关系]]捕获时序三维运动，以 B-spline 压缩表征降低预测难度。

**具体实现**:
- 在主摄像头图像上铺设 20×20 = Q = 400 个锚点网格
- 每个锚点的位置为 $p_{q,t} = (u_{q,t}, v_{q,t}, d_{q,t})$：归一化图像坐标（$[-1,1]$）与深度
- 以当前帧为参考计算位移 $\Delta p_{q,t}$，并统一转换到参考相机坐标系以消除相机运动
- 仿真标注：通过几何与实体姿态直接计算；真实世界标注：通过 [[TraceForge]] / [[TraceGen]] 流水线生成
- 每个锚点的运动序列用 $K=10$ 个三次 B-spline 控制点压缩

#### 模块 2: Tri-MoT 结构化注意力

**设计动机**: 让运动专家监督塑造动作专家的视觉表征，同时确保动作专家在推理时不依赖运动分支。

**具体实现**:
- **当前视频 token**: 仅与自身交互（因果自注意力）
- **未来视频 / 运动 / 动作 token**: 各自与本组内部 + 当前视频 token 交互
- **关键约束**: 动作 token 不与运动 token 交互，确保推理时可直接移除运动分支
- 运动专家用 2×2 stride-2 卷积将 B-spline 系数嵌入，每个 token 打包 4 条轨迹，配合 Fourier 位置编码

---

## 关键公式

### 公式 1: [[Action Chunking|策略定义]]

$$
A \sim \pi_\theta(\cdot \mid I_0, \ell, s_0), \quad A \in \mathbb{R}^{H_A \times D_A}
$$

**含义**: 策略从当前 RGB 观测、语言指令和本体感知状态采样动作块。

**符号说明**:
- $I_0$: 当前 RGB 观测
- $\ell$: 语言指令
- $s_0$: 本体感知状态
- $H_A$: 动作 horizon（32）
- $D_A$: 动作维度

---

### 公式 2: [[点追踪|运动位移定义]]

$$
\Delta p_{q,t} = p_{q,t} - p_{q,0}, \quad t = 0, \ldots, H_M
$$

**含义**: 以当前帧为基准计算每个锚点的三维运动位移，初始位移为零。

**符号说明**:
- $p_{q,t} = (u_{q,t}, v_{q,t}, d_{q,t})$: 锚点 $q$ 在时刻 $t$ 的归一化图像坐标和深度
- $H_M$: 运动 horizon（32）
- $\Delta p_{q,0} = 0$: 当前帧位移为零

---

### 公式 3: [[B-spline|B-spline 运动压缩]]

$$
\widetilde{\Delta P}_q \approx B C_q, \quad C_q \in \mathbb{R}^{K \times 3}
$$

**含义**: 将缩放后的运动序列用 B-spline 基矩阵分解为 $K$ 个控制点，实现紧凑化表征。

**符号说明**:
- $\widetilde{\Delta P}_q$: 锚点 $q$ 的缩放后运动序列
- $B$: 固定 B-spline 基矩阵
- $C_q$: $K=10$ 个控制点（每个锚点）

---

### 公式 4: [[Flow Matching|流匹配插值]]

$$
x_{r,\sigma_r} = (1-\sigma_r) x_r + \sigma_r \epsilon_r, \quad v_r^* = \epsilon_r - x_r
$$

**含义**: 对每个分支的目标加入线性插值噪声，目标速度场为噪声与干净目标之差。

**符号说明**:
- $r \in \{V, M, A\}$: 分支（视频 / 运动 / 动作）
- $x_r$: 干净目标
- $\epsilon_r$: 高斯噪声
- $\sigma_r \in [0,1]$: 噪声水平
- $v_r^*$: 目标速度

---

### 公式 5: [[Flow Matching|流匹配损失]]

$$
\mathcal{L}_r^{\mathrm{FM}} = \mathbb{E}\left[ w_r(\sigma_r) \| v_{\theta,r} - v_r^* \|^2 \right]
$$

**含义**: 每个分支的流匹配损失，以噪声水平相关的权重 $w_r(\sigma_r)$ 加权。

**符号说明**:
- $w_r(\sigma_r)$: 噪声水平相关权重函数
- $v_{\theta,r}$: 模型预测速度

---

### 公式 6: 运动平衡损失

$$
\mathcal{L}_M = \mathbb{E}\left[ w_M(\sigma_M) \left( \alpha \ell_{\mathrm{mov}} + (1-\alpha) \ell_{\mathrm{stat}} \right) \right]
$$

**含义**: 对运动锚点和静止锚点分别计算均值系数速度误差并加权平衡，避免静态主导造成的梯度偏置。

**符号说明**:
- $\ell_{\mathrm{mov}}$: 运动锚点的均值系数速度误差
- $\ell_{\mathrm{stat}}$: 静止锚点的均值系数速度误差
- $\alpha$: 两组的平衡权重

---

### 公式 7（无编号）: 净系数估计

$$
\hat{C}_0 = C_{\sigma_M} - \sigma_M v_{\theta,M}
$$

**含义**: 从当前噪声状态和预测速度估计干净 B-spline 控制点，用于静止惩罚项。

**符号说明**:
- $\hat{C}_0$: 估计的干净控制点，通过 $B\hat{C}_0$ 解码为预测轨迹 $\widehat{\Delta p}_{q,t}$

---

### 公式 8: 静止运动惩罚

$$
\mathcal{L}_{\mathrm{zero}} = \mathbb{E}_{\mathrm{stat}} \left[ \underset{q \in \mathcal{S}}{\mathrm{mean}} \; \underset{t \in \mathcal{T}_q}{\mathrm{mean}} \; \rho_\beta\!\left(\widehat{\Delta p}_{q,t}\right) \right]
$$

**含义**: 对静止锚点在预测轨迹上施加 Smooth-L1 惩罚，防止模型对静止区域预测出虚假运动。

**符号说明**:
- $\mathcal{S}$: 静止锚点集合
- $\mathcal{T}_q$: 锚点 $q$ 的未来时刻集合
- $\rho_\beta$: 通道均值 Smooth-L1 惩罚项

---

### 公式 9: [[Flow Matching|联合训练目标]]

$$
\mathcal{L} = \lambda_V \mathcal{L}_V + \lambda_M \mathcal{L}_M + \lambda_{\mathrm{zero}} \mathcal{L}_{\mathrm{zero}} + \lambda_A \mathcal{L}_A
$$

**含义**: 视频预测、平衡运动、静止惩罚、动作生成四个分支的加权求和联合损失。

**符号说明**:
- $\lambda_\bullet$: 各分支损失权重
- $\mathcal{L}_V$: 视频分支流匹配损失
- $\mathcal{L}_A$: 动作分支流匹配损失
- 训练开始时先 2,000 步预热视频和运动专家，再加入动作专家

---

## 关键图表

### Figure 1: 系统概览

![Figure 1](https://arxiv.org/html/2610.36840v1/motionwam_teaser.png)

**说明**: Trace-WAM 总览图，包含三个部分：(左) 运动标注示意（仿真几何标注 vs 真实世界 TraceForge 标注），(中) [[Tri-MoT]] 三分支训练架构与单分支推理对比，(右) RoboCasa-GR1、LIBERO-Plus 等基准上的主要实验结果和真实机器人执行成功率。

---

### Figure 2: 深度通道验证与深度消融

（图片未收录于 arXiv HTML，详见项目主页）

**说明**: (左) 在 32 步轨迹上统计深度变化存活率曲线，约 30% 的锚点发生显著深度变化，表明深度维度对操作任务具有信息量；(右) 深度消融实验柱状图，去掉深度通道（仅 2D 运动）使 LIBERO-Plus 平均成功率从 84.5% 降至 82.4%（-2.1%）。

---

### Figure 3: Tri-MoT 架构与注意力掩码

![Figure 3](https://arxiv.org/html/2610.36840v1/motionwam_architecture.png)

**说明**: 展示 [[Tri-MoT]] 的详细结构与结构化注意力机制。当前视频 token 仅自注意力，未来视频/运动/动作 token 各自可看到当前视频，动作 token 不与运动 token 交互（蓝色×区域）。训练时三分支并行，推理时仅保留当前视频 + 动作分支。

---

### Figure 4: RoboCasa-GR1 少样本 OOD 泛化

（图片未收录于 arXiv HTML，详见项目主页）

**说明**: 在 RoboCasa-GR1 基准上对多种方法进行少样本 OOD 泛化测试的柱状图。Trace-WAM 在平均 OOD 成功率 58.6% 上领先于 [[Fast-WAM]]（44.8% 少样本）和 Joint-WAM（54.8% 少样本），展示了 3D 运动监督在分布外泛化上的优势。

---

### Figure 5: 真实机器人实验（Piper 机械臂）

![Figure 5](https://arxiv.org/html/2610.36840v1/piper_realworld_overview.png)

**说明**: Piper 机械臂真实环境实验设置与代表性执行帧示意。三个任务各 50 条演示、20 次测试：Place Bread in Bowl (85% vs 80%)、Place and Close Drawer (95% vs 55%)、Wipe Blackboard (75% vs 55%)，Trace-WAM 平均 85.0% vs Fast-WAM 63.3%（+21.7%）。

---

### Figure 6: 定性运动预测可视化（LIBERO-Plus）

![Figure 6](https://arxiv.org/html/2610.36840v1/motion_prediction_visualization.png)

**说明**: 在 LIBERO-Plus 任务上对 32 步 horizon 的 B-spline 运动预测进行可视化。展示 [[Tri-MoT]] 运动专家能够准确预测机器人末端执行器和任务相关物体（如杯子、容器）的三维运动轨迹，而背景静止锚点保持接近零位移（静止惩罚有效）。

---

### Table 1: RoboCasa-GR1 域内性能（成功率 %）

| 方法 | 预训练 | 少样本 | 全数据 |
|------|--------|--------|--------|
| GR00T N1.6 | ✓ | — | 47.6 |
| StarVLA-OFT | × | — | 48.8 |
| StarVLA-α | × | — | 53.8 |
| ABot-M0 | ✓ | — | 58.3 |
| RLDX-1 | ✓ | — | 58.7 |
| JoyAI-RA | ✓ | — | 63.2 |
| DIAL | × | 58.3 | 70.2 |
| LAWA w/ ego | ✓ | 65.6 | 80.8 |
| LAWA w/o ego | × | 59.7 | 76.3 |
| [[Fast-WAM]] | × | 44.8 | 74.6 |
| Joint-WAM | × | 54.8 | 78.3 |
| **Trace-WAM (Ours)** | × | **63.6** | **80.3** |

**关键发现**: 少样本场景下 Trace-WAM 提升最显著（+18.8% vs Fast-WAM），表明 3D 运动监督在数据稀缺时尤为有效。

---

### Table 2: LIBERO 域内性能（成功率 %）

| 方法 | P.T. | Spatial | Object | Goal | Long | Avg. |
|------|------|---------|--------|------|------|------|
| OpenVLA | ✓ | 84.7 | 88.4 | 79.2 | 53.7 | 76.5 |
| OpenVLA-OFT | ✓ | 97.6 | 98.4 | 97.9 | 94.5 | 97.1 |
| π0 | ✓ | 96.8 | 98.8 | 95.8 | 85.2 | 94.2 |
| π0.5 | ✓ | 98.8 | 98.2 | 98.0 | 92.4 | 96.9 |
| LingBot-VA | ✓ | 98.5 | 99.6 | 97.2 | 98.5 | 98.5 |
| [[Fast-WAM]] | × | 97.2 | 100 | 97.0 | 95.2 | 97.6 |
| Joint-WAM | × | 99.6 | 99.4 | 98.2 | 96.8 | 98.5 |
| [[GeoSem-WAM]] | × | 99.0 | 100.0 | 98.2 | 97.0 | 98.6 |
| **Trace-WAM (Ours)** | × | 98.2 | 99.8 | 99.2 | 97.6 | **98.7** |

**关键发现**: LIBERO 各子集接近饱和，Trace-WAM 以 98.7% 平均成功率与最优方法持平，Long 子集（97.6%）相对 Fast-WAM（95.2%）提升最明显。

---

### Table 3: LIBERO → LIBERO-Plus 零样本迁移（成功率 %）

| 方法 | P.T. | Camera | Robot | Language | Light | Background | Noise | Layout | Avg. |
|------|------|--------|-------|----------|-------|------------|-------|--------|------|
| OpenVLA-OFT | ✓ | 56.4 | 31.9 | 79.5 | 88.7 | 93.3 | 75.8 | 74.2 | 69.6 |
| π0-FAST | ✓ | 65.1 | 21.6 | 61.0 | 73.2 | 73.2 | 74.4 | 68.8 | 61.6 |
| [[Fast-WAM]] | × | 40.0 | 68.5 | 65.6 | 89.8 | 66.9 | 63.3 | 76.8 | 66.2 |
| Joint-WAM | × | 59.4 | 78.3 | 92.7 | 95.7 | 65.2 | 75.6 | 84.5 | 78.7 |
| [[GeoSem-WAM]] | × | — | — | — | — | — | — | — | — |
| ImageWAM | × | 80.8 | 50.3 | 91.4 | 98.1 | 85.5 | 93.8 | 80.5 | 83.1 |
| **Trace-WAM (Ours)** | × | 66.0 | **87.4** | 91.9 | 97.2 | 76.1 | 88.6 | 85.7 | **84.5** |

**关键发现**: LIBERO-Plus 更能体现泛化能力差异。Trace-WAM 在 7 种扰动类型上全面超越 Fast-WAM，Robot（+18.9%）和 Noise（+25.3%）扰动类型提升最显著，表明 3D 运动监督能帮助策略关注几何不变的运动规律。

---

### Table 4: 消融实验（LIBERO-Plus，成功率 %）

| 变体 | Camera | Robot | Language | Light | Background | Noise | Layout | Avg. |
|------|--------|-------|----------|-------|------------|-------|--------|------|
| *运动监督消融* | | | | | | | | |
| w/o Motion FM-split | 55.7 | 84.5 | 89.2 | 96.1 | 70.5 | 85.1 | 83.7 | 80.4 |
| w/o Static-Motion Penalty | 60.9 | 86.9 | 93.4 | 97.3 | 75.3 | 88.1 | 85.0 | 83.6 |
| *运动条件消融* | | | | | | | | |
| Motion Inject（注入） | 61.7 | 82.8 | 93.4 | 96.3 | 73.1 | 81.0 | 84.9 | 81.6 |
| 视频-运动双向注意力 | 54.4 | 83.5 | 94.0 | 96.2 | 68.3 | 82.3 | 84.8 | 80.3 |
| *运动表征消融* | | | | | | | | |
| w/o Depth（2D 运动）| 65.1 | 86.3 | 91.9 | 95.4 | 70.3 | 82.6 | 85.8 | 82.4 |
| w/o B-spline（原始运动）| 64.2 | 85.5 | 91.3 | 97.2 | 75.1 | 86.8 | 85.4 | 83.4 |
| **Trace-WAM（完整）** | **66.0** | **87.4** | **91.9** | **97.2** | **76.1** | **88.6** | **85.7** | **84.5** |

**关键发现**: 
- 去掉运动/静止 FM-split 损失最大（-4.1%），说明平衡两类锚点的监督信号至关重要
- 使用双向注意力让动作专家直接消费运动特征反而更差（-4.2%），证明"运动仅影响视觉表征"的设计是正确的
- 深度通道贡献 -2.1%，B-spline 压缩贡献 -1.1%，两者均有价值

---

### Table 5: 运动预测精度（LIBERO-Plus，越低越好）

| 指标 | 表征方式 | 8 步 | 16 步 | 32 步 |
|------|---------|------|-------|-------|
| 图像平面 ADE (px) | 原始运动 | 0.92 | 1.25 | 1.87 |
| 图像平面 ADE (px) | **B-spline** | **0.67** | **0.95** | **1.43** |
| 深度 MAE (mm) | 原始运动 | 1.634 | 2.297 | 3.427 |
| 深度 MAE (mm) | **B-spline** | **1.277** | **1.808** | **2.768** |

**关键发现**: B-spline 压缩在所有时间步和指标上均优于直接预测原始运动，证明平滑连续的参数化有助于运动预测精度。

---

## 实验

### 数据集

| 数据集 | 特点 | 用途 |
|--------|------|------|
| [[LIBERO]] | 4 套子集（Spatial/Object/Goal/Long），100 任务 | 域内训练 + 测试 |
| [[LIBERO-Plus]] | 7 种 OOD 扰动（摄像头/机器人/语言/光照/背景/噪声/布局）| 零样本 OOD 测试 |
| [[RoboCasa-GR1]] | GR1 人形机器人操作，少样本与全数据两种设置 | 域内 + 少样本 OOD |
| 真实世界（Piper 臂）| 3 任务，50 条演示/任务，20 次测试/任务 | 真实机器人验证 |

### 实现细节

- **Backbone**: 冻结因果视频 VAE（编码 RGB → 潜变量 token）+ 冻结文本编码器
- **视频下采样**: 4× → 9 帧/chunk
- **运动锚点**: 400 个（20×20 网格）
- **B-spline 控制点**: $K=10$，预测 $H_M=32$ 步
- **训练**: 先 2,000 步预热视频 + 运动专家，再联合训练动作专家；VAE 和文本编码器全程冻结
- **真实机器人**: AgileX Piper 机械臂，[[TraceForge]]/[[TraceGen]] 生成运动标注

### 可视化结果

LIBERO-Plus 上的 B-spline 运动预测（Figure 6）显示：运动专家能准确预测机器人末端执行器的弧形运动轨迹，同时将背景静止锚点维持在接近零位移。这种对称的运动感知与静止感知对比验证了静止运动惩罚（$\mathcal{L}_{\mathrm{zero}}$）的有效性。

---

## 批判性思考

### 优点

1. **推理零开销**: 运动分支仅在训练时使用，推理与 [[Fast-WAM]] 完全相同，工程友好
2. **3D 感知的有效性**: 通过深度消融和 B-spline 消融定量验证每个设计选择，结论清晰
3. **强泛化证据**: LIBERO-Plus 全 7 种扰动类型均超越基线，且真实机器人实验也验证了提升

### 局限性

1. **依赖运动标注**: 真实世界需要 [[TraceForge]] / [[TraceGen]] 流水线生成运动标注，增加数据准备复杂度
2. **固定锚点分辨率**: 20×20 均匀网格可能错过任务关键但分布集中的小物体（如螺丝钉）
3. **遮挡处理**: 被遮挡点的运动标注方式未详述，实际场景中遮挡可能导致标注噪声

### 潜在改进方向

1. **自适应锚点采样**: 基于任务语言指令或注意力图动态放置锚点，聚焦关键区域
2. **在线运动标注**: 探索无需离线 TraceForge 的自监督运动估计方案
3. **多摄像头融合**: 将多视角的 3D 运动信息进行一致性融合以提升深度估计鲁棒性

### 可复现性评估

- [ ] 代码开源（项目主页有 GitHub 链接但截至论文发布尚未发布）
- [ ] 预训练模型（未发布）
- [x] 训练细节完整（架构维度、超参数、训练步数均明确）
- [x] 数据集可获取（LIBERO、RoboCasa 均为公开数据集）

---

## 关联笔记

### 基于

- [[Fast-WAM]]: 推理架构的直接基础（Fast-WAM 推理路径）
- [[Flow Matching]]: 所有分支采用流匹配训练目标
- [[TraceForge]] / [[TraceGen]]: 真实世界运动标注流水线
- [[MoT]]: Mixture-of-Transformer 专家路由机制

### 对比

- [[Fast-WAM]]: 最直接的消融基线，Trace-WAM 是其运动监督增强版
- [[GeoSem-WAM]]: 同样使用辅助几何监督（静态深度/分割），但缺乏时序运动信息
- [[LAWA]]: 使用 ego-centric 视频增强，在全数据 RoboCasa-GR1 上略优（80.8% vs 80.3%）

### 方法相关

- [[Tri-MoT]]: 本文提出的三分支 MoT 架构
- [[World Action Model]]: 上位概念
- [[B-spline]]: 运动轨迹压缩方法
- [[Action Chunking]]: 动作输出格式
- [[点追踪]]: 持续点对应关系的追踪技术

### 硬件/数据相关

- [[LIBERO]]: 主要域内评测基准
- [[LIBERO-Plus]]: OOD 泛化评测基准
- [[RoboCasa-GR1]]: 人形机器人操作基准
- [[AgileX PiperX]]: 真实机器人实验平台

---

## 速查卡片

> [!summary] Trace-WAM (arXiv 2610.36840)
> - **核心**: 以未来 3D 运动轨迹（图像位移 + 深度）为辅助监督，引导 WAM 表征学习
> - **方法**: Tri-MoT 三分支 MoT；运动 token 训练时影响视觉表征、推理时移除；B-spline 压缩轨迹
> - **结果**: LIBERO-Plus 84.5%（+18.3% vs Fast-WAM），RoboCasa-GR1 少样本 63.6%（+18.8%），真实 Piper 85.0%（+21.7%）
> - **代码**: https://trace-wam.github.io

---

*笔记创建时间: 2026-10-29*
