---
title: "Perceive, Predict, and Plan: a Unified World Action Model for Egocentric Dexterous Manipulation"
method_name: "P3WAM"
authors: [Zirui Wu, Chen Guo, Tianjian Jiang, Zeren Jiang, Zeming Li, Lingchen Yang, Longteng Duan, Ying Xue, Handi Yin, Shenyuan Gao, Martin R. Oswald, Jun Zhang, Ping Tan, Jie Song]
year: 2026
venue: arXiv
tags: [world-action-model, egocentric-manipulation, video-diffusion, hand-reconstruction, dexterous-robot]
zotero_collection: Robotics/World Model
image_source: online
arxiv_html: https://arxiv.org/html/2610.22363v1/
created: 2026-10-17
---

# 论文笔记：Perceive, Predict, and Plan: a Unified World Action Model for Egocentric Dexterous Manipulation

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | HKUST (GZ)、Primatix、ETH Zurich、VGG Oxford、HKUST、University of St. Gallen |
| 日期 | October 2026 |
| 项目主页 | [p3-wam.pages.dev](https://p3-wam.pages.dev/) |
| 对比基线 | [[HaMeR]]、[[WiLoR]]、[[HaWoR]]、[[Dyn-HaMR]] |
| 链接 | [arXiv](https://arxiv.org/abs/2610.22363) / Code（forthcoming）|

---

## 一句话总结

> P3-WAM 用单一 [[Video Diffusion Transformer]] 统一处理自我中心灵巧操作的三个任务——感知（3D 手部重建）、预测（动作条件视频生成）、规划（联合生成动作与视觉未来）——三任务联合训练相互增益。

---

## 核心贡献

1. **统一三任务模型**: 单一模型在同一框架内处理 Perceive（视频→手部+相机轨迹）、Predict（动作→未来视频）、Plan（初始帧+文本+目标帧→动作+视频），三个模式共享权重。
2. **混合变换器架构**: 基于 [[Wan 2.2 A14B]] 扩散变换器，采用 [[Mixture-of-Transformers]] 双流设计：视频流与动作流共享基础权重，动作分支附加 [[LoRA]] 适配（rank 128），通过双向交叉注意力对齐视觉与动作表示。
3. **统一训练提升全部能力**: 与单任务训练相比，统一训练让 Perceive MPJPE 下降 23.5%（113.5→86.9 mm），Predict PSNR 提升 0.78 dB，Plan VLM 评分全面提升。

---

## 问题背景

### 要解决的问题

自我中心灵巧操作场景中，感知（3D 手部运动重建）、预测（给定动作生成未来视频）和规划（给定初始观测和指令生成动作及其视觉结果）三个任务通常被独立处理，无法共享知识。

### 现有方法的局限

- **感知侧**：[[HaMeR]]、[[WiLoR]]、[[HaWoR]] 等方法仅做手部重建，忽视了动作与视觉的联合建模；[[Dyn-HaMR]] 虽引入时序动态但相机空间误差仍大（231.54 mm）。
- **预测侧**：现有动作条件视频生成方法未利用感知和规划数据，生成质量受限（backbone PSNR 仅 15.93 dB）。
- **规划侧**：世界动作模型（WAM）研究多关注宏观机器人操作，缺乏对灵巧手操作的专项建模。

### 本文的动机

感知、预测、规划本质上依赖相同的"动作→视觉结果"知识：感知任务提供真实手部轨迹监督，预测任务提供视觉生成能力，规划任务整合两者。因此三任务联合训练可以互相增益，形成更强大的世界动作模型。

---

## 方法详解

### 模型架构

P3-WAM 采用 **双流 [[Video Diffusion Transformer]]** 架构，基于 [[Wan 2.2 A14B]]（Wan2.2-Fun-A14B-Control）预训练权重：

- **输入**: 初始帧 $I_0$（始终可观测）+ 文本指令 $l$（可选）+ 目标帧 $I_T$（仅 Plan 模式）
- **视频流**: 处理 RGB 视频潜变量 $\mathbf{z}_v$，用 [[Plücker Embeddings]] 相机射线图和手部骨架作为空间条件
- **动作流**: 处理手部状态 $H_t$（[[MANO]] 参数）和相机状态 $P_t$，编码为紧凑动作 token（每帧一个相机 token + 每手一个 token）
- **架构核心**: [[Mixture-of-Transformers]] 双流共享基础权重，动作分支附加 [[LoRA]]（rank 128）；双流通过双向注意力联合建模
- **输出**: 去噪视频 + 手部 MANO 参数序列 + 相机位姿序列
- **总参数**: ~14B（基于 Wan2.2-Fun-A14B-Control）

### 核心模块

#### 模块 1: 随机模态遮蔽（Randomized Modality Masking）

**设计动机**: 通过控制哪些帧和动作状态是"可观测"的，在同一框架内表达三种不同任务模式。

**具体实现**:
- 第一帧始终可观测（$\bar{m}_{v,0} = 1$）
- Perceive：视频全部可观测，动作全部缺失（预测动作）
- Predict：动作全部可观测，未来帧缺失（生成视频）
- Plan：仅初始帧可观测，动作和视频均需生成；目标帧若给定则作为最终帧条件
- 缺失动作用可学习查询嵌入 $\mathbf{e}^{\text{query}}_{a,t}$ 代替

#### 模块 2: 动作自编码器（Action Autoencoder）

**设计动机**: 将高维 [[MANO]] 参数和相机位姿压缩为低维动作 token，与视频 token 在同一注意力空间对齐。

**具体实现**:
- 冻结编码器将每帧 MANO 旋转、形状参数 $\bm{\beta}$、根位置、手部存在性和相机位姿编码为动作 latent $\mathbf{z}_a$
- 编码后每帧只有 2 个 token（左右手各一 + 相机），极大压缩了动作序列长度
- 解码器在推理时将预测的 latent 还原为 MANO 参数和相机轨迹

#### 模块 3: 目标帧条件（Visual Goal Conditioning）

**设计动机**: 为 Plan 模式提供明确的终态约束，引导长时序规划。

**具体实现**:
- 目标帧 $I_T$ 编码为终端视频 latent，作为最终帧的可观测条件
- 多阶段任务：自动将长任务拆分为规划窗口，用视觉子目标衔接
- 无目标图像时：仅基于文本指令和初始帧生成（见 Figure 13）

---

## 关键公式

### 公式 1: [[Flow Matching|视频流匹配损失]]

$$
\mathcal{L}_{\mathrm{video}} = \mathbb{E}_{\tau, \bm{\epsilon}, \mathbf{m}} \left[ \frac{w(\tau)}{\max(1, |\mathcal{U}_v|)} \sum_{k \in \mathcal{U}_v} \left\| \hat{\mathbf{v}}_{\theta,k} - (\bm{\epsilon}_k - \mathbf{z}_{v,k}) \right\|_2^2 \right]
$$

**含义**: 对未观测视频帧集合 $\mathcal{U}_v$ 计算预测速度场与目标速度场的均方误差，归一化到未观测帧数。

**符号说明**:
- $\tau \sim \mathcal{U}(0,1)$：[[Flow Matching]] 时间步，控制噪声程度
- $w(\tau)$：时间步权重函数
- $\bm{\epsilon}_k$：纯噪声
- $\mathbf{z}_{v,k}$：第 $k$ 帧视频 latent（干净）
- $\hat{\mathbf{v}}_{\theta,k}$：网络预测的速度场
- $\mathcal{U}_v$：未观测帧的下标集合

### 公式 2: [[Flow Matching|总优化目标]]

$$
\mathcal{L} = \mathcal{L}_{\mathrm{video}} + \mathcal{L}_{\mathrm{param}} + \mathcal{L}_{\mathrm{spatial}} + \mathcal{L}_{\mathrm{temporal}} + \mathcal{L}_{\mathrm{presence}} + \mathcal{L}_{\mathrm{camera}}
$$

**含义**: 六个损失项之和，联合优化视频生成质量和动作预测精度。

**符号说明**:
- $\mathcal{L}_{\mathrm{video}}$：视频流匹配损失（公式 1）
- $\mathcal{L}_{\mathrm{param}}$：MANO 旋转和形状参数损失
- $\mathcal{L}_{\mathrm{spatial}}$：3D 和 2D 关节空间损失
- $\mathcal{L}_{\mathrm{temporal}}$：时序运动平滑损失（关节运动 + 轨迹）
- $\mathcal{L}_{\mathrm{presence}}$：手部存在性二元交叉熵损失
- $\mathcal{L}_{\mathrm{camera}}$：相机旋转、平移、FOV 损失

### 公式 3: [[Flow Matching|带遮蔽的视频输入]] (Eq. 3)

$$
\tilde{\mathbf{z}}_{v,\ell} = \bar{m}_{v,\ell}\, \mathbf{z}_{v,\ell} + (1 - \bar{m}_{v,\ell}) \left[ (1-\tau)\, \mathbf{z}_{v,\ell} + \tau\, \bm{\epsilon}_\ell \right]
$$

$$
\tau_{v,\ell} = (1 - \bar{m}_{v,\ell})\, \tau
$$

**含义**: 可观测帧保持原始 latent，未观测帧按时间步 $\tau$ 添加噪声；网络对未观测帧预测速度场，对可观测帧感知为"无噪声"。

**符号说明**:
- $\bar{m}_{v,\ell} \in \{0, 1\}$：第 $\ell$ 帧的可观测性掩码（1 = 可观测）
- $\bm{\epsilon}_\ell$：高斯噪声
- $\tau_{v,\ell}$：第 $\ell$ 帧实际使用的噪声程度

### 公式 4: [[MANO|带遮蔽的动作输入]] (Eq. 4)

$$
\tilde{\mathbf{z}}_{a,t} = \mathbf{m}_{a,t} \odot \mathbf{z}_{a,t} + (1 - \mathbf{m}_{a,t}) \odot \mathbf{e}^{\mathrm{query}}_{a,t}
$$

**含义**: 可观测动作使用真实 latent，缺失动作替换为可学习查询嵌入，在统一框架内表示"未知动作"。

**符号说明**:
- $\mathbf{m}_{a,t} \in \{0,1\}^d$：动作可观测性掩码向量
- $\mathbf{z}_{a,t}$：第 $t$ 帧真实动作 latent
- $\mathbf{e}^{\mathrm{query}}_{a,t}$：可学习查询嵌入（代表缺失动作）

### 公式 5: [[MANO|MANO 参数损失]] (Eq. 5)

$$
\mathcal{L}_{\mathrm{param}} = \left\langle \frac{1}{N_j} \sum_{j=1}^{N_j} \lambda_j\, d_R(\hat{\mathbf{R}}_{t,j}, \mathbf{R}_{t,j}) \right\rangle + \lambda_{\mathrm{shape}} \left\langle \rho(\hat{\bm{\beta}} - \bm{\beta}) \right\rangle
$$

**含义**: 对每个关节旋转计算测地线距离，加上形状参数的鲁棒回归损失。

**符号说明**:
- $d_R(\cdot, \cdot)$：SO(3) 上的测地线距离
- $\hat{\mathbf{R}}_{t,j}, \mathbf{R}_{t,j}$：第 $t$ 帧第 $j$ 关节的预测和真实旋转矩阵
- $\rho(\cdot)$：鲁棒损失函数（如 Huber）
- $\hat{\bm{\beta}}, \bm{\beta}$：预测和真实 MANO 形状参数

### 公式 6: [[MPJPE|空间关节损失]] (Eq. 6)

$$
\mathcal{L}_{\mathrm{spatial}} = \lambda_{\mathrm{3D}} \left\langle \rho(\hat{\mathbf{J}}_t - \mathbf{J}_t) \right\rangle + \lambda_{\mathrm{2D}} \left\langle \rho\!\left( \pi_{K_t}(\hat{\mathbf{J}}_t) - \pi_{K_t}(\mathbf{J}_t) \right) \right\rangle
$$

**含义**: 同时惩罚 3D 关节位置误差（对应 MPJPE）和 2D 投影误差，确保姿态在三维和图像空间均准确。

**符号说明**:
- $\hat{\mathbf{J}}_t, \mathbf{J}_t$：预测和真实 3D 关节位置
- $\pi_{K_t}$：相机内参 $K_t$ 定义的投影函数

### 公式 7: 时序运动损失 (Eq. 7)

$$
\mathcal{L}_{\mathrm{temporal}} = \lambda_{\mathrm{art}} \mathcal{L}_{\mathrm{art}} + \lambda_{\mathrm{traj}} \mathcal{L}_{\mathrm{traj}}
$$

**含义**: 联合约束关节运动平滑性（$\mathcal{L}_{\mathrm{art}}$，关节角速度）和根轨迹平滑性（$\mathcal{L}_{\mathrm{traj}}$，根位置速度）。

### 公式 8: 手部存在性损失 (Eq. 8)

$$
\mathcal{L}_{\mathrm{presence}} = -\lambda_{\mathrm{presence}} \left\langle p_t \log \hat{p}_t + (1 - p_t) \log(1 - \hat{p}_t) \right\rangle
$$

**含义**: 二元交叉熵损失，预测每帧左右手是否可见，处理手部出入画面的情况。

**符号说明**:
- $p_t \in \{0,1\}$：真实手部存在性标签
- $\hat{p}_t \in (0,1)$：预测的手部存在概率

### 公式 9: 相机损失 (Eq. 9)

$$
\mathcal{L}_{\mathrm{camera}} = \lambda_{\mathrm{cam},R} \left\langle \|\hat{\mathbf{R}}_t^{\mathrm{cam}} - \mathbf{R}_t^{\mathrm{cam}}\|_F^2 \right\rangle + \lambda_{\mathrm{cam},t} \left\langle \|\hat{\mathbf{t}}_t^{\mathrm{cam}} - \mathbf{t}_t^{\mathrm{cam}}\|_2^2 \right\rangle + \lambda_{\mathrm{fov}} \left\langle \|\hat{\bm{\phi}}_t - \bm{\phi}_t\|_2^2 \right\rangle
$$

**含义**: 分别约束相机旋转（Frobenius 范数）、平移（L2）和视场角 FOV（L2）的预测误差。

**符号说明**:
- $\hat{\mathbf{R}}_t^{\mathrm{cam}}, \mathbf{R}_t^{\mathrm{cam}}$：预测和真实相机旋转矩阵
- $\hat{\mathbf{t}}_t^{\mathrm{cam}}, \mathbf{t}_t^{\mathrm{cam}}$：预测和真实相机平移（相对第一帧）
- $\hat{\bm{\phi}}_t, \bm{\phi}_t$：预测和真实视场角参数

### 公式 10: [[Classifier-Free Guidance (CFG)|动作引导]] (Eq. 10, Appendix F.4)

$$
\hat{\mathbf{v}}_\gamma = \hat{\mathbf{v}}_\emptyset + \gamma \left( \hat{\mathbf{v}}_{\mathrm{act}} - \hat{\mathbf{v}}_\emptyset \right)
$$

**含义**: 推理时用动作引导系数 $\gamma$ 控制动作条件的强度；$\gamma=0$ 为无条件生成，$\gamma=1$ 为完全条件生成。

**符号说明**:
- $\hat{\mathbf{v}}_\emptyset$：无动作条件下的预测速度场
- $\hat{\mathbf{v}}_{\mathrm{act}}$：有动作条件下的预测速度场
- $\gamma$：动作引导系数（类似 [[Classifier-Free Guidance (CFG)]] 的 guidance scale）

---

## 关键图表

### Figure 1: 系统概览 (Teaser)

![Figure 1](https://arxiv.org/html/2610.22363v1/01-teaser.png)

**说明**: P3-WAM 三种模式的统一展示。Perceive 模式从自我中心视频重建 3D 手部和相机轨迹；Predict 模式给定动作生成未来视频；Plan 模式从初始帧和文本指令联合生成动作和视觉序列，最终动作轨迹通过 [[Human-to-Robot Retargeting]] 部署到灵巧机器人。

### Figure 2: 训练框架（模态遮蔽示意）

*（arXiv HTML 暂无直接图片链接，详见项目主页）*

**说明**: 展示三种模式如何通过随机模态遮蔽统一训练：Perceive = 视频可观测 + 动作缺失；Predict = 动作可观测 + 未来帧缺失；Plan = 仅初始帧可观测。

### Figure 3: 方法总体架构

![Figure 3](https://arxiv.org/html/2610.22363v1/03-method-overview.png)

**说明**: 双流 [[Mixture-of-Transformers]] 架构。视频流（基于 [[Wan 2.2 A14B]] DiT）接收带噪视频 latent，动作流处理手部和相机 token；两流通过双向注意力共享信息。[[Plücker Embeddings]] 相机射线图和手部骨架作为视频流的空间条件。

### Figure 4: 视觉目标条件控制

![Figure 4](https://arxiv.org/html/2610.22363v1/04-goal-image-control.png)

**说明**: 目标图像作为空间终态约束，左侧为单阶段任务（单一目标帧），右侧为多阶段任务分解（用视觉子目标衔接多个规划窗口）。

### Figure 5: Perceive 定性对比

![Figure 5](https://arxiv.org/html/2610.22363v1/05-recon-qualitative-comparison.png)

**说明**: P3-WAM 在四个数据集上与 [[HaMeR]]、[[Dyn-HaMR]] 的定性比较，展示 P3-WAM 在遮挡手部和复杂姿态下的优势。

### Figure 6: Predict 定性对比

![Figure 6](https://arxiv.org/html/2610.22363v1/06-wm-qualitative-comparison-compact.png)

**说明**: 动作条件视频生成定性对比，P3-WAM 生成的视频帧质量更高，手部动作与条件动作一致性更好。

### Figure 7: Plan 定性对比（有目标帧）

![Figure 7](https://arxiv.org/html/2610.22363v1/07-plan-goal-only-qualitative.png)

**说明**: Plan 模式（给定目标图像）与视频生成 baseline 的对比，P3-WAM 能准确执行目标过渡和物理交互。

### Figure 8: 机器人部署

![Figure 8](https://arxiv.org/html/2610.22363v1/08-robot-deployment-compact.png)

**说明**: 端到端部署流程：初始观测帧 → P3-WAM 生成人手轨迹 → [[Human-to-Robot Retargeting]]（Do as I Do 方法）重定向到灵巧机器人手 → 开环执行。

### Figure 9: Plan 条件遮蔽与目标帧编码

*（arXiv HTML 暂无直接图片链接）*

**说明**: 展示 Plan 模式中目标图像如何编码为终端视频 latent，以及条件遮蔽的具体实现。

### Figure 10: 人工评估问卷界面

![Figure 10](https://arxiv.org/html/2610.22363v1/figures/09-questionnaire-stimulus-en.png)

**说明**: 人工评估设置——评审者看到指令、初始帧和目标帧，以及一个匿名视频，进行盲评。

### Figure 11a: 身份一致性评估界面

![Figure 11a](https://arxiv.org/html/2610.22363v1/figures/10-questionnaire-identity-en.png)

**说明**: 身份一致性维度的具体评估界面。

### Figure 11b: 多维度评分界面

![Figure 11b](https://arxiv.org/html/2610.22363v1/figures/11-questionnaire-ratings-en.png)

**说明**: 问卷评分界面，每个维度（动作过程、指令遵循、物理交互、时序一致性、目标过渡）单独评分 1-5，并提供"不可判断"选项。

### Figure 12: 单模式 vs. 统一训练定性对比

![Figure 12](https://arxiv.org/html/2610.22363v1/12-training-mode-qualitative.png)

**说明**: 定性展示统一训练对三个模式的提升——Perceive 的姿态更准确，Predict 的视频质量更高，Plan 的动作更合理。

### Figure 13: Plan 无目标帧定性对比

![Figure 13](https://arxiv.org/html/2610.22363v1/13-plan-no-goal-qualitative.png)

**说明**: 无目标图像条件下，P3-WAM 仅凭初始帧和文本指令执行规划，与视频生成 baseline 对比。

### Table 1: 统一训练 vs. 单模式训练消融（宏平均）

| 配置 | Perceive MPJPE (mm) ↓ | Perceive PA-MPJPE (mm) ↓ | Predict PSNR (dB) ↑ | Predict FVD ↓ |
|------|----------------------|--------------------------|---------------------|---------------|
| Single-mode（仅各自任务训练） | 113.5 | 12.97 | 17.36 | 1105.95 |
| **Unified（P3-WAM）** | **86.9** | **12.23** | **18.14** | **1009.84** |

**关键发现**: 统一训练相比单模式训练，[[MPJPE]] 下降 23.5%，PSNR 提升 0.78 dB，[[FVD]] 下降 8.7%，证明三任务互相增益。

### Table 2: Plan VLM 评分（统一 vs. 单模式，1-5 分）

| 配置 | 动作过程 | 指令遵循 | 物理交互 | 时序一致性 | 目标过渡 |
|------|---------|---------|---------|-----------|---------|
| Single-mode | — | — | — | 2.15 | 1.98 |
| **Unified（P3-WAM）** | — | — | — | **3.57** | **3.29** |

**关键发现**: 统一训练在所有五个维度上全面提升，时序一致性提升 +1.42，目标过渡提升 +1.31。

### Table 3: Perceive 与专用基线对比（相机空间 MPJPE，mm，宏平均）

| 方法 | Camera MPJPE-p ↓ | 4 数据集最优数 |
|------|-----------------|--------------|
| [[HaMeR]] | 145.34 | 0 |
| [[Dyn-HaMR]] | 231.54 | 0 |
| **P3-WAM（Ours）** | **63.05** | **4/4** |

**关键发现**: P3-WAM 在全部四个数据集（[[EgoDex]]、[[ARCTIC]]、[[HOT3D]]、[[EgoLive]]）上均优于专用手部重建方法，相机空间误差比 HaMeR 低 56.7%。

### Table 4: Predict 与视频生成 baseline 对比（全帧 PSNR/FVD，宏平均）

| 方法 | Full-frame PSNR (dB) ↑ | FVD ↓ |
|------|------------------------|-------|
| Wan2.2-Fun-Control（backbone）| 15.93 | — |
| **P3-WAM（Ours）** | **19.02** | **621.73** |

**关键发现**: P3-WAM 在全部四个数据集全帧 PSNR 上均达最优，相比 backbone 提升 +3.09 dB。

### Table 5: Plan 与视频生成 baseline 对比（VLM 评分）

| 方法 | 动作过程 ↑ | 指令遵循 ↑ | 物理交互 ↑ | 时序一致性 ↑ | 目标过渡 ↑ |
|------|-----------|-----------|-----------|------------|-----------|
| Best baseline | — | — | — | *最高* | — |
| **P3-WAM（Ours）** | **最高** | **最高** | **最高** | 最高 -0.033 | **最高** |

**关键发现**: P3-WAM 在动作过程、指令遵循、物理交互、目标过渡四个维度均优于所有 baseline，仅在时序/身份一致性上略低 0.033 分。

---

## 实验

### 数据集

| 数据集 | 特点 | 用途 |
|--------|------|------|
| [[EgoDex]] | 自我中心灵巧操作视频，含 3D 手部标注 | Perceive / Predict / Plan 训练和评估 |
| [[ARCTIC]] | 双手操作关节物体，含精细手部标注 | Perceive 评估 |
| [[HOT3D]] | 自我中心 3D 手部与物体交互数据集 | Perceive 评估 |
| [[EgoLive]] | 自我中心日常操作视频，真实环境多样性 | Perceive 评估 |
| Plan 评估集 | 188 视频对，17 个任务类别 | Plan 定性和 VLM 评估 |

### 实现细节

- **Backbone**: Wan2.2-Fun-A14B-Control（[[Wan 2.2 A14B]] 视频扩散变换器）
- **动作分支 LoRA rank**: 128
- **动作 token**: 每帧 2 个（左手 + 右手）+ 1 个相机 token
- **目标函数**: 公式 (2) 六项损失联合优化
- **推理**:
  - Perceive：单次前向传播（无迭代去噪）
  - Predict / Plan：迭代去噪，可观测帧 latent 固定
- **训练步数**: 对比实验在 5000 步；完整训练更多步
- **硬件**: 未明确报告（推测多卡 A100/H100）

### 可视化结果

- Perceive：明显改善遮挡和快速运动场景下的手部重建，[[Plücker Embeddings]] 相机条件有效解决相机运动歧义
- Predict：生成视频在手部动作细节和物体交互上与条件动作高度一致
- Plan：多阶段任务分解有效，视觉子目标引导下能完成需要多步骤的桌面操作

---

## 批判性思考

### 优点

1. **统一框架优雅**: 随机模态遮蔽的设计极其简洁，仅通过改变哪些帧/动作可观测，即实现三种截然不同的任务模式。
2. **预训练充分利用**: 以 [[Wan 2.2 A14B]]（14B 参数）为基础，继承了强大的视频生成先验，LoRA 适配开销小。
3. **跨任务增益得到验证**: Table 1 清楚表明联合训练对各子任务均有实质性提升，不是简单的参数共享。
4. **机器人部署闭环**: 从视频感知到动作轨迹到机器人执行，提供了完整 pipeline 展示。

### 局限性

1. **单目深度歧义**: 从单目视频恢复全局 3D 手部位置本质上存在尺度歧义，相机运动建模有助但无法完全解决。
2. **开环机器人执行**: 机器人实验采用开环执行，无法根据实际环境反馈纠正误差，鲁棒性有限。
3. **评估数据集重叠风险**: EgoDex、ARCTIC 等数据集部分用于训练，评估的泛化性需进一步验证。
4. **推理速度**: 基于 14B 参数扩散模型的迭代去噪推理速度对实时机器人应用是挑战。

### 潜在改进方向

1. 引入触觉或深度传感器提供几何约束，解决单目深度歧义。
2. 探索闭环控制，将视觉反馈与预测轨迹在线纠正结合。
3. 模型蒸馏或流步数减少（如 [[Flow Matching]] 的少步采样）加速推理以达到实时要求。

### 可复现性评估

- [ ] 代码开源（forthcoming）
- [ ] 预训练模型（未发布）
- [x] 训练细节（appendix 有较完整描述）
- [x] 数据集可获取（EgoDex、ARCTIC、HOT3D 均公开）

---

## 关联笔记

### 基于

- [[Wan 2.2 A14B]]: 基础视频扩散变换器（Wan2.2-Fun-A14B-Control）
- [[Mixture-of-Transformers]]: 双流共享权重架构设计
- [[MANO]]: 手部参数化模型，用于动作表示
- [[Flow Matching]]: 视频生成训练目标

### 对比

- [[HaMeR]]: 手部重建专用模型，Perceive baseline
- [[WiLoR]]: 手部重建专用模型，Perceive baseline
- [[HaWoR]]: 考虑时序动态的手部重建，Perceive baseline
- [[Dyn-HaMR]]: 动态场景手部重建，Perceive baseline

### 方法相关

- [[Plücker Embeddings]]: 相机射线图表示，视频流空间条件
- [[LoRA]]: 动作分支参数高效微调
- [[Classifier-Free Guidance (CFG)]]: 推理时动作引导（公式 10）
- [[Human-to-Robot Retargeting]]: 机器人部署的动作重定向方法

### 硬件/数据相关

- [[EgoDex]]: 主要训练和评估数据集
- [[ARCTIC]]: Perceive 评估数据集
- [[HOT3D]]: Perceive 评估数据集
- [[EgoLive]]: Perceive 评估数据集
- [[MPJPE]]: 手部重建评估指标
- [[FVD]]: 视频生成质量评估指标

---

## 速查卡片

> [!summary] P3-WAM (2026)
> - **核心**: 单一模型统一处理自我中心手部操作的感知、预测、规划三任务，联合训练互相增益
> - **方法**: Wan2.2-Fun-A14B 视频扩散 + [[Mixture-of-Transformers]] 双流 + 随机模态遮蔽
> - **结果**: Perceive MPJPE 63.05 mm（vs HaMeR 145.34 mm），Predict PSNR 19.02 dB，Plan 四维 VLM 最优
> - **代码**: https://p3-wam.pages.dev/（代码 forthcoming）

---

*笔记创建时间: 2026-10-17*
