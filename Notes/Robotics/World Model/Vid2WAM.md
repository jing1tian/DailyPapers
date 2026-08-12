---
title: "Vid2WAM: Distilling Video Diffusion Priors into World Action Models"
method_name: "Vid2WAM"
authors: [Chenhao Qiu, Ruixiang Wang, Runyi Zhao, Sixu Lin, Songen Gu, Shufeng Nan, Guiliang Liu, Kui Jia, Yanwei Fu, Simo Wu]
year: 2026
venue: arXiv
tags: [world-action-model, video-diffusion, knowledge-distillation, robot-manipulation, inverse-dynamics-model, novel-task-generalization, dual-channel-distillation]
zotero_collection: Robotics/World Model
image_source: online
arxiv_html: https://arxiv.org/html/2608.08558
created: 2026-08-12
---

# 论文笔记：Vid2WAM: Distilling Video Diffusion Priors into World Action Models

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Sun Yat-sen University（Kui Jia）、Fudan University（Yanwei Fu）、McGill University（Guiliang Liu）等 |
| 日期 | August 2026 |
| 项目主页 | [vid2wam-website](https://qch-fa.github.io/vid2wam-website/) |
| 对比基线 | [[Fast-WAM]]、[[π₀.₅]]、Motus |
| 链接 | [arXiv](https://arxiv.org/abs/2608.08558) / Code 暂未开放 |

---

## 一句话总结

> Vid2WAM 将大型[[视频扩散模型]]的未来预测先验蒸馏进紧凑的 [[WAM|World Action Model]]，通过**双通道**（视频潜变量 + [[IDM|逆动力学模型]]伪动作）与**源感知残差适配器**实现教师无关的高效推理，在新任务泛化上较 [[Fast-WAM]] 提升高达 12.5 个百分点。

---

## 核心贡献

1. **视频基础模型→WAM 蒸馏框架**: 突破"未来监督必须来自专家演示"的假设，用适配后的视频扩散模型离线生成任务条件 rollout 作为 student WAM 的未来监督，无需目标任务的 expert trajectory。
2. **双通道、源感知迁移策略**: 同时蒸馏未来视觉动态（future latent）和 [[Inverse Dynamics Model|IDM]] 推断的伪动作，并引入**源感知残差适配器**消除伪动作噪声对共享动作骨干的干扰。
3. **推理效率保持**: 视频教师与 IDM 仅用于离线生成，部署时完全丢弃，仅保留紧凑的 student WAM，推理延迟与 Fast-WAM 持平（~212 ms）。

---

## 问题背景

### 要解决的问题

[[World Action Model]]（WAM）利用未来预测监督来学习时空结构化的策略表征，但现有 WAM 均假设**未来监督必须来自目标任务的专家演示**——future video targets 和 action labels 都从机器人轨迹录制中取得。这使 WAM 的泛化能力受限于演示数据的规模、多样性和采集成本。

### 现有方法的局限

- **演示依赖**: Fast-WAM、Flash-WAM 等方法的未来监督均来自录制的 expert trajectory，新任务必须重新采集。
- **在线生成方法（Video Gen + IDM）**: 推理时在线生成 rollout 再用 IDM 恢复动作，延迟极高（Teacher Policy 约 4894 ms，约为 Fast-WAM 的 23×），且生成错误直接影响控制。
- **动作唯一迁移（DreamGen 等）**: 仅将生成视频通过 IDM 恢复伪动作监督策略，生成的未来帧本身被丢弃，造成信息瓶颈，伪动作噪声无法有效隔离。

### 本文的动机

大型[[视频基础模型]]已学习了丰富的物体运动、人-物交互和长时域视觉动力学知识。轻量适配后即可对新任务生成条件 rollout，扩展训练状态分布无需完整专家轨迹。将这些 rollout 同时用于：
(a) 直接监督 student 的**视频预测分支**（保留预测性先验）；
(b) 通过 IDM 恢复**具身伪动作**（提供动作 grounding）。

源感知适配器在不污染共享骨干的前提下消化伪动作噪声。

---

## 方法详解

### 模型架构

![Figure 2: Vid2WAM Overview](https://arxiv.org/html/2608.08558v1/method.png)

Vid2WAM 包含三个主要组件：**视频教师**（仅训练期使用）、**IDM**（仅训练期使用）、**student WAM**（部署期唯一模型）。

- **输入**: 语言指令 $\ell$ + 当前视觉观测 $o_0$ + 本体感知状态 $e_0$
- **Backbone**: 与 [[Fast-WAM]] 共享的视频-动作联合 DiT
- **核心模块**: [[Source-Aware Residual Adapter|源感知残差适配器]] 分离真实/伪数据域；[[Inverse Dynamics Model|IDM]] 从生成视频恢复伪动作
- **输出**: 动作块 $a_{1:H}$（推理时直接由 student WAM 生成）
- **总参数**: Student WAM ≈ 5B；视频教师 Wan2.1-14B（仅离线）

### 核心模块

#### 模块 1：视频教师适配（Teacher Adaptation）

**设计动机**: 利用[[视频扩散模型]]的运动预测和指令跟随能力，通过少量机器人演示视频做轻量微调，保留大规模预训练的先验。

**具体实现**:
- Teacher $T_\psi$ 初始化自 **LVP 检查点**（[[Wan2.1|Wan2.1-14B]] 在机器人操作数据上预训练的版本）
- 仅在有限演示视频上微调，使教师适应机器人外观和物理约束
- 微调后冻结，仅用于离线 rollout 生成

公式（Eq. 1）：

$$
\hat{v}_{1:T} = T_{\psi}(o_0, \ell)
$$

**符号说明**:
- $\hat{v}_{1:T}$：教师生成的未来视频帧序列（$T$ 帧）
- $o_0$：当前观测（初始帧）
- $\ell$：语言指令

#### 模块 2：逆动力学模型（IDM）

**设计动机**: 从生成的视频帧序列中恢复具身动作，提供动作 grounding，无需目标任务的 expert 动作标注。

**具体实现**:
- 使用 [[ResNet-50]] 骨干 + 动作/本体感知预测头
- 用 action-labeled 轨迹训练，无需任务特定语义标注，因此可用与目标任务无关的交互数据
- 训练后冻结，作为伪动作标注器

公式（Eq. 2）：

$$
(\hat{a}_{1:H}, \hat{e}_{1:H}) = \operatorname{IDM}(\hat{v}_{1:T})
$$

**符号说明**:
- $\hat{a}_{1:H}$：IDM 恢复的伪动作块（$H$ 步）
- $\hat{e}_{1:H}$：IDM 恢复的伪机器人状态序列
- $\hat{v}_{1:T}$：教师生成视频帧

**IDM 训练目标**（Eq. S4, S5）：

$$
\hat{a}_t = q_\psi\!\left(I_{t-\delta}, I_t, I_{t+\delta}\right)
$$

$$
\mathcal{L}_{\mathrm{IDM}} = \mathbb{E}\!\left[\operatorname{SmoothL1}\!\left(\frac{\hat{a}_t - \mu_a}{\sigma_a},\ \frac{a_t - \mu_a}{\sigma_a}\right)\right]
$$

**符号说明**:
- $q_\psi$：IDM 网络（ResNet-50）
- $I_{t-\delta}, I_t, I_{t+\delta}$：前后帧三元组（时间窗 $\delta$）
- $\mu_a, \sigma_a$：动作均值与标准差（归一化）

#### 模块 3：伪数据生成与标注（Pseudo-Data Pipeline）

**流程**:
1. 收集额外初始状态池（真实观测 / 仿真截图 / GPT-Image-2 合成图像）
2. 冻结教师生成视频 rollout $\hat{v}_{1:T}$
3. 用 **student VAE** 重编码到 student 潜空间，得 $\hat{z}_{1:T}$（保证监督与预测在同一潜空间）
4. IDM 推断伪动作块 $\hat{a}_{1:H}$ 和伪状态序列 $\hat{e}_{1:H}$
5. 存入离线蒸馏缓冲区：$(o_0, \ell, \hat{z}_{1:T}, \hat{a}_{1:H}, \hat{e}_{1:H})$

#### 模块 4：源感知残差适配器（Source-Aware Residual Adapter）

**设计动机**: 直接混合真实专家动作与 IDM 伪动作会引入干扰（伪动作含视频生成误差 + IDM 误差）。轻量残差适配器为两个域提供独立修正，共享动作骨干保持默认路径。

**公式**（Eq. 3）：

$$
\operatorname{Ada}_d(x) = x + \alpha\, W_{\mathrm{up}}^d\, \sigma\!\left(W_{\mathrm{down}}^d\, \operatorname{LN}(x)\right)
$$

**含义**: 为域 $d \in \{\text{real}, \text{pseudo}\}$ 的动作 token $x$ 添加瓶颈残差修正，$W_{\mathrm{up}}^d$ 初始化为零（适配器初始为恒等映射），训练时逐渐学习域特定偏置。

**符号说明**:
- $d$：数据域（`real` 或 `pseudo`）
- $W_{\mathrm{down}}^d, W_{\mathrm{up}}^d$：域特定瓶颈投影矩阵（$W_{\mathrm{up}}^d$ 零初始化）
- $\sigma$：非线性激活
- $\alpha$：残差贡献系数
- $\operatorname{LN}$：Layer Norm

> 推理时仅保留 `real` 域适配器；`pseudo` 域适配器和未来视频预测头被丢弃。

---

## 关键公式

### 公式 1：[[Flow Matching|Flow Matching 线性插值]]

$$
y_t = (1 - t)\, y + t\, \epsilon
$$

**含义**: 将干净目标 $y$ 沿时间 $t$ 线性插向高斯噪声 $\epsilon$，构造 flow matching 的中间噪声样本。

**符号说明**:
- $y$：干净目标（动作块或视频潜变量）
- $\epsilon \sim \mathcal{N}(0, I)$：高斯噪声
- $t \in (0, 1)$：噪声时间步

### 公式 2：[[Flow Matching|Flow Matching 损失]]

$$
\mathcal{L}_{\mathrm{FM}}(y, d) = \mathbb{E}_{y,\epsilon,t}\!\left[\left\|f_\theta(y_t, t, c;\, d) - (\epsilon - y)\right\|_2^2\right]
$$

**含义**: 令模型 $f_\theta$ 预测噪声-目标速度场 $(\epsilon - y)$，等价于 flow matching 的速度匹配目标；域 $d$ 决定使用哪个适配器。

**符号说明**:
- $f_\theta$：student WAM 的 DiT 预测网络
- $c$：条件信息（观测 $o_0$、指令 $\ell$、本体感知 $e_0$）
- $d$：数据域（`real` 或 `pseudo`）
- $y_t$：见 Eq. 4

### 公式 3：域分离训练损失（Eq. 6）

$$
\begin{aligned}
\mathcal{L}_{\mathrm{real}} &= \mathcal{L}_{\mathrm{FM}}(a,\, \text{real}) + \beta\, \mathcal{L}_{\mathrm{FM}}(z,\, \text{real}) \\
\mathcal{L}_{\mathrm{pseudo}} &= \mathcal{L}_{\mathrm{FM}}(\hat{a},\, \text{pseudo}) + \beta\, \mathcal{L}_{\mathrm{FM}}(\hat{z},\, \text{pseudo})
\end{aligned}
$$

**含义**: 真实域损失使用专家动作 $a$ 和真实视频潜变量 $z$；伪域损失使用 IDM 伪动作 $\hat{a}$ 和教师生成潜变量 $\hat{z}$；$\beta$ 平衡动作与视频监督比例。

**符号说明**:
- $a, z$：真实演示的动作块与视频潜变量
- $\hat{a}, \hat{z}$：伪数据缓冲区中的 IDM 伪动作与教师生成潜变量
- $\beta$：动作 vs. 视频损失权重

### 公式 4：总训练目标（Eq. 7）

$$
\mathcal{L} = \lambda_{\mathrm{real}}\, \mathcal{L}_{\mathrm{real}} + \lambda_{\mathrm{pseudo}}\, \mathcal{L}_{\mathrm{pseudo}}
$$

**含义**: 联合优化真实域和伪域损失，$\lambda$ 控制两域贡献比例。

**符号说明**:
- $\lambda_{\mathrm{real}}, \lambda_{\mathrm{pseudo}}$：真实域与伪域的损失权重

### 公式 5：教师微调目标（Eq. S1–S3，Shifted Cosine Schedule）

$$
\tau = \frac{st}{1 + (s-1)t},\quad s = 3
$$

$$
z_\tau = (1 - \tau)\, z + \tau\, \epsilon
$$

$$
\mathcal{L}_{\mathrm{teacher}} = \mathbb{E}_{z,\epsilon,t}\!\left[\left\|v_\Theta(z_\tau, \tau, c) - (\epsilon - z)\right\|_2^2\right]
$$

**含义**: 教师视频模型微调时采用 shifted cosine 时间步重采样（$s=3$），使模型更关注高噪声区间；$v_\Theta$ 为教师 DiT 网络。

**符号说明**:
- $\tau$：shifted 时间步（由 $t$ 变换）
- $s = 3$：shift 强度
- $v_\Theta$：教师视频 DiT（Wan2.1-14B fine-tuned）

---

## 关键图表

### Figure 1：方法对比与主要结果

![Figure 1: Paradigm Comparison and Results](https://arxiv.org/html/2608.08558v1/intro.png)

**说明**: **左**：三种视频驱动机器人策略范式对比——(a) 在线 Video Gen + IDM：推理时代价高昂；(b) [[Fast-WAM]]：从真实视频-动作轨迹联合训练；(c) Vid2WAM：离线蒸馏，部署时仅用 student WAM。**右上**：Vid2WAM 在各基准上持续改善 novel task 成功率。**右下**：Vid2WAM 同时达到最高新任务成功率与最低推理延迟。

### Figure 2：Vid2WAM 训练与推理流程概览

![Figure 2: Vid2WAM Overview](https://arxiv.org/html/2608.08558v1/method.png)

**说明**: **训练阶段**：冻结教师生成未来视频 → student VAE 重编码为潜变量 $\hat{z}$ → IDM 恢复伪动作 $\hat{a}$ → 双通道监督 student（视频潜 + 动作），源感知适配器分离两域。**推理阶段**：教师与 IDM 丢弃，仅单帧通过 student 视频骨干提取潜在世界特征，直接预测动作块。

### Figure 3：新任务未来视频预测定性对比

![Figure 3: Qualitative Future-Video Predictions](https://arxiv.org/html/2608.08558v1/illustration_compact.png)

**说明**: 相同初始观测和语言指令下，[[Fast-WAM]]（上行）预测的未来序列往往缺乏一致的任务进度，Vid2WAM（下行）能更准确地预测朝向目标的运动。表明视频基础模型先验有效提升了未来预测质量，并间接改善动作生成。

### Figure 4：RoboTwin 2.0 代表性新任务成功率

（Figure 4 为 HTML 内嵌图，无独立 URL，图见论文 PDF 第 6 页）

**说明**: 在若干代表性 RoboTwin 2.0 新任务上，Vid2WAM 在 clean 和 randomized 评估条件下均持续优于 Fast-WAM，验证了视频先验对未见任务的泛化价值。

### Figure 5：真实世界双臂操作执行

![Figure 5: Real-World Executions](https://arxiv.org/html/2608.08558v1/realworld.png)

**说明**: 展示 Vid2WAM 在真实双臂机器人上执行 seen task 和 novel task（无真实专家轨迹）的关键帧，从左到右时间递进。包括抽取试管、取纸巾、番茄入篮等任务，证明方法在真实世界中的可行性。

### Figure S1：教师生成 Rollout 示例

![Figure S1: Teacher Rollouts](https://arxiv.org/html/2608.08558v1/teacher_rollout_illustrate.png)

**说明**: 跨 LIBERO、RoboTwin 和真实世界场景的教师生成 rollout 示例。每行 6 帧按时间顺序排列，条件为语言指令。涵盖少量数据和新任务两种制度，以及 clean 和 randomized 真实世界初始状态。

### Figure S2：真实世界双臂实验平台

![Figure S2: Real-World Setup](https://arxiv.org/html/2608.08558v1/realworld_setup.png)

**说明**: 实验平台由两个 AgileX Piper 机械臂、头顶 Intel RealSense D435 相机、两个腕部 D435 相机、遥操作界面和可重配置工作台组成。头顶相机提供俯视视角作为空间监督；腕部相机提供第一视角。

### Figure S3：GPT-Image-2 初始状态增广

![Figure S3: Initial State Augmentation](https://arxiv.org/html/2608.08558v1/illustration_imagegen.png)

**说明**: 针对真实世界新任务（仅有 2 个真实初始观测），用 GPT-Image-2 从真实头顶俯视图合成 60 张多样化初始状态图（改变物体位置、光照等），作为教师生成 rollout 的条件输入，大幅扩展新任务的分布覆盖。

---

### Table 1：RoboTwin 2.0 综合成功率（%）

| Method | Low-data Clean | Low-data Rand. | Novel Clean | Novel Rand. | Novel Subset Clean | Novel Subset Rand. |
|--------|---------------|---------------|------------|------------|-------------------|-------------------|
| π₀.₅ | 48.7 | 42.8 | 67.9 | 66.5 | 46.5 | 46.3 |
| Motus | 49.9 | 43.6 | 75.1 | 76.3 | 48.9 | 51.5 |
| Fast-WAM | 52.1 | 39.1 | 75.7 | 74.7 | 45.0 | 42.8 |
| **Vid2WAM（Ours）** | **55.4** | **45.4** | **78.3** | **78.5** | **54.7** | **55.3** |

**表格说明**: Novel subset（仅 15 个未见任务）上，Fast-WAM 从 novel 均值 75.7% 骤降至 45.0%，而 Vid2WAM 仅轻微下降（78.3% → 54.7%），比 Fast-WAM 高 9.7 / 12.5 个点，说明视频蒸馏先验对新任务泛化帮助最大。

### Table 2：LIBERO-Plus 鲁棒性成功率（%，部分列）

| Regime | Method | Spatial | Object | Goal | Long | Overall |
|--------|--------|---------|--------|------|------|---------|
| Low-data | Motus | 34.2 | 35.4 | 29.4 | 12.4 | 27.9 |
| Low-data | Fast-WAM | 39.8 | 54.0 | 14.6 | 25.8 | 33.6 |
| Low-data | **Ours** | **50.0** | **55.0** | **27.8** | 23.0 | **39.0** |
| Novel | Motus | 52.2 | 44.0 | 31.8 | 39.6 | 41.9 |
| Novel | Fast-WAM | 48.0 | 59.6 | 37.0 | 38.6 | 45.8 |
| Novel | **Ours** | **52.4** | **60.2** | **41.4** | **38.6** | **48.1** |

**表格说明**: LIBERO-Plus 对 7 类扰动（相机、机器人状态、语言、光照、背景、传感器噪声、布局）均匀测试，Vid2WAM 在 overall 指标上两种制度均最优，光照扰动下 low-data 提升尤为显著（71.6% vs. Fast-WAM 57.8%）。

### Table 3：LIBERO 四套件平均成功率（%）

| Regime | Method | Spatial | Object | Goal | Long | Avg. |
|--------|--------|---------|--------|------|------|------|
| Low-data | π₀.₅ | 91.2 | 95.6 | 84.4 | 76.4 | 86.9 |
| Low-data | Motus | 93.0 | 94.4 | 85.0 | 78.4 | 87.7 |
| Low-data | Fast-WAM | 89.6 | 97.8 | 82.8 | 78.8 | 87.3 |
| Low-data | **Ours** | **93.0** | **98.6** | **87.6** | **79.4** | **89.7** |
| Novel | π₀.₅ | 77.2 | 80.2 | 78.4 | 75.8 | 77.9 |
| Novel | Motus | 77.4 | 72.2 | 68.8 | 69.8 | 72.1 |
| Novel | Fast-WAM | 76.8 | 79.0 | 77.8 | 73.0 | 76.7 |
| Novel | **Ours** | **79.4** | 79.4 | **79.2** | **75.2** | **78.3** |

**表格说明**: Low-data 制度下 Vid2WAM 在每个 suite 上取得最佳或并列最佳，平均 89.7%；Novel 制度下同样达到最高平均（78.3%），证明方法的数据效率和新任务泛化能力。

### Table 4：真实世界双臂操作成功率（每任务 20 次）

| Regime | Task | π₀.₅ | Motus | Fast-WAM | **Ours** |
|--------|------|-------|-------|----------|---------|
| Seen | Click Bell | 30% | 70% | 20% | **65%** |
| Seen | Item Handover | 25% | 60% | 25% | **60%** |
| Seen | Carry Basket | 45% | 45% | 20% | **50%** |
| Seen | Open Pan Place | 5% | 40% | 10% | **40%** |
| Seen | Open Drawer Place | 60% | 55% | 30% | **65%** |
| Seen | Insert Test Tube | 15% | 0% | 10% | **25%** |
| Novel | Pick Test Tube | 0% | 10% | 5% | **30%** |
| Novel | Take Tissue | 0% | 0% | 5% | **15%** |
| Novel | Tomato Basket | 0% | 20% | 25% | **30%** |

**表格说明**: Vid2WAM 在所有 9 个任务上达到最优或竞争力表现。尤其在**新任务**（无真实专家轨迹）上，其他基线普遍失败，Vid2WAM 仅凭 2 个初始观测 + GPT-Image-2 增广即能成功执行透明试管拔取和可变形纸巾提取。

### Table 5：消融实验（LIBERO 低数据制度）

| 变体 | Spatial | Object | Goal | Long | Avg. |
|------|---------|--------|------|------|------|
| Fast-WAM（无蒸馏） | 89.6 | 97.8 | 82.8 | 78.8 | 87.3 |
| **Vid2WAM（Full）** | **93.0** | **98.6** | **87.6** | **79.4** | **89.7** |
| Dual Action DiT（两个独立 DiT） | 92.2 | 97.8 | 85.8 | 80.6 | 89.1 |
| No Adapter（无源感知适配器） | 93.8 | 97.6 | 84.2 | 78.4 | 88.5 |
| Pseudo Action Only（仅伪动作） | 90.4 | 96.0 | 85.0 | 74.0 | 86.4 |
| Future Latent Only（仅视频潜变量） | 92.8 | 98.2 | 86.4 | 75.2 | 88.2 |
| Teacher Policy（在线教师+IDM） | 76.0 | 85.2 | 65.6 | 58.2 | 71.3 |

**关键发现**:
- **双通道 > 单通道**: Future Latent Only（88.2%）和 Pseudo Action Only（86.4%）均不及 Full（89.7%），两路监督互补。
- **适配器有效**: No Adapter（88.5%）低于 Full，且 Long suite 下跌明显（78.4% vs 79.4%），证明源感知隔离的必要性。
- **在线推理有害**: Teacher Policy（71.3%）远低于所有蒸馏变体，说明增益来自离线知识蒸馏 + 真实动作 grounding，而非更强的在线控制器。
- **共享骨干优于双 DiT**: Dual Action DiT（89.1%）略低于共享骨干 + 适配器设计（89.7%）。

### Table 6：推理延迟对比（单张 RTX 4090）

| Method | Params | Mean (ms) | Std (ms) | Min (ms) | Max (ms) |
|--------|--------|-----------|----------|----------|----------|
| π₀.₅ | 3.3B | 76.7 | 0.4 | 76.2 | 77.6 |
| Fast-WAM | 5B | 209.1 | 4.7 | 204.8 | 220.2 |
| **Vid2WAM（Ours）** | **5B** | **212.5** | **11.7** | **199.0** | **227.8** |
| Motus | 8B | 2414.9 | 22.8 | 2380.0 | 2456.3 |
| Teacher Policy | 14B | 4894.5 | 106.6 | 4785.8 | 4998.9 |

**表格说明**: Vid2WAM 与 Fast-WAM 延迟几乎持平（212.5 ms vs. 209.1 ms），比 Motus 快 **11.4×**，比在线 Teacher Policy 快 **23×**，验证了离线蒸馏策略的推理效率优势。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| [[RoboTwin2\|RoboTwin 2.0]] | 50 任务，100 trials/任务 | 双臂操作，clean + randomized | 仿真主基准 |
| [[LIBERO]] | 4 套件 × 10 任务，50 trials | 多样化操作，有空间/目标/长序列变体 | 仿真低数据/新任务 |
| [[LIBERO-Plus]] | 7 类扰动，500 variants/套件 | 相机/光照/背景/噪声等鲁棒性压力测试 | 仿真鲁棒性 |
| 真实双臂 | 6 seen × 60 + 3 novel × 0 real traj | AgileX Piper，3 相机视角 | 真实世界验证 |

### 实现细节

- **视频教师**: LVP 检查点（Wan2.1-14B），在训练集上轻量微调
- **IDM Backbone**: [[ResNet-50]]，SmoothL1 损失，动作归一化训练
- **伪监督截断**: 仿真实验中限定前 8 秒（减少长时域生成误差影响）
- **真实新任务增广**: GPT-Image-2 从 2 个真实初始观测合成 60 张多样化条件图
- **视频损失掩码**: 真实新任务中空间视频损失仅计算头顶相机有效区域
- **硬件**: 推理测试在单张 RTX 4090 上进行

### 可视化结果

- Figure 3 显示 Vid2WAM 的未来预测更连贯，能正确预测任务目标状态（Fast-WAM 经常预测错误方向或静止帧）
- Figure S1 表明教师 rollout 在跨场景、跨任务上视觉质量较高，具备多样化的物体交互模式

---

## 批判性思考

### 优点

1. **假设突破**: 彻底解放 WAM 对专家演示的未来监督依赖，思路简洁清晰。
2. **双通道蒸馏设计巧妙**: 视频潜变量监督保留了预测性先验，而不仅仅依赖有损的 IDM 恢复——信息瓶颈问题被显式处理。
3. **源感知适配器轻量有效**: 用零初始化残差适配器隔离域噪声，无需修改主干架构，泛化性好。
4. **推理开销为零**: 教师与 IDM 完全离线，部署参数量与 Fast-WAM 相同。
5. **真实世界验证充分**: 包含 novel task（0 专家轨迹）场景，结果有说服力。

### 局限性

1. **教师质量瓶颈**: 视频教师微调仍需一定量的真实演示视频（few-shot），对极端新的具身形态或场景适配能力待验证。
2. **IDM 误差传播**: 消融显示 Pseudo Action Only（86.4%）甚至低于 Fast-WAM（87.3%），说明仅靠 IDM 信号有负效果，IDM 质量直接影响伪动作可靠性。
3. **真实世界 novel task 数据依赖图像增广**: 依赖 GPT-Image-2 合成初始状态，引入额外成本和 API 依赖，且合成图像分布偏差未深入分析。
4. **与 DreamGen 等差距分析不够充分**: 论文未直接与 DreamGen 对比，相关工作描述的优势仍停留在原理层面。

### 潜在改进方向

1. 探索教师生成视频的在线 refinement（配合 RLHF 或判别器过滤低质量 rollout）以提升 IDM 恢复质量。
2. 将源感知适配器推广到多具身形态迁移场景（cross-embodiment）。
3. 结合视觉语言模型（VLM）作为第三监督通道，提供语义级别的对齐约束。

### 可复现性评估

- [ ] 代码开源（暂未开放）
- [ ] 预训练模型（未提供）
- [x] 训练细节完整（方法和附录中有充分描述）
- [x] 数据集可获取（RoboTwin 2.0、LIBERO 均为公开 benchmark）

---

## 关联笔记

### 基于

- [[Fast-WAM]]: WAM 核心架构基础，视频-动作联合 DiT 训练范式
- [[Flash-WAM]]: WAM 家族另一高效变体
- [[WAM]]: World Action Model 基础概念
- [[Flow Matching]]: 训练目标采用 flow matching 损失

### 对比

- [[Fast-WAM]]: 最直接 baseline，Vid2WAM 在新任务上超出 9.7~12.5 个百分点
- [[π₀.₅]]: 代表广泛机器人数据预训练路线，验证视频蒸馏路线的互补价值
- [[Inverse Dynamics Model]]: IDM 用于伪动作恢复，与纯 IDM 方法的区别在于额外使用视频潜变量监督

### 方法相关

- [[Source-Aware Residual Adapter]]: 核心新概念，域感知残差适配器
- [[Knowledge Distillation]]: 视频基础模型→WAM 的知识迁移框架
- [[Inverse Dynamics Model]]: IDM 伪动作恢复
- [[Flow Matching]]: 动作与视频的联合生成目标
- [[Action Chunking]]: 动作块预测方式

### 硬件/数据相关

- [[RoboTwin2|RoboTwin 2.0]]: 主仿真 benchmark（50 双臂操作任务）
- [[LIBERO]]: 辅助仿真 benchmark
- [[LIBERO-Plus]]: 鲁棒性扩展 benchmark

---

## 速查卡片

> [!summary] Vid2WAM: Distilling Video Diffusion Priors into World Action Models
> - **核心**: 将视频扩散模型的未来预测先验蒸馏进 WAM，免去 expert demo 对未来监督的依赖
> - **方法**: 双通道蒸馏（视频潜变量 + IDM 伪动作）+ 源感知残差适配器，教师离线使用
> - **结果**: RoboTwin 2.0 新任务 +9.7~12.5 pts vs Fast-WAM；推理延迟与 Fast-WAM 持平（~212 ms）
> - **代码**: 暂未开放 | [项目主页](https://qch-fa.github.io/vid2wam-website/)

---

*笔记创建时间: 2026-08-12*
