---
title: "TACO: TActile World Model as a Self-COrrector for Scalable VLA Post-Training"
method_name: "TACO"
authors: [Shengbang Liu, Yueru Jia, Yuyang Yan, Jiaming Liu, Xinran Zhang, Qiuxuan Feng, Yandong Guo, Shiji Zhou, Boxin Shi, Shanghang Zhang]
year: 2026
venue: arXiv
tags: [vla, tactile-sensing, world-model, robot-manipulation, post-training, contact-rich, flow-matching]
zotero_collection: Robotics/VLA
image_source: online
arxiv_html: https://arxiv.org/html/2607.02840
created: 2026-07-08
---

# 论文笔记：TACO: TActile World Model as a Self-COrrector for Scalable VLA Post-Training

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Peking University, MEGVII Technology |
| 日期 | July 2026 |
| 项目主页 | [taco-wm.github.io](https://taco-wm.github.io/) |
| 对比基线 | [[BC]] (Filtered BC), [[VLA]] (Base Policy) |
| 链接 | [arXiv](https://arxiv.org/abs/2607.02840) |

---

## 一句话总结

> TACO 通过触觉感知世界模型构建"识别–想象–标注"循环，为 [[VLA]] 生成可扩展的纠错数据，在 6 个接触丰富的操作任务上平均提升 44% 成功率。

---

## 核心贡献

1. **触觉感知世界模型（Visuo-Tactile World Model）**: 以 [[Wan2.2]]-TI2V-5B 为基础，实现视频与力信号的联合去噪，通过[[时序 RoPE 对齐|Temporal RoPE Alignment]] 将力矩 token 映射至视频 latent 时间轴，生成局部一致的视触觉纠错片段。
2. **统一进度-动作模型（UPA Model）**: 结合 [[DINOv2]] 视觉通路与 MLP 触觉通路，联合预测 7-DoF 纠错动作和任务进度 $p_t \in [0,1]$，驱动失败状态识别与合成数据标注。
3. **知识隔离触觉适应（Knowledge-Insulated Tactile Adaptation）**: 对 [[VLA]] 骨干施加 stop-gradient，仅通过 [[AdaRMS|adaRMSNorm]] 条件化将触觉学习路由至动作专家，保护预训练视觉-语言先验不被侵蚀。

---

## 问题背景

### 要解决的问题

[[VLA]] 在接触丰富任务（插花、擦白板、扭瓶盖等）中，细微扰动（滑移、压力不足、异常扭矩）导致失败，这些失败难以从视觉单独检测。现有方法依赖人工介入收集恢复数据，成本高且难以扩展。

### 现有方法的局限

- **纯视觉 VLA**: 无法感知接触状态，对接触失败不敏感。
- **人工干预收集**: 需人类反复示范恢复动作，数据收集代价极高。
- **Filtered BC**: 仅筛选真实成功轨迹，无法合成失败-恢复过渡数据。
- **视觉世界模型**: 忽略[[力矩感知]]，无法预测接触动态中的力反馈信息。

### 本文的动机

触觉信号能精确指示接触失败状态，且力矩传感器数据易于获取。若能构建视触觉联合世界模型，可在不增加人工标注的前提下，离线生成大量失败→恢复轨迹，实现可扩展的 VLA 后训练。

---

## 方法详解

### 模型架构

TACO 采用**双模型 + 迭代循环**架构：

- **输入**: 语言指令 $l$ + RGB 观测 $I_t$ + 12-D 力矩信号 $F_t \in \mathbb{R}^{12}$（左右各 6-DoF）
- **世界模型 Backbone**: [[Wan2.2]]-TI2V-5B（图文生成视频基础模型）
- **感知 Backbone**: [[DINOv2]]（视觉通路）+ MLP（触觉通路）
- **策略 Backbone**: [[PaliGemma]] 22B VLM + 300M 动作专家
- **输出**: 纠错动作 $\hat{a}_{t:t+T}$（$T=49$ 步）
- **总参数**: 世界模型 ~5B，策略 ~22.3B

### 核心模块

#### 模块 1: 视触觉生成模型（Visuo-Tactile Generation Model）

**设计动机**: 利用 [[Flow Matching]] 实现视频与力信号联合去噪，生成局部视触觉一致的纠错片段。

**具体实现**:
- 以 [[Wan2.2]]-TI2V-5B 为骨干，冻结大部分参数
- 力序列 $F \in \mathbb{R}^{B \times T \times 12}$ 经力矩 tokenizer $\mathcal{T}_\eta$ 转换为 $X^f \in \mathbb{R}^{B \times T \times d}$
- 视频 latent token $X^v \in \mathbb{R}^{B \times N_v \times d}$ 与力 token 拼接：$X = [X^v; X^f]$
- [[时序 RoPE 对齐|Temporal RoPE]] 将力 token 位置映射至视频时间轴，实现同步去噪
- 条件输入：当前观测 $I_t$，力信号 $F_t$，语言指令 $l$

#### 模块 2: 统一进度-动作模型（Unified Progress-Action Model, UPA）

**设计动机**: 同时检测失败状态和标注纠错动作，避免人工介入。

**具体实现**:
- [[DINOv2]] 视觉通路 + 方向感知解码器，提取视觉特征
- MLP 触觉通路，处理 12-D 力矩输入
- 联合预测 7-DoF 动作 $\hat{a}_t \in \mathbb{R}^7$ 和任务进度 $\hat{p}_t \in [0,1]$
- 进度停滞判断：$p_{t+\Delta} - p_t < \varepsilon$ 时标记为失败邻近状态

#### 模块 3: 知识隔离触觉适应（Knowledge-Insulated Tactile Adaptation）

**设计动机**: 在触觉微调时保护 [[PaliGemma]] 预训练的视觉-语言先验。

**具体实现**:
- 对 VLM 骨干输出表示施加 **stop-gradient**（$z_t$ 不参与梯度传播）
- 触觉学习仅通过 [[AdaRMS|adaRMSNorm]] 条件化路由至动作专家
- 仅更新：力矩编码器、适应层、动作专家
- [[优势条件训练|Advantage Conditioning]]：二元优势标签 $y \in \{0,1\}$ 区分恢复行为 ($y=1$) 与失败行为 ($y=0$)

### Recognize–Imagine–Label 循环

**Phase 1 – Recognize（识别）**:
部署当前策略 $\pi^{(k)}$，获取真实滚出 $\mathcal{D}_{roll}^{(k)}$，用 UPA 模型识别失败邻近状态锚点 $\mathcal{S}_{anchor}^{(k)}$。

**Phase 2 – Imagine（想象）**:
从锚点状态条件生成 $T=49$ 步视触觉纠错片段 $(\hat{I}_{t:t+T}, \hat{F}_{t:t+T})$。

**Phase 3 – Label（标注）**:
UPA 模型对想象片段预测纠错动作 $\hat{a}_{t:t+T}$ 和进度 $\hat{p}_{t:t+T}$，分配二元优势标签，混合真实成功数据与合成纠错数据训练新策略 $\pi^{(k+1)}$。

---

## 关键公式

### 公式 1: [[Flow Matching|视触觉联合流匹配损失]]

$$
\mathcal{L}_{joint} = \|u_\psi^v - (\xi_1^v - \xi_0^v)\|_2^2 + \lambda_f \|u_\psi^f - (\xi_1^f - \xi_0^f)\|_2^2
$$

**含义**: 同时优化视频流场和力信号流场的预测误差，实现联合视触觉去噪。

**符号说明**:
- $u_\psi^v$: 预测的视频流场
- $u_\psi^f$: 预测的力信号流场
- $\xi_1^v, \xi_1^f$: 干净的视频 latent 和力序列
- $\xi_0^v, \xi_0^f$: 对应的高斯噪声
- $\lambda_f$: 力去噪项平衡权重

### 公式 2: [[时序 RoPE 对齐|Temporal RoPE Alignment]]

$$
\rho(i) = \text{round}\!\left(\frac{i}{T-1} \times (f-1)\right), \quad i = 0, \ldots, T-1
$$

**含义**: 将第 $i$ 个力 token 映射到视频 latent 的时间轴位置，实现力-视频的同步去噪。

**符号说明**:
- $\rho(i)$: 第 $i$ 个力 token 的时间位置
- $T$: 力 token 序列长度
- $f$: 视频 latent 的时间维度长度
- $i$: 力 token 下标

### 公式 3: [[统一进度-动作模型|UPA 联合预测]]

$$
(\hat{a}_t, \hat{p}_t) = U_\phi(I_t, F_t)
$$

**含义**: UPA 模型从当前 RGB 帧和力矩信号联合预测纠错动作和任务进度。

**符号说明**:
- $\hat{a}_t \in \mathbb{R}^7$: 预测的 7-DoF 末端执行器动作
- $\hat{p}_t \in [0,1]$: 归一化任务进度预测值
- $I_t$: 时刻 $t$ 的 RGB 帧
- $F_t \in \mathbb{R}^{12}$: 左右各 6-DoF 力矩读数

### 公式 4: [[统一进度-动作模型|UPA 联合训练损失]]

$$
\mathcal{L}_{UPA} = \text{SmoothL1}(\hat{a}_t, a_t) + m_t \|\hat{p}_t - p_t\|_2^2
$$

**含义**: 联合优化动作预测（Smooth L1）和任务进度预测（MSE），$m_t$ 掩码过滤无有效进度标签的步骤。

**符号说明**:
- $\text{SmoothL1}$: 平滑 L1 损失，对动作异常值鲁棒
- $a_t$: 真实动作标签
- $m_t \in \{0,1\}$: 有效进度标签指示符
- $p_t$: 真实归一化进度值

### 公式 5: [[失败邻近状态检测|Failure-Adjacent Anchor Selection]]

$$
\mathcal{S}_{anchor}^{(k)} = \{(\tau, t) \mid \tau \in \mathcal{D}_{roll}^{(k)},\ p_{t+\Delta} - p_t < \varepsilon\}
$$

**含义**: 在真实滚出中识别任务进度停滞的时刻，作为纠错想象的锚点。

**符号说明**:
- $\tau$: 滚出轨迹
- $t$: 时间步
- $\Delta$: 进度观测窗口长度
- $\varepsilon$: 进度停滞阈值
- $p_t$: 时刻 $t$ 的 UPA 预测进度值

### 公式 6: [[视触觉生成|Imagined Correction Generation]]

$$
(\hat{I}_{t:t+T},\ \hat{F}_{t:t+T}) \sim G_\psi(\cdot \mid I_t, F_t, l)
$$

$$
(\hat{a}_{t:t+T},\ \hat{p}_{t:t+T}) = U_\phi(\hat{I}_{t:t+T},\ \hat{F}_{t:t+T})
$$

**含义**: 世界模型 $G_\psi$ 从锚点条件生成纠错视触觉片段，UPA 模型 $U_\phi$ 对其进行动作标注。

**符号说明**:
- $G_\psi$: 视触觉生成模型（[[Wan2.2]]-TI2V-5B）
- $T = 49$: 纠错段长度（时间步数）
- $l$: 语言指令条件

### 公式 7: [[优势条件训练|动作专家损失（含力矩与优势条件化）]]

$$
\mathcal{L}_\pi = \mathbb{E}\left[\|u_\theta(x_\sigma, \sigma \mid z_t,\ \tilde{c}_{adaRMS}) - (\varepsilon - a_t)\|_2^2\right]
$$

$$
c_{adaRMS} = c_t + \lambda_f c_f + \lambda_a c_a
$$

**含义**: 动作专家在 [[CFG|Classifier-Free Guidance]] 下预测流速度，条件融合流时间步 $c_t$、力矩 $c_f$、优势 $c_a$，通过 [[AdaRMS|adaRMSNorm]] 注入而不破坏 VLM 骨干。

**符号说明**:
- $x_\sigma = \sigma\varepsilon + (1-\sigma)a_t$: 噪声动作块
- $\sigma$: 噪声水平
- $z_t$: VLM 骨干前缀表示（stop-gradient）
- $c_t$: 流时间步条件
- $c_f$: 力矩历史条件
- $c_a$: 优势标签条件（$y \in \{0,1\}$）
- $\lambda_f, \lambda_a$: 权重系数
- $\tilde{c}_{adaRMS}$: 随机替换 null 的条件（[[CFG]] 训练）

### 公式 8: 视触觉 Token 拼接

$$
X^f = \mathcal{T}_\eta(F) \in \mathbb{R}^{B \times T \times d}, \quad X = [X^v;\ X^f] \in \mathbb{R}^{B \times (N_v + T) \times d}
$$

**含义**: 力序列经 tokenizer 转化后与视频 latent token 拼接，送入统一 Transformer 做联合去噪。

**符号说明**:
- $B$: batch size；$N_v$: 视频 token 数；$d$: 隐维度
- $\mathcal{T}_\eta$: 力矩 tokenizer（可学习）
- $X^v$: 视频 latent token；$X^f$: 力矩 token

---

## 关键图表

### Figure 1: Overview / 系统概览

![Figure 1](https://arxiv.org/html/2607.02840v1/x1.png)

**说明**: TACO 整体框架。基于真实机器人滚出，执行 Recognize–Imagine–Label 循环：识别失败邻近状态，通过视触觉世界模型生成局部纠错，再由 UPA 模型标注纠错动作，并以[[优势条件训练]]微调策略。

### Figure 2: TACO Framework / 迭代框架

![Figure 2](https://arxiv.org/html/2607.02840v1/x2.png)

**说明**: TACO 迭代后训练流程，展示三个阶段（Recognize / Imagine / Label）的数据流向。每次迭代后策略改进的真实滚出数据进入下一轮循环，实现自增强。

### Figure 3: Tactile-Aware World Model Architecture

![Figure 3](https://arxiv.org/html/2607.02840v1/x3.png)

**说明**: 视触觉世界模型架构细节。展示 [[Wan2.2]] 视频基础模型如何通过 [[时序 RoPE 对齐]] 融合力矩 token，以及想象片段经 UPA 模型转化为纠错动作序列的完整流程。

### Figure 4: Visualization of Imagined Corrections

![Figure 4](https://arxiv.org/html/2607.02840v1/x4.png)

**说明**: 想象纠错数据可视化（顶部：失败滚出；底部：想象的局部纠正）。展示模型在"扭瓶盖"等任务上生成局部视觉和力觉一致性恢复序列的能力。

### Figure 5: Ablation Study

![Figure 5](https://arxiv.org/html/2607.02840v1/image/Ablation.png)

**说明**: 消融研究柱状图，比较移除视频（V）、力矩（F）、动作（A）、进度（R）各模态时的任务成功率，验证各组件的必要性。移除触觉生成后成功率从 82% 下降至 28%。

### Figure 6: Action Distribution Analysis

![Figure 6](https://arxiv.org/html/2607.02840v1/image/action_distribution_v3.png)

**说明**: 在 Insert Flower 任务上投影 40 次成功滚出的末端执行器 XY 位姿分布。TACO（迭代 1/2）相比 Filtered BC 覆盖更广的动作空间，说明纠错数据拓宽了策略的行为分布。

### Figure 7: Generalization Performance

![Figure 7](https://arxiv.org/html/2607.02840v1/image/generalization_new.png)

**说明**: 三类泛化场景（未见背景 / 未见物体 / 未见位置）的对比实验结果。TACO 在所有泛化测试中均优于 Filtered BC 和基础策略。

### Figure 8: Real-World Robot Setup

![Figure 8](https://arxiv.org/html/2607.02840v1/x5.png)

**说明**: 实验硬件配置。[[Franka Research 3]] 单臂平台，并联夹爪，指尖安装 Xense 触觉传感器（6-DoF 力矩感知）。六个任务：Insert Flower、Wipe Whiteboard、Twist Bottle Cap、Play Xylophone、Toast Bread、Move Hanoi Rings。

### Figure 9: Robot Execution Progress (Appendix)

![Figure 9](https://arxiv.org/html/2607.02840v1/image/task_visualization2.jpg)

**说明**: 六个真实任务的前置摄像头关键帧序列，展示基础策略失败与 TACO 成功执行的对比。

### Figure 10: Additional Ablation Studies (Appendix)

![Figure 10](https://arxiv.org/html/2607.02840v1/image/ablation_appendix2.png)

**说明**: 附录消融。(a) Toast Bread 上真实/想象数据比例扩展（1:8 达 97% 成功率）；(b) 优势条件训练的影响；(c) 失败邻近锚点选择策略对比（均匀 vs. 进度停滞）。

### Figure 11: Failure Case Analysis (Appendix)

![Figure 11](https://arxiv.org/html/2607.02840v1/x6.png)

**说明**: Filtered BC、TACO（无知识隔离）和 TACO 在 Wipe Whiteboard、Move Hanoi Rings、Twist Bottle Cap 上的失败案例分析。知识隔离对避免 VLM 先验退化至关重要。

### Figure 12: Additional Generalization Experiments (Appendix)

![Figure 12](https://arxiv.org/html/2607.02840v1/image/generalization_appendix.png)

**说明**: Insert Flower 任务在三类泛化场景下的补充实验（附录），与 Figure 7 互补。

### Figure 13: Visualization of Imagined Corrections (Appendix)

![Figure 13](https://arxiv.org/html/2607.02840v1/image/Visualization_appendix.jpg)

**说明**: 附录中的额外想象纠错可视化（对应 Figure 4 的扩展版本），展示更多任务上的视触觉生成效果。

---

### Table 1: 真实任务定量结果（成功率 SR / 完成步数 CS）

| 方法 | Insert Flower | Wipe Whiteboard | Twist Bottle Cap | Play Xylophone | Toast Bread | Move Hanoi Rings | **Average** |
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Base Policy | SR: 0.50, CS: 250 | SR: 0.51, CS: 151 | SR: 0.45, CS: 131 | SR: 0.46, CS: 132 | SR: 0.30, CS: 183 | SR: 0.08, CS: 266 | SR: 0.38, CS: 185.5 |
| Filtered BC (Iter 1) | SR: 0.55, CS: 274 | SR: 0.54, CS: 120 | SR: 0.50, CS: 62 | SR: 0.49, CS: 128 | SR: 0.32, CS: 189 | SR: 0.07, CS: 120 | SR: 0.41, CS: 148.8 |
| TACO w/o KI (Iter 1) | SR: 0.55, CS: 233 | SR: 0.33, CS: 100 | SR: 0.55, CS: 58 | SR: 0.58, CS: 144 | SR: 0.48, CS: 193 | SR: 0.42, CS: 201 | SR: 0.49, CS: 154.8 |
| **TACO (Iter 1)** | **SR: 0.70, CS: 207** | **SR: 0.55, CS: 95** | **SR: 0.85, CS: 56** | **SR: 0.63, CS: 115** | **SR: 0.70, CS: 184** | **SR: 0.51, CS: 194** | **SR: 0.66, CS: 141.8** |
| Filtered BC (Iter 2) | SR: 0.52, CS: 289 | SR: 0.57, CS: 133 | SR: 0.48, CS: 79 | SR: 0.51, CS: 125 | SR: 0.36, CS: 177 | SR: 0.11, CS: 130 | SR: 0.43, CS: 155.5 |
| TACO w/o KI (Iter 2) | SR: 0.62, CS: 223 | SR: 0.35, CS: 98 | SR: 0.65, CS: 51 | SR: 0.52, CS: 120 | SR: 0.51, CS: 191 | SR: 0.37, CS: 196 | SR: 0.50, CS: 146.5 |
| **TACO (Iter 2)** | **SR: 0.93, CS: 169** | **SR: 0.65, CS: 87** | **SR: 0.98, CS: 52** | **SR: 0.78, CS: 97** | **SR: 0.81, CS: 177** | **SR: 0.79, CS: 184** | **SR: 0.82, CS: 127.7** |

**关键发现**: TACO Iter 2 平均 SR 达 0.82，相比基础策略 +44%，相比 Filtered BC +39%，相比 TACO w/o KI +32%。知识隔离对 Wipe Whiteboard 至关重要（w/o KI 仅 0.35，full TACO 达 0.65）。

### Table 2: 预训练公共机器人数据集

| 数据集 | 机器人平台 | 轨迹数 |
|--------|------------|--------|
| DROID | Franka Panda | 201,119 |
| AgiBot | AgiBot G1 | 3,017 |
| RoboMIND | Franka/UR/Ark/Agilex/TienKung | 1,721,985 |

**说明**: 用于视触觉生成模型预训练的公开数据集规模。RoboMIND 提供最大量的多机器人平台轨迹数据。

### Table 3: 方法对比（组件差异）

| 方法 | 真实成功数据 | 想象纠错数据 | 知识隔离 | 目的 |
|------|:---:|:---:|:---:|------|
| Base Policy | ✗ | ✗ | ✗ | 仅 50 次示范暖启动 |
| Filtered BC | ✓ | ✗ | ✗ | 测试真实成功数据是否足够 |
| TACO w/o KI | ✓ | ✓ | ✗ | 测试知识隔离的必要性 |
| **TACO** | ✓ | ✓ | ✓ | 完整方法 |

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| 自采演示数据 | 50 次/任务 | 真实机器人接触丰富任务 | 基础策略暖启动 |
| DROID | 201K 轨迹 | 多场景 Franka 操作 | 世界模型预训练 |
| RoboMIND | 1.72M 轨迹 | 多机器人平台大规模 | 世界模型预训练 |
| 想象纠错数据 | 可扩展（1:8 最优） | 合成视触觉纠错片段 | 迭代后训练 |

### 实现细节

- **策略 Backbone**: [[PaliGemma]] 22B + 300M 动作专家
- **世界模型**: [[Wan2.2]]-TI2V-5B，冻结主干微调力矩通路
- **感知模型**: [[DINOv2]] + 方向感知解码器 + MLP 触觉通路
- **优化器**: AdamW，余弦调度，300 步 warmup，峰值 LR $5 \times 10^{-5}$
- **Batch Size**: 32
- **训练步数**: 30,000 步
- **硬件**: 8 GPU（完全分片数据并行）
- **力矩传感器**: Xense，12-D（左右各 6-DoF 力 + 力矩）

### 可视化结果

- 想象纠错片段与真实恢复轨迹在视觉外观和力矩曲线上保持局部一致性
- 行动分布分析显示 TACO 迭代后覆盖更广的末端执行器位姿空间（超出原始示范流形）
- 泛化实验中，TACO 对未见物体（海绵替换橡皮）能自适应调整力策略

---

## 批判性思考

### 优点
1. **触觉-视觉联合建模**: 在世界模型生成和标注两个环节都引入触觉，而非仅作辅助，从根本上提升接触感知能力。
2. **无需额外人工介入**: Recognize–Imagine–Label 循环完全自动化，真正实现可扩展后训练。
3. **知识隔离设计精巧**: stop-gradient + adaRMSNorm 条件化精确控制触觉信息注入位置，避免灾难性遗忘。

### 局限性
1. **离线生成**: 纠错数据在部署前离线生成，无法在实时失败时在线恢复。
2. **局部接触失败**: 主要针对局部接触性失败（滑移、压力不足），对长时域、语义层面的失败效果未验证。
3. **触觉传感器依赖**: 需要配备 Xense 等精确力矩传感器，限制了在低成本机器人上的应用。

### 潜在改进方向
1. 在线纠错：将世界模型推理嵌入策略执行循环，实现实时失败检测与恢复。
2. 扩展至长时域失败：结合任务规划层，处理非局部性失败（如错误子任务完成）。

### 可复现性评估
- [ ] 代码开源（项目主页已公开但代码未发布）
- [ ] 预训练模型（未提供）
- [x] 训练细节完整（超参数、硬件配置详尽）
- [ ] 数据集可获取（自采数据未公开；公开预训练数据集可获取）

---

## 关联笔记

### 基于
- [[Wan2.2]]: 视触觉生成模型的基础骨干（Wan2.2-TI2V-5B）
- [[PaliGemma]]: VLA 策略的 VLM 骨干（22B）
- [[Flow Matching]]: 联合视触觉去噪的训练范式
- [[DINOv2]]: UPA 模型视觉通路骨干

### 对比
- [[BC]]: Filtered BC 基线，仅使用真实成功数据，无合成纠错
- [[VLA]]: 基础策略（Base Policy），无后训练

### 方法相关
- [[AdaRMS]]: 触觉和优势条件注入动作专家的机制
- [[CFG]]: 优势条件化训练中的分类器自由引导
- [[RoPE]]: 力 token 时序对齐的 Temporal RoPE
- [[World Model]]: TACO 本质是触觉感知世界模型
- [[优势条件训练]]: 二元优势标签区分恢复与失败行为

### 硬件/数据相关
- [[Franka Research 3]]: 实验使用的机械臂平台
- [[触觉传感器]]: Xense 指尖力矩传感器

---

## 速查卡片

> [!summary] TACO: TActile World Model as a Self-COrrector for Scalable VLA Post-Training
> - **核心**: 触觉世界模型驱动的 VLA 自动后训练，无需人工介入
> - **方法**: Recognize–Imagine–Label 循环 + 视触觉联合生成 + 知识隔离适应
> - **结果**: 6 个接触丰富任务平均 SR 0.82（vs. 基础策略 0.38），+44%
> - **项目**: [taco-wm.github.io](https://taco-wm.github.io/)

---

*笔记创建时间: 2026-07-08*
