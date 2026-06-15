---
title: "Flash-WAM: Modality-Aware Distillation for World Action Models"
method_name: "Flash-WAM"
authors: [Arman Akbari, Ci Zhang, Arash Akbari, Lin Zhao, Yixiao Chen, Weiwei Chen, Xuan Zhang, Geng Yuan, Yanzhi Wang]
year: 2026
venue: arXiv
tags: [world-action-model, consistency-distillation, inference-efficiency, robot-manipulation, flow-matching, diffusion-policy, real-time-control]
zotero_collection: Robotics/World Model
image_source: online
arxiv_html: https://arxiv.org/html/2606.05254v1
created: 2026-06-06
---

# 论文笔记：Flash-WAM: Modality-Aware Distillation for World Action Models

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | （论文未公开列出机构） |
| 日期 | June 2026 |
| 项目主页 | [flashwam.github.io](https://flashwam.github.io) |
| 对比基线 | [[LingBot-VA]], [[DMD]] |
| 链接 | [arXiv](https://arxiv.org/abs/2606.05254) / Code（即将开放） |

---

## 一句话总结

> Flash-WAM 通过模态感知一致性蒸馏，将联合视频-动作 WAM 的推理延迟从 8.1 秒压缩至 348 毫秒（23× 加速），同时在 RoboTwin 2.0 上保持 85.54% 成功率，并在真实 Unitree G1 人形机器人上验证了实时可行性。

---

## 核心贡献

1. **诊断失败根源**: 从理论上证明标准 [[LCM|一致性蒸馏（LCM）]] 参数化在 $\sigma \to 0$ 时存在二次消失梯度（$|b(\sigma)| \propto \sigma^2$），对集中于低噪声区的动作流几乎无学习信号，导致朴素联合蒸馏成功率从 91.25% 崩溃至 23.97%。
2. **模态感知一致性函数**: 为动作流设计线性参数化（$f^a = \mathbf{x}_\sigma - \sigma v_\theta$，线性梯度缩放，无超参数）；为视频流保留方差保持 LCM 参数化。Proposition 1 给出充要条件的理论证明。
3. **实时部署验证**: 在仿真（RoboTwin 2.0：85.54%，19×；LIBERO：95.7%，13.7×）和真实 Unitree G1 人形机器人（60% 平均成功率，1v/2a）上均达到 ≤500ms 的实时控制预算。

---

## 问题背景

### 要解决的问题

[[World Action Model|WAM（世界动作模型）]] 需要联合对未来视频和机器人动作进行扩散去噪。以 [[LingBot-VA]] 为例，每个控制 chunk 需要 25 步视频去噪 + 50 步动作去噪，在 NVIDIA L40S 上延迟高达 8.1 秒，远超实时控制的 500ms 预算（2 Hz）。

### 现有方法的局限

- **朴素联合 LCM**：直接将标准一致性蒸馏应用于视频和动作两个流，成功率从 91.25% 崩溃至 23.97%。
- **Video-only LCM**：只蒸馏视频流、动作流用教师推理，成功率 78.79%，但无法实现真正的端到端单步推理。
- **[[DMD|DMD2]]**：分布匹配蒸馏，需要额外对抗训练 critic，工程复杂度高，1v/2a 成功率仅 78.74%。

### 本文的动机

视频流和动作流使用**非对称 SNR 偏移噪声调度**：视频偏移因子 $s^v = 5.0$（训练质量集中于高噪声），动作偏移因子 $s^a = 1.0$（均匀覆盖全噪声范围，低噪声区有大量质量）。标准 LCM 参数化在低噪声区的梯度为 $\mathcal{O}(\sigma^2)$，对动作流产生"梯度真空"。针对不同模态的噪声特性分别选择最优一致性函数是解决问题的关键。

---

## 方法详解

### 模型架构

Flash-WAM 是在 [[LingBot-VA]] 教师模型上进行**模态感知一致性蒸馏**的框架：

- **输入**: 语言指令 $\mathbf{C}$ + 当前视觉观测 $o_t$
- **教师模型**: [[LingBot-VA]]（[[Cascaded WAM]]，先视频后动作）
- **核心机制**: 动作流用线性参数化 $f^a$，视频流用方差保持 $f^v$，联合蒸馏损失 $\mathcal{L} = \mathcal{L}^v + \lambda_a \mathcal{L}^a$
- **输出**: 单步推理得到视频帧预测 + 动作块 $a_{t:t+k}$
- **推理配置**: 1v/2a（19×加速）或 1v/1a（23×加速）

### 核心模块

#### 模块1: 非对称 SNR 偏移噪声调度

**设计动机**: [[流匹配]] 中的 SNR Shift 技术可以调整训练质量在噪声频谱上的分布，视频和动作使用不同偏移系数以匹配各自生成难度。

**具体实现**: 对偏移系数 $s$ 进行重参数化：

$$
\sigma = \frac{s\tilde{\sigma}}{1 + (s-1)\tilde{\sigma}}, \quad \tilde{\sigma} \sim \mathcal{U}[0, 1]
$$

视频使用 $s^v = 5.0$（偏高噪声），动作使用 $s^a = 1.0$（均匀）。两流在 [[Joint Self-Attention]] 中交互，但独立采样噪声水平。

#### 模块2: 动作流线性参数化（核心创新）

**设计动机**: Proposition 1 证明，最优一致性函数要求 $|b(\sigma)| = \mathcal{O}(\sigma)$（$b'(0) \neq 0$），而标准 LCM 的 $b'(0) = 0$ 违反此条件。

**具体实现**:

$$
a(\sigma) = 1, \quad b(\sigma) = -\sigma
$$

$$
f^a(\mathbf{x}^a_\sigma, \sigma) = \mathbf{x}^a_\sigma - \sigma \cdot v_\theta(\mathbf{x}^a_\sigma, \sigma)
$$

等价于[[流匹配]]的 $\hat{\mathbf{x}}_0 = \mathbf{x}_\sigma - \sigma v_\theta$，无额外超参数，$b'(0) = -1 \neq 0$ 满足最优条件。

#### 模块3: 视频流方差保持参数化

**设计动机**: 视频流集中于高噪声区间，标准方差保持 LCM 参数化在此区间本身已是最优，无需修改。

**具体实现**:

$$
f^v(\mathbf{x}^v_\sigma, \sigma) = c_{\text{skip}}(\sigma)\,\mathbf{x}^v_\sigma + c_{\text{out}}(\sigma)\,\hat{\mathbf{x}}^v_0
$$

$$
c_{\text{skip}}(\sigma) = \frac{\sigma_d^2}{\sigma^2 + \sigma_d^2}, \quad c_{\text{out}}(\sigma) = \frac{\sigma\,\sigma_d}{\sqrt{\sigma^2 + \sigma_d^2}}
$$

其中 $\sigma_d = 0.5$，保证输出方差在不同 $\sigma$ 下稳定。

#### 模块4: 动作正则化

**设计动机**: 单步动作预测误差在自回归展开时被放大（horizon effect），正则化约束能改善长视野任务稳定性。

**具体实现**: 在蒸馏损失中加入动作正则项，权重 $\lambda_r = 0.2$，约束学生动作预测向教师靠拢。

---

## 关键公式

### 公式1: [[流匹配|Flow Matching 训练目标]]

$$
\mathcal{L}_{\text{FM}} = \mathbb{E}_{\mathbf{x}_0, \boldsymbol{\epsilon}, \sigma} \left\| v_\theta(\mathbf{x}_\sigma, \sigma) - (\boldsymbol{\epsilon} - \mathbf{x}_0) \right\|^2
$$

**含义**: 训练速度场 $v_\theta$ 预测从干净样本到噪声的方向，是教师模型 LingBot-VA 的训练目标。

**符号说明**:
- $\mathbf{x}_0$: 干净样本（视频帧或动作块）
- $\mathbf{x}_\sigma = (1-\sigma)\mathbf{x}_0 + \sigma\boldsymbol{\epsilon}$: 噪声插值样本
- $\boldsymbol{\epsilon} \sim \mathcal{N}(0, I)$: 标准高斯噪声
- $v_\theta$: 速度场网络（教师模型参数）

### 公式2: [[SNR Shift|SNR 偏移噪声调度]]

$$
\sigma = \frac{s\tilde{\sigma}}{1 + (s-1)\tilde{\sigma}}, \quad \tilde{\sigma} \sim \mathcal{U}[0, 1]
$$

**含义**: 通过偏移系数 $s$ 将训练质量集中到高噪声区（大 $s$）或均匀覆盖（$s=1$）。

**符号说明**:
- $s$: SNR 偏移系数，$s^v = 5.0$（视频），$s^a = 1.0$（动作）
- $\tilde{\sigma}$: 原始均匀采样噪声水平
- 当 $s=1$ 时退化为均匀调度 $\sigma = \tilde{\sigma}$

### 公式3: [[一致性蒸馏|一致性函数通用族]]

$$
f(\mathbf{x}_\sigma, \sigma) = a(\sigma)\,\mathbf{x}_\sigma + b(\sigma)\,v_\theta(\mathbf{x}_\sigma, \sigma)
$$

**含义**: 所有一致性蒸馏方法可统一写成此形式，边界条件 $a(0) = 1, b(0) = 0$ 确保 $\sigma=0$ 时输出干净样本。

**符号说明**:
- $a(\sigma), b(\sigma)$: 模态特定的标量函数
- 标准 LCM: $b(\sigma) \propto \sigma^2/\sigma_d$（二次，次优）
- Flash-WAM 动作流: $b(\sigma) = -\sigma$（线性，最优）

### 公式4: [[一致性蒸馏|Proposition 1 — 最优梯度缩放充要条件]]

$$
|b(\sigma)| = \mathcal{O}(\sigma) \text{ as } \sigma \to 0 \iff b'(0) \neq 0
$$

**含义**: 一致性函数在低噪声区域保持线性梯度缩放的充要条件。标准 LCM 的 $b'(0) = 0$ 违反此条件，导致动作流梯度为 $\mathcal{O}(\sigma^2)$（二次消失）。线性参数化的 $b(\sigma) = -\sigma$ 满足 $b'(0) = -1 \neq 0$，达到最优。

### 公式5: [[Flash-WAM|动作流一致性函数（线性参数化）]]

$$
f^a(\mathbf{x}^a_\sigma, \sigma) = \mathbf{x}^a_\sigma - \sigma \cdot v_\theta(\mathbf{x}^a_\sigma, \sigma)
$$

**含义**: 动作流专用一致性函数，$b(\sigma) = -\sigma$ 保证低噪声区梯度不消失，等价于流匹配的直接 $\hat{\mathbf{x}}_0$ 估计。

**符号说明**:
- $\mathbf{x}^a_\sigma$: 动作流带噪样本
- $v_\theta(\mathbf{x}^a_\sigma, \sigma)$: 速度场网络对动作流的预测
- $b(\sigma) = -\sigma$: 线性参数化，$b'(0) = -1 \neq 0$

### 公式6: [[LCM|视频流一致性函数（方差保持参数化）]]

$$
f^v(\mathbf{x}^v_\sigma, \sigma) = \frac{\sigma_d^2}{\sigma^2 + \sigma_d^2}\,\mathbf{x}^v_\sigma + \frac{\sigma\,\sigma_d}{\sqrt{\sigma^2 + \sigma_d^2}}\,\hat{\mathbf{x}}^v_0
$$

**含义**: 视频流专用一致性函数，保持输出方差在不同噪声水平下稳定，适配视频流的高噪声区间特性。

**符号说明**:
- $\sigma_d = 0.5$: 方差保持参数（数据尺度）
- $\hat{\mathbf{x}}^v_0 = \mathbf{x}^v_\sigma - \sigma\,v_\theta(\mathbf{x}^v_\sigma, \sigma)$: 速度场对干净视频样本的估计
- $c_{\text{skip}} = \sigma_d^2/(\sigma^2+\sigma_d^2)$: 输入混合权重
- $c_{\text{out}} = \sigma\sigma_d/\sqrt{\sigma^2+\sigma_d^2}$: 预测混合权重

### 公式7: [[一致性蒸馏|一致性蒸馏损失]]

$$
\mathcal{L}_{\text{CD}} = d\!\left(f_{\theta_S}(\mathbf{x}_{\sigma_s}, \sigma_s),\ f_{\theta_{S'}}(\tilde{\mathbf{x}}_{\sigma_e}, \sigma_e)\right)
$$

**含义**: 要求一致性函数在 ODE 轨迹的相邻噪声点对同一干净样本给出一致估计，$\theta_{S'}$ 为带 stop-gradient 的 EMA 学生网络。

**符号说明**:
- $\sigma_s > \sigma_e$: 起点和终点噪声水平（ODE 前向方向）
- $\tilde{\mathbf{x}}_{\sigma_e}$: 单步 ODE 求解从 $\sigma_s$ 前进到 $\sigma_e$ 的中间样本
- $d(\cdot, \cdot)$: Huber 距离（$c = 0.001$）
- $\theta_{S'}$: EMA 目标学生网络参数

### 公式8: [[Flash-WAM|联合训练目标]]

$$
\mathcal{L}^v = d\!\left(f^v_{\theta_S}(\mathbf{x}^v_{\sigma_s}, \sigma_s),\ f^v_{\theta_{S'}}(\tilde{\mathbf{x}}^v_{\sigma_e}, \sigma_e)\right)
$$

$$
\mathcal{L}^a = d\!\left(f^a_{\theta_S}(\mathbf{x}^a_{\sigma_s}, \sigma_s),\ f^a_{\theta_{S'}}(\tilde{\mathbf{x}}^a_{\sigma_e}, \sigma_e)\right)
$$

$$
\mathcal{L} = \mathcal{L}^v + \lambda_a\,\mathcal{L}^a
$$

**含义**: 视频流和动作流分别用各自的模态感知一致性函数计算损失，加权求和形成联合训练目标。

**符号说明**:
- $\lambda_a = 1.0$: 动作流损失权重
- $\mathcal{L}^v$: 视频流一致性损失（方差保持参数化）
- $\mathcal{L}^a$: 动作流一致性损失（线性参数化）

---

## 关键图表

### Figure 1: 推理延迟与成功率总览

![Figure 1a - per-chunk 推理延迟对比](https://arxiv.org/html/2606.05254v1/x2.png)

![Figure 1b - RoboTwin 2.0 成功率对比](https://arxiv.org/html/2606.05254v1/x3.png)

**说明**: (a) per-chunk 推理延迟从 LingBot-VA 教师的 8.1s 压缩至 Flash-WAM 的 348ms（23× 加速），满足 500ms 实时控制预算。(b) Flash-WAM 在 RoboTwin 2.0 上达到 85.54%（1v/2a），远超 Naive Joint LCM（24%）和 DMD2（78.74%），是唯一接近教师水平的蒸馏方法。

### Figure 2: Flash-WAM 方法总览

![Figure 2 - Overview](https://arxiv.org/html/2606.05254v1/x4.png)

**说明**: 三栏展示框架。**左**：诊断动机——展示 Naive Joint LCM 失效（动作流在低 σ 区域梯度消失，成功率 24%），与 Flash-WAM 的恢复（85.5%）对比。**中**：Flash-WAM 训练流水线——分别使用 $f^v$（方差保持）和 $f^a$（线性）对视频流和动作流蒸馏，EMA 学生网络提供目标。**右**：部署阶段——蒸馏后学生模型以单步去噪自回归生成视频帧和动作，实现实时控制。

### Figure 3: 真实机器人评测套件

![Figure 3 - Unitree G1](https://arxiv.org/html/2606.05254v1/x5.png)

**说明**: Unitree G1 人形机器人上的三项操作任务评测套件（T1、T2、T3），分别测试不同难度的桌面操控。Flash-WAM（1v/2a）平均 60%，相比无蒸馏的步数压缩（40%）提升 20 个百分点。

### Figure 4: RoboTwin 定性对比

![Figure 4 - Qualitative Comparison](https://arxiv.org/html/2606.05254v1/x6.png)

**说明**: RoboTwin 任务 "pick_diverse_bottles" 的开环视频生成对比。朴素 LCM 出现瓶状物形态退化，场景混乱；Flash-WAM 预测视频质量接近教师模型，物体形态清晰、场景结构完整。

### Table 1: RoboTwin 2.0 主实验结果

| 方法 | $N_v$ | $N_a$ | Clean | Rand. | 平均 | 加速 |
|------|--------|--------|-------|-------|------|------|
| π₀ | – | – | 65.92 | 58.40 | 62.2% | – |
| π₀.₅ | – | – | 82.74 | 76.76 | 79.8% | – |
| X-VLA | – | – | 72.9 | 72.8 | 72.8% | – |
| Motus | – | – | 88.66 | 87.02 | 87.8% | – |
| LingBot-VA（教师） | 25 | 50 | 91.64 | 90.86 | 91.25% | 1.0× |
| LingBot-VA + DMD2 | 1 | 2 | 85.08 | 72.36 | 78.74% | 19.0× |
| Video-only LCM | 1 | 2 | 80.66 | 76.92 | 78.79% | – |
| Naive Joint LCM | 1 | 2 | 25.88 | 22.07 | 23.97% | – |
| **Flash-WAM** | **1** | **2** | **88.42** | **82.66** | **85.54%** | **19.0×** |
| LingBot-VA + DMD2 | 1 | 1 | 52.66 | 48.46 | 50.56% | 23.3× |
| Video-only LCM | 1 | 1 | 77.90 | 69.46 | 73.68% | – |
| Naive Joint LCM | 1 | 1 | 39.68 | 32.96 | 36.32% | – |
| **Flash-WAM** | **1** | **1** | **82.56** | **80.26** | **81.41%** | **23.3×** |

**关键发现**: 朴素联合 LCM 崩溃至 24%，Flash-WAM 在 1v/2a 达到 85.54%，超越 DMD2（78.74%）6.8 个百分点；1v/1a 下 DMD2 仅 50.56%，Flash-WAM 达到 81.41%。

### Table 2: LIBERO 基准测试

| 方法 | $N_v$ | $N_a$ | Spatial | Object | Goal | Long | 平均 | 加速 |
|------|--------|--------|---------|--------|------|------|------|------|
| π₀ | – | – | 96.8 | 98.8 | 95.8 | 85.2 | 94.1% | – |
| X-VLA | – | – | 98.2 | 98.6 | 97.8 | 97.6 | 98.1% | – |
| LingBot-VA（教师） | 20 | 50 | 98.5 | 99.8 | 98.0 | 98.3 | 98.6% | 1.0× |
| Video-only LCM | 1 | 2 | 95.1 | 92.0 | 96.0 | 97.8 | 95.2% | 13.7× |
| **Flash-WAM** | **1** | **2** | **97.0** | **92.8** | **96.4** | **98.0** | **95.7%** | **13.7×** |
| Video-only LCM | 1 | 1 | 95.0 | 91.5 | 95.0 | 95.4 | 94.2% | 16.3× |
| **Flash-WAM** | **1** | **1** | **96.0** | **92.6** | **96.0** | **95.8** | **95.1%** | **16.3×** |

**关键发现**: LIBERO 任务相对简单，Flash-WAM（95.7%）与 Video-only LCM（95.2%）差距较小，但仍一致优于基线，实现 13-16× 加速。

### Table 3: 真实机器人（Unitree G1）成功率

| 方法 | $N_v/N_a$ | T1 | T2 | T3 | 平均 |
|------|------------|-----|-----|-----|------|
| LingBot-VA（教师） | 33/10 | 50% | 70% | 80% | 66.7% |
| LingBot-VA（减少步数，无蒸馏） | 1/2 | 30% | 30% | 60% | 40.0% |
| Video-only LCM | 1/2 | 30% | 50% | 50% | 43.3% |
| **Flash-WAM** | **1/2** | **50%** | **60%** | **70%** | **60.0%** |
| LingBot-VA（减少步数，无蒸馏） | 1/1 | 10% | 30% | 30% | 23.3% |
| Video-only LCM | 1/1 | 20% | 40% | 40% | 33.3% |
| **Flash-WAM** | **1/1** | **40%** | **50%** | **60%** | **50.0%** |

**关键发现**: 直接减少推理步数（无蒸馏）真实成功率仅 40%；Flash-WAM 1v/2a 恢复至 60%，与教师（66.7%）差距仅 6.7 个百分点，证明蒸馏在真实世界中的有效性。

### Table 4: RoboTwin 2.0 消融实验（各 Horizon）

| 方法 | $N_v/N_a$ | H=1 Clean | H=1 Rand. | H=2 Clean | H=2 Rand. | H=3 Clean | H=3 Rand. | Clean Avg | Rand. Avg |
|------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|
| LingBot-VA（教师） | 25/50 | 94.18 | 93.56 | 90.35 | 86.95 | 93.22 | 93.28 | 92.93 | 91.55 |
| Video-only LCM | 1/2 | 87.10 | 82.73 | 73.13 | 68.19 | 62.50 | 68.25 | 80.66 | 76.92 |
| Video-only LCM + reg | 1/2 | 91.53 | 88.50 | 83.00 | 74.69 | 68.00 | 62.75 | 86.92 | 82.02 |
| Naive Joint LCM | 1/2 | 41.00 | 35.13 | 4.00 | 3.13 | 0.00 | 0.00 | 25.88 | 20.08 |
| **Flash-WAM** | **1/2** | **92.30** | **88.47** | **84.88** | **76.63** | **73.50** | **63.25** | **88.42** | **82.66** |
| Video-only LCM | 1/1 | 85.57 | 78.17 | 72.06 | 61.81 | 43.75 | 34.75 | 77.90 | 69.46 |
| Video-only LCM + reg | 1/1 | 66.87 | 61.07 | 39.19 | 35.56 | 10.25 | 4.75 | 53.48 | 48.40 |
| Naive Joint LCM | 1/1 | 54.63 | 46.00 | 21.56 | 15.63 | 0.00 | 0.00 | 39.68 | 32.96 |
| **Flash-WAM** | **1/1** | **87.30** | **86.93** | **78.44** | **72.63** | **63.50** | **60.75** | **82.56** | **80.26** |

**关键发现**: 随 horizon 增大（H=3），朴素 LCM 成功率归零（0%），Flash-WAM 仍保持 73%/60%（Clean/Rand.），说明模态感知设计对长任务视野尤为重要。Video-only LCM + reg 在 1v/1a 时从 78% 下降到 53%，说明辅助正则化不能替代正确的参数化。

### Table 5 & 6: 训练超参数

| 超参数 | 微调阶段（LIBERO） | 蒸馏阶段 |
|--------|------------------|----------|
| 优化器 | AdamW (β₁=0.9, β₂=0.95) | AdamW (β₁=0.9, β₂=0.999) |
| 学习率 | 1×10⁻⁵ | 5×10⁻⁶ |
| 权重衰减 | 0.1 | – |
| Gradient Clipping | 2.0 | 2.0 |
| Warmup 步数 | 10 | 100 |
| Effective Batch Size | 120 | 48 |
| 训练步数 | 4,000 | 2,000 |
| 硬件 | 4× H100, ~24h | 4× H100, ~24h |
| $\lambda_a$ | – | 1.0 |
| $\lambda_r$ | – | 0.2 |
| EMA 衰减 $\alpha$ | – | 0.995 |
| 数据尺度 $\sigma_d$ | – | 0.5 |
| CFG 范围 | – | [2.0, 10.0] |
| 损失函数 | – | Huber (c=0.001) |

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| RoboTwin 2.0 | 多任务仿真 | Clean/Randomized 难度，H=1/2/3 Horizon | 主要评测 + 消融 |
| LIBERO (4 suites) | Spatial/Object/Goal/Long | 多样化桌面操作，Long 任务尤其困难 | 仿真评测 |
| Unitree G1 真实场景 | 3 项操作任务（T1/T2/T3） | 真实人形机器人部署，每任务 10 次试验 | 真实世界验证 |

### 实现细节

- **Backbone**: [[LingBot-VA]]（像素空间 [[Cascaded WAM]]）
- **图像分辨率**: 128×128
- **动作维度**: 30，每视频帧 4 个动作，帧块大小 K=4
- **SNR 偏移**: $s^v = 5.0$，$s^a = 1.0$
- **推理部署**: NVIDIA L40S GPU，目标 ≤500ms/chunk

### 可视化结果

Figure 4 定性对比显示，朴素 LCM 生成的 "pick_diverse_bottles" 视频中出现瓶状物形变、颜色混乱等退化现象，而 Flash-WAM 的视频预测质量接近教师，物体轮廓清晰，动作轨迹与视频预测协调一致。

---

## 批判性思考

### 优点

1. **理论贡献扎实**: Proposition 1 提供一致性函数最优性的充要条件，设计决策有数学支撑而非经验调参。
2. **方法简洁无超参数**: 动作流一致性函数 $f^a = \mathbf{x}_\sigma - \sigma v_\theta$ 直接推导，无额外引入超参数，比 DMD2 的对抗训练简单得多。
3. **工程落地验证**: 同时在仿真和真实 Unitree G1 上验证，latency 数据具体（348ms），可信度高。
4. **框架通用性**: 模态感知蒸馏思路可推广到其他存在异构噪声调度的多模态扩散模型。

### 局限性

1. **仅针对 Cascaded WAM**: 理论和实验均基于 LingBot-VA（先视频后动作），对 Joint WAM（视频-动作联合去噪，如 Motus、UWM）的适用性未验证。
2. **真实机器人性能差距**: 1v/2a 真实成功率 60%，教师仅 66.7%（绝对成功率本身不高），任务种类仅 3 项。
3. **代码和模型未开源**: 无法复现和二次开发。

### 潜在改进方向

1. 扩展到 Joint WAM（视频-动作联合去噪 transformer）的模态感知蒸馏
2. 与自适应步数调度（类似 [[SANTS]]）结合，实现"步数 + 参数"双重优化
3. 探索渐进式多步蒸馏（Progressive Distillation），先蒸馏到 N/2 步再蒸馏到 1 步

### 可复现性评估

- [ ] 代码开源（"coming soon"）
- [ ] 预训练模型（HuggingFace，"coming soon"）
- [x] 训练细节完整（Table 5/6 超参数完整）
- [x] 数据集可获取（RoboTwin 2.0、LIBERO 公开）

---

## 关联笔记

### 基于

- [[LingBot-VA]]: 教师模型，Flash-WAM 直接在此基础上蒸馏
- [[一致性蒸馏]]: 核心蒸馏框架，Flash-WAM 提出其针对多模态异构噪声的改进版本
- [[LCM]]: 被改进的视频流参数化基础（方差保持部分沿用）
- [[流匹配]]: WAM 训练的基础生成范式，线性参数化等价于流匹配的 $\hat{\mathbf{x}}_0$ 估计

### 对比

- [[SANTS]]: 同为 WAM 推理加速，SANTS 通过自适应步数调度减少 NFE（延迟降至 523ms），Flash-WAM 通过蒸馏实现单步推理（348ms）
- [[DMD]]: 分布匹配蒸馏，Flash-WAM 在 RoboTwin 1v/2a 上显著优于 DMD2（85.54% vs 78.74%）

### 方法相关

- [[模态感知蒸馏]]: Flash-WAM 的核心创新概念
- [[World Action Model]]: Flash-WAM 针对的模型类别
- [[Cascaded WAM]]: LingBot-VA 的架构范式
- [[条件流匹配]]: WAM 使用的条件生成技术

### 硬件/数据相关

- [[RoboTwin]]: 主要评测仿真环境
- [[LIBERO]]: 次要评测基准

---

## 速查卡片

> [!summary] Flash-WAM
> - **核心**: 证明标准 LCM 对 WAM 动作流存在梯度消失，提出模态感知一致性函数解决联合视频-动作蒸馏失效问题
> - **方法**: 动作流用线性参数化 $f^a = x_\sigma - \sigma v_\theta$，视频流用方差保持 LCM，联合训练
> - **结果**: RoboTwin 85.54%（19×加速），LIBERO 95.7%（13.7×），真实 G1 机器人 60%；8.1s → 348ms
> - **代码**: flashwam.github.io（即将开放）

---

*笔记创建时间: 2026-06-06*
