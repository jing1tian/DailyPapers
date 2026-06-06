---
title: "Flash-WAM: Modality-Aware Distillation for World Action Models"
method_name: "Flash-WAM"
authors: [Arman Akbari, Ci Zhang, Arash Akbari, Lin Zhao, Yixiao Chen, Weiwei Chen, Xuan Zhang, Geng Yuan, Yanzhi Wang]
year: 2026
venue: arXiv
tags: [world-action-model, consistency-distillation, flow-matching, inference-acceleration, robot-manipulation]
zotero_collection: Robotics/World Model
image_source: mixed
arxiv_html: https://arxiv.org/html/2606.05254
created: 2026-06-06
---

# 论文笔记：Flash-WAM: Modality-Aware Distillation for World Action Models

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Northeastern University, Tencent |
| 日期 | June 2026 |
| 项目主页 | — |
| 对比基线 | [[LingBot-VA]] |
| 链接 | [arXiv](https://arxiv.org/abs/2606.05254) / Code: — |

---

## 一句话总结

> Flash-WAM 通过模态感知一致性蒸馏，将视频+动作联合生成的 WAM 推理压缩至单步，实现 23× 加速（8.1s → 348ms），同时保持 85.5% 的仿真成功率。

---

## 核心贡献

1. **问题诊断**: 从理论上证明标准 [[LCM]] 参数化在低噪声（σ→0）动作域产生二次消失梯度，无法直接用于视频+动作联合蒸馏
2. **模态感知一致性函数**: 为动作流设计线性参数化（Linear Parametrization）、为视频流保留方差保持（Variance-Preserving）LCM 参数化，分别对应不同 SNR 区间的最优梯度缩放
3. **单步联合推理**: 在 [[RoboTwin 2.0]] 上以 1v/2a 配置达到 85.5% 成功率（vs 教师模型 91.25%），真实世界 Unitree G1 人形机器人上达到 60% 平均成功率

---

## 问题背景

### 要解决的问题

[[WAM|World Action Model]] 在推理时需要对视频和动作进行多步扩散去噪（[[LingBot-VA]] 需要 Nv=25 步视频去噪 + Na=50 步动作去噪），导致每个 chunk 的推理延迟高达 8.1 秒，无法满足实时控制需求。

### 现有方法的局限

标准的 [[一致性蒸馏]] / [[LCM]] 参数化设计用于图像生成（高噪声区间），直接应用于 WAM 的联合视频-动作生成会失效——因为视频流和动作流使用**不对称的 SNR 偏移噪声调度**（视频：sv=5.0，动作：sa=1.0），使两种模态处于结构上不同的噪声区间。具体失败原因是：LCM 的参数化在 σ→0 时梯度信号二次消失（|b_LCM(σ)| = σ²/σ_d），导致低噪声的动作流几乎无法学习，Naive Joint LCM 在 RoboTwin 2.0 上只有 24% 成功率（vs 教师 91.25%）。

[[DMD2|DMD]] 等分布匹配蒸馏方法需要额外的 critic 网络进行对抗训练，工程复杂度高，且适配多模态场景同样困难。

### 本文的动机

核心洞察：**一致性函数应当是模态特定的（modality-specific）**，而非全局统一。通过 Proposition 1 的梯度分析可知，在低噪声区间（动作流），需要 |b(σ)| = O(σ) 且 b'(0) ≠ 0 的参数化；在高噪声区间（视频流），方差保持的 LCM 参数化本来就合适。因此，分别为两种模态设计不同的一致性函数，在同一联合训练目标下即可解决这一根本矛盾。

---

## 方法详解

### 模型架构

Flash-WAM 基于 [[LingBot-VA]] 教师模型，采用**模态感知一致性蒸馏**框架：

- **输入**: 语言指令 $l$ + 视觉观测 $o_t$
- **教师模型**: [[LingBot-VA]]（Nv=25 步视频去噪 + Na=50 步动作去噪）
- **核心组件**: 模态感知一致性函数（动作流用线性参数化，视频流用方差保持参数化）
- **输出**: 视频帧预测 + 动作块 $a_{t:t+k}$
- **推理**: 每个模态只需单步去噪（Nv=1, Na=1 或 Nv=1, Na=2）

### 核心模块

#### 模块 1: 一致性函数的通用形式

**设计动机**: 利用 [[一致性蒸馏]] 的自洽性约束，将多步 ODE 求解压缩为单步

**具体实现**: 一致性函数将带噪样本 $\mathbf{x}_\sigma$ 直接映射到干净样本估计：

$$
f(\mathbf{x}_\sigma, \sigma) = a(\sigma)\,\mathbf{x}_\sigma + b(\sigma)\,v_\theta(\mathbf{x}_\sigma, \sigma)
$$

边界条件：$a(0) = 1, b(0) = 0$（确保 σ=0 时输出干净样本）。

[[LCM]] 使用的参数化为：$c_\text{skip}(σ) = σ_d^2/(σ^2+σ_d^2)$，$c_\text{out}(σ) = σ σ_d / \sqrt{σ^2+σ_d^2}$，满足边界条件但在 σ→0 时有 $|b_\text{LCM}(σ)| \approx σ^2/σ_d$（二次消失）。

#### 模块 2: 动作流线性参数化（Flash-WAM 核心创新）

**设计动机**: 根据 Proposition 1，在 σ→0 附近，梯度最优缩放要求 $|b(σ)| = O(σ)$ 且 $b'(0) \neq 0$。LCM 参数化不满足此条件（$b'(0)=0$），而线性参数化满足。

**具体实现**:

$$
a(σ) = 1, \quad b(σ) = -σ
$$

$$
f^a(\mathbf{x}^a_\sigma, \sigma) = \mathbf{x}^a_\sigma - \sigma\,v_\theta(\mathbf{x}^a_\sigma, \sigma)
$$

这等价于[[流匹配]] 中的 $\hat{\mathbf{x}}_0 = \mathbf{x}_\sigma - \sigma v_\theta$，无额外超参数，梯度缩放为 O(σ)。

#### 模块 3: 视频流方差保持参数化

**设计动机**: 视频流的噪声调度偏移更大（sv=5.0），处于高噪声区间，LCM 参数化在此区间提供稳定的方差保持，不存在梯度消失问题。

**具体实现**:

$$
f^v(\mathbf{x}^v_\sigma, \sigma) = c_\text{skip}(\sigma)\,\mathbf{x}^v_\sigma + c_\text{out}(\sigma)\,\hat{\mathbf{x}}^v_0
$$

其中：

$$
c_\text{skip}(\sigma) = \frac{\sigma_d^2}{\sigma^2 + \sigma_d^2}, \quad c_\text{out}(\sigma) = \frac{\sigma\,\sigma_d}{\sqrt{\sigma^2 + \sigma_d^2}}
$$

---

## 关键公式

### 公式 1: [[流匹配|流匹配目标]]

$$
\mathcal{L}_\text{FM} = \mathbb{E}_{\mathbf{x}_0, \epsilon, \sigma}\left[\left\|v_\theta(\mathbf{x}_\sigma, \sigma) - (\epsilon - \mathbf{x}_0)\right\|^2\right]
$$

**含义**: 流匹配训练目标，让速度场预测网络 $v_\theta$ 预测从干净样本到噪声的方向

**符号说明**:
- $\mathbf{x}_0$: 干净样本
- $\epsilon$: 标准高斯噪声
- $\sigma$: 噪声水平，按 SNR 偏移调度采样：$\sigma = s\tilde{\sigma}/(1+(s-1)\tilde{\sigma})$，$\tilde{\sigma} \sim U[0,1]$
- $s$: SNR 偏移因子（视频 sv=5.0，动作 sa=1.0）

### 公式 2: [[一致性蒸馏|一致性蒸馏损失]]

$$
\mathcal{L}_\text{CD} = d\!\left(f_{\theta_S}(\mathbf{x}_{\sigma_s}, \sigma_s),\; f_{\theta_{S'}}(\tilde{\mathbf{x}}_{\sigma_e}, \sigma_e)\right)
$$

**含义**: 要求一致性函数在相邻噪声水平下对同一干净样本给出一致估计，$\theta_{S'}$ 为带 stop-gradient 的 EMA 学生网络

**符号说明**:
- $\sigma_s$: 起始噪声水平
- $\sigma_e < \sigma_s$: 目标噪声水平（经单步 ODE 前进后）
- $\tilde{\mathbf{x}}_{\sigma_e}$: 单步 ODE 求解得到的中间样本
- $d(\cdot, \cdot)$: Huber 距离（c=0.001）

### 公式 3: [[Flash-WAM|动作一致性函数（线性参数化）]]

$$
f^a(\mathbf{x}^a_\sigma, \sigma) = \mathbf{x}^a_\sigma - \sigma\,v_\theta(\mathbf{x}^a_\sigma, \sigma)
$$

**含义**: 动作流专用一致性函数，通过线性缩放 b(σ)=-σ 保证 σ→0 时梯度不消失

**符号说明**:
- $\mathbf{x}^a_\sigma$: 动作流带噪样本
- $v_\theta$: 速度场预测网络（与视频流共享）
- $b(\sigma) = -\sigma$: 线性参数化，满足 $|b'(0)| = 1 \neq 0$

### 公式 4: [[LCM|视频一致性函数（方差保持参数化）]]

$$
f^v(\mathbf{x}^v_\sigma, \sigma) = \frac{\sigma_d^2}{\sigma^2+\sigma_d^2}\,\mathbf{x}^v_\sigma + \frac{\sigma\,\sigma_d}{\sqrt{\sigma^2+\sigma_d^2}}\,\hat{\mathbf{x}}^v_0
$$

**含义**: 视频流专用一致性函数，保持方差不变（适合高噪声区间）

**符号说明**:
- $\sigma_d = 0.5$: 方差保持参数
- $\hat{\mathbf{x}}^v_0 = \mathbf{x}^v_\sigma - \sigma\,v_\theta(\mathbf{x}^v_\sigma, \sigma)$: 速度场对干净样本的估计
- $c_\text{skip}, c_\text{out}$: 随噪声水平变化的混合权重

### 公式 5: [[Flash-WAM|联合训练目标]]

$$
\mathcal{L} = \mathcal{L}^v + \lambda_a\,\mathcal{L}^a
$$

**含义**: 联合优化视频流和动作流的一致性损失，加权组合

**符号说明**:
- $\mathcal{L}^v$: 视频流一致性损失（使用方差保持参数化）
- $\mathcal{L}^a$: 动作流一致性损失（使用线性参数化）
- $\lambda_a = 1.0$: 动作流损失权重

### 公式 6: [[一致性蒸馏|梯度消失分析（Proposition 1）]]

$$
\left\|\nabla_\theta \mathcal{L}_\text{CD}\right\| \propto |b(\sigma)| \cdot \left\|\nabla_\theta v_\theta\right\|
$$

**含义**: 一致性蒸馏损失的梯度幅度与 $|b(\sigma)|$ 成正比。当 σ→0 时，LCM 有 $|b_\text{LCM}(\sigma)| \sim \sigma^2/\sigma_d$（二次消失），线性参数化有 $|b_\text{linear}(\sigma)| = \sigma$（线性，不消失），后者的梯度信号强 $O(1/\sigma)$ 倍。

---

## 关键图表

### Figure 1: 性能与延迟对比

![[Flash-WAM_fig4.png]]

**说明**: (a) LCM 参数化的梯度缩放 |b(σ)| 随噪声水平 σ 的变化。蓝色（LCM）在 σ→0 时二次消失，橙色（线性参数化）保持线性增长，确保动作流梯度不消失。

![[Flash-WAM_fig5.png]]

**说明**: 线性参数化（橙色）vs 方差保持 LCM（蓝色虚线）vs 纯线性（蓝色实线）的梯度曲线对比。

### Figure 2: Flash-WAM 方法概览（架构三部分）

![[Flash-WAM_fig6.jpeg]]

![[Flash-WAM_fig7.png]]

**说明**: (左) 诊断动机——说明 Naive Joint LCM 失败原因（动作流梯度消失）；(中) Flash-WAM 训练流水线——模态感知一致性函数的联合训练；(右) 部署阶段——学生模型自回归地以单步去噪生成视频帧和动作。RoboTwin `pick_diverse_bottles` 任务的定性生成结果。

### Figure 3: 真实世界评测环境（Unitree G1）

![[Flash-WAM_fig8.jpeg]]

![[Flash-WAM_fig10.jpeg]]

**说明**: Unitree G1（EmbodyX）人形机器人上的操控任务评测套件（Task 1: 标记目标抓取，Task 2: 小件拾取）。Flash-WAM 在 1v/2a 配置下平均成功率 60%，相比教师模型（66.7%）下降较小。

### Table 1: RoboTwin 2.0 主实验结果

| Method | Nv | Na | Clean | Rand. | Average | Speedup |
|--------|----|----|-------|-------|---------|---------|
| π0 | — | — | 65.92 | 58.40 | 62.2 | — |
| π0.5 | — | — | 82.74 | 76.76 | 79.8 | — |
| X-VLA | — | — | 72.9 | 72.8 | 72.8 | — |
| Motus | — | — | 88.66 | 87.02 | 87.8 | — |
| LingBot-VA (teacher) | 25 | 50 | 91.64 | 90.86 | 91.25 | 1.0× |
| LingBot-VA + DMD2 | 1 | 2 | 85.08 | 72.36 | 78.74 | 19.0× |
| Video-only LCM | 1 | 2 | 80.66 | 76.92 | 78.79 | — |
| Naive Joint LCM | 1 | 2 | 25.88 | 22.07 | 23.97 | — |
| **Flash-WAM** | **1** | **2** | **88.42** | **82.66** | **85.54** | **—** |
| LingBot-VA + DMD2 | 1 | 1 | 52.66 | 48.46 | 50.56 | 23.3× |
| Video-only LCM | 1 | 1 | 77.90 | 69.46 | 73.68 | — |
| Naive Joint LCM | 1 | 1 | 39.68 | 32.96 | 36.32 | — |
| **Flash-WAM** | **1** | **1** | **82.56** | **80.26** | **81.41** | **—** |

**说明**: Flash-WAM 在 1v/2a 配置下比 Video-only LCM 高 6.75%，比 Naive Joint LCM 高 61.57%，比 DMD2 高 6.8%。

### Table 2: LIBERO 基准结果

| Method | Nv | Na | Spatial | Object | Goal | Long | Average | Speedup |
|--------|----|----|---------|--------|------|------|---------|---------|
| π0 | — | — | 96.8 | 98.8 | 95.8 | 85.2 | 94.1 | — |
| X-VLA | — | — | 98.2 | 98.6 | 97.8 | 97.6 | 98.1 | — |
| LingBot-VA (teacher) | 20 | 50 | 98.5 | 99.8 | 98.0 | 98.3 | 98.6 | 1.0× |
| Video-only LCM | 1 | 2 | 95.1 | 92.0 | 96.0 | 97.8 | 95.2 | 13.7× |
| **Flash-WAM** | **1** | **2** | **97.0** | **92.8** | **96.4** | **98.0** | **95.7** | **13.7×** |
| Video-only LCM | 1 | 1 | 95.0 | 91.5 | 95.0 | 95.4 | 94.2 | 16.3× |
| **Flash-WAM** | **1** | **1** | **96.0** | **92.6** | **96.0** | **95.8** | **95.1** | **16.3×** |

**说明**: LIBERO 上 Flash-WAM 以 95.7% 达到近 teacher 水平，13.7× 加速。

### Table 3: 真实机器人实验结果（Unitree G1）

| Method | Nv/Na | Task 1 | Task 2 | Task 3 | Average |
|--------|-------|--------|--------|--------|---------|
| LingBot-VA (teacher) | 33/10 | 50% | 70% | 80% | 66.7% |
| LingBot-VA (reduced NFE) | 1/2 | 30% | 30% | 60% | 40.0% |
| Video-only LCM | 1/2 | 30% | 50% | 50% | 43.3% |
| **Flash-WAM** | **1/2** | **50%** | **60%** | **70%** | **60.0%** |
| LingBot-VA (reduced NFE) | 1/1 | 10% | 30% | 30% | 23.3% |
| Video-only LCM | 1/1 | 20% | 40% | 40% | 33.3% |
| **Flash-WAM** | **1/1** | **40%** | **50%** | **60%** | **50.0%** |

**说明**: 真实世界 Flash-WAM 1v/2a 达 60%，比 Video-only LCM 高 16.7%，比暴力降低 NFE 高 20%。

### Table 4: 消融实验（RoboTwin 2.0）

| Method | Config | Average |
|--------|--------|---------|
| Video-only LCM | 1v/2a | 78.79% |
| Video-only LCM + reg | 1v/2a | 86.92% |
| Naive Joint LCM | 1v/2a | 23.97% |
| **Flash-WAM** | **1v/2a** | **85.54%** |
| Video-only LCM | 1v/1a | 73.68% |
| Video-only LCM + reg | 1v/1a | 53.48% |
| Naive Joint LCM | 1v/1a | 36.32% |
| **Flash-WAM** | **1v/1a** | **81.41%** |

**关键发现**: 添加 MSE 正则化到 Video-only LCM 在 1v/2a 时可提升到 86.92%，但在 1v/1a 时大幅下降到 53.48%，说明辅助损失无法替代正确的动作流蒸馏参数化。Flash-WAM 在两种配置下都表现稳定。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| RoboTwin 2.0 | 多任务 | 仿真，含 Clean/Random 两种设置 | 主要训练与评测 |
| LIBERO | 4 个 Suite（Spatial/Object/Goal/Long） | 仿真，任务多样性高 | 评测 |
| Unitree G1 真实数据 | 3 个操控任务 | 真实人形机器人 | 真实世界评测 |

### 实现细节

- **架构**: 128×128 图像分辨率，30 维动作，4 帧/视频，4 chunk
- **流匹配参数**: sv=5.0（视频），sa=1.0（动作）
- **一致性蒸馏参数**: λa=1.0，λr=0.2（正则项），α=0.995（EMA 衰减），σd=0.5
- **损失**: Huber（c=0.001），CFG 范围 [2.0, 10.0]
- **优化器**: AdamW（β1=0.9, β2=0.999），lr=5×10⁻⁶，2000 训练步
- **硬件**: 4× H100 GPU，约 24 小时/套件
- **LIBERO 微调**: AdamW（β1=0.9, β2=0.95），lr=1×10⁻⁵，Effective Batch Size 120，4000 步

### 可视化结果

在 `pick_diverse_bottles` 任务的定性对比中，Flash-WAM 生成的视频帧轨迹更连贯，动作与视频预测更协调，而 Naive Joint LCM 的视频生成出现混乱、动作不连续的现象。

---

## 批判性思考

### 优点

1. **理论扎实**: Proposition 1 提供了严格的梯度分析，将模态选择从经验调参上升为有理论依据的设计原则
2. **工程简洁**: 动作流线性参数化无额外超参数，比 DMD2 的对抗训练简单得多
3. **真实世界验证**: 在 Unitree G1 人形机器人上的真实评测增强了可信度
4. **兼容性强**: 蒸馏框架可直接作用于已有的 WAM 教师模型，无需重新训练

### 局限性

1. **依赖教师模型质量**: 作为蒸馏方法，上限受教师模型 LingBot-VA 的 91.25% 约束，真实机器人上教师本身只有 66.7%
2. **仅针对单类 WAM 架构**: 实验仅基于 LingBot-VA（像素空间 [[Cascaded WAM]]），是否适用于潜空间或 Joint WAM 架构尚未验证
3. **NFE 配置固定**: 当前只考虑 Nv=1+Na=1/2 的配置，中间配置（如 Nv=2）的性能权衡未完全探索

### 潜在改进方向

1. 扩展到 Joint WAM 架构（视频和动作同步生成的 transformer）
2. 结合 Progressive Distillation 实现渐进式多步压缩
3. 探索自适应 λa 权重调度，根据任务难度动态调整动作流蒸馏强度

### 可复现性评估

- [ ] 代码开源（未公开）
- [ ] 预训练模型（未公开）
- [x] 训练细节完整（Table 5/6 提供详细超参数）
- [x] 数据集可获取（RoboTwin 2.0、LIBERO 均为公开数据集）

---

## 关联笔记

### 基于

- [[LingBot-VA]]: 教师模型，Flash-WAM 直接蒸馏此模型
- [[一致性蒸馏]]: 核心蒸馏框架
- [[LCM]]: 被改进的视频流参数化基础
- [[流匹配]]: WAM 训练的基础生成范式

### 对比

- [[DMD]]: 另一类蒸馏方法，需要对抗训练，Flash-WAM 在简洁性和性能上均优
- [[Pi05]]: VLA 基线，在 RoboTwin 2.0 上 79.8%，低于 Flash-WAM 85.5%

### 方法相关

- [[WAM]]: Flash-WAM 所属模型类别
- [[Cascaded WAM]]: LingBot-VA（教师）的架构范式
- [[条件流匹配]]: WAM 使用的条件生成技术

### 硬件/数据相关

- [[RoboTwin 2.0]]: 主要评测基准
- [[LIBERO]]: 次要评测基准
- Unitree G1: 真实机器人测试平台

---

## 速查卡片

> [!summary] Flash-WAM: Modality-Aware Distillation for World Action Models
> - **核心**: 模态感知一致性蒸馏，解决 WAM 中视频+动作联合蒸馏的梯度消失问题
> - **方法**: 动作流用线性参数化 f^a=x_σ-σ·v_θ，视频流用方差保持 LCM，联合训练
> - **结果**: 23× 加速（8.1s→348ms），RoboTwin 85.5%，LIBERO 95.7%，真实机器人 60%
> - **代码**: 未公开

---

*笔记创建时间: 2026-06-06*
