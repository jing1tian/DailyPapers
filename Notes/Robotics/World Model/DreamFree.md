---
title: "Future Representation Distillation for Imagination-Free World Action Models"
method_name: "DreamFree"
authors: [Jaehyeong Jo, Sangwon Jang, Sihyun Yu, Seonghyeon Ye, Yekyung Nah, Hojung Jung, Kimin Lee, Sung Ju Hwang]
year: 2026
venue: arXiv
tags: [world-action-model, representation-distillation, video-diffusion, robot-manipulation, flow-matching, real-time-control, embodied-ai]
zotero_collection: Robotics/World Model
image_source: online
created: 2026-11-05
---

# 论文笔记：Future Representation Distillation for Imagination-Free World Action Models

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | KAIST |
| 日期 | November 2026 |
| 项目主页 | [harryjo97.github.io/dreamfree](https://harryjo97.github.io/dreamfree/) |
| 对比基线 | [[DreamZero]], [[π₀.₅]], [[Fast-WAM]] |
| 链接 | [arXiv](https://arxiv.org/abs/2611.04527) / [Code](https://github.com/harryjo97/dreamfree) |

---

## 一句话总结

> DreamFree 通过**未来表征蒸馏**将冻结视频扩散模型的预测性知识注入视觉编码器，使其仅从当前观测即可预测未来相关表征，从而在无需生成未来视频的情况下实现 11Hz 实时机器人控制，真实世界操作成功率达 64%，优于 DreamZero（56%）和 π₀.₅（16%）。

---

## 核心贡献

1. **想象力自由的 WAM**: 引入 DreamFree，将预训练视频扩散模型重新用作预测性视觉编码器，通过[[Future Representation Distillation|未来表征蒸馏]]使编码器从当前观测预测未来视觉表征，推理时无需迭代视频去噪。
2. **性能与效率双优**: 以 14B 视频骨干实现 11Hz 闭环控制，相比同等骨干的 DreamZero 加速 **17.8×**，真实世界操作成功率提升 8 pp；延迟与 π₀.₅ 相当但成功率高出 4×。
3. **视频先验可扩展**: 更大的视频骨干（5B→14B）带来 +13.2 pp 成功率提升，额外无标注视频数据进一步改善性能，表明 DreamFree 能将更强的视频先验转化为更好的机器人控制。
4. **无动作标注任务迁移**: 仅从视频演示（包括来自不同机体的视频）即可学习新任务，无需任何动作标注，复合任务平均成功率从 0% 提升至 21.7%。

---

## 问题背景

### 要解决的问题

[[世界动作模型（WAM）]] 通过预训练[[视频扩散模型]]的预测性知识来指导机器人动作，当前主流做法是通过**未来视频预测**来利用该知识（联合去噪视频和动作，或从预测视频中解码动作）。然而：
- **推理成本高**: 迭代视频去噪代价巨大，限制了部署频率
- **预测误差传播**: 视频预测的不准确性会导致动作预测错误
- **粒度过剩**: 动作预测是否需要与视频生成同等程度的视觉细节？

### 现有方法的局限

| 类别 | 代表方法 | 局限 |
|------|----------|------|
| 视频生成 WAM | [[DreamZero]] | 推理延迟高（1632ms），视频预测不准时动作失败 |
| 无视频推理 WAM | [[Fast-WAM]] | RoboCasa 成功率仅 44.4%，骨干从 5B 扩到 14B 几乎无提升 |
| VLA | [[π₀.₅]] | 仅有静态图像-文本先验，缺乏时空动力学知识 |

### 本文的动机

核心假设：**视频模型的预测性知识可以有效编码在视觉表征中**，从而支持动作预测而无需迭代视频去噪。

关键洞察：在特征空间中蒸馏预测信息——训练视觉编码器从当前观测预测冻结视频扩散模型的"未来感知"表征，使预测性知识以单次前向传播即可获得，不需要未来视频生成。

---

## 方法详解

### 模型架构

DreamFree 采用**两阶段训练、单阶段推理**架构：

- **Stage 1（预测性视觉编码器训练）**: 利用[[Future Representation Distillation|未来表征蒸馏]] + 视频生成目标，使编码器学会从当前帧预测未来帧的表征
- **Stage 2（动作模块训练）**: 冻结视觉编码器，训练动作读出模块和动作解码器，从预测性视觉表征中提取动作
- **推理**: 仅需编码器单次前向传播 + 动作 DiT 的 N 步去噪，无视频去噪

```
训练阶段1:
  视频帧 (当前+未来) → [冻结 VDM Teacher] → 目标表征 ↓
  当前帧 → [可训练 Encoder] → 当前帧隐状态 → [Projector] → 对齐目标
  损失: L = L_gen + L_distill

训练阶段2:
  当前帧 → [冻结 Encoder] → h_k → [Readout] → c (64 token)
  c + 语言 ℓ + 本体状态 q + 历史 s → [Action DiT] → 动作块 a_{1:h}

推理:
  当前帧 → [Encoder (单次)] → [Readout] → [Action DiT × N步] → 动作块
```

### 核心模块

#### 模块1: 预测性视觉编码器（Predictive Visual Encoder）

**设计动机**: 利用预训练视频扩散骨干的时空动力学知识，同时仅依赖当前帧进行动作预测。

**具体实现**:
- 从预训练[[Video Diffusion Transformer|视频 DiT]]（[[Wan 2.2-5B]] 或 [[Wan 2.2 A14B]]）初始化
- 对骨干双向视频自注意力施加**结构化掩码**（见图2(a) 左侧）：
  - 当前帧 token 仅彼此互相注意（禁止看未来）
  - 未来帧 token 仅注意当前帧和自身，禁用文本交叉注意力
- 训练时接收**干净当前帧 + 带噪未来帧**
- 训练后，编码器在推理时仅接收当前帧（timestep=0）

#### 模块2: 未来表征蒸馏（Future Representation Distillation）

**设计动机**: 受[[REPA]]启发，将冻结教师 VDM 的"未来感知"特征蒸馏到只看当前观测的学生编码器中。

**具体实现**:
- 教师 $\boldsymbol{g}$（冻结 VDM）接收完整带噪视频，提取第 $j$ 层隐状态作为蒸馏目标
- 学生 $\boldsymbol{u}_\theta$ 只处理当前帧，提取第 $i$ 层隐状态 $\boldsymbol{h}_i$
- 轻量级 Transformer 解码器投影器 $\boldsymbol{p}_\psi$ 将当前帧表征扩展到与教师匹配的时空 token 布局
- 仅对**未来 token 位置**（非当前帧）做相似度对齐（排除当前帧 token 以确保未来预测的监督是有效的）

#### 模块3: 动作读出模块（Action Readout Module）

**设计动机**: 从视觉编码器的大量 token 中提取动作相关信息，压缩为固定长度的动作条件 token。

**具体实现**:
- 受[[Perceiver-IO]]启发的潜在查询读出（Latent-Query Readout）
- 64 个可学习查询，通过自注意力和交叉注意力（以编码器第 $k$ 层隐状态为 K/V）更新
- 两层解码器块，每层含自注意力、交叉注意力、1024→4096→1024 GELU MLP，输出 64 个动作面向 token $\boldsymbol{c}$

#### 模块4: 动作解码器（Action DiT）

**设计动机**: 以预测性视觉表征为条件，用流匹配方法预测动作块。

**具体实现**:
- 30 个 DiT 块，隐层宽度 1024，前馈宽度 4096，24 头注意力（总参数 ~1B）
- 以动作面向 token $\boldsymbol{c}$、语言 $\ell$、本体状态 $\boldsymbol{q}$、可选历史 $\boldsymbol{s}$ 为条件
- [[Flow Matching|流匹配]]训练，预测 32 步动作块

---

## 关键公式

### 公式1: [[Future Representation Distillation|未来表征蒸馏损失]]

$$
\mathcal{L}_{\text{distill}}(\theta, \psi) := \mathbb{E}_{\boldsymbol{x}^*, \boldsymbol{\epsilon}_d} \left[ -\text{sim}\left( \boldsymbol{p}_\psi(\boldsymbol{h}_i),\, \boldsymbol{g}_j(\boldsymbol{x}_{t_d}, t_d, \ell) \right) \right]
$$

**含义**: 最大化学生编码器（仅看当前帧）投影与教师 VDM（看完整视频）表征之间的余弦相似度，将"预测性知识"从教师注入学生。

**符号说明**:
- $\boldsymbol{u}_\theta$: 可训练学生编码器（预测性视觉编码器）
- $\boldsymbol{g}$: 冻结教师视频扩散模型
- $\boldsymbol{p}_\psi$: 可训练投影器（轻量级 Transformer 解码器，仅训练时使用）
- $\boldsymbol{h}_i$: 学生编码器第 $i$ 层的当前帧隐状态
- $\boldsymbol{x}^*$: 干净视频潜在表示，$\boldsymbol{x}^*_0$ 为第一帧
- $\boldsymbol{\epsilon}_d \sim \mathcal{N}(0, \boldsymbol{I})$: 高斯噪声（教师输入用）
- $t_d \in [0,1]$: 控制教师输入噪声水平的**固定**时间步（仿真取 0.7，DROID 取 0.8）
- $\boldsymbol{x}_{t_d} = (1-t_d)\boldsymbol{x}^* + t_d \boldsymbol{\epsilon}_d$: 加噪后的完整视频
- $\text{sim}(\cdot, \cdot)$: 在逐 token 非仿射 LayerNorm 后的平均余弦相似度

### 公式2: [[Flow Matching|视频生成正则化损失]]

$$
\mathcal{L}_{\text{gen}}(\theta) := \mathbb{E}_{\boldsymbol{x}^*, \boldsymbol{\epsilon}, t} \left\| \boldsymbol{u}_\theta(\boldsymbol{x}_t, t, \ell) - (\boldsymbol{\epsilon} - \boldsymbol{x}^*) \right\|_2^2
$$

**含义**: 保留预训练视频骨干的视频生成能力作为正则项，防止蒸馏过程中丢失预测性能力。条件：当前帧 latent token 保持干净（timestep=0），未来帧 latent 被噪声扰动。

**符号说明**:
- $\boldsymbol{x}_t = (1-t)\boldsymbol{x}^* + t\boldsymbol{\epsilon}$: $t$ 时刻带噪视频（未来帧加噪，当前帧干净）
- $\boldsymbol{\epsilon} \sim \mathcal{N}(0, \boldsymbol{I})$: 高斯噪声（生成损失用）
- $t \in [0,1]$: 随机时间步，用 $p_{\text{train}}$ 分布采样
- $(\boldsymbol{\epsilon} - \boldsymbol{x}^*)$: 流匹配速度场目标

### 公式3: 总训练目标（Stage 1）

$$
\mathcal{L}(\theta, \psi) := \mathcal{L}_{\text{gen}}(\theta) + \mathcal{L}_{\text{distill}}(\theta, \psi)
$$

**含义**: 视觉编码器同时优化视频生成能力（保留预训练先验）和未来表征蒸馏（注入预测性知识），两者互补。

### 公式4: [[Flow Matching|动作流匹配损失]]（Stage 2）

$$
\mathcal{L}_{\text{act}} := \mathbb{E}_{\boldsymbol{a}^*, \boldsymbol{\epsilon}, t} \left\| \boldsymbol{u}_\phi(\boldsymbol{a}_t, t \mid \boldsymbol{c}, \boldsymbol{q}, \boldsymbol{s}, \ell) - (\boldsymbol{\epsilon} - \boldsymbol{a}^*) \right\|_2^2
$$

**含义**: 以预测性视觉 token $\boldsymbol{c}$、语言 $\ell$、本体状态 $\boldsymbol{q}$、历史 $\boldsymbol{s}$ 为条件，训练动作解码器预测动作块。

**符号说明**:
- $\boldsymbol{u}_\phi$: 动作解码器（1B DiT）
- $\boldsymbol{a}^*$: 干净动作块，$\boldsymbol{a}_t = (1-t)\boldsymbol{a}^* + t\boldsymbol{\epsilon}$
- $\boldsymbol{c} = \boldsymbol{r}_\omega(\text{LN}(\boldsymbol{h}_k))$: 动作读出模块输出的 64 个视觉 token
- $\boldsymbol{q}$: 本体感受状态（proprioception）
- $\boldsymbol{s}$: 可选动作历史（DROID 中使用前 8 步历史）

---

## 关键图表

### Figure 1: Overview — DreamFree 系统概览

![Figure 1: Overview](https://harryjo97.github.io/dreamfree/assets/figures/droid_success_vs_latency_v3.png)

**说明**: DreamFree 将冻结 VDM 的预测知识蒸馏入视觉编码器，推理时仅需单次编码器前向传播生成动作块。右图显示真实世界评估中：DreamFree（Wan-14B）以 91ms 延迟达到 64% 成功率，DreamZero 以 1632ms 仅达 56%，π₀.₅ 以 81ms 仅达 16%，DreamFree 实现 **17.8× 推理加速**。

### Figure 2: 两阶段训练架构

![Figure 2: Architecture](https://harryjo97.github.io/dreamfree/assets/figures/method_v2.png)

**说明**: **(a) Stage 1 — 未来表征蒸馏**：学生编码器用结构化掩码仅从当前帧提取特征，投影器将当前帧表征扩展到未来帧 token 布局，与冻结 VDM 教师表征做余弦对齐。**(b) Stage 2 — 动作 DiT 训练**：冻结编码器，读出模块提取 64 个动作 token，动作 DiT 以此为条件预测动作块。

### Figure 3: 推理架构（Imagination-Free 推理）

![Figure 3: Inference](https://harryjo97.github.io/dreamfree/assets/figures/inference.png)

**说明**: 推理时：(1) 当前帧 + 语言指令 → 视觉编码器（5B 或 14B，timestep=0，单次传播）→ 隐状态 $\boldsymbol{h}_k$；(2) 读出模块 → 64 个动作 token $\boldsymbol{c}$；(3) Action DiT（N 步去噪）→ 32 步动作块，取前 R 步执行。全程无任何视频去噪或 CFG。

### Figure 4: 表征分析（Q1）

**说明**（无法从项目页获取对应图片，见论文 Figure 3）: 四种表征的动作探测（Action Probing）和未来探测（Future Probing）归一化 MSE 对比：

| 条件 | 输入 | Action Probing ↓ | Future Probing ↓ |
|------|------|-----------------|-----------------|
| 当前帧 only | 仅当前帧 | 最高（基准） | 最高 |
| VDM（无蒸馏） | 仅当前帧 | 中等 | 中等 |
| **DreamFree Encoder** | **仅当前帧** | **接近 VDM(未来帧)** | **接近 VDM(未来帧)** |
| VDM（参考） | 当前+未来帧 | 参考上界 | 参考上界 |
| Action Readout（DreamFree） | 仅当前帧 | **最低** | — |

**关键发现**: 蒸馏使编码器以单张当前帧实现接近"直接观测未来"的动作解码能力；动作读出模块进一步提取动作相关信息。

### Figure 5: Fast-WAM vs DreamFree 对比

![Figure 5: Fast-WAM comparison](https://harryjo97.github.io/dreamfree/assets/figures/fastwam_test.png)

**说明**: 展示 DreamFree 与联合 WAM（视频生成）和 Fast-WAM（无视频生成）的训练推理流程对比。DreamFree 通过明确的蒸馏监督使编码器特征包含未来预测信息，这是 Fast-WAM 仅依靠生成损失共享权重所无法实现的。

### Table 1: LIBERO 仿真基准（平均成功率 %）

| 方法 | Avg. SR |
|------|---------|
| VLAs | |
| π₀ | 94.1 |
| π₀.₅ | 96.9 |
| OpenVLA-OFT | 97.1 |
| WAMs | |
| Motus | 97.7 |
| LingBot-VA | 98.5 |
| Fast-WAM | 97.6 |
| **DreamFree（Ours）** | **98.6** |

LIBERO 详细分套结果（Table 7）：

| Method | Spatial | Object | Goal | Long | Avg |
|--------|---------|--------|------|------|-----|
| LingBot-VA | 98.5 | 99.6 | 97.2 | 98.5 | 98.5 |
| Fast-WAM | 98.2 | 100.0 | 97.0 | 95.2 | 97.6 |
| **DreamFree** | **99.0** | **99.8** | **98.8** | 96.6 | **98.6** |

**说明**: DreamFree 在无跨机体预训练的情况下达到 SOTA 性能，验证预测性视觉表征对分布内操作的有效性。

### Table 2: RoboCasa-Kitchen 仿真基准（24 任务平均成功率 %）

| 方法 | Avg. SR |
|------|---------|
| π₀ | 62.5 |
| GR00T-N1.5 | 64.1 |
| FLARE | 66.4 |
| Cosmos Policy | 67.1 |
| Fast-WAM 5B | 44.4 |
| Fast-WAM 14B | 44.7 |
| DreamFree 5B | 58.5 |
| DreamFree 14B | **71.7** |
| DreamFree 5B + extra videos | 68.8 |
| **DreamFree 14B + extra videos** | **74.4** |

**关键发现**: 
- 骨干从 5B 扩到 14B：DreamFree 提升 **+13.2 pp**，Fast-WAM 仅提升 **+0.3 pp**（几乎无提升）
- 额外无标注视频数据使 14B 模型再提升 **+2.7 pp**
- Fast-WAM 表明仅靠生成损失共享权重不足以利用更大视频先验

### Table 3: 消融实验（视觉编码器训练目标）

| 配置 | LIBERO | RoboCasa |
|------|--------|----------|
| **Ours（完整）** | **98.6** | **71.7** |
| w/o $\mathcal{L}_{\text{gen}}$ | 98.0 | 55.6 |
| w/o $\mathcal{L}_{\text{distill}}$ | 94.6 | 47.2 |

**关键发现**: 去掉蒸馏目标 $\mathcal{L}_{\text{distill}}$ 影响最大（RoboCasa 下降 24.5 pp）；生成目标也不可缺少。两者互补，蒸馏是核心。

### Figure 6: 真实世界 DROID 评估（Q4）

![Figure 6: Real-world DROID](https://harryjo97.github.io/dreamfree/assets/figures/droid_real_two_row.png)

**说明**: 10 个真实任务每任务 5 次测试（π₀.₅、DreamZero、DreamFree 对比）。DreamFree 在涉及可变形物体操作（折叠手套、用毛巾覆盖草莓）和连续重排（将螺栓放入碗、将帽子挂上三脚架）的任务上明显优于 DreamZero（DreamZero 在这些任务全部失败）。

**汇总结果**:
| 指标 | DreamFree | DreamZero | π₀.₅ |
|------|-----------|-----------|-------|
| Avg. Success | **64%** | 56% | 16% |
| Avg. Progress | **83%** | 71.8% | 33.5% |

### Figure 7: 推理延迟与任务完成时间

![Figure 7: Latency](https://harryjo97.github.io/dreamfree/assets/figures/droid_latency_completion.png)

**说明**: 左图：H200 GPU 上每动作块延迟（DreamFree 91ms vs DreamZero 1632ms，17.8× 加速；π₀.₅ 81ms，DreamFree 1.1×）。右图：放置 2/4/6 个积木的完成时间，DreamFree 在难度最高的 6 个积木任务中远快于对比方法，体现实时推理优势。

### Figure 8: 任务迁移实验（Q5，论文 Figure 8）

**说明**（见论文图 8）: 三个迁移场景的成功率和进度对比：

**场景1 — 原子到复合任务迁移**（RoboCasa, 同机体）:
- Base（仅原子任务）: 0% 成功率
- + 复合任务视频（无蒸馏，仅 $\mathcal{L}_{\text{gen}}$）: 0% 成功率
- **+ 复合任务视频（含蒸馏）**: **21.7%** 成功率，每个任务至少 1 次成功

**场景2 — 跨机体迁移**（Franka Panda 视频 → ALOHA-AgileX 执行）:
- Base: 6.7% 成功率
- **+ 不同机体视频（含蒸馏）**: **30%** 成功率

**场景3 — 策略训练后视频适配**（RoboTwin, 10 视频演示, 10 分钟微调）:
- Base: 10% 成功率
- + 单视角视频微调: 20% 成功率
- **+ 多视角视频微调**: **45% → 65%** 成功率

---

## 实验结果

### 核心性能汇总

| 基准 | DreamFree（Ours）| 最佳对比方法 | 提升 |
|------|-----------------|-------------|------|
| LIBERO Avg. SR | **98.6%** | OpenVLA-OFT 97.1% | +1.5 pp |
| RoboCasa 14B | **71.7%** | Cosmos Policy 67.1% | +4.6 pp |
| DROID 真实成功率 | **64%** | DreamZero 56% | +8 pp |
| DROID 推理延迟 | **91ms (11Hz)** | DreamZero 1632ms | 17.8× 加速 |
| RoboCasa 骨干扩展 5B→14B | **+13.2 pp** | Fast-WAM +0.3 pp | 44× 更有效 |

### 实验

#### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| [[LIBERO]] | 4 套 × 10 任务 | 标准 VLA 基准 | 仿真训练/测试 |
| [[RoboCasa]]-Kitchen | 24 任务，50 demos/任务 | 厨房多任务，物体场景变化 | 仿真训练/测试 |
| RoboCasa365 | 65 原子任务 | 额外无标注视频 | 视觉编码器预训练 |
| [[DROID]] | 大规模实地 | 多样化真实操作 | 真实世界评估（10 任务）|
| ABC-130k | ~100h 子集 | 双臂机器人 | YAM 双臂实验 |

### 实现细节

- **视频骨干**: Wan2.2-TI2V-5B 和 Wan2.2-I2V-A14B（高噪声专家）
- **动作解码器**: ~1B DiT（30 个 DiT 块，宽度 1024）
- **动作块长度**: 32 步
- **时序下采样**: 4×（9 帧/块）
- **优化器**: AdamW，学习率 1e-4
- **LoRA**: rank-128（LIBERO/RoboCasa），DROID 使用全量训练
- **输入分辨率**: LIBERO 224×448 / RoboCasa 352×352 / DROID 352×640
- **教师噪声**: $t_d$ = 0.7（仿真）/ 0.8（DROID）
- **动作采样**: LIBERO/RoboCasa 10 步欧拉，DROID 16 步，flow shift=5，无 CFG
- **硬件**: 16×NVIDIA H200，约 1 周（DROID 14B）

### 推理优化

- 仅对当前帧做 VAE 编码（非完整视频序列）
- 融合注意力核（fused attention kernels）和融合 QKV/KV 投影
- 视觉和动作 Transformer 块用 `torch.compile` 编译
- CUDA graph 重放 VAE 编码、特征提取和动作采样循环
- 语言嵌入跨查询缓存，动作条件 K/V 每块计算一次复用

---

## 批判性思考

### 优点
1. **推理效率突破**: 通过将预测性知识蒸馏入单次前向传播，以 VLA 相当的延迟获得 WAM 级别的操作性能，解决了 WAM 领域的核心效率瓶颈。
2. **无动作标注学习**: Stage 1 完全不需要动作标注，可利用任意视频数据（包括不同机体），大大扩展了可用训练数据的范围。
3. **可扩展的视频先验**: 明确证明更大视频骨干（5B→14B）能带来可观的性能提升，Fast-WAM 无法做到这一点，说明蒸馏机制才是关键。
4. **真实世界鲁棒性**: 在可变形物体和连续重排任务上超越视频生成 WAM，说明无需生成视频时不准确预测问题的困扰也相应消除。

### 局限性
1. **训练复杂性**: 三阶段流水线（骨干适配→视觉编码器训练→动作模块训练）增加了工程复杂度和计算成本，DROID 14B 需 16 块 H200 训练约 1 周。
2. **体化预训练未探索**: 论文仅在 DROID 上验证了 14B 骨干，未评估更大规模具身预训练（受限于计算资源）。
3. **部分任务失败**: 泵压任务（Press pump）仍失败，推测是动作泛化问题而非视觉理解问题；视频微调受限于基础动作技能的完备性。
4. **联合训练未探索**: 视觉编码器和动作模块联合训练（带梯度阻断）的可行性未验证（参考 Knowledge Insulation）。

### 潜在改进方向
1. 探索编码器和动作模块联合训练，简化流水线
2. 结合更大规模具身预训练数据（如 Open X-Embodiment）
3. 研究 $t_d$ 的自适应调整策略（当前为固定超参数）

### 可复现性评估
- [x] 代码开源（https://github.com/harryjo97/dreamfree）
- [ ] 预训练模型（代码已开源但模型权重状态未明确）
- [x] 训练细节完整（详细 Appendix A/B/C）
- [x] 数据集可获取（LIBERO/RoboCasa/DROID 均为公开数据集）

---

## 关联笔记

### 基于
- [[世界动作模型（WAM）]]: DreamFree 是一种新型 WAM，通过表征蒸馏而非未来视频生成来利用视频先验
- [[REPA]]: 未来表征蒸馏的设计灵感来源，原为生成模型加速的表征对齐技术
- [[Wan 2.2-5B]] / [[Wan 2.2 A14B]]: 视频扩散骨干，高噪声专家版本

### 对比
- [[DreamZero]]: 同骨干的视频生成 WAM，DreamFree 推理速度快 17.8×，真实操作成功率高 8 pp
- [[Fast-WAM]]: 同为无视频生成推理的 WAM，但成功率（44.4%）和骨干扩展性远不如 DreamFree
- [[π₀.₅]]: VLA 基线，DreamFree 以相当延迟达 4× 更高真实世界成功率

### 方法相关
- [[Future Representation Distillation|未来表征蒸馏]]: DreamFree 的核心技术贡献
- [[Flow Matching]]: Stage 1 生成损失和 Stage 2 动作解码损失均基于流匹配
- [[Action Chunking|动作块]]: 预测 32 步动作块，每次执行前 R 步
- [[Perceiver-IO]]: 动作读出模块的设计参考

### 硬件/数据相关
- [[LIBERO]]: 仿真评估基准之一（4 套共 40 任务）
- [[DROID]]: 真实世界评估基准（10 任务，5 rollout/任务）
- [[RoboCasa]]: 厨房多任务仿真基准（24 任务，50 demos/任务）

---

## 速查卡片

> [!summary] Future Representation Distillation for Imagination-Free WAMs (DreamFree)
> - **核心**: 将冻结 VDM 的预测性表征蒸馏入视觉编码器，推理时单次前向传播即可获得"未来感知"视觉特征
> - **方法**: Stage1 蒸馏（$\mathcal{L}_{\text{gen}} + \mathcal{L}_{\text{distill}}$）+ Stage2 动作读出 DiT
> - **结果**: 真实世界成功率 64%（>DreamZero 56%），11Hz 推理（17.8× 加速），LIBERO 98.6%，RoboCasa-14B 71.7%
> - **代码**: [github.com/harryjo97/dreamfree](https://github.com/harryjo97/dreamfree)

---

*笔记创建时间: 2026-11-05*
