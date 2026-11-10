---
title: "LeWAM: A JEPA World Action Model with Diffusion-Steering-Based MPC"
method_name: "LeWAM"
authors: [Shashank Hegde, Alexander Popov, Elie Aljalbout, Nikolai Smolyanskiy]
year: 2026
venue: arXiv
tags: [world-action-model, jepa, diffusion-policy, flow-matching, model-predictive-control, robot-manipulation, imitation-learning]
zotero_collection: Robotics/World Model
image_source: online
arxiv_html: https://arxiv.org/html/2610.12407v2
created: 2026-11-10
---

# 论文笔记：LeWAM: A JEPA World Action Model with Diffusion-Steering-Based MPC

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | NVIDIA |
| 日期 | October 2026 |
| 项目主页 | — |
| 对比基线 | [[Dreamer]], [[LeWM]], [[Cosmos Policy]] |
| 链接 | [arXiv](https://arxiv.org/abs/2610.12407) |

---

## 一句话总结

> LeWAM 是一个基于 [[JEPA]] 的无解码器世界动作模型，通过在噪声空间（而非动作空间）进行 [[MPC|模型预测控制]]，将 [[扩散策略|扩散策略]] 与世界模型统一到一个 backbone 中，同时获得无干扰因素的视觉表征和优越的规划性能。

---

## 核心贡献

1. **无解码器 JEPA 世界动作模型**: 将 [[JEPA]] 表征目标与动作预测统一到同一编码器，完全避免像素重建，使隐空间专注于任务相关状态，对视觉干扰因素免疫。
2. **四模式统一 Transformer**: 一个双向 [[Transformer]] backbone 通过掩码输入同时承担前向动力学、逆向动力学、后向动力学和策略四种功能，所有模式在每个 batch 共同训练。
3. **扩散引导（Diffusion Steering）MPC**: 在 [[流匹配|整流流]] 策略的噪声空间 $\varepsilon$ 而非动作空间进行采样规划，配合球面投影约束候选集，显著提升闭环规划性能。

---

## 问题背景

### 要解决的问题

现有的基于重建的世界动作模型（如 [[Dreamer]]、[[Cosmos Policy|RecWAM]]）需要将视觉容量浪费在与任务无关的像素细节上，导致隐空间被干扰因素污染。同时，在动作空间直接做 [[MPC]] 容易利用动力学模型误差优化出幻觉的未来，实际闭环性能差。

### 现有方法的局限

- **重建型世界模型**（Dreamer）：像素解码消耗模型容量，隐变量编码干扰信息（$R^2_\text{distractor} = 0.19$），无法做到干净的状态表征。
- **纯 JEPA 世界模型**（[[LeWM]]）：表征质量好、无干扰，但缺乏策略头，无法直接用于在线控制。
- **动作空间 MPC**：规划器利用动力学模型错误，优化出不可执行的幻觉轨迹，实际成功率接近 0（3–8%）。

### 本文的动机

JEPA 目标（预测下一帧隐变量而非像素）天然筛去视觉无关信息。将策略头接入同一编码器并以梯度信号对齐动作与表征，同时用噪声空间 MPC 绕开动作空间误差利用问题。

---

## 方法详解

### 模型架构

LeWAM 采用 **无解码器单 Backbone** 架构：

- **输入**: 图像观测序列（3 帧上下文 + 1 帧预测目标），对应 3 段动作块
- **编码器**: [[ViT]] -tiny/14 at 224px，从头训练，输出每帧 384-D [CLS] 隐向量，[[SIGReg]] 正则防止坍塌
- **Backbone**: 深度 8、宽度 512、16 头双向 [[Transformer]]，读取帧隐变量与动作 token
- **策略头**: [[流匹配|整流流]]（Rectified Flow），推理时 8 步 Euler 解码
- **输出**: 5 步动作块 $a_{t:t+5}$（机器人操作 35-D，Transport 70-D）
- **总参数**: 41.0M

### 核心模块

#### 模块 1: 四模式统一训练

**设计动机**: 一个 backbone 通过掩码不同输入实现四种监督信号共同训练。

**具体实现**（掩码选择激活不同模式，所有模式每 batch 共同计算，损失求和）:

- **前向动力学**（Forward）: 掩码最后一帧隐向量，用 MSE 回归下一隐状态（权重 ×2）
- **后向动力学**（Backward）: 掩码第一帧，从后续帧 + 动作回归前一隐状态（权重 ×1）
- **逆动力学**（Inverse）: 保留所有帧，替换目标动作块为加噪整流流样本，预测速度（权重 ×1）
- **策略**（Policy）: 掩码最后一帧 + 加噪目标块，用 **unknown-action token** 替换过去动作防止复制（权重 ×1）

#### 模块 2: SIGReg 正则化

**设计动机**: 无 EMA target 也无 stop-gradient 情况下防止 [[Representation Collapse|隐变量坍塌]]。

**具体实现**: 用 [[SIGReg]]（Spectral Information Geometry Regularization）将帧隐向量正则化朝各向同性高斯分布靠拢，使用 17 knots、1024 projections，权重 0.09 × 4 模式 = 0.36。

#### 模块 3: Diffusion Steering MPC

**设计动机**: 在 [[整流流]] 策略的噪声空间 $\varepsilon \sim \mathcal{N}(0,I)$ 而非动作空间做 [[CEM]]/[[MPPI]]/[[CMA-ES]] 搜索，避免动力学模型误差的利用。

**具体实现**:
- 决策变量改为 $H$ 段噪声 $\varepsilon_{1:H}$
- 每段噪声通过策略头 $g(\varepsilon_h; \hat{\mathbf{z}}_{h-1})$ 解码为动作块
- 对想象的最终隐状态与目标隐状态的距离打分（Eq. 1–2）
- **球面投影**: 在解码前将 $\varepsilon$ 投影到半径 $\sqrt{A}$ 的球面，保证候选集在训练分布的 "typical set" 内

---

## 关键公式

### 公式 1: [[整流流|整流流训练目标]]

$$
a_\sigma = (1-\sigma)\,a + \sigma\,\varepsilon, \quad \sigma \sim \mathcal{U}(0,1), \quad \text{target velocity: } \varepsilon - a
$$

**含义**: 在干净动作 $a$ 和标准高斯噪声 $\varepsilon$ 之间线性插值得到中间状态 $a_\sigma$，策略头预测从噪声到动作的速度向量。

**符号说明**:
- $a$: 真实动作块
- $\varepsilon \sim \mathcal{N}(0, I)$: 标准高斯噪声
- $\sigma \sim \mathcal{U}(0,1)$: 插值系数（训练时随机采样）
- $a_\sigma$: 插值中间状态

### 公式 2: [[整流流|整流流推理]]（8 步 Euler）

$$
a = g(\varepsilon;\,\mathbf{z}_t), \quad \varepsilon \sim \mathcal{N}(0, I)
$$

**含义**: 推理时从高斯噪声出发，经 8 步 Euler 积分通过策略头 $g$ 解码为动作块。

**符号说明**:
- $g(\cdot;\,\mathbf{z}_t)$: 以当前帧隐状态 $\mathbf{z}_t$ 为条件的策略头
- $\mathbf{z}_t$: 当前帧序列的隐向量上下文

### 公式 3: [[MPC|目标隐状态代价]] (Eq. 1)

$$
\hat{c}(\varepsilon_{1:H}) = \big\lVert \hat{f}(\hat{\mathbf{z}}_{H-1}, a_H) - z_g \big\rVert_2
$$

**含义**: 对一组候选噪声方案 $\varepsilon_{1:H}$，评估最终想象隐状态与目标隐状态 $z_g$ 的 L2 距离作为代价。

**符号说明**:
- $\hat{f}(\cdot)$: 前向动力学预测函数
- $\hat{\mathbf{z}}_{H-1}$: 经 $H-1$ 步想象后的隐状态上下文
- $z_g$: 目标帧的隐向量（从最近 demo 帧前 25 步取得）
- $H$: 规划时间段数

### 公式 4: [[Diffusion-Steering|Diffusion Steering 展开]] (Eq. 2)

$$
\begin{aligned}
a_h &= g(\varepsilon_h;\,\hat{\mathbf{z}}_{h-1}) \\
\hat{z}_h &= \hat{f}(\hat{\mathbf{z}}_{h-1}, a_h) \\
\hat{\mathbf{z}}_h &= \operatorname{shift}(\hat{\mathbf{z}}_{h-1}, \hat{z}_h), \quad \hat{\mathbf{z}}_0 = \mathbf{z}_t
\end{aligned}
$$

**含义**: 在想象的未来序列中，每一步先用噪声 $\varepsilon_h$ 解码动作，再通过动力学模型预测下一隐状态，滑动更新上下文窗口 $\hat{\mathbf{z}}$。

**符号说明**:
- $a_h$: 第 $h$ 段解码动作块
- $\hat{z}_h$: 第 $h$ 步预测隐状态
- $\operatorname{shift}(\cdot)$: 滑动窗口更新操作
- $\hat{\mathbf{z}}_0 = \mathbf{z}_t$: 初始化为当前观测隐状态

### 公式 5: [[典型集投影|噪声球面投影]]

$$
\varepsilon \leftarrow \sqrt{A} \cdot \frac{\varepsilon}{\lVert \varepsilon \rVert}
$$

**含义**: 将候选噪声投影到半径 $\sqrt{A}$ 的球面，使每个候选 $\varepsilon$ 都处于整流流训练时的 "typical set"，保证解码出的动作块来自训练分布。

**符号说明**:
- $A$: 动作块维度（决定高斯噪声的典型范数 $\sqrt{A}$）
- $\varepsilon$: 候选噪声向量
- 搜索分布本身仍在欧氏空间进行（仅解码前投影）

---

## 关键图表

### Figure 1: 四模式训练 / Training Modes

![Figure 1 - Training Modes](https://arxiv.org/html/2610.12407v2/figures/method/modes.png)

**说明**: 同一训练窗口（3 帧上下文 + 1 帧目标 + 3 段动作块）通过不同掩码激活四种训练模式：前向、后向、逆向、策略。无任何像素解码，所有模式共享一个编码器和 backbone。

### Figure 2: 模型架构 / Architecture

![Figure 2 - Architecture](https://arxiv.org/html/2610.12407v2/figures/method/arch.png)

**说明**: 单一 ViT-tiny 编码器（带 [[SIGReg]] 正则）输出帧隐向量，送入一个双向 [[Transformer]] backbone，该 backbone 通过输入掩码兼容四种功能：前向/后向/逆向动力学与策略头。

### Figure 3: 任务环境（含干扰因素）/ Task Environments

**PushT（带画点和画条干扰）**

![Figure 3a - PushT](https://arxiv.org/html/2610.12407v2/figures/tasks/pusht.png)

**Lift（带晃动灰色方块干扰）**

![Figure 3b - Lift](https://arxiv.org/html/2610.12407v2/figures/tasks/lift.png)

**Can**

![Figure 3c - Can](https://arxiv.org/html/2610.12407v2/figures/tasks/can.png)

**Square**

![Figure 3d - Square](https://arxiv.org/html/2610.12407v2/figures/tasks/square.png)

**ToolHang**

![Figure 3e - ToolHang](https://arxiv.org/html/2610.12407v2/figures/tasks/tool_hang.png)

**Transport**

![Figure 3f - Transport](https://arxiv.org/html/2610.12407v2/figures/tasks/transport.png)

**说明**: 所有任务均在视觉干扰因素下评测（PushT 画点/条，robomimic 系列晃动方块），仅依赖像素输入，测试模型的干扰鲁棒性。

### Figure 4: 想象展开可视化 / Imagination Rollouts

**PushT 想象展开**

![Figure 4a - PushT Imagination](https://arxiv.org/html/2610.12407v2/figures/results/pusht_imagination_holdout.png)

**Lift 想象展开**

![Figure 4b - Lift Imagination](https://arxiv.org/html/2610.12407v2/figures/results/lift_imagination_holdout.png)

**说明**: 给定 3 帧真实上下文后，LeWAM 用前向动力学在隐空间想象 5 帧未来（真实 GT 动作），验证动力学模型的准确性。

### Figure 5: 闭环平均成功率 / Average Closed-Loop Success

![Figure 5 - Average Closed-Loop](https://arxiv.org/html/2610.12407v2/figures/results/closed_loop_ep300_average.png)

**说明**: LeWAM 的策略头闭环性能接近参考流匹配策略（上界），DS-CEM 噪声空间规划进一步提升表现。动作空间 MPC 对所有模型均造成严重性能下降（3–8%）。

### Figure 6: 各任务闭环成功率（附录）/ Per-Task Results

![Figure 6 - Per-Task Results](https://arxiv.org/html/2610.12407v2/figures/results/closed_loop_ep300_per_task.png)

**说明**: 分任务展示所有模型与规划器组合的闭环成功率（5 seeds × 50 评测 seeds），LeWAM + DS-CEM 在多数任务上取得最高成绩。

### Table 1: 世界动作模型特性对比

| 模型 | 无解码器 | 动作梯度至编码器 | 策略头 | 前向动力学 | 逆向动力学 | 后向动力学 | 多模态 | 单 Backbone |
|------|----------|----------------|--------|------------|------------|------------|--------|-------------|
| Dreamer | — | — | ✓ | ✓ | — | — | — | — |
| TD-MPC2 | ✓ | — | ✓ | ✓ | — | — | — | — |
| DINO-WM / V-JEPA 2-AC | ✓ | — | — | ✓ | — | — | — | — |
| LeWM | ✓ | — | — | ✓ | — | — | — | — |
| UWM | — | — | ✓ | ✓ | ✓ | — | ✓ | ✓ |
| Cosmos Policy | — | — | ✓ | ✓ | ✓ | — | ✓ | ✓ |
| **LeWAM（ours）** | **✓** | **✓** | **✓** | **✓** | **✓** | **✓** | **✓** | **✓** |

**表格说明**: LeWAM 是唯一同时满足所有特性的模型，尤其是动作损失反向传播到编码器（Alignment）和后向动力学两项。

### Table 2: 线性探测结果（6 任务平均）

| 模型 | $R^2$ 任务状态 ↑ | $R^2$ 干扰因素 ↓ | Pearson r 任务状态 ↑ | Pearson r 干扰因素 ↓ | MSE 任务状态 ↓ | MSE 干扰因素 ↑ |
|------|-----------------|-----------------|---------------------|---------------------|----------------|----------------|
| Dreamer | 0.68 ± 0.00 | 0.19 ± 0.01 | 0.83 ± 0.00 | 0.34 ± 0.01 | 0.32 ± 0.00 | 0.81 ± 0.01 |
| LeWM + diff. head | 0.62 ± 0.01 | 0.00 ± 0.01 | 0.78 ± 0.01 | 0.03 ± 0.01 | 0.39 ± 0.01 | 1.04 ± 0.01 |
| RecWAM | 0.78 ± 0.01 | 0.73 ± 0.01 | 0.87 ± 0.00 | 0.85 ± 0.00 | 0.22 ± 0.00 | 0.27 ± 0.01 |
| **LeWAM（ours）** | **0.80 ± 0.00** | **0.00 ± 0.01** | **0.89 ± 0.00** | **0.03 ± 0.01** | **0.21 ± 0.00** | **1.10 ± 0.01** |

**表格说明**: LeWAM 实现最高任务状态 R²（0.80）同时干扰因素 R² 为零——只有重建型模型（RecWAM $R^2_\text{dist}=0.73$）才对干扰物建模。

### Table 3: 四模式消融（robomimic, 50 episodes）

| 任务 | 变体 | $R^2$ 状态 ↑ | 策略成功率 % ↑ | DS-CEM 成功率 % ↑ |
|------|------|-------------|---------------|------------------|
| Lift | 全四模式 | **0.423 ± 0.019** | **94.0 ± 4.2** | **97.6 ± 0.8** |
| Lift | 去前向 | 0.323 ± 0.045 | 91.6 ± 2.3 | 95.6 ± 2.9 |
| Lift | 去逆向 | 0.386 ± 0.031 | 92.0 ± 5.5 | 95.6 ± 2.9 |
| Lift | 去策略 | 0.359 ± 0.019 | — | 0.8 ± 1.6 |
| Lift | 去后向 | 0.370 ± 0.056 | 88.0 ± 4.6 | 93.6 ± 2.3 |
| Can | 全四模式 | **0.839 ± 0.004** | **82.8 ± 6.1** | **78.8 ± 4.5** |
| Can | 去前向 | 0.781 ± 0.018 | 77.6 ± 8.3 | 74.4 ± 7.7 |
| Can | 去逆向 | 0.817 ± 0.013 | 79.6 ± 8.0 | 73.2 ± 8.2 |
| Can | 去策略 | 0.761 ± 0.010 | — | 0.0 ± 0.0 |
| Can | 去后向 | 0.801 ± 0.015 | 77.6 ± 3.9 | 77.6 ± 2.7 |
| Square | 全四模式 | **0.896 ± 0.007** | 80.4 ± 1.5 | **87.2 ± 4.7** |
| Square | 去前向 | 0.876 ± 0.003 | 71.2 ± 8.4 | 73.6 ± 1.5 |
| Square | 去逆向 | 0.885 ± 0.003 | **82.0 ± 3.8** | 81.6 ± 3.2 |
| Square | 去策略 | 0.858 ± 0.015 | — | 0.0 ± 0.0 |
| Square | 去后向 | 0.872 ± 0.007 | 74.0 ± 7.7 | 82.8 ± 3.7 |

**关键发现**: 去掉策略模式导致 DS-CEM 成功率归零（扩散引导必须有策略头）；每种模式缺失都会不同程度降低表征质量和闭环性能。

### Table 4: 球面投影消融（DS-CEM 成功率 %）

| 任务 | 有 $\sqrt{A}$ 球面投影 | 无投影 |
|------|----------------------|--------|
| Lift | **97.6 ± 0.8** | 93.0 ± 3.0 |
| Can | **78.8 ± 4.5** | 23.0 ± 8.0 |
| Square | **87.2 ± 4.7** | 75.0 ± 3.0 |

**关键发现**: 在 Can 任务上投影使成功率从 23% 提升至 79%，说明保持噪声候选集在训练分布内对规划质量至关重要。

### Table 5: LeWAM 超参数配置

| 项目 | 设置 |
|------|------|
| 数据 | 每任务 200 demos，224px，frameskip 5，3 帧上下文 + 1 帧目标，5 步动作块（操作 35-D，Transport 70-D），z-score 归一化 |
| 训练 | 300 epochs，batch 128，AdamW（lr 5×10⁻⁵，weight decay 10⁻³），warm-up + cosine，bf16，grad clip 1.0 |
| 编码器 | ViT-tiny/14 at 224px，[CLS] token，MLP projector（hidden 2048）→ 384-D |
| 正则化 | SIGReg，权重 0.36（0.09 × 4 模式），17 knots，1024 projections |
| Backbone | 双向 Transformer，depth 8，width 512，16 头 × 64，MLP 2048 |
| 模式权重 | 前向 ×2，逆向/策略/后向 ×1 |
| 策略头 | 整流流，推理 8 步 Euler |
| 评测 | 每 seed 50 个固定起始，robomimic 任务时间步：Lift 100，Can/Square 400，ToolHang/Transport 700 |
| 规划器 | 64 样本，3 迭代，8 elites（CEM/MPPI/CMA-ES） |

---

## 实验

### 数据集与任务

| 数据集 | 规模 | 特点 | 任务 |
|--------|------|------|------|
| PushT | 200 demos | 2D 推挡，视觉干扰（点+条） | 推目标到指定位置 |
| robomimic Lift | 200 demos | 单臂拾取，晃动灰色干扰方块 | 抓起方块 |
| robomimic Can | 200 demos | 单臂，晃动干扰方块 | 将罐子放入目标区 |
| robomimic Square | 200 demos | 单臂，晃动干扰方块 | 方形零件精密插入 |
| robomimic ToolHang | 200 demos | 单臂，168px 侧视 | 悬挂工具（高精度） |
| robomimic Transport | 200 demos | 双臂，双肩相机拼接 256px | 跨容器传送物体 |

### 实现细节

- **Backbone**: ViT-tiny/14 编码器（DINOv2-small 结构），从零训练
- **优化器**: AdamW，lr = 5×10⁻⁵，weight decay = 10⁻³
- **Batch Size**: 128
- **训练轮数**: 300 epochs
- **精度**: bf16
- **模型大小**: LeWAM 41.0M 参数（与所有 baseline 对齐至 ~41-43M）
- **相机**: PushT 96px→224px，robomimic 84-256px→224px（见附录 B）

### 可视化结果

想象展开（Figure 4）显示 LeWAM 能在隐空间准确预测未来 5 帧状态（给定真实动作），验证其前向动力学模型的有效性，PushT 和 Lift 任务下均有良好预测。

---

## 批判性思考

### 优点

1. **一体化设计**: 四模式单 backbone 统一表征、动力学和策略，参数高效且各模式互相强化。
2. **表征质量**: 在所有基线中唯一同时实现高任务状态 R²（0.80）和零干扰因素 R²（0.00）。
3. **噪声空间 MPC 的洞察**: 识别到动作空间 MPC 对小数据集容易误用动力学误差这一根本问题，噪声投影的解法简洁有效。
4. **消融完整**: 四模式、投影、规划器类型均有系统消融，结论可信。

### 局限性

1. **目标隐向量依赖 Demo**: 规划需要目标帧的隐向量 $z_g$（从最近 demo 帧前 25 步取得），推理时不完全"无需 demo"。
2. **规划速度**: 采样 MPC 需要多次前向动力学展开，实时性有待评估。
3. **仅 sim 实验**: 所有实验在仿真环境中进行，实际机器人验证缺失。
4. **单相机观测**: 仅使用单相机（除 Transport），多相机/3D 感知未探索。

### 潜在改进方向

1. **目标条件无关化**: 探索不依赖目标隐向量的规划（如从语言目标）。
2. **实机迁移**: 在现实机器人上验证，处理更多动态无关特征。
3. **更快规划**: 结合 MPPI 并行化或 diffusion guidance 加速推理。

### 可复现性评估

- [ ] 代码开源（论文中未提及）
- [ ] 预训练模型
- [x] 训练细节完整（附录 B-D 详细）
- [x] 数据集可获取（robomimic、PushT 均公开）

---

## 关联笔记

### 基于

- [[LeWM]]: 前置工作，LeWAM 在此基础上增加动作梯度和四模式训练
- [[JEPA]]: 核心表征学习框架，联合嵌入预测架构
- [[Dreamer]]: 重建型世界模型基线
- [[扩散策略|Diffusion Policy]]: 策略头的理论基础

### 对比

- [[Dreamer]]: 重建型 baseline，$R^2_\text{dist}=0.19$ 验证像素重建的问题
- [[Cosmos Policy]]: RecWAM baseline，重建型 WAM，$R^2_\text{dist}=0.73$
- [[LeWM]]: 无策略头的 JEPA 世界模型版本

### 方法相关

- [[流匹配|整流流 (Rectified Flow)]]: 策略头使用的生成模型框架
- [[MPC|模型预测控制]]: 规划框架
- [[CEM]]: 交叉熵方法，默认规划器
- [[MPPI]]: 另一规划器选项
- [[CMA-ES]]: 另一规划器选项
- [[SIGReg]]: 防坍塌正则化方法
- [[ViT]]: 视觉编码器架构

### 硬件/数据相关

- [[robomimic]]: 主实验数据集
- [[PushT]]: 2D 操作任务数据集

---

## 速查卡片

> [!summary] LeWAM: JEPA World Action Model with Diffusion Steering MPC
> - **核心**: 无解码器 JEPA WAM，动作梯度对齐表征与策略
> - **方法**: 四模式单 Backbone + 噪声空间 DS-CEM 规划
> - **结果**: 任务状态 R²=0.80，干扰 R²=0.00，DS-CEM 在 Can 任务 78.8% vs 动作空间 MPC ~5%
> - **代码**: 未公开

---

*笔记创建时间: 2026-11-10*
