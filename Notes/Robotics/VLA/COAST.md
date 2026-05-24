---
title: "Contrastive Conceptor Activation Steering (COAST): Unlocking Vision-Language-Action Models through Hidden States"
method_name: "COAST"
authors: [Miranda Muqing Miao, Subin Kim, Brandon Yang, Lyle Ungar]
year: 2025
venue: arXiv (Submitted to NeurIPS 2026)
tags: [vla, activation-steering, inference-time, conceptor, representation-engineering, robot-manipulation, training-free]
zotero_collection: Robotics/VLA
image_source: online
arxiv_html: https://arxiv.org/html/2605.17144v1
created: 2026-05-20
---

# 论文笔记：COAST

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | University of Pennsylvania |
| 日期 | May 2026 |
| 项目主页 | N/A |
| 对比基线 | [[CAA]]、[[SAE Steering]]、[[SFT]] |
| 链接 | [arXiv](https://arxiv.org/abs/2605.17144) / Code: N/A |

---

## 一句话总结

> COAST 是一种无需训练的推理时激活引导方法，通过构建成功/失败对比[[Conceptor|概念算子]]对冻结 VLA 的残差流施加乘法门控，在仿真中提升 20%+、在真实机器人上提升 40%+。

---

## 核心贡献

1. **Contrastive Conceptor 引导**: 从少量成功/失败 rollout 激活中构建[[Conceptor|概念算子]]，通过布尔 AND-NOT 运算隔离任务关键子空间，推理时对 VLA 激活施加乘法门控。
2. **几何洞察**: 发现跨任务"失败表征共享结构"但"成功表征保持任务特异性"，揭示 VLA 隐藏状态中存在未被利用的任务相关知识。
3. **高效超参数选择**: 三阶段启发式选择仅评估超参数网格 3-8%，即可恢复 oracle 性能的 93%。

---

## 问题背景

### 要解决的问题

现有 [[VLA|视觉-语言-动作模型]] 在部署时常因任务分布偏移或模型能力不足导致失败，而重新训练成本极高。如何在不修改模型权重的情况下，在推理阶段提升 VLA 的成功率？

### 现有方法的局限

- [[SFT|监督微调]]：需要大量成功示例，成本高，且容易遗忘。
- [[CAA|对比激活加法]]：使用加法向量进行干预，未充分利用子空间结构。
- [[SAE Steering|稀疏自编码器引导]]：需要训练 SAE，且对 VLA 动作专家适配性弱。

### 本文的动机

VLA 模型在成功和失败 rollout 中的隐藏状态呈现出可分离的子空间结构。通过线性算子（Conceptor）捕获这些子空间，可以在推理时引导模型激活向成功区域移动，而无需任何梯度更新。

---

## 方法详解

### 模型架构

COAST 采用**推理时激活引导**框架，作用于冻结的 VLA 模型：

- **目标网络**: 支持 [[Flow Matching|流匹配]] VLA（[[Pi05|π0.5]]、[[GR00T|GR00T N1.5]]）、[[自回归策略|自回归]] VLA（π0-FAST）以及[[Diffusion Policy|扩散策略]]
- **作用位置**: 动作专家（Action Expert）的[[Residual Stream|残差流]]，特定层 $\ell$ 的激活
- **核心模块**: [[Conceptor|概念算子]] 用于子空间建模，[[Boolean Logic|布尔运算]] 用于对比合成
- **输出**: 门控后的激活 $h'$，经下游处理生成动作

### 核心模块

#### 模块 1: Conceptor 构建

**设计动机**: 利用[[Conceptor|主成分分析的软化变体]]的软投影特性，通过孔径参数 $\alpha$ 控制子空间覆盖范围。

**具体实现**:
- 收集成功/失败 rollout 中特定层的激活，均值中心化后组成矩阵 $\tilde{X}$
- 计算协方差矩阵 $R = \frac{1}{N}\tilde{X}^T\tilde{X}$
- 通过闭式解构造 Conceptor：$C = R(R + \alpha^{-2}I)^{-1}$
- 特征值结构：高方差方向通过（$\mu_i \approx 1$），低方差方向被抑制（$\mu_i \approx 0$）

#### 模块 2: 对比 Conceptor 合成

**设计动机**: 通过[[Conceptor Boolean Logic|布尔 AND-NOT 运算]]隔离"成功但非失败"的子空间，消除任务无关的共享失败方向。

**具体实现**:
- 布尔 NOT：$\neg C = I - C$（翻转子空间）
- 布尔 AND：$A \land B = (A^{-1} + B^{-1} - I)^{-1}$（软交集）
- 合成引导 Conceptor：$C_{\text{steer}} = C_{\text{success}} \land \neg C_{\text{failure}}$
- 最终引导矩阵：$M = (1-\beta)I + \beta C_{\text{steer}}$（插值控制引导强度）

#### 模块 3: 推理时乘法门控

**设计动机**: 不同于加法干预，乘法门控保持激活方向的相对缩放，更适合结构化子空间投影。

**具体实现**:
- 对残差流激活施加：$h' = hM^\top$
- 支持全局（Global）策略：对整个轨迹统一引导
- 支持逐步（Per-step）策略：根据去噪步骤动态调整，适合长时序任务

---

## 关键公式

### 公式 1: [[Conceptor|Conceptor 优化目标]]

$$
\min_C \frac{1}{N}\|\tilde{X} - \tilde{X}C\|_F^2 + \alpha^{-2}\|C\|_F^2
$$

**含义**: 最小化激活重构误差，同时通过孔径参数 $\alpha$ 正则化 Conceptor 复杂度。

**符号说明**:
- $C$: Conceptor 矩阵（正半定线性算子）
- $\tilde{X}$: 均值中心化后的激活矩阵，形状 $N \times d$
- $N$: rollout 激活样本数
- $\alpha$: 孔径参数，控制正则化强度
- $\|\cdot\|_F$: Frobenius 范数

### 公式 2: [[Conceptor|Conceptor 特征值变换]]

$$
\mu_i = \frac{\lambda_i}{\lambda_i + \alpha^{-2}}
$$

**含义**: Conceptor 的特征值由激活协方差特征值 $\lambda_i$ 和孔径 $\alpha$ 共同决定，实现软带通滤波。

**符号说明**:
- $\mu_i$: Conceptor $C$ 的第 $i$ 个特征值
- $\lambda_i$: 协方差矩阵 $R$ 的第 $i$ 个特征值
- $\alpha$: 孔径参数

### 公式 3: [[Conceptor Boolean Logic|布尔 AND 运算]]

$$
A \land B = (A^{-1} + B^{-1} - I)^{-1}
$$

**含义**: Conceptor 代数中的软交集运算，计算同时被两个算子保留的激活方向。

**符号说明**:
- $A, B$: 两个 Conceptor 矩阵
- $I$: 单位矩阵

### 公式 4: [[Contrastive Conceptor|对比 Conceptor 构建]]

$$
C_{\text{steer}} = C_{\text{success}} \land \neg C_{\text{failure}}
$$

**含义**: 将成功 Conceptor 与失败 Conceptor 的补集做交集，隔离"成功特有"的判别子空间。

**符号说明**:
- $C_{\text{steer}}$: 引导用 Conceptor
- $C_{\text{success}}$: 从成功 rollout 激活构建的 Conceptor
- $C_{\text{failure}}$: 从失败 rollout 激活构建的 Conceptor
- $\neg$: 布尔 NOT 运算，$\neg C = I - C$

### 公式 5: [[Activation Steering|乘法门控矩阵]]

$$
M = (1 - \beta)I + \beta C_{\text{steer}}
$$

**含义**: 在原始激活和完全引导激活之间进行插值，$\beta$ 控制引导强度。

**符号说明**:
- $M$: 门控矩阵
- $\beta \in [0, 1]$: 引导强度参数
- $I$: 单位矩阵（无引导的恒等变换）

### 公式 6: [[Activation Steering|残差流激活引导]]

$$
h' = hM^\top
$$

**含义**: 对动作专家残差流激活施加乘法门控，在前向传播中引导模型激活向成功子空间移动。

**符号说明**:
- $h$: 原始残差流激活向量
- $h'$: 引导后的激活向量
- $M^\top$: 门控矩阵的转置

### 公式 7: [[Subspace Similarity|成功-失败子空间重叠相似度]]

$$
\text{sim}(C^+, C^-) = \frac{\text{tr}(C^+ C^-)}{\sqrt{\text{tr}((C^+)^2)\,\text{tr}((C^-)^2)}}
$$

**含义**: 量化成功与失败 Conceptor 间的子空间重叠程度，与引导增益正相关（$\rho = 0.59, p = 0.002$）。

**符号说明**:
- $C^+$: 成功 Conceptor
- $C^-$: 失败 Conceptor
- $\text{tr}(\cdot)$: 矩阵迹运算

### 公式 8: [[Cross-Task Transfer|跨任务失败子空间包容度]]

$$
\frac{\text{tr}(C_{\text{src}}^f C_{\text{tgt}}^f)}{\text{tr}((C_{\text{src}}^f)^2)}
$$

**含义**: 衡量源任务失败子空间对目标任务失败子空间的包容程度，预测跨任务迁移引导效果（$r = 0.30$-$0.49, p < 0.005$）。

**符号说明**:
- $C_{\text{src}}^f$: 源任务失败 Conceptor
- $C_{\text{tgt}}^f$: 目标任务失败 Conceptor

---

## 关键图表

### Figure 1: COAST 方法概览

![Figure 1](https://arxiv.org/html/2605.17144v1/x1.png)

**说明**: COAST 的整体流程。在推理时，从少量 rollout 激活中构建对比 [[Conceptor|概念算子]]，通过乘法门控引导冻结 VLA 的[[Residual Stream|残差流]]激活，无需修改模型权重。

### Figure 2: 实验环境概览

![Figure 2](https://arxiv.org/html/2605.17144v1/x2.png)

**说明**: 三个仿真 benchmark（MetaWorld ML45、LIBERO-10、RoboCasa 子集）加上 DROID 平台的三个真实机器人任务。难度依次递增，覆盖不同操作复杂度。

### Figure 3: 判别子空间的低秩结构与可预测性

![Figure 3](https://arxiv.org/html/2605.17144v1/x4.png)

**说明**:
- **面板 A**: MetaWorld ML45 上成功/失败 Conceptor 的特征值谱快速衰减，表明结果相关计算占据 1024 维残差流的约 1%（低秩但非秩一）。
- **面板 B**: 逐任务引导增益与成功-失败子空间重叠的散点图，相关系数 $\rho = 0.59$，$p = 0.002$，验证重叠程度可预测引导效果。

### Figure 4: 引导后激活向成功区域偏移

![Figure 4](https://arxiv.org/html/2605.17144v1/x5.png)

**说明**:
- **面板 A**: 将逐步激活投影到顶部特征向量上，显示引导后激活质心向成功区域移动。
- **面板 B**: 沿归一化轨迹时间展示投影演化，证明引导在去噪步骤中保持轨迹稳定性。

### Figure 5: 跨任务失败几何共享

![Figure 5](https://arxiv.org/html/2605.17144v1/x6.png)

**说明**:
- **左图**: 多任务激活的联合 PCA 可视化，失败激活在各任务中分布于共享区域，成功激活保持任务特异性。
- **右图**: 失败子空间包容度与迁移增益的相关散点图（$r = 0.30$-$0.49$，$p < 0.005$）；成功子空间包容度与迁移增益无相关（$|r| < 0.13$，$p > 0.4$）。

### Table 1: 各 Benchmark 全量性能对比

Contrastive conceptors 在每对模型-benchmark 上均取得最大且最稳定的提升。

#### RoboCasa 子集（7 任务）

| Task | π0.5 Base | +SFT | +SAE | +CAA | +Glob. | +Per. | +Pos. | GR00T Base | +Glob. | +Per. | DP Base | +Glob. | +Per. |
|------|-----------|------|------|------|--------|-------|-------|-----------|--------|-------|---------|--------|-------|
| Close Fridge | 0.20 | 0.03 | 0.47 | 0.27 | 0.47 | 0.40 | 0.40 | 0.67 | 0.93 | 1.00 | 0.43 | 0.56 | 0.60 |
| Coffee Mug | 0.13 | 0.23 | 0.33 | 0.13 | 0.33 | 0.33 | 0.53 | 0.20 | 0.27 | 0.33 | 0.13 | 0.23 | 0.16 |
| Open Drawer | 0.53 | 0.47 | 0.60 | 0.53 | 0.67 | 0.53 | 0.67 | 0.53 | 0.80 | 0.53 | 0.10 | 0.33 | 0.30 |
| Stand Mixer | 0.67 | 0.30 | 0.53 | 0.53 | 0.73 | 0.80 | 0.53 | 0.60 | 0.80 | 0.80 | 0.63 | 0.83 | 0.83 |
| PP Cabinet | 0.60 | 0.50 | 0.67 | 0.67 | 0.74 | 0.74 | 0.67 | 0.73 | 0.80 | 0.80 | 0.30 | 0.46 | 0.56 |
| PP Stove | 0.33 | 0.43 | 0.53 | 0.53 | 0.53 | 0.67 | 0.60 | 0.73 | 0.87 | 0.75 | 0.10 | 0.13 | 0.13 |
| Kettle | 0.33 | 0.20 | 0.27 | 0.27 | 0.40 | 0.40 | 0.53 | 0.67 | 0.80 | 0.73 | 0.56 | 0.60 | 0.66 |
| **Mean** | **0.40** | 0.31 | 0.49 | 0.42 | 0.55 | 0.55 | 0.56 | **0.59** | **0.75** | **0.71** | **0.32** | **0.45** | **0.46** |
| Δ | – | −0.09 | +0.09 | +0.02 | +0.15 | +0.15 | +0.16 | – | +0.16 | +0.12 | – | +0.13 | +0.14 |
| p_t | – | .195 | .172 | .642 | .002** | .009** | .044* | – | .002** | .039* | – | .004** | .006** |

#### LIBERO-10（10 任务）

| Task | π0.5 Base | +SAE | +CAA | +Glob. | +Per. | π0-FAST Base | +Glob. | +Per. |
|------|-----------|------|------|--------|-------|--------------|--------|-------|
| Stove+Moka | 0.53 | 0.67 | 0.67 | 0.93 | 0.87 | 0.80 | 1.00 | 0.87 |
| Bowl+Drawer | 0.40 | 0.80 | 0.47 | 0.73 | 0.80 | 0.67 | 0.93 | 0.93 |
| Mug+Micro | 0.13 | 0.20 | 0.13 | 0.40 | 0.60 | 0.47 | 0.73 | 0.80 |
| Two Mokas | 0.20 | 0.13 | 0.07 | 0.47 | 0.53 | 0.33 | 0.53 | 0.53 |
| Soup+Cheese | 0.80 | 0.73 | 0.80 | 0.93 | 1.00 | 0.67 | 0.87 | 0.87 |
| Soup+Tomato | 0.40 | 0.80 | 0.53 | 0.80 | 0.80 | 0.67 | 0.87 | 0.93 |
| Cheese+Butter | 0.60 | 0.80 | 0.60 | 0.93 | 0.93 | 1.00 | – | – |
| Mugs+Plates | 0.07 | 0.33 | 0.07 | 0.60 | 0.60 | 0.60 | 0.87 | 0.87 |
| Mug+Choc | 0.53 | 0.73 | 0.60 | 0.87 | 0.87 | 0.73 | 0.93 | 0.93 |
| Book+Caddy | 0.67 | 0.87 | 0.80 | 0.93 | 1.00 | 0.60 | 0.87 | 0.87 |
| **Mean** | **0.43** | 0.61 | 0.47 | **0.76** | **0.80** | **0.65** | **0.84** | **0.84** |
| Δ | – | +0.17 | +0.04 | +0.33 | **+0.37** | – | +0.23 | +0.23 |
| p_t | – | .009** | .157 | <.001*** | <.001*** | – | <.001*** | <.001*** |

#### MetaWorld ML45（10 任务代表）

| Task | π0.5 Base | +SAE | +CAA | +Glob. | +Per. | π0-FAST Base | +Glob. | +Per. |
|------|-----------|------|------|--------|-------|--------------|--------|-------|
| coffee-push | 0.80 | 1.00 | 0.87 | 1.00 | 1.00 | 0.88 | 0.94 | 0.94 |
| push | 0.93 | 0.93 | 0.93 | 1.00 | 1.00 | 0.75 | 0.94 | 0.88 |
| pick-place | 0.87 | 0.80 | 0.87 | 1.00 | 1.00 | 0.69 | 0.88 | 0.88 |
| plate-slide-back | 0.60 | 0.53 | 0.53 | 0.93 | 0.93 | 0.88 | 0.88 | 0.88 |
| faucet-close | 0.80 | 1.00 | 0.93 | 1.00 | 0.93 | 0.44 | 0.69 | 0.69 |
| pick-place-wall | 0.20 | 0.33 | 0.40 | 0.87 | 1.00 | 0.69 | 0.81 | 0.75 |
| reach | 0.93 | 0.93 | 1.00 | 0.93 | 0.93 | 0.81 | 0.81 | 0.81 |
| coffee-pull | 0.93 | 1.00 | 1.00 | 0.93 | 0.93 | 0.62 | 0.69 | 0.62 |
| disassemble | 0.60 | 0.80 | 0.73 | 0.93 | 0.93 | 0.56 | 0.69 | 0.56 |
| stick-push | 0.20 | 0.40 | 0.60 | 0.80 | 0.73 | 0.81 | 0.88 | 0.88 |
| **Mean** | **0.69** | 0.77 | 0.79 | **0.94** | **0.94** | **0.71** | **0.82** | **0.79** |
| Δ | – | +0.08 | +0.10 | +0.25 | **+0.25** | – | +0.11 | +0.08 |
| p_t | – | .064 | .039* | .008** | .012* | – | .003** | .023* |

**关键发现**: COAST（Global/Per-step）在所有 benchmark 上均显著优于 SFT、SAE 引导和 CAA，且 SFT 反而降低了部分模型的性能。

### Table 2: 跨任务迁移实验

跨任务 Conceptor 迁移：源任务的失败子空间包容度预测迁移成功，$r = 0.30$-$0.49$，$p < 0.005$。

#### LIBERO-10

| Target Task | Self SR Δ | Top-1 Source (Δ) | Top-2 Source (Δ) |
|-------------|-----------|-----------------|-----------------|
| Stove+Moka | +0.40 | Two Mokas (PS) +0.60 | Soup+Tomato (G) +0.33 |
| Bowl+Drawer | +0.40 | Soup+Tomato (PS) +0.20 | Mug+Pudding (PS) +0.14 |
| Mug+Micro | +0.47 | Two Mokas (PS) +0.20 | Mug+Pudding (PS) −0.13 |
| Two Mokas | +0.40 | Mugs+Plates (PS) +0.40 | Mug+Pudding (PS) −0.06 |
| Soup+Cheese | +0.20 | Two Mokas (G) +0.67 | Soup+Tomato (G) +0.47 |
| Soup+Tomato | +0.40 | Bowl+Drawer (PS) +0.27 | Bowl+Drawer (PS) +0.27 |
| Cheese+Butter | +0.33 | Mug+Micro (G) +0.80 | Soup+Tomato (PS) +0.60 |
| Mugs+Plates | +0.60 | Mug+Micro (G) +0.60 | Soup+Tomato (PS) +0.40 |
| Mug+Pudding | +0.34 | Mug+Micro (G) +0.67 | Soup+Tomato (PS) +0.40 |
| Book+Caddy | +0.33 | Soup+Tomato (PS) +0.53 | Soup+Tomato (G) +0.47 |

#### RoboCasa

| Target Task | Self SR Δ | Top-1 Source (Δ) | Top-2 Source (Δ) |
|-------------|-----------|-----------------|-----------------|
| CloseFridge | +0.27 | PP_Stove (PS) +0.07 | OpenDrawer (PS) −0.20 |
| CoffeeSetup | +0.20 | Kettle (G) +0.00 | OpenDrawer (PS) −0.20 |
| OpenDrawer | +0.14 | CoffeeSetup (PS) +0.60 | CoffeeSetup (PS) +0.47 |
| StandMixer | +0.13 | CoffeeSetup (G) +0.54 | CoffeeSetup (G) +0.53 |
| PP_Cabinet | +0.13 | CoffeeSetup (PS) +0.54 | CoffeeSetup (G) +0.47 |
| PP_Stove | +0.34 | CoffeeSetup (G) +0.40 | CoffeeSetup (PS) +0.34 |
| Kettle | +0.07 | CoffeeSetup (PS) +0.14 | CloseFridge (PS) +0.07 |

**关键发现**: 跨任务迁移有时超越自适应引导（如 Cheese+Butter 的 Top-1 达 +0.80 vs Self +0.33），验证了失败几何结构的跨任务共享性。

---

## 实验

### 数据集 / Benchmarks

| Benchmark | 规模 | 特点 | 用途 |
|-----------|------|------|------|
| [[MetaWorld ML45]] | 45 任务，10 任务评测 | 多样操作任务 | 仿真测试 |
| [[LIBERO-10]] | 10 任务 | 长时序操作 | 仿真测试 |
| [[RoboCasa]] | 7 任务子集 | 厨房场景，难度较高 | 仿真测试 |
| [[DROID]] | 3 真实任务 | Open Drawer、Close Microwave、Put Duck in Cabinet | 真实机器人测试 |

### 实现细节

- **激活提取**: 对动作 token 做均值池化，取特定层 $\ell$ 的残差流激活
- **超参数**: 孔径 $\alpha$、引导强度 $\beta$、层索引 $\ell$
- **高效超参数选择**: 三阶段启发式（层选择→孔径选择→强度扫描），仅评估 3-8% 网格，恢复 93% oracle 性能
- **所需 rollout 数**: 少量成功/失败示例（具体数量未公开）
- **计算开销**: 与 SAE/CAA 相近，推理延迟最小

### 真实机器人结果（DROID 平台）

每任务 15 次独立试验：
- **Open Drawer**: COAST 取得最大增益
- **Close Microwave**: 仅需 1 次成功 rollout，成功率提升至 46%
- **Put Duck in Cabinet**: 三任务平均提升 **40%**

### 消融 / Baseline 对比

| 变体 | 描述 |
|------|------|
| +SFT | 监督微调（需要大量数据，通常降低性能） |
| +SAE | 稀疏自编码器引导 |
| +CAA | 对比激活加法（加法向量干预） |
| +Glob. | COAST 全局策略（整轨迹统一引导） |
| +Per. | COAST 逐步策略（按去噪步动态调整） |
| +Pos. | 仅成功 Conceptor（$C_{\text{steer}} = C_{\text{success}}$，无对比项） |

---

## 批判性思考

### 优点

1. **无需训练**: 推理时干预，不修改模型权重，部署成本极低。
2. **数据高效**: 仅需少量成功/失败 rollout，可从 1 次成功示例启动。
3. **几何可解释性**: Conceptor 子空间结构和相关分析提供了清晰的成功/失败几何直觉。
4. **跨架构泛化**: 在 Flow Matching（π0.5、GR00T）、自回归（π0-FAST）和扩散策略上均有效。
5. **跨任务迁移**: 发现的失败几何共享性支持零样本迁移引导。

### 局限性

1. **需要失败 rollout**: 对于已经接近完美的任务，失败样本可能难以获取或不代表真实失败模式。
2. **子空间不相交时增益有限**: 若成功/失败 Conceptor 重叠低，判别信号弱，效果受限。
3. **超参数选择泛化性**: 三阶段启发式在所测试架构之外的泛化性尚不明确。
4. **解释性风险**: 引导作用于内部激活，难以从输出端检测，存在隐蔽性风险。

### 潜在改进方向

1. 扩展至在线自适应：边部署边收集 rollout，动态更新 Conceptor。
2. 探索多任务联合 Conceptor：显式利用失败几何共享性构建通用引导器。
3. 与 [[Reinforcement Learning|RLHF]] 或奖励模型结合，自动化 rollout 成功/失败标注。

### 可复现性评估

- [ ] 代码开源（暂无）
- [ ] 预训练模型（依赖 π0.5、GR00T N1.5 等已有模型）
- [x] 训练细节完整（超参数选择流程描述清楚）
- [x] 数据集可获取（MetaWorld、LIBERO、RoboCasa 均公开）

---

## 关联笔记

### 基于

- [[Conceptor]]: Jaeger (2014) 提出用于循环网络的概念算子，本文将其迁移至 Transformer 激活引导
- [[Pi05|π0.5]]: 主要测试 VLA 之一
- [[GR00T|GR00T N1.5]]: 主要测试 VLA 之一
- π0-FAST: 测试的自回归 VLA

### 对比

- [[CAA]]: 对比激活加法，加法向量干预，COAST 使用乘法门控
- [[SAE Steering]]: 稀疏自编码器引导，需要额外训练
- [[SFT]]: 监督微调，需要大量数据，常导致性能下降

### 方法相关

- [[Conceptor]]: 核心数学工具，正半定线性算子
- [[Residual Stream]]: 作用位置，Transformer 残差流
- [[Activation Steering]]: 更广泛的激活引导研究方向
- [[Representation Engineering]]: 相关的内部表征操控框架

### 硬件/数据相关

- [[DROID]]: 真实机器人实验平台
- [[MetaWorld]]: 仿真 benchmark
- [[LIBERO]]: 仿真 benchmark

---

## 速查卡片

> [!summary] COAST
> - **核心**: 推理时对 VLA 残差流施加对比 Conceptor 乘法门控，无需训练
> - **方法**: 从少量成功/失败 rollout 构建 Conceptor，布尔 AND-NOT 合成引导算子
> - **结果**: 仿真 +20%+，真实机器人 +40%（MetaWorld 0.69→0.94，LIBERO 0.43→0.80）
> - **代码**: 暂未开源

---

*笔记创建时间: 2026-05-20*
