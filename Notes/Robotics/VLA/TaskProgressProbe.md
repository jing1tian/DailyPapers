---
title: "Decoding Task Progress from VLA Representations"
method_name: "TaskProgressProbe"
authors: [Atiksh Bhardwaj, Edward Weiyi Duan, Prithwish Dan, Wei-Chiu Ma, Preston Culbertson]
year: 2026
venue: arXiv
tags: [vla-interpretability, linear-probe, task-progress, ood-detection, representation-learning, manipulation]
zotero_collection: Robotics/VLA
image_source: online
arxiv_html: https://arxiv.org/html/2608.13474v1
created: 2026-08-15
---

# 论文笔记：Decoding Task Progress from VLA Representations

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Cornell University |
| 日期 | August 2026 |
| 项目主页 | — |
| 对比基线 | [[SAFE]] (SAFE-MLP / SAFE-LSTM) |
| 链接 | [arXiv](https://arxiv.org/abs/2608.13474) |

---

## 一句话总结

> 从 [[Pi05|π₀.₅]] 的[[Residual Stream|残差流]]激活中，任务进度信号可以被[[线性探测|线性探针]]读出，且该信号在预训练 [[PaliGemma]] 骨干中便已存在；基于此构造的无标签 OOD 检测器与有监督基线持平。

---

## 核心贡献

1. **形式化框架**：严格区分[[弱可解码性]]、[[强可解码性]]和[[可操控性]]三个层次，为 VLA 可解释性研究提供精确概念工具
2. **进度可解码性验证**：证明任务进度特征在 [[Pi05|π₀.₅]] 的残差流中弱可解码（跨任务 R²≈0.85）且在对比训练后强可解码（响应语言扰动），但**不可操控**——意味着该特征可观测却不参与因果决策
3. **无标签 OOD 检测**：将进度探针作为 OOD 检测器，在跨 OOD 模式泛化测试中超过有监督基线（per-episode AUROC 0.960 vs. 0.894）

---

## 问题背景

### 要解决的问题

VLA 模型（如 [[Pi05|π₀.₅]]）在黑盒状态下部署，内部表示的语义完全未知。本文追问：这些模型是否在内部编码了**任务进度**——即"当前完成了多少比例的轨迹"？

### 现有方法的局限

- [[SAFE]] 等有监督失败检测方法需要大量 OOD 标注数据，泛化性差
- 可解释性研究多集中于语言模型，VLA 的[[Residual Stream|残差流]]分析几乎空白
- [[Activation Steering]] 研究缺乏与"可解码性"的明确区分，混淆了相关性与因果性

### 本文的动机

任务进度是机器人策略执行中最基本的时序语义之一。若 VLA 内部已隐式编码该信号，则可在不增加额外监督的情况下免费获得一个轨迹状态估计器，进而用于 OOD 检测、策略监控等下游任务。

---

## 方法详解

### 核心概念体系

本文在生成式策略（Generative Policy）框架下给出三个层次递进的形式化定义：

**[[弱可解码性]]（Weak Decodability）**：存在线性探针 φ，使其在分布内激活上能以低风险预测目标特征 ζ。

**[[强可解码性]]（Strong Decodability）**：同一探针无需重新训练，即可对反事实变换输入 $\mathcal{T}(\mathbf{x})$ 正确预测对应目标 $\zeta(\mathcal{T}(\mathbf{x}))$。

**[[可操控性]]（Steerability）**：沿探针方向注入激活后，策略的输出行为能匹配目标值所对应的反事实行为。

### 模型架构

本文以 [[Pi05|π₀.₅]] 为探测对象：它以 [[PaliGemma]] 为视觉语言骨干，配合动作专家模块，在 [[VLABench]] 的 10 个操作任务上微调。

实验层次结构：SigLIP 视觉编码器 → PaliGemma Gemma 语言层（共 18 层）→ 动作专家

探针 φ 作用于**第 0 层（弱解码性 / OOD 检测）**和**第 10 层（强解码性测试）**的[[Residual Stream|残差流]]激活 $\mathbf{z}^i$。

### 探针训练

**弱解码性探针**：标准[[线性探测|线性回归]]，L₁ 损失，Adam 优化，轨迹级 80/20 划分。

**强解码性探针**：对比训练——同一时间步的原始轨迹与反事实轨迹（通过替换指令语言中的目标名词生成）配对，使用带时间依赖边距的 Hinge Loss + 二元交叉熵锚点，sigmoid 锐化目标。

---

## 关键公式

### 公式1: [[Residual Stream|残差流递推]]

$$
\mathbf{z}^{i+1} = \mathbf{z}^i + \mathbf{f}^i(\mathbf{z}^i), \quad i = 0, \ldots, L-1
$$

**含义**：Transformer 每层通过残差连接将子层输出叠加到主信息流，形成可探测的累积表示。

**符号说明**：
- $\mathbf{z}^i$：第 $i$ 层的残差流激活向量
- $\mathbf{f}^i$：第 $i$ 层的注意力 + MLP 子层组合

---

### 公式2: [[线性探测|线性探针]]

$$
\varphi(\mathbf{z}^i) = \mathbf{w}^\top \mathbf{z}^i + b
$$

**含义**：从残差流激活中线性解码标量目标（任务进度 τ）。

**符号说明**：
- $\mathbf{w}$：探针权重向量
- $b$：偏置
- $\mathbf{z}^i$：第 $i$ 层激活

---

### 公式3: 探针风险函数

$$
R(\varphi; \mathcal{D}) = \frac{1}{K} \sum_{k=1}^{K} \mathcal{L}\!\left(\varphi\!\left(\mathbf{z}^i_k\right);\, \zeta_k\right)
$$

**含义**：在数据集 $\mathcal{D}$ 上的平均预测损失，用 L₁ 损失实现。

**符号说明**：
- $K$：样本总数
- $\zeta_k$：第 $k$ 个样本的目标标签（任务进度值）

---

### 公式4: [[任务进度]]定义

$$
\tau(\mathbf{x}_t) = 1 - \frac{\varphi_\xi(\mathbf{x}_t)}{T} \in [0, 1]
$$

**含义**：将当前时间步归一化为"剩余完成比例"，轨迹开始时 τ=1，结束时 τ=0。

**符号说明**：
- $\varphi_\xi(\mathbf{x}_t) = t$：当前时间步
- $T$：轨迹总长度
- $\tau \in [0,1]$：归一化任务进度

---

### 公式5: 激活注入（Steerability 测试，Equation 1）

$$
\tilde{\mathbf{z}}^i(\mathbf{x}) = \mathbf{z}^i(\mathbf{x}) + \frac{\zeta(\mathcal{T}(\mathbf{x})) - \zeta(\mathbf{x})}{\|\mathbf{w}\|^2} \cdot \mathbf{w}
$$

**含义**：沿探针方向注入反事实进度差值，尝试"操控"模型行为到目标状态。

**符号说明**：
- $\mathcal{T}(\mathbf{x})$：反事实输入（语言名词替换后的版本）
- $\zeta(\cdot)$：目标进度值
- $\mathbf{w}$：探针权重向量

---

### 公式6: 可操控性风险（Equation 2）

$$
R_S(\varphi; \mathcal{D}, \mathcal{T}) = \frac{1}{K} \sum_{k=1}^{K} d(\tilde{\pi}_k,\, \pi^*_k)
$$

**含义**：衡量注入激活后输出动作与反事实目标动作之间的距离。结果显示注入无效，**π₀.₅ 的任务进度不可操控**。

**符号说明**：
- $\tilde{\pi}_k$：注入后的策略输出动作
- $\pi^*_k$：反事实目标动作
- $d(\cdot,\cdot)$：动作空间距离（L₂）

---

### 公式7: 进度残差 OOD 检测器（Equation 3）

$$
V_\tau(t) = \tau_\text{pred} - \left(1 - \frac{t}{\mathbb{E}[T \mid \ell]}\right)
$$

**含义**：进度预测值与期望进度之差，偏差超过阈值时触发 OOD 报警（卡滞检测）。

**符号说明**：
- $\tau_\text{pred}$：探针预测的当前任务进度
- $t$：当前时间步
- $\mathbb{E}[T \mid \ell]$：给定语言指令 $\ell$ 下的期望轨迹长度

---

## 关键图表

### Figure 1: 方法概览

![Figure 1](https://arxiv.org/html/2608.13474v1/x1.png)

**说明**：线性探针 φ 作用于 [[Pi05|π₀.₅]] 的残差流激活 $\mathbf{z}^0$，解码任务进度 τ，支持三条下游分析：OOD 检测、语言接地诊断、可操控性研究。

---

### Figure 2: 弱解码性跨层结果

![Figure 2](https://arxiv.org/html/2608.13474v1/figures/figure1.png)

**说明**：左图：在微调后的 [[Pi05|π₀.₅]] 各层（SigLIP → PaliGemma Gemma → 动作专家）的弱解码性。第 0 层已有强信号。右图：探针容量 sweep，真实标签曲线远高于乱序标签，排除虚假相关。

---

### Figure 3: 各训练阶段对比

![Figure 3](https://arxiv.org/html/2608.13474v1/figures/comparison_progress_1x4.png)

**说明**：四个模型阶段的进度可解码性对比：仅 [[PaliGemma]] 骨干 → 微调前 [[Pi05]] 基模型 → VLABench 微调 π₀ → VLABench 微调 [[Pi05|π₀.₅]]。关键发现：**任务进度特征在 PaliGemma 预训练阶段便已出现**，并非机器人微调的产物。

---

### Figure 4: 对比训练实现强解码性

![Figure 4](https://arxiv.org/html/2608.13474v1/x2.png)

**说明**：原始指令（蓝）vs. 语言名词替换后（橙）的进度轨迹。左：朴素第 0 层探针——两者曲线几乎重合，语言不敏感。右：第 10 层[[强可解码性|对比探针]]——名词替换后曲线正确分离，R²=0.33，实现[[强可解码性]]，但代价是原始进度跟踪精度下降。

---

### Figure 5: 可操控性测试失败

![Figure 5](https://arxiv.org/html/2608.13474v1/figures/injection_combined.png)

**说明**：左图：在第 0 层注入 τ_boost=0.2 后，进度信号仅在注入层附近改变，未能传播到后续层。右图：注入后动作与目标动作的 L₂ 距离未改善（与未注入基线相当）。**结论：进度特征可观测但不因果**，[[可操控性]]不成立。

---

### Figure 6: OOD 定性滚动展示

![Figure 6](https://arxiv.org/html/2608.13474v1/figures/qual_ood.png)

**说明**：在 Gaussian Blur OOD 模式下（注入位置 k=0.25, 0.5, 0.75）的典型轨迹。$V_\tau(t)$ 在扰动注入后出现明显残差偏离，成功触发 OOD 报警。

---

### Figure 7: 跨任务泛化 Scaling

![Figure 7](https://arxiv.org/html/2608.13474v1/figures/generalization_scaling.png)

**说明**：训练任务数量 k 与未见任务 R² 的关系。左：各组合的分布。右：均值与最优组合。关键：**3 个训练任务后均值趋于饱和（R²≈0.71），8 个任务达到 R²≈0.85**，但最优组合的方差在大 k 时上升。

---

### Figure 8: 语言敏感度 vs. 进度跟踪精度

![Figure 8](https://arxiv.org/html/2608.13474v1/x3.png)

**说明**：各层的语言敏感度 Δ（左）与分布内进度 R²（右）。语言敏感度在第 10 层峰值，R² 在第 0 层最高，两者形成典型的 tradeoff——深层更具语义，浅层更具进度信息。

---

### Figure 9: 嵌入探针 vs. 原始观测探针

![Figure 9](https://arxiv.org/html/2608.13474v1/figures/rawdata.png)

**说明**：以 VLA 嵌入为输入的探针（R²=0.807）vs. 直接以原始图像像素为输入（R²=−2627.7）。**任务进度信息存在于表示空间，而非原始像素**，证明 VLA 做了有意义的特征提取。

---

### Table 1: 跨任务泛化 OOD 检测性能（AUROC）

| 检测器 | per-replan S | per-replan U | per-episode S | per-episode U |
|--------|:------------:|:------------:|:-------------:|:-------------:|
| $V_\tau$（本文） | 0.796±0.090 | 0.833±0.058 | **0.910±0.036** | 0.851±0.056 |
| Mahalanobis | **0.986±0.010** | 0.859±0.040 | 0.775±0.013 | 0.719±0.016 |
| VAE | 0.859±0.036 | 0.796±0.056 | 0.782±0.035 | 0.778±0.034 |
| SAFE-MLP | 0.917±0.025 | **0.935±0.013** | 0.904±0.021 | **0.917±0.015** |
| SAFE-LSTM | 0.827±0.041 | 0.849±0.048 | 0.836±0.035 | 0.838±0.030 |

S=Seen tasks，U=Unseen tasks。$V_\tau$ 在 per-episode 粒度上表现最佳（seen），在 per-replan 上略逊于 SAFE-MLP 但**无需任何 OOD 标注数据**。

---

### Table 2: 跨 OOD 模式泛化（未见扰动类型）

| 检测器 | per-replan U | per-episode U |
|--------|:------------:|:-------------:|
| $V_\tau$（本文） | **0.871±0.104** | **0.960±0.013** |
| SAFE-MLP | 0.844±0.044 | 0.894±0.063 |
| SAFE-LSTM | 0.641±0.159 | 0.857±0.103 |

**关键发现**：在跨 OOD 模式泛化（未见扰动类型）场景下，$V_\tau$ 超越所有有监督基线，per-episode AUROC 达到 0.960。

---

### Table 3: 各 OOD 扰动模式结果（per-episode，未见模式）

| OOD 模式 | $V_\tau$ AUROC |
|---------|:--------------:|
| Gaussian Noise | 0.963±0.008 |
| Color Shift | 0.950±0.006 |
| Gaussian Blur | 0.977±0.004 |
| Occlusion | 0.948±0.007 |

四种模式下均保持强性能，对光照、相机抖动、遮挡等真实场景扰动鲁棒。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| [[VLABench]] | 10 任务 × 500 轨迹/任务 | 桌面操作，多目标选择 | 训练 + 测试 |
| VLABench OOD | 4 种扰动模式 | Gaussian noise, Color shift, Blur, Occlusion | OOD 检测评估 |

### 实现细节

- **Target Model**：[[Pi05|π₀.₅]] 微调版（VLABench fine-tuned）
- **探针层**：Layer 0（弱解码 / OOD 检测），Layer 10（强解码测试）
- **优化器**：Adam，学习率 1e-3
- **Batch Size**：64
- **训练轮数**：300 epochs
- **数据划分**：80/20 轨迹级（episode-level split）
- **对比训练**：Hinge Loss（边距 0.5~2.5 时间依赖）+ BCE 锚点（λ≈1.5），每时间步生成 ≥5 条反事实影子轨迹

### 可视化结果

- 弱解码性探针输出的进度曲线呈单调递减，与真实进度高度吻合（R²≈0.807）
- 对比探针在目标词汇替换后正确分离两条轨迹的预测曲线
- OOD 扰动注入后进度残差 $V_\tau$ 出现明显阶跃

---

## 批判性思考

### 优点
1. **概念清晰**：弱/强可解码性/可操控性三分法填补了 VLA 可解释性研究的概念空白，后续工作可直接沿用
2. **免监督 OOD**：在跨模式泛化场景下无需标注数据即超越有监督方法，实用价值高
3. **历史追溯**：证明进度信号来源于 PaliGemma 预训练，而非任务微调，为骨干模型分析提供了新视角

### 局限性
1. **单一模型族**：仅在 π₀.₅ 上验证，是否可迁移到其他 VLA 架构（如 OpenVLA、RoboVLMs）未知
2. **进度语义歧义**：以时间步归一化代替语义完成度，对于非线性进展的任务（如开瓶盖）可能失效
3. **可操控性测试局限**：仅沿单一探针方向、固定幅度注入，未探索非线性或多方向注入

### 潜在改进方向
1. 设计语义进度标注（如关键帧标注）替代时间步代理，提升强解码性精度
2. 将进度探针与在线规划结合，实现任务进度感知的自适应重规划
3. 探索可操控性失败原因：是激活传播被下游层抑制，还是任务进度根本不参与动作生成？

### 可复现性评估
- [ ] 代码开源
- [x] 预训练模型（π₀.₅ 已公开）
- [x] 训练细节完整（见 Appendix）
- [x] 数据集可获取（VLABench）

---

## 关联笔记

### 基于
- [[Pi05]]：被探测的目标 VLA 模型
- [[PaliGemma]]：π₀.₅ 的视觉语言骨干
- [[线性探测]]：核心探针方法论
- [[VLABench]]：实验平台

### 对比
- [[SAFE]]：有监督 VLA 失败检测基线，需要 OOD 标注
- [[Activation Steering]]：可操控性的对照参考框架

### 方法相关
- [[Residual Stream]]：探针作用位置
- [[弱可解码性]]：三分类概念之一
- [[强可解码性]]：三分类概念之一，需对比训练实现
- [[可操控性]]：三分类概念之一，本文证明不成立
- [[任务进度]]：被解码的目标特征
- [[Out-of-Distribution Generalization]]：下游应用场景

### 数据集相关
- [[VLABench]]：10 个桌面操作任务

---

## 速查卡片

> [!summary] Decoding Task Progress from VLA Representations
> - **核心**: π₀.₅ 内部线性编码了任务进度，来自预训练骨干而非机器人微调
> - **方法**: 线性探针 + 对比训练实现弱/强可解码性；激活注入证明不可操控
> - **结果**: 跨 OOD 模式 per-episode AUROC 0.960，超越有监督 SAFE-MLP（0.894）
> - **代码**: 暂未开源

---

*笔记创建时间: 2026-08-15*
