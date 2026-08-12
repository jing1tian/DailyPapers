---
title: "JEPA-WAM: Learning Vision-Language-Action Policies with Joint-Embedding World Modeling"
method_name: "JEPA-WAM"
authors: [Yihan Lin, Jiawei He, Shifeng Bao, Chen Zhao, Yang Li, Xiaobo Wang, Yan Wang, Cheng Chi, Jing Zhang]
year: 2026
venue: arXiv
tags: [world-model, jepa, vla, self-supervised-learning, latent-world-model, flow-matching, ood-generalization, bimanual-manipulation]
zotero_collection: 3-Robotics/World Model
image_source: online
arxiv_html: https://arxiv.org/html/2608.09381v1
created: 2026-08-12
---

# 论文笔记：JEPA-WAM — Learning Vision-Language-Action Policies with Joint-Embedding World Modeling

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | 多机构合作（9 位作者，包括 Cheng Chi、Jing Zhang） |
| 日期 | August 2026 |
| 项目主页 | [spritewithoutice.github.io/JEPA_WAM](https://spritewithoutice.github.io/JEPA_WAM/) |
| 对比基线 | [[π0.5]] · [[VLA-JEPA]] · [[LaWAM]] · [[Being-H0.7]] · [[Cosmos-Policy]] · [[ABot-M0]] |
| 链接 | [arXiv](https://arxiv.org/abs/2608.09381) / [HTML](https://arxiv.org/html/2608.09381v1) |

---

## 一句话总结

> JEPA-WAM 在冻结的 [[V-JEPA]] 嵌入空间内，通过共享预测器将**联合当前-未来帧目标**的潜在转移预测与连续动作生成耦合，无需大规模预训练即在 LIBERO-Plus OOD 基准达到 79.2%，叠加到 [[π0.5]] 后进一步提升至 86.3%。

---

## 核心贡献

1. **[[Joint Current-Future Target|联合当前-未来帧目标]]**: 将当前帧与未来帧联合输入 V-JEPA 编码器生成"空间结构化联合目标"，同时捕获稳定区域（静态背景）与变化区域（运动物体），保留 patch 级对应关系。
2. **共享预测器架构**: 以 [[Qwen]] 2.5-0.5B 语言模型为骨干，转移预测与动作生成共享一套 Transformer 权重，转移监督直接塑造 backbone 表示，而 64 个动作占位符 token 提供独立读出路径，防止两个目标相互干扰。
3. **无需大规模预训练的强泛化**: 不依赖任何机器人策略预训练，在 LIBERO-Plus 7 类视觉/空间扰动下达到最优；作为辅助监督叠加到 [[π0.5]] 后，实真实世界双臂操作 OOD 成功率从 22.5% 提升至 84.7%。

---

## 问题背景

### 要解决的问题

在视觉操作任务中，如何在不生成像素级未来帧（开销大）的前提下，让策略网络通过隐式世界建模学到**可泛化的视觉时序结构**，从而在 OOD（域外）分布下保持高成功率。

### 现有方法的局限

- **显式帧生成 WAM**（如 [[LaWAM]]、[[Cosmos-Policy]]）：在高分辨率视频上生成像素级未来帧，计算开销极大，且生成质量不稳定。
- **独立特征差分**（Endpoint Difference）：仅编码当前帧与未来帧各自的特征再做差，无法捕获帧间 patch 级的对应关系，丢失轨迹内部结构。
- **iREPA 风格空间对齐**（先对特征做空间上采样再做预测）：人为引入分辨率失配，损失空间结构信息（消融 -4.5%）。
- **全隐藏状态作为动作条件**：用预测器全部输出状态同时监督转移和驱动动作，两目标耦合导致互相干扰（消融 -6.1%）。

### 本文的动机

[[V-JEPA]] 预训练编码器已学到丰富的视频物理先验；若在其嵌入空间内：
1. 将当前帧与未来帧**联合编码**而非独立编码 → 迫使目标同时包含稳定背景与运动信息，产生有意义的 patch 级监督；
2. 用**共享 Transformer 骨干**同时完成转移预测和动作生成 → 转移监督自然塑造面向控制的表示，而无需额外冻结、正则化或 EMA。

---

## 方法详解

### 模型架构

JEPA-WAM 采用 **Latent World Action Model** 架构，在推理时仅需当前观测和语言指令：

- **输入**: 语言指令 $l$ + 多视角当前观测 $o_t$（训练时额外需要未来观测 $o_{t+\delta}$）
- **视觉编码器**: 冻结的 [[V-JEPA]] 2.1 [[ViT]]-L/16（300M 参数），每视角产生 24×24 patch 网格，特征维度 1024，多视角按固定顺序拼接，不做 pooling
- **Backbone**: [[Qwen]] 2.5-0.5B 语言模型（896 维隐藏状态）
- **动作专家**: 独立的 MLP 流，以共享 backbone 输出的 64 个动作占位符 token $C_t$ 为条件
- **总参数（JEPA-WAM 独立版）**: 约 0.5B + 视觉编码器
- **推理延迟**: 85 ms / 步（11.76 Hz），快于 [[ABot-M0]]（125 ms，7.99 Hz）

### 核心模块

#### 模块 1：[[Joint Current-Future Target|联合当前-未来帧目标（Joint Current-Future Target）]]

**设计动机**: 若分别编码当前帧和未来帧再做差（endpoint difference），网络只需在各自嵌入中找对应关系，损失了帧间 patch 级动态；联合编码则让 [[Stop-Gradient|stop-gradient]] 后的 [[V-JEPA]] 编码器"同时看"两帧，迫使目标携带帧间变化的细粒度信息。

**具体实现**:
- 将当前帧 $o_t$ 与未来帧 $o_{t+\delta}$ 在时间维度 stack 后，输入冻结的 V-JEPA 编码器
- 输出的 patch 级特征序列 $Y \in \mathbb{R}^{N_{\text{vis}} \times 1024}$ 即为监督目标
- 对目标分支施加 [[Stop-Gradient]]，防止梯度回传到冻结编码器
- 时间偏移 $\delta$：LIBERO 取 31 帧，RoboTwin 取 50 帧

#### 模块 2：共享预测器 + 动作占位符读出路径

**设计动机**: 避免转移监督（需要捕获所有 patch 的视觉时序结构）与动作监督（只需提取任务相关的行为信号）互相污染。

**具体实现**:
- 共享 [[Qwen]] 2.5-0.5B 骨干接收视觉 patch token + 语言 token + 64 个可学习动作占位符 token
- **转移预测头**: 从对应视觉位置输出预测目标 $\hat{Y} \in \mathbb{R}^{N_{\text{vis}} \times 1024}$（$Q^{\text{wm}}_t$），用于计算转移损失
- **动作条件 $C_t$**: 从 64 个动作占位符 token 的隐藏状态读出，注入独立动作专家
- **训练时**: 未来帧 token 被 [[Stop-Gradient]] 掩码，不影响动作路径

#### 模块 3：[[CFM|条件流匹配（CFM）]] 动作生成

**设计动机**: 使用 [[Flow Matching]] 替代 diffusion，采样效率更高，避免离散化噪声调度问题。

**具体实现**:
- 以 $C_t$（动作占位符隐藏状态）为条件，预测速度场 $A_\psi$
- 推理时通过 ODE 积分从噪声 $\epsilon$ 流向动作 $a$
- [[Action Chunking]] 动作块长度：LIBERO 为 8 步，RoboTwin 为 50 步

#### 模块 4：迁移到预训练 VLA（[[π0.5]] + JEPA 监督）

**设计动机**: JEPA-WAM 的转移监督可作为辅助损失插入现有预训练 VLA，无需修改动作路径。

**具体实现**:
- 在 [[π0.5]] 的 VLM 前缀中插入 64 个可学习未来 token
- 将 VLM 未来 token 的隐藏状态 reshape 为 8×8 粗空间网格
- 经轻量投影 + 空间上采样匹配 V-JEPA 分辨率，计算辅助转移损失
- 辅助损失权重 $\lambda_{\text{wm}} = 0.1$，训练前 1K 步线性 warmup
- 原始动作查询被未来 token 掩码，保持动作路径不变
- [[LoRA]] 微调：rank 32，scaling 64，dropout 0.1

---

## 关键公式

### 公式 1：[[Joint Current-Future Target|联合帧目标编码]]

$$
Y = \text{SG}\!\left(\text{Enc}_{\text{VJEPA}}\!\left([o_t \| o_{t+\delta}]\right)\right)
$$

**含义**: 将当前帧与未来帧在时间维度拼接后，通过冻结的 V-JEPA 编码器得到 patch 级联合目标 $Y$，[[Stop-Gradient]] 防止梯度回传。

**符号说明**:
- $o_t \in \mathbb{R}^{C \times H \times W}$：当前多视角观测
- $o_{t+\delta}$：时间偏移 $\delta$ 步后的未来观测
- $\text{SG}(\cdot)$：[[Stop-Gradient]] 算子
- $Y \in \mathbb{R}^{N_{\text{vis}} \times 1024}$：patch 级联合目标，$N_{\text{vis}}$ 为可见 patch 数

### 公式 2：[[Joint Current-Future Target|转移预测损失]]（patch 级余弦距离）

$$
\mathcal{L}_{\text{wm}} = \frac{1}{B \cdot N_{\text{vis}}} \sum_{b=1}^{B} \sum_{i=1}^{N_{\text{vis}}} \left(1 - \cos\!\left(\hat{Y}_i^b,\, Y_i^b\right)\right)
$$

**含义**: 在 batch 内对所有可见 patch 计算预测值与目标值的 [[Cosine Similarity|余弦距离]] 之和，驱动共享预测器学习空间结构化的视觉时序特征。

**符号说明**:
- $B$：batch 大小
- $\hat{Y}_i^b$：共享预测器对第 $b$ 样本第 $i$ 个 patch 的预测输出
- $Y_i^b$：对应的联合编码目标（stop-gradient）
- $\cos(\cdot, \cdot)$：余弦相似度

### 公式 3：[[CFM|条件流匹配动作损失]]

$$
\mathcal{L}_{\text{act}} = \mathbb{E}_{\tau \sim \mathcal{U}(0,1),\, \epsilon \sim \mathcal{N}(0, I)}\!\left[\left\|A_\psi\!\left(a_\tau,\, \tau,\, s_t,\, C_t\right) - (a - \epsilon)\right\|_2^2\right]
$$

**含义**: 以动作占位符隐藏状态 $C_t$ 为条件，训练速度场 $A_\psi$ 拟合从噪声 $\epsilon$ 到真实动作 $a$ 的直线流方向，推理时对 ODE 积分生成动作。

**符号说明**:
- $\tau \sim \mathcal{U}(0,1)$：[[Flow Matching|流时间步]]，均匀采样
- $\epsilon \sim \mathcal{N}(0, I)$：标准高斯噪声
- $a_\tau = (1-\tau)\epsilon + \tau a$：插值后的中间噪声动作
- $A_\psi$：速度场网络（动作专家）
- $s_t$：当前状态（视觉 + 语言上下文）
- $C_t$：64 个动作占位符 token 隐藏状态，来自共享 backbone

### 公式 4：联合训练目标

$$
\mathcal{L} = \mathcal{L}_{\text{act}} + 0.5 \cdot \mathcal{L}_{\text{wm}}
$$

**含义**: 动作生成损失与转移预测损失的加权组合，权重 0.5 使两者在量级上平衡。

**符号说明**:
- $\mathcal{L}_{\text{act}}$：[[CFM|条件流匹配]] 动作损失
- $0.5 \cdot \mathcal{L}_{\text{wm}}$：转移监督辅助损失（π0.5 迁移版本中权重改为 $\lambda_{\text{wm}} = 0.1$）

---

## 关键图表

### Figure 1：JEPA-WAM 系统概览与性能对比

![Figure 1](https://arxiv.org/html/2608.09381v1/x1.png)

**说明**: 上方展示 JEPA-WAM 整体框架——冻结 [[V-JEPA]] 编码器产生联合目标，共享预测器同时完成转移预测与动作生成；下方对比 LIBERO、RoboTwin 2.0 及真实世界任务上的性能，可视化 JEPA 监督对 [[π0.5]] 的提升幅度。

### Figure 2：潜在 WAM 范式对比

![Figure 2](https://arxiv.org/html/2608.09381v1/x2.png)

**说明**: 对比三类 Latent WAM 设计：(a) 解耦独立预测器（转移与动作分支完全分离）；(b) 仅动作生成（无世界模型监督）；(c) **JEPA-WAM**（共享预测器耦合两目标）。核心区别在于转移监督是否直接塑造动作 backbone 的表示。

### Figure 3：JEPA-WAM 架构详解

![Figure 3](https://arxiv.org/html/2608.09381v1/x3.png)

**说明**: 冻结 V-JEPA 编码器构建空间结构化 [[Joint Current-Future Target|联合当前-未来帧目标]]（目标分支仅训练时使用）；[[Qwen]] 2.5-0.5B 共享预测器处理当前观测，输出视觉 token 位置 $Q^{\text{wm}}_t$（用于转移损失）和动作占位符隐藏状态 $C_t$（用于条件动作生成）；两路读出互不干扰。

### Figure 4：向预训练 VLA 迁移的转移监督

![Figure 4](https://arxiv.org/html/2608.09381v1/x4.png)

**说明**: 将 JEPA 转移监督叠加到 [[π0.5]] 的方案。在 VLM 前缀插入 64 个可学习未来 token，其隐藏状态经轻量上采样对齐 V-JEPA 分辨率，计算辅助 $\mathcal{L}_{\text{wm}}$；原始动作查询路径（Expert Query）完全不变，推理时去掉目标分支即可零开销部署。

### Figure 5：真实世界双臂操作评测

![Figure 5](https://arxiv.org/html/2608.09381v1/x5.png)

**说明**: 五项双臂操作任务（面包摆放、水果摆放、积木堆叠等）的 ID/OOD 成功率对比。**π0.5 + JEPA** 在 OOD 下达到 84.7%，远超 π0.5 独立基线（22.5%）；JEPA-WAM 独立版 OOD 59.8% 亦超过 π0.5 基线 ID 的 51.8%。

### Figure 6（附录）：时序间隔分类混淆矩阵

![Figure 6](https://arxiv.org/html/2608.09381v1/Appendix/Figures/joint_target_gap_confusion.png)

**说明**: 固定未来帧、变化真实时序间隔的分类实验。行为真实间隔，列为预测间隔。**Future-only**（仅未来帧特征）几乎无对角结构；**Endpoint Difference**（特征差）呈弱对角；**Joint Target**（本文）显示最清晰的对角结构（准确率 67.2% vs. 47.0%），证明联合目标显著保留帧间时序信息。

### Table 1：LIBERO 域内结果（%）

| Method | LIBERO-Spatial | LIBERO-Object | LIBERO-Goal | LIBERO-Long | Avg |
|--------|---------------|--------------|------------|------------|-----|
| 其他基线 | — | — | — | — | — |
| **JEPA-WAM** | — | — | — | — | **96.7** |
| **π0.5 + JEPA** | — | — | — | — | **97.8** |
| π0.5 baseline | — | — | — | — | 96.9 |

**关键发现**: 域内性能 JEPA-WAM 与 π0.5 基线相当（96.7% vs 96.9%），叠加 JEPA 监督后小幅提升至 97.8%，说明转移监督不损害分布内任务能力。

### Table 2：LIBERO-Plus OOD 泛化结果（%）

| Method | Camera | Robot | Language | Lighting | Background | Noise | Layout | Avg |
|--------|--------|-------|---------|---------|-----------|-------|--------|-----|
| π0.5 baseline | — | — | — | — | — | — | — | 84.5 |
| VLA-JEPA | — | — | — | — | — | — | — | ~78.0 |
| **JEPA-WAM** | **79.2** | — | 68.2 | **93.3** | — | — | — | **79.2** |
| **π0.5 + JEPA** | — | — | — | — | — | — | — | **86.3** |

**关键发现**: JEPA-WAM 在无预训练方法中最优（79.2%）；相机视角扰动（79.2%）和光照扰动（93.3%）表现最强；语言扰动（68.2%）最弱，与方法语言无关的转移目标设计相符。

### Table 3：RoboTwin 2.0 结果（%）

| Method | Clean | Random |
|--------|-------|--------|
| **JEPA-WAM** | **79.9** | **36.9** |
| π0.5 baseline | — | — |
| **π0.5 + JEPA** | **84.6** | **37.5** |

**关键发现**: RoboTwin 2.0 的 Domain Randomization 场景下，Random（强视觉随机化）比 Clean 成功率低约 43 pp，但 JEPA 监督在两种设置下均带来一致提升。

### Table 4：消融实验（LIBERO-Plus，%）

| 设计选择 | 变体 | 性能 | 增益 |
|--------|------|------|------|
| 视觉表示 | DINO+SigLIP | 73.2 | — |
| 视觉表示 | **V-JEPA（本文）** | **77.0** | +3.8 |
| 目标构造 | Future-only | 77.3 | — |
| 目标构造 | **Joint（本文）** | **79.2** | +1.9 |
| 空间对齐 | iREPA | 74.7 | — |
| 空间对齐 | **Direct（本文）** | **79.2** | +4.5 |
| 监督层 | Lower-16 | 76.5 | — |
| 监督层 | **Final（本文）** | **79.2** | +2.7 |
| 动作条件 | Full hidden | 73.1 | — |
| 动作条件 | **Placeholders（本文）** | **79.2** | +6.1 |

**关键发现**: 动作占位符读出路径（+6.1%）和直接空间对齐（+4.5%）是最重要的设计选择；V-JEPA 作为特征骨干优于 DINO+SigLIP（+3.8%），验证了预训练视频表示的优势。

### Table 5：时序结构分析（附录）

| 方法 | 时序间隔分类准确率 | 轨迹残差 $R^2$ |
|------|-----------------|--------------|
| Endpoint Difference | 47.0% | 0.485 |
| **Joint Target（本文）** | **67.2%** | **0.582** |
| 增益 | +20.1% | +0.097 |

**关键发现**: 联合目标在时序间隔分类（+20.1%）和轨迹残差预测（$R^2$ +0.097）上均显著超越端点差分法，证明联合编码捕获了超越一阶位移的帧间轨迹内部结构。

---

## 实验

### 数据集

| 数据集 / 环境 | 规模 | 特点 | 用途 |
|------------|------|------|------|
| [[LIBERO]] | 130 任务，5 类别 | 桌面操作，标准 benchmark | 域内训练/测试 |
| [[LIBERO-Plus]] | 同上，7 种 OOD 扰动 | 相机/机器人/语言/光照/背景/噪声/布局 | OOD 泛化测试 |
| [[RoboTwin 2.0]] | 多任务 | 域随机化，物理参数随机 | 仿真 OOD |
| 真实世界双臂 | 5 任务 | 面包/水果/积木等，有 OOD 变化 | 真实部署验证 |

### 实现细节

- **视觉编码器**: 冻结 [[V-JEPA]] 2.1 [[ViT]]-L/16（300M 参数，不参与梯度更新）
- **骨干网络**: [[Qwen]] 2.5-0.5B（[[LoRA]] rank 32, scaling 64, dropout 0.1）
- **优化器**: [[AdamW]]，峰值学习率 $2 \times 10^{-4}$，余弦衰减
- **Batch Size**: 128
- **训练步数**: 60K steps
- **精度**: BF16
- **硬件**: 8× GPU
- **时间偏移 $\delta$**: 31 帧（LIBERO）/ 50 帧（RoboTwin）
- **动作块长度**: 8（LIBERO）/ 50（RoboTwin）

### 可视化结果

- **时序间隔混淆矩阵**（Figure 6）：联合目标呈现清晰对角结构，证明其保留了帧间时序顺序信息。
- **真实机器人评测**（Figure 5）：π0.5 + JEPA 在 5 项双臂任务 OOD 场景下均显著超越独立 π0.5 基线，尤其在外观分布偏移下改善最大。

---

## 批判性思考

### 优点

1. **计算高效**: 在推理时完全移除目标分支（冻结编码器 + 未来帧），无额外开销，仅 85 ms / 步。
2. **即插即用迁移**: 可作为纯辅助损失叠加到任意预训练 VLA（如 [[π0.5]]），不改变原有动作路径，风险极低。
3. **实验分析扎实**: 从时序间隔分类、轨迹残差 $R^2$ 到 7 类 OOD 扰动逐类分析，消融实验验证每个设计选择的具体贡献值，可信度高。

### 局限性

1. **语言条件转移缺失**: 转移目标仅是视觉性的（联合帧编码），与语言指令无关；同一观测下不同语言指令可能产生不同轨迹，但监督目标相同，存在目标歧义。
2. **语言扰动泛化弱**: LIBERO-Plus 中语言变化扰动成功率最低（68.2%），与上述局限直接相关。
3. **依赖 V-JEPA 预训练**: 方法强绑定 V-JEPA 特征空间，若换用其他视觉骨干（如 DINO+SigLIP）性能下降 3.8%，迁移灵活性受限。

### 潜在改进方向

1. 引入**语言条件化转移目标**（Language-Conditioned Latent Target），将指令信息融入联合目标的构建过程，改善语言泛化。
2. 将 JEPA 监督扩展到**多视角时序一致性**，进一步挖掘多相机观测的几何对应关系。
3. 探索**在线 RL 微调**场景下 JEPA 监督的持续作用，验证其是否加速奖励学习收敛。

### 可复现性评估

- [ ] 代码开源（项目页面存在但代码链接未确认）
- [ ] 预训练模型（未明确提供）
- [x] 训练细节完整（超参数、架构、步数均在论文中给出）
- [x] 数据集可获取（LIBERO、RoboTwin 2.0 均为公开 benchmark）

---

## 关联笔记

### 基于

- [[V-JEPA]]: 冻结视频 JEPA 编码器，提供空间结构化视频先验
- [[JEPA]]: 联合嵌入预测架构的基础范式
- [[CFM]]: 条件流匹配，用于连续动作生成

### 对比

- [[π0.5]]: 大规模预训练 VLA 基线；本文作为辅助监督叠加后进一步提升
- [[VLA-JEPA]]: 同为 JEPA + VLA 方向，是主要竞争基线
- [[LaWAM]]: 显式潜在帧生成 WAM，计算开销更大
- [[Being-H0.7]]: 参与 LIBERO-Plus 对比的强基线
- [[Cosmos-Policy]]: 参与对比的预训练 WAM 方法
- [[ABot-M0]]: 在推理速度（Hz）上的对比参考

### 方法相关

- [[Joint Current-Future Target]]: 本文核心贡献，联合当前-未来帧 patch 级目标
- [[Action Chunking]]: 动作块生成方式
- [[Stop-Gradient]]: 目标分支梯度截断
- [[LoRA]]: 用于 π0.5 迁移的参数高效微调
- [[Qwen]]: 共享预测器骨干网络
- [[Cosine Similarity]]: 转移损失的度量函数

### 硬件/数据相关

- [[LIBERO]]: 域内训练与测试 benchmark
- [[LIBERO-Plus]]: OOD 扰动评估 benchmark
- [[RoboTwin 2.0]]: 域随机化仿真 benchmark
- [[ViT]]: V-JEPA 使用的视觉骨干（ViT-L/16）

---

## 速查卡片

> [!summary] JEPA-WAM (2026)
> - **核心**: 在冻结 V-JEPA 空间内用联合当前-未来帧目标作转移监督，与动作生成共享预测器
> - **方法**: Joint Current-Future Target + 共享 Transformer 骨干 + CFM 动作专家
> - **结果**: LIBERO-Plus 79.2%（无预训练最优）；π0.5+JEPA 86.3%；真实世界 OOD 84.7%
> - **代码**: [项目主页](https://spritewithoutice.github.io/JEPA_WAM/)

---

*笔记创建时间: 2026-08-12*
