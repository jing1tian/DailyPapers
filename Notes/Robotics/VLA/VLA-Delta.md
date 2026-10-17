---
title: "VLA-Δ: Learning Physical State Transition for Vision-Language-Action Models"
method_name: "VLA-Delta"
authors: [Qi Lu, Jiarun Li, Ce Zhang, Miao Pan, Yi Pan, Xu Wang, Wenqi Zhang, Xuhong Zhang]
year: 2026
venue: arXiv
tags: [vision-language-action, physical-state-transition, world-model, flow-matching, robot-manipulation, diffusion-transformer]
zotero_collection: Robotics/VLA
image_source: online
arxiv_html: https://arxiv.org/html/2610.21306
created: 2026-10-17
---

# 论文笔记：VLA-Δ: Learning Physical State Transition for Vision-Language-Action Models

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Zhejiang University (OmniAI Group, ZJU ACES Lab); Central South University |
| 日期 | October 2026 |
| 项目主页 | https://github.com/ZJU-OmniAI/VLA-Delta |
| 对比基线 | [[pi0.5]], [[ZR-0]], [[GR00T N1]], [[WorldVLA]], [[DreamVLA]] |
| 链接 | [arXiv](https://arxiv.org/abs/2610.21306) / [Code](https://github.com/ZJU-OmniAI/VLA-Delta) |

---

## 一句话总结

> VLA-Δ 通过在预训练阶段将物理状态变化（结构化槽位、光流、差异文本）作为显式监督目标，使 [[VLA]] 模型真正理解物理动态，推理时零额外开销。

---

## 核心贡献

1. **VLA-Δ 框架**: 提出以当前-未来物理状态差异为显式学习目标的三层监督体系（结构化槽位 + 稠密光流 + 差异文本），训练后的物理状态表示 $Z_t$ 直接用于条件化动作生成。
2. **大规模带标注预训练数据集**: 构建包含约 484 小时、2400 万帧演示的数据集，配套自动化标注流程（Qwen3.8 + SAM 3 + UniDepth V2 + MegaFlow + GPT-5.6 质检），涵盖 9 大物理状态转变类别。
3. **SOTA 性能**: 在 LIBERO（98.4%）和 RoboTwin 2.0（90.06% Clean）上达到最优，并在真实机器人实验中超越 [[pi0.5]] 5.5 个百分点。

---

## 问题背景

### 要解决的问题

大多数 [[VLA]] 模型直接将视觉-语言观测映射到动作，缺乏对中间物理状态变化的显式监督，导致模型倾向于过拟合观测-动作的表面对应关系，而非理解物理动态（例如过早闭合夹爪、抓取失败后继续执行）。

### 现有方法的局限

- **直接 VLA**：无物理监督，依赖隐式表征
- **Reasoning VLA**（如 [[ECoT]]）：提供任务语义推理，但缺乏视觉运动层面的状态变化
- **World Action Model**（如 [[WorldVLA]]）：生成未来图像帧计算开销大，且在像素空间预测冗余信息
- **Latent-WAM**（如 [[DreamVLA]]）：在潜变量空间预测，但仍缺乏结构化物理状态监督

### 本文的动机

物理状态的改变（目标位置变化、夹爪接触、障碍物风险等）是比原始像素或任务文本更直接、更紧凑的动作条件信号。通过在预训练阶段施加多层物理状态转变监督，模型可以学习到语义丰富且几何结构化的转变表示 $Z_t$，推理时只需用该表示条件化 [[Diffusion Transformer|扩散 Transformer]] 动作专家，无需解码中间目标。

---

## 方法详解

### 模型架构

![Figure 2: VLA-Δ 整体架构](https://arxiv.org/html/2610.21306v1/2-method.png)

VLA-Δ 采用 **编码器-查询-扩散动作专家** 架构：

- **输入**: 语言指令 $l$ + 视觉观测 $o_t$ + 本体感知状态 $r_t$
- **Backbone**: [[Qwen3-VL-2B-Instruct]]（视觉语言主干）
- **核心模块**: 可学习的 [[Physical State Difference Queries|物理状态差异查询]] $Q$，附加于上下文末尾
- **输出**: [[Action Chunking|动作块]] $a_{t:t+k}$（通过 [[Flow Matching]] 生成）
- **动作专家**: 9 层 [[Diffusion Transformer]]，基于 [[GR00T N1]] 设计
- **初始化**: 从 ZR-0base 权重初始化

### 核心模块

#### 模块1: 物理状态差异查询（Physical State Difference Queries）

**设计动机**: 利用 [[Cross-Attention]] 机制将所有观测与指令信息压缩为紧凑的状态转变表示。

**具体实现**:
- 可学习 Queries $Q \in \mathbb{R}^{N_q \times d}$ 追加到上下文序列末尾
- Readout 操作提取最终层隐状态：$Z_t = \text{Readout}_Q(F_\theta([C_t; Q])) \in \mathbb{R}^{N_q \times d}$
- 差异文本 token 仅可以 attend 到 Queries 和前面的文本，强迫所有观测信息必须经过 Queries 这一瓶颈传递
- Queries 分为 7 个功能组（G1-G7），对应不同物理状态类别

#### 模块2: 结构化转变槽位（Structured Transition Slots）

**设计动机**: 将物理状态转变分解为 9 个可监督的结构化类别，提供精确的几何与语义约束。

**9 个槽位类别**:
1. **任务进度**（Task Progress）：多分类，任务完成度阶段
2. **任务阶段**（Task Phase）：多分类，细粒度操作子步骤
3. **目标位置**（Target Position）：回归，目标物体在图像中的位置
4. **夹爪位置**（Gripper Position）：回归，夹爪在图像中的 3D 位置
5. **运动/相对位移**（Motion/Relative Displacement）：回归，物体相对位移向量
6. **可交互物体位置**（Interactable Position）：回归，相关交互目标的位置
7. **夹爪状态转变**（Gripper State Transitions）：分类，开/闭变化
8. **接触转变**（Contact Transitions）：分类，接触状态变化
9. **障碍物风险**（Obstacle Risk）：分类，障碍物碰撞风险等级

#### 模块3: 稠密光流预测（Dense Optical Flow）

**设计动机**: 提供像素级运动监督，补充槽位监督无法覆盖的稠密运动信息。

**具体实现**:
- 轻量级解码器读取最后 $N_{\text{flow}}$ 个差异 Queries
- 预测当前帧到未来帧之间的稠密 2D 光流场
- 损失加权强调大运动区域

#### 模块4: 未来状态差异文本（Future-State Difference Text）

**设计动机**: 利用 [[VLM]] 原生自回归能力生成结构化差异描述，提供高层语义监督。

**生成内容**: 任务进度描述、空间位置变化、接触状态转变、约束与障碍信息。

---

## 关键公式

### 公式1: [[Physical State Difference Queries|查询状态提取]]

$$
Z_t = \text{Readout}_Q(F_\theta([C_t; Q])) \in \mathbb{R}^{N_q \times d}
$$

**含义**: 将上下文 $C_t$（视觉+语言观测）与可学习差异 Queries $Q$ 拼接后送入骨干网络，提取 Query 位置的最终层隐状态作为物理状态转变表示。

**符号说明**:
- $C_t$: 当前时刻的视觉-语言上下文
- $Q$: 可学习的物理状态差异 Queries
- $F_\theta$: 视觉语言主干网络
- $N_q$: Query 数量
- $d$: 隐层维度

### 公式2: [[Structured Transition Slots|结构化槽位损失]]

$$
\mathcal{L}_{\text{slot}} = \sum_{i=1}^{9} w_i \mathcal{L}_i(\hat{y}_i, y_i)
$$

**含义**: 对 9 个结构化物理状态类别（分类/回归）加权求和，每类使用对应的分类交叉熵或回归损失。

**符号说明**:
- $w_i$: 第 $i$ 个槽位类别的权重
- $\hat{y}_i$: 预测的槽位输出（槽位头 $h_{\text{slot}}$ 从 $Z_t^{\text{slot}}$ 读取）
- $y_i$: 标注的真实槽位值

### 公式3: [[Dense Optical Flow|光流损失]]

$$
\mathcal{L}_{\text{flow}} = \mathcal{L}_{\text{all}} + \beta_{\text{motion}} \mathcal{L}_{\text{motion}}
$$

**含义**: 总光流损失由全局重建损失与大运动区域加权损失组成，$\beta_{\text{motion}}$ 控制运动区域的额外惩罚力度。

**符号说明**:
- $\mathcal{L}_{\text{all}}$: 全像素光流重建损失
- $\mathcal{L}_{\text{motion}}$: 仅在大运动幅度像素上计算的损失
- $\beta_{\text{motion}}$: 大运动区域权重系数

### 公式4: [[Autoregressive Generation|差异文本损失]]

$$
\mathcal{L}_{\text{Text}} = -\frac{1}{|\Omega_T|} \sum_{j \in \Omega_T} \log p_\theta(t_j \mid C, Q, t_{<j}; M_{\text{attn}})
$$

**含义**: 使用骨干网络原生自回归目标，对目标差异文本 token 集合 $\Omega_T$ 计算交叉熵，带有特殊注意力掩码 $M_{\text{attn}}$（限制文本 token 只能 attend 到 Queries）。

**符号说明**:
- $\Omega_T$: 差异文本 token 的位置集合
- $t_j$: 第 $j$ 个目标文本 token
- $M_{\text{attn}}$: 限制注意力范围的掩码

### 公式5: [[Flow Matching|动作损失]]

$$
\mathcal{L}_{\text{act}} = \mathbb{E}\left[\frac{1}{|\Omega_a|} \sum_{k \in \Omega_a} \left([v_\psi(A_t^\tau, \tau \mid Z_t, E_\eta(r_t))]_k - [v_t^*]_k\right)^2\right]
$$

其中插值：

$$
A_t^\tau = (1 - \tau)\xi + \tau A_t, \quad v_t^* = A_t - \xi, \quad \xi \sim \mathcal{N}(0, I)
$$

**含义**: [[Flow Matching]] 目标，训练 [[Diffusion Transformer]] 动作专家预测从噪声到真实动作的流速场，以物理状态转变表示 $Z_t$ 和本体状态编码 $E_\eta(r_t)$ 为条件。

**符号说明**:
- $A_t^\tau$: 时间步 $\tau \in [0, 1]$ 处的插值动作
- $\xi \sim \mathcal{N}(0, I)$: 高斯噪声（流的起点）
- $v_t^*$: 目标流速（真实动作 - 噪声）
- $v_\psi$: 动作专家网络
- $E_\eta$: 本体状态编码器

### 公式6: [[Multitask Learning|预训练总损失]]

$$
\mathcal{L} = \lambda_{\text{Text}} \mathcal{L}_{\text{Text}} + \lambda_{\text{slot}} \mathcal{L}_{\text{slot}} + \lambda_{\text{flow}} \mathcal{L}_{\text{flow}} + \lambda_{\text{act}} \mathcal{L}_{\text{act}}
$$

**含义**: 预训练阶段同时优化四个目标，推理时仅执行 Query 提取与动作生成，无额外解码开销。

**符号说明**:
- $\lambda_{\text{Text}}, \lambda_{\text{slot}}, \lambda_{\text{flow}}, \lambda_{\text{act}}$: 各损失的权重系数

### 公式7: 槽位预测头（Appendix）

$$
u_{t,i} = W_i \cdot \text{LN}_i(\bar{z}_{t,g(i)}) + b_i
$$

**含义**: 第 $i$ 个槽位的预测头，对对应 Query 组 $g(i)$ 的均值隐状态进行层归一化和线性变换。

**符号说明**:
- $\bar{z}_{t,g(i)}$: Query 组 $g(i)$ 的均值隐状态
- $g(i)$: 第 $i$ 个槽位所属的 Query 组
- $\text{LN}_i$: 第 $i$ 个槽位的 Layer Normalization

---

## 关键图表

### Figure 1: 不同 VLA 范式对比

![Figure 1: 五种 VLA 转变建模范式对比](https://arxiv.org/html/2610.21306v1/1-introduction.png)

**说明**: 对比五种范式：(a) 直接 VLA（无物理监督）、(b) Reasoning VLA（如 [[ECoT]]，任务语义）、(c) WAM（生成未来图像帧）、(d) Latent-WAM（预测未来潜变量）、(e) **VLA-Δ**（多层物理状态转变监督）。

### Figure 2: VLA-Δ 整体架构

![Figure 2: VLA-Δ 整体架构图](https://arxiv.org/html/2610.21306v1/2-method.png)

**说明**: 展示 [[Physical State Difference Queries]] 如何从三路监督（结构化槽位、稠密光流、差异文本）学习物理状态转变表示 $Z_t$，并通过 [[Cross-Attention]] 条件化 [[Diffusion Transformer]] 动作专家生成动作。

### Figure 3: 物理状态转变标注示例与自动化流程

![Figure 3: 数据标注流程](https://arxiv.org/html/2610.21306v1/3-Slot.png)

**说明**: 展示自动化标注流水线：Qwen3.8 生成语义分割和结构化标签 → SAM 3 生成目标/夹爪/障碍物 Bounding Box 与掩码 → UniDepth V2 估计深度 → MegaFlow 计算光流 → GPT-5.6 复杂样本质检。

### Figure 4: LIBERO-Plus 扰动鲁棒性

![Figure 4: LIBERO-Plus 四种扰动下性能对比](https://arxiv.org/html/2610.21306v1/4-Libero-Plus.png)

**说明**: VLA-Δ 在语言扰动（Language Perturbation）下达到 89.0%，显著超越 [[pi0.5]] 的 82.6%，在光照、背景、布局扰动下也保持竞争力。

### Figure 5: 真实机器人实验结果

![Figure 5: AgileX PiPER 真实机器人评估](https://arxiv.org/html/2610.21306v1/5-Real-Robot_Evaluation.png)

**说明**: 在 AgileX PiPER 6-DoF 双臂机器人上评估 5 项任务（障碍物规避、长视野、细粒度、空间推理、泛化），VLA-Δ 平均 68.0%，超越 [[pi0.5]] 5.5 个百分点，障碍物规避任务（42.5%）表现最突出。

### Figure 6: Query 隐状态 t-SNE 可视化

![Figure 6: t-SNE 可视化 Slot Query 隐状态](https://arxiv.org/html/2610.21306v1/6-tsne.png)

**说明**: [[t-SNE]] 可视化显示 7 个 [[Physical State Difference Queries|Query 组]] 的隐状态形成清晰分离的簇，且在仅用动作损失微调后该结构仍然保持，说明预训练物理状态表示具有稳健的结构化特性。

### Figure 7: Query 注意力热图

![Figure 7: 毛巾折叠和挂杯任务的注意力热图](https://arxiv.org/html/2610.21306v1/7-Heat-map.png)

**说明**: 完整 VLA-Δ 模型的注意力集中于目标物体、夹爪、接触区域，而消融版本（无物理状态监督）更多 attend 到背景区域，验证物理状态监督引导模型关注动作相关区域。

### Figure 8: 标注示例

![Figure 8a: 当前-未来帧对与差异文本+槽位标注](https://arxiv.org/html/2610.21306v1/dataset.png)

![Figure 8b: 装盘任务槽位标注](https://arxiv.org/html/2610.21306v1/Slot-1.png)

![Figure 8c: 挂杯任务槽位标注](https://arxiv.org/html/2610.21306v1/Slot-2.png)

**说明**: 展示三种任务下的完整标注样例，包括当前帧-未来帧对、差异文本描述以及 9 类结构化槽位的具体标注值。

### Table 1: LIBERO 成功率对比（%）

| Method | Spatial | Object | Goal | Long | Average |
|--------|---------|--------|------|------|---------|
| TraceVLA | 92.2 | 92.2 | 91.4 | 84.6 | 90.1 |
| OpenVLA | 84.7 | 88.4 | 79.2 | 53.7 | 76.5 |
| SpatialVLA | 95.2 | 91.4 | 93.0 | 84.4 | 91.0 |
| WorldVLA | 97.0 | 96.6 | 97.0 | 88.0 | 94.7 |
| CoT-VLA | 97.4 | 98.0 | 96.4 | 91.4 | 95.8 |
| ThinkAct | 97.8 | 98.6 | 97.0 | 92.8 | 96.6 |
| SmolVLA | 97.4 | 97.4 | 97.0 | 93.2 | 96.3 |
| DreamVLA | 97.4 | 98.2 | 97.4 | 94.6 | 96.9 |
| π0 | 97.0 | 96.4 | 97.8 | 94.8 | 96.5 |
| UniVLA | 98.0 | 98.8 | 98.0 | 95.0 | 97.5 |
| GR00T-N1.7 | 97.8 | 98.4 | 97.4 | 94.4 | 97.0 |
| OpenVLA-OFT | 97.0 | 98.0 | 97.8 | 93.4 | 96.6 |
| VLA-Adapter | 97.6 | 98.2 | 97.4 | 93.8 | 96.8 |
| π0.5 | 98.8 | 98.2 | 98.0 | 92.4 | 96.9 |
| ZR-0 | 97.4 | 99.4 | 97.8 | 96.4 | 97.8 |
| **VLA-Δ（Ours）** | **98.8** | **99.8** | **98.6** | **96.4** | **98.4** |

**关键发现**: VLA-Δ 在 LIBERO 四个子任务上全面超越所有基线，平均成功率 98.4% 为最优，Object 子任务达到 99.8%。

### Table 2: RoboTwin 2.0 成功率对比（50 任务平均，%）

| Method | Clean | Random |
|--------|-------|--------|
| X-VLA | 72.88 | 72.84 |
| π0 | 65.92 | 58.40 |
| π0.5 | 82.74 | 76.76 |
| Motus | 88.52 | 87.02 |
| ZR-0 | 88.70 | 87.98 |
| **VLA-Δ（Ours）** | **90.06** | **89.74** |

**关键发现**: VLA-Δ 在清洁和随机初始化条件下均最优。Clean-to-Random 性能下降仅 0.32%（对比 ZR-0 的 0.72%、Motus 的 1.50%），说明物理状态监督显著提升了对初始化扰动的鲁棒性。

### Table 3: LIBERO-Plus 扰动鲁棒性（%）

| 扰动类型 | π0.5 | ZR-0 | **VLA-Δ** |
|---------|------|------|-----------|
| Language | 82.6 | 85.3 | **89.0** |
| Light | — | — | competitive |
| Background | — | — | competitive |
| Layout | — | — | competitive |

**关键发现**: 语言扰动下 VLA-Δ 提升 6.4%（vs π0.5），表明物理状态转变监督增强了对语言多样性的泛化能力。

### Table 4: 预训练数据集统计

| 数据集 | 帧数 | 时长 | 任务数 |
|--------|------|------|--------|
| MolmoAct | 1.10M | 30.7h | 96 |
| DROID | 27.6M | 511.7h | 49,630 |
| RH20T | 3.9M | 108.3h | 141 |
| **总计** | **~24M（筛选后）** | **~484h** | **—** |

### Table 5: 标注质量审计（300 个样本）

| 标注类别 | 有效比例 |
|---------|---------|
| 运动/位移 | 93.00% |
| 任务进度 | 97.33% |
| 任务阶段 | 98.00% |
| 目标位置 | 99.33% |
| 物体约束与目标位置 | 99.67% |

### Table 6: 消融实验（LIBERO 40 任务平均成功率，%）

| 变体 | 成功率 |
|------|--------|
| Dense VLM features（无预训练辅助） | 95.70 |
| Dense VLM features + 差异文本预训练 | 95.85 |
| Difference Queries only | 96.25 |
| Queries + 差异文本预训练 | 96.45 |
| Query + 差异文本 | 97.20 |
| Query + 结构化槽位 | 96.90 |
| Query + 光流 | 97.35 |
| **Full VLA-Δ（三路监督）** | **98.40** |

**关键发现**: 三路监督缺一不可：光流贡献最大的单路提升，结构化槽位提供最强的可解释性，差异文本贡献高层语义，三路组合产生 2.7% 的综合增益。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| DROID | 27.6M 帧，511.7h | 49,630 任务，大规模多样 | 预训练 |
| MolmoAct | 1.10M 帧，30.7h | 96 任务，带语言标注 | 预训练 |
| RH20T | 3.9M 帧，108.3h | 141 任务，触觉+视觉 | 预训练 |
| LIBERO | 50 任务 × 50 ep | 4 子套件（Spatial/Object/Goal/Long） | 评估 |
| LIBERO-Plus | 40 任务 × 50 ep × 4 扰动 | 语言/光照/背景/布局扰动 | 鲁棒性评估 |
| RoboTwin 2.0 | 50 任务 × 100 ep | 27,500 训练演示，动作块 16 | 评估 |
| Real Robot | 5 任务 × 40 ep | AgileX PiPER 6-DoF 双臂 | 真实机器人评估 |

### 实现细节

- **Backbone**: [[Qwen3-VL-2B-Instruct]]
- **动作专家**: 9 层 [[Diffusion Transformer]]，基于 [[GR00T N1]] 架构
- **初始化**: ZR-0base 权重
- **动作生成**: [[Flow Matching]]，动作块 16 步，执行前 8 步后重新规划
- **评估协议**: LIBERO 每任务 50 次，RoboTwin 每任务 100 次
- **标注工具**: Qwen3.8（语义）+ SAM 3（分割）+ UniDepth V2（深度）+ MegaFlow（光流）+ GPT-5.6（质检）

### 可视化结果

- t-SNE 分析确认 7 个 Query 组在物理状态转变空间中形成可解释的离散簇，且微调后持续保持
- 注意力热图证实物理状态监督引导模型关注任务关键区域而非背景

---

## 批判性思考

### 优点

1. **物理监督层次丰富**: 三路监督（结构化、稠密几何、语义文本）从不同粒度覆盖物理状态变化，相互补充
2. **推理零开销**: 所有辅助目标仅在预训练阶段使用，推理时完全透明
3. **自动化标注流水线**: 结合多个专业模型实现高质量标注，93-99% 的验证通过率且有人工审计

### 局限性

1. **标注质量依赖**: GPT-5.6 生成的 Bounding Box 不是独立的人工真值，复杂场景的标注噪声难以消除
2. **Zotero 分类依赖单一视角**: 实验仅在桌面操作任务上验证，移动底座、人形机器人等场景泛化未知
3. **数据集覆盖不均衡**: DROID 占预训练数据 85% 以上，可能导致对 DROID 场景的过拟合

### 潜在改进方向

1. 将在线轨迹修正引入推理阶段，实现更动态的物理状态感知
2. 扩展到触觉、听觉等多模态物理状态监督

### 可复现性评估

- [x] 代码开源（GitHub: ZJU-OmniAI/VLA-Delta）
- [ ] 预训练模型（未提及公开发布）
- [x] 训练细节完整（附录 D 详细描述槽位头和 Query 组分配）
- [x] 数据集可获取（DROID、RH20T、MolmoAct 均为公开数据集）

---

## 关联笔记

### 基于

- [[GR00T N1]]: 动作专家架构基础，VLA-Δ 使用其 9 层 Diffusion Transformer 设计
- [[Qwen3-VL]]: 视觉语言骨干网络
- [[Flow Matching]]: 动作生成范式

### 对比

- [[pi0.5]]: 主要对比基线，真实机器人实验超越 5.5%
- [[ZR-0]]: 基于相同初始化权重，VLA-Δ 在 LIBERO 和 RoboTwin 上均优于它
- [[WorldVLA]]: 代表 WAM 范式（生成未来图像），VLA-Δ 避免其高计算开销
- [[DreamVLA]]: 代表 Latent-WAM 范式，VLA-Δ 在结构化监督上超越
- [[ECoT]]: 代表 Reasoning VLA 范式（仅语义推理）

### 方法相关

- [[Physical State Transition]]: 核心监督信号，区别于像素级或任务级监督
- [[Optical Flow]]: 稠密运动监督的技术基础
- [[Diffusion Transformer]]: 动作生成专家架构
- [[Action Chunking]]: 动作块生成策略

### 数据相关

- [[DROID]]: 主要预训练数据集（511.7h）
- [[RH20T]]: 多模态机器人操作数据集
- [[LIBERO]]: 标准仿真评测基准
- [[RoboTwin]]: 大规模仿真评测平台

---

## 速查卡片

> [!summary] VLA-Δ: Learning Physical State Transition for VLAs
> - **核心**: 用物理状态转变（槽位+光流+差异文本）作为预训练显式监督，推理零开销
> - **方法**: 可学习差异 Queries 压缩三路物理监督 → 条件化 Diffusion Transformer 动作专家
> - **结果**: LIBERO 98.4%、RoboTwin 90.06%、真实机器人超越 π0.5 5.5%
> - **代码**: https://github.com/ZJU-OmniAI/VLA-Delta

---

*笔记创建时间: 2026-10-17*
