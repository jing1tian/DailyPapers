---
title: "Generalizable VLA Finetuning via Representation Anchoring and Language-Action Alignment"
method_name: "AnchorAlign"
authors: [Dwip Dalal, Shivansh Patel, Chahit Jain, Jeonghwan Kim, Utkarsh Mishra, Alex Baratian, Hyeonjeong Ha, Heng Ji, Svetlana Lazebnik, Unnat Jain]
year: 2026
venue: arXiv
tags: [vla, behavior-cloning, knowledge-distillation, representation-learning, generalization, catastrophic-forgetting, robot-manipulation, finetuning]
zotero_collection: Robotics/VLA
image_source: online
arxiv_html: https://arxiv.org/html/2607.13429v1
created: 2026-07-17
---

# 论文笔记：Generalizable VLA Finetuning via Representation Anchoring and Language-Action Alignment

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | UIUC, Texas A&M University, UC Irvine |
| 日期 | July 2026 |
| 项目主页 | [anchoralignvla.github.io](https://anchoralignvla.github.io) |
| 对比基线 | [[VLA-Adapter]], [[OpenVLA-OFT]], [[MolmoAct]] |
| 链接 | [arXiv](https://arxiv.org/abs/2607.13429) |

---

## 一句话总结

> Anchor-Align 通过层级表示蒸馏与语言-动作对齐两个损失项，防止 VLA 微调时灾难性遗忘，在无需额外数据的情况下显著提升真实机器人任务的泛化能力。

---

## 核心贡献

1. **[[Vision-Language Anchoring|表示锚定]]**: 维护冻结预训练 VLM 副本作为 teacher，对所有 decoder 层施加层级蒸馏损失，防止[[行为克隆]]微调导致的[[灾难性遗忘]]
2. **[[Language-Action Alignment|语言-动作对齐]]**: 将连续动作块自动转换为离散运动方向标签（6 方向），在相同观测上同时监督语言头和动作头，消除语言-动作预测矛盾
3. **零额外数据方案**: 两个目标完全复用演示数据中已有的监督信号，无需额外标注或数据收集

---

## 问题背景

### 要解决的问题

标准[[行为克隆]]微调会渐进式覆盖 [[VLA（视觉-语言-动作模型）|VLM]] 的预训练表示，导致视觉语义泛化能力下降——模型"记住"训练场景的动作轨迹，而非理解语言指令的语义。

### 现有方法的局限

1. **协同训练**（[[Co-training]]）：将机器人演示与视觉语言数据混合训练，但无法防止表示漂移，且语言头和动作头在**不同观测**上接受监督，造成语言-动作矛盾（Pearson r = −0.03）
2. **冻结骨干**（Frozen backbone）：完全冻结 [[大型视觉语言模型|VLM]] 无法适应机器人视觉域，在 LIBERO-PRO 平均成功率仅 43.1%

### 本文的动机

VLM 预训练建立了强大的视觉语义能力，应当"保留并接地"而非"交换给控制"。实验证明，标准 BC 在 10K 步内损失 **94%** 的 GQA 视觉推理准确率——这种[[灾难性遗忘]]是可以通过蒸馏约束避免的。

---

## 方法详解

### 模型架构

Anchor-Align 以[[VLA（视觉-语言-动作模型）]]为基础架构，采用**双约束微调**策略：

- **输入**: 语言指令 $l$ + 观测 $o_t$（[[DINOv2]] + SigLIP 双编码，256 patches/图像，共 512 patches）
- **Backbone**: Prismatic-Qwen2.5-0.5B，通过 [[LoRA]]（rank=64，全层应用）适配
- **核心模块**: [[Vision-Language Anchoring|表示锚定]] + [[Language-Action Alignment|语言-动作对齐]]
- **动作头1**: [[Bridge Attention]] + 回归（VLA-Adapter 架构）
- **动作头2**: [[Flow Matching|流匹配]] FM-DiT（StarVLA 架构）
- **总额外参数**: Projection Head 约 803,712（1.5 MB bfloat16）

### 核心模块

#### 模块1: Vision-Language Anchoring（表示锚定）

**设计动机**: 利用[[知识蒸馏]]防止[[灾难性遗忘]]，无需额外数据。

**具体实现**:

- 维护一个**冻结的预训练 VLM 副本**作为 anchor（Teacher）
- 对所有 decoder 层施加层级 Frobenius 范数蒸馏损失
- 仅对视觉和文本 token 位置 $\mathbf{m}$ 计算对齐（不包括动作 token）

#### 模块2: Language-Action Alignment（语言-动作对齐）

**设计动机**: 消除[[Co-training|协同训练]]中语言头和动作头接受不同观测监督造成的矛盾。

**具体实现**:

- 将 K 步[[动作分块|动作块]]的平移分量取平均得到运动方向向量 $\bar{\mathbf{v}}$
- 将最大绝对值分量映射到 6 方向标签之一：`{forward, backward, left, right, up, down}`
- 通过学习的投影矩阵 $\mathbf{W}_{proj}$ 和冻结的语言建模头 $\mathbf{W}_{lm}$ 预测方向
- 在**相同观测**上同时监督语言头（方向分类）和动作头（行为克隆）

---

## 关键公式

### 公式1: [[行为克隆|总训练目标]]

$$
\mathcal{L}_{total} = \mathcal{L}_{action} + \lambda_{anchor} \cdot \mathcal{L}_{anchor} + \lambda_{align} \cdot \mathcal{L}_{align}
$$

**含义**: Anchor-Align 的总损失由行为克隆损失、表示锚定损失和语言-动作对齐损失三项组成。

**符号说明**:

- $\mathcal{L}_{action}$: 行为克隆动作预测损失
- $\mathcal{L}_{anchor}$: 层级表示蒸馏损失
- $\mathcal{L}_{align}$: 语言-动作方向对齐损失
- $\lambda_{anchor} = 1.0$，$\lambda_{align} = 0.5$: 权重系数

### 公式2: [[Vision-Language Anchoring|单层锚定损失]]

$$
\mathcal{L}_{anchor}^{(u)} = \left\|\mathbf{H}_u^S[\mathbf{m}] - \mathbf{H}_u^A[\mathbf{m}]\right\|_F^2
$$

**含义**: 第 $u$ 层的 student（训练中的 VLA）与 anchor（冻结 VLM）在视觉和文本 token 位置上的隐状态差异，用 Frobenius 范数度量。

**符号说明**:

- $\mathbf{H}_u^S$: 第 $u$ 层 student backbone 的隐状态
- $\mathbf{H}_u^A$: 第 $u$ 层 anchor（冻结）VLM 的隐状态
- $\mathbf{m}$: 视觉和文本 token 的位置集合
- $\|\cdot\|_F^2$: Frobenius 范数的平方

### 公式3: [[知识蒸馏|总锚定损失]]

$$
\mathcal{L}_{anchor} = \frac{1}{|\mathcal{D}|} \sum_{u \in \mathcal{D}} \mathcal{L}_{anchor}^{(u)}
$$

**含义**: 对所有 decoder 层的锚定损失取平均，实现全层级的表示对齐。

**符号说明**:

- $\mathcal{D}$: decoder 层集合
- $|\mathcal{D}|$: decoder 层总数

### 公式4: [[Language-Action Alignment|语言动作预测]]

$$
\mathbf{a}^{lang} = \mathbf{W}_{lm} \cdot \mathbf{W}_{proj} \cdot \mathbf{h}^{pre} \in \mathbb{R}^{|\mathcal{V}|}
$$

**含义**: 将最后一条指令 token 的最后层隐状态，经学习的投影矩阵和冻结的语言建模头，转换为词表空间上的方向预测 logits。

**符号说明**:

- $\mathbf{h}^{pre}$: 最后一条指令 token 的最后层隐状态
- $\mathbf{W}_{proj} \in \mathbb{R}^{896 \times 896}$: 学习的投影矩阵（约 803K 参数）
- $\mathbf{W}_{lm}$: 冻结的预训练语言建模头
- $\mathcal{V}$: 词表集合

### 公式5: [[Language-Action Alignment|对齐损失]]

$$
\mathcal{L}_{align} = \operatorname{CE}\!\left(\mathbf{a}^{lang},\ \hat{a}^{lang}\right)
$$

**含义**: 预测的方向 logits 与从动作块自动导出的真实方向标签之间的[[交叉熵]]损失。

**符号说明**:

- $\mathbf{a}^{lang}$: 预测的方向 logits（词表维度）
- $\hat{a}^{lang}$: 从演示[[动作分块|动作块]]中自动导出的真实方向标签
- $\operatorname{CE}$: 交叉熵损失函数

### 公式6: [[动作分块|动作平均（方向标签生成）]]

$$
\bar{\mathbf{v}} = \frac{1}{K} \sum_{k=1}^{K} \mathbf{A}_{:,k,1:3} \in \mathbb{R}^{B \times 3}
$$

**含义**: 对动作块中 K 步的平移分量求平均，得到整体运动方向向量，再取最大绝对值分量对应方向作为标签。

**符号说明**:

- $\mathbf{A} \in \mathbb{R}^{B \times K \times 7}$: 批次动作块（7 维：3 平移 + 3 旋转 + 1 夹爪）
- $B$: batch size
- $K$: chunk 长度
- $\mathbf{A}_{:,k,1:3}$: 第 $k$ 步的平移分量（前 3 维）

---

## 关键图表

### Figure 1: OOD 语义泛化测试对比

![Figure 1](https://arxiv.org/html/2607.13429v1/x1.png)

**说明**: 标准 BC 微调后，即使指令改为"抓粉色杯子"，模型仍然伸向绿色杯子（训练先验）。Anchor-Align 保留了 [[VLA（视觉-语言-动作模型）|VLM]] 的语义表示，能正确响应新指令。直观展示了[[灾难性遗忘]]问题及本方法的解决效果。

### Figure 2: Anchor-Align 方法总览

![Figure 2](https://arxiv.org/html/2607.13429v1/x2.png)

**说明**: 展示[[Vision-Language Anchoring|表示锚定]]和[[Language-Action Alignment|语言-动作对齐]]两个核心模块。锚定模块通过 Anchor Loss 在每个 transformer 层蒸馏冻结 VLM 的表示；对齐模块通过 Align Loss 在相同观测上同时监督语言方向预测和动作生成。

### Figure 3: 语言-动作对齐原理

![Figure 3](https://arxiv.org/html/2607.13429v1/x3.png)

**说明**: 标准 BC（红色）只监督动作预测，语言头输出与所需运动方向矛盾。[[Language-Action Alignment|语言-动作对齐]]（绿色）从真实动作目标中导出离散方向标签，强制语言输出与动作生成达成一致。蓝色为当前观测中的实际动作方向。

### Figure 4: LIBERO-PRO 语义扰动泛化

![Figure 4](https://arxiv.org/html/2607.13429v1/x4.png)

**说明**: 上排为位置交换轴（position-swap），下排为对象交换轴（object-swap）。标准 BC 执行与训练场景绑定的轨迹（伸向原始位置或错误目标）；Anchor-Align 能根据扰动后的观测正确定位目标。

### Figure 5: 真实机器人泛化测试

![Figure 5](https://arxiv.org/html/2607.13429v1/x5.png)

**说明**: 三行分别为不同扰动场景（组合对象布局、空间重排、语义扰动）。每行左侧为训练场景，右侧为测试场景。标准 BC 失败，Anchor-Align 能跨布局变化完成任务。

### Figure 6: 杂乱场景中的目标选择

![Figure 6](https://arxiv.org/html/2607.13429v1/x6.png)

**说明**: 从高度杂乱的桌面出发，Anchor-Align VLA 能在众多干扰物中正确抓取语言指定的目标物体，展示了对当前场景的实时语言接地能力。

### Figure 7: 真实机器人多架构多扰动成功率

![Figure 7a](https://arxiv.org/html/2607.13429v1/x7.png)

![Figure 7b](https://arxiv.org/html/2607.13429v1/x8.png)

![Figure 7c](https://arxiv.org/html/2607.13429v1/x9.png)

**说明**: 在两种 VLA 骨干（VLA-Adapter 和 [[StarVLA]]）上，三类泛化测试（空间重排、杂乱场景、组合布局）的成功率对比（每条件 20 次 rollout）。Anchor-Align 在所有条件和骨干上均有提升，验证方法的架构无关性。

### Figure 8: 失败模式分析

![Figure 8](https://arxiv.org/html/2607.13429v1/x10.png)

**说明**: 对比标准 BC 和 Anchor-Align 的真实机器人失败原因分布。Anchor-Align 将语义错误从 7 降至 0，将错误对象选取从 10 降至 1，记忆化错误从约 15 降至 9，主要剩余失败为抓取执行和高精度操作问题。

### Figure 9: 视觉推理能力保留曲线

![Figure 9](https://arxiv.org/html/2607.13429v1/x11.png)

**说明**: 展示微调过程中 GQA 视觉推理准确率的变化。标准 BC（橙色）在 10K 步内损失 94% 的预训练视觉语义能力；Anchor-Align（绿色）保留 70%；冻结 VLM（虚线）为上界。量化证明"动作性能与视觉语言推理并非根本对立"。

### Table 1: LIBERO-PRO 和 LIBERO-Plus 基准对比

| Method | Lang. Reph. | Object Swap | Pos. Swap | Mean (PRO) | Lang. Instr. | Bg. Text. | Robot Init | Cam. View | Obj. Layout | Light Cond. | Sensor Noise | Mean (Plus) |
|--------|------------|------------|----------|------------|------------|----------|-----------|----------|------------|-----------|------------|-------------|
| Co-training + KI | 54.0 | 77.4 | 0.0 | 43.8 | 48.0 | 82.6 | 25.7 | 64.6 | 65.7 | 73.3 | 49.0 | 57.1 |
| MolmoAct | 77.8 | 82.4 | 0.0 | 53.4 | 79.5 | 84.1 | 47.4 | 10.1 | 76.5 | 77.4 | 53.4 | 60.8 |
| OpenVLA-OFT | 74.4 | 95.2 | 0.0 | 56.5 | 81.5 | 95.7 | 40.3 | 94.7 | 88.6 | 95.5 | 28.2 | 74.1 |
| VLA-Adapter [Frozen] | 56.0 | 73.4 | 0.0 | 43.1 | 41.5 | 70.9 | 35.1 | 94.4 | 62.3 | 84.9 | 36.2 | 59.9 |
| VLA-Adapter | 91.1 | 89.6 | 2.3 | 61.0 | 85.1 | 90.7 | 52.6 | 92.6 | 93.2 | 93.2 | 89.5 | 85.1 |
| **Anchor-Align VLA** | **97.0** | **96.2** | **22.6** | **71.9** | **87.2** | **99.6** | **59.1** | **96.3** | **97.4** | **99.0** | **96.9** | **90.3** |

**关键发现**: Anchor-Align 在最困难的位置交换（Pos. Swap）上从 2.3% 大幅提升至 22.6%（+20.3%），LIBERO-PRO 平均成功率提升 10.9 点，LIBERO-Plus 提升 5.2 点。

### Table 2: CALVIN ABC→D 长视野任务

| Method | 1/5 | 2/5 | 3/5 | 4/5 | 5/5 | Avg Len |
|--------|-----|-----|-----|-----|-----|---------|
| UniVLA | 95.5 | 85.8 | 75.4 | 66.9 | 56.5 | 3.8 |
| OpenVLA-OFT | 96.3 | 89.1 | 82.4 | 75.8 | 66.5 | 4.1 |
| OpenHelix | 97.1 | 91.4 | 82.8 | 72.6 | 64.1 | 4.1 |
| VLA-Adapter | 98.3 | 94.0 | 87.5 | 80.0 | 73.1 | 4.3 |
| **Anchor-Align VLA** | **99.1** | **95.8** | **90.6** | **84.7** | **77.9** | **4.5** |

**关键发现**: Anchor-Align 在 5 指令链上完成率从 73.1% 提升至 77.9%，平均 rollout 长度从 4.3 到 4.5，达到 SOTA。

### Table 3: 消融实验

| 方法 | 关键组件 | LIBERO-PRO | LIBERO-Plus |
|------|---------|-----------|------------|
| VLA-Adapter | 标准 BC | 61.0 | 85.1 |
| Align VLA | 仅对齐 | 65.9 | 88.6 |
| Anchor VLA | 仅锚定 | 68.1 | 87.3 |
| **Anchor-Align VLA** | **完整方法** | **71.9** | **90.3** |

**关键发现**: 两个组件均独立有效，结合后效果最佳；锚定组件单独贡献略大于对齐组件（+7.1 vs +4.9 on PRO）。

### Table 4: 正则化控制实验

| 方法 | 关键技术 | LIBERO-PRO | LIBERO-Plus |
|------|---------|-----------|------------|
| VLA-Adapter | 标准 BC | 61.0 | 85.1 |
| Shuffle（控制） | 乱序方向标签 | 61.4 | 84.9 |
| Scatter（控制） | 随机分散标签 | 63.3 | 85.7 |
| **Anchor-Align VLA** | **真实方向对齐** | **71.9** | **90.3** |

**关键发现**: Shuffle 和 Scatter 控制组接近标准 VLA 微调性能，证明提升来自真实的语言-动作对齐语义，而非辅助任务的正则化效果。

### Table 5: 语言-动作对齐量化

| 模型 | 成功率 | 对齐率 | [[Pearson Correlation\|Pearson r]] |
|------|-------|-------|----------------------------------|
| Co-training + KI | 43.8% | 14.3% | +0.10 |
| VLA-Adapter | 61.0% | 16.8% | −0.03 |
| **Anchor-Align VLA** | **71.9%** | **78.4%** | **+0.51** |

**关键发现**: Pearson r=+0.51 表明语言-动作对齐率与任务成功率显著正相关；标准 BC 甚至呈现负相关（r=−0.03），说明其语言头输出与动作存在系统性矛盾。

---

## 实验

### 数据集与基准

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| [[LIBERO-PRO]] | OOD 扰动套件 | 语言重述、对象交换、位置交换 3 轴 | OOD 泛化测试 |
| [[LIBERO-Plus]] | 感知鲁棒套件 | 背景纹理、光照、相机视角等 7 维扰动 | 感知鲁棒性测试 |
| [[CALVIN]] ABC→D | 长视野链式任务 | 在 A/B/C 训练，D 环境零样本测试 | 长视野任务测试 |
| 真实机器人 | 20 rollouts/条件 | xArm7，空间重排/杂乱/组合布局 | 真实场景验证 |

### 实现细节

- **Backbone**: Prismatic-Qwen2.5-0.5B
- **适配器**: [[LoRA]]，rank=64，全层应用
- **视觉编码**: DINOv2 + SigLIP 双编码，256 patches/图像（共 512）
- **动作头1**: [[Bridge Attention]] + 回归（VLA-Adapter 架构）
- **动作头2**: [[Flow Matching|流匹配]] FM-DiT（[[StarVLA]] 架构）
- **投影头**: $\mathbf{W}_{proj} \in \mathbb{R}^{896 \times 896}$，约 803,712 参数（1.5 MB）
- **超参数**: $\lambda_{anchor}=1.0$，$\lambda_{align}=0.5$
- **硬件**: 4 × NVIDIA GH200，bfloat16 混合精度，PyTorch 2.7
- **真实机器人**: UFactory xArm7（7-DOF）

### 可视化结果

- 语义扰动实验中，标准 BC 将 90% 的 rollout 错误地抓向训练先验对象，Anchor-Align 实现 100% 正确识别新颜色指令
- GQA 视觉推理曲线清晰展示两者在微调过程中表示能力的分化（10K 步内差距达 60+ 个百分点）

---

## 批判性思考

### 优点

1. **零额外数据**: 两个目标完全复用演示数据中已有监督信号，实用性强
2. **架构无关**: 在回归头（VLA-Adapter）和[[Flow Matching|流匹配]]头（[[StarVLA]]）两种异构 VLA 上均有效
3. **可量化分析**: 通过 GQA、对齐率、[[Pearson Correlation|皮尔逊相关]]等指标提供了清晰的机制解释
4. **真实场景提升显著**: 真实机器人平均成功率接近翻倍（28.3%→54.2% / 36.7%→60.0%）

### 局限性

1. **方向标签简化**: 仅 6 个方向标签，旋转密集任务（如拧螺丝）的表达力不足
2. **轻量级骨干**: 使用 0.5B 参数的 VLM，能否 scale 到更大模型需验证
3. **抓取精度问题残余**: 失败模式分析显示抓取执行仍是主要错误来源，方法对精度操作改善有限

### 潜在改进方向

1. 扩展方向标签至更细粒度（如加入旋转方向、速度幅度），覆盖更复杂任务
2. 探索锚定 loss 的 token 选择策略（如仅对关键视觉 token 锚定，降低计算开销）
3. 将对齐机制拓展至多步推理场景（如与 [[ECoT]] 框架结合）

### 可复现性评估

- [x] 代码开源（anchoralignvla.github.io）
- [ ] 预训练模型（未提供）
- [x] 训练细节完整（LoRA rank、λ 值、硬件均有说明）
- [x] 数据集可获取（LIBERO、CALVIN 均公开）

---

## 关联笔记

### 基于

- [[VLA（视觉-语言-动作模型）]]: 核心基础架构
- [[行为克隆]]: 微调的基础方法，也是本文改进的对象
- [[知识蒸馏]]: 锚定模块的核心技术

### 对比

- [[OpenVLA-OFT]]: LIBERO 基准主要对比基线
- [[MolmoAct]]: 视觉语言模型转 VLA 方法对比
- [[UniVLA]]: CALVIN 对比基线

### 方法相关

- [[Vision-Language Anchoring]]: 核心方法1——层级 VLM 表示蒸馏
- [[Language-Action Alignment]]: 核心方法2——动作转方向标签联合监督
- [[LoRA]]: 参数高效微调
- [[Bridge Attention]]: VLA-Adapter 动作头注意力机制
- [[Flow Matching]]: StarVLA 动作头生成方法

### 硬件/数据相关

- [[LIBERO-PRO]]: OOD 泛化基准
- [[LIBERO-Plus]]: 感知鲁棒性基准
- [[CALVIN]]: 长视野任务基准

---

## 速查卡片

> [!summary] AnchorAlign
> - **核心**: 防止 VLA 微调中 VLM 表示漂移，同时对齐语言和动作预测
> - **方法**: 层级蒸馏（Vision-Language Anchoring）+ 自动方向标签监督（Language-Action Alignment）
> - **结果**: 真实机器人成功率接近翻倍；LIBERO-PRO mean +10.9%；CALVIN avg len 4.5（SOTA）
> - **代码**: [anchoralignvla.github.io](https://anchoralignvla.github.io)

---

*笔记创建时间: 2026-07-17*
