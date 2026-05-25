---
title: "PhysX-Omni: Unified Simulation-Ready Physical 3D Generation for Rigid, Deformable, and Articulated Objects"
method_name: "PhysX-Omni"
authors: [Ziang Cao, Yinghao Liu, Haitian Li, Runmao Yao, Fangzhou Hong, Zhaoxi Chen, Liang Pan, Ziwei Liu]
year: 2026
venue: arXiv
tags: [3d-generation, simulation-ready, physical-properties, articulated-objects, deformable-objects, vision-language-model, embodied-ai]
zotero_collection: Robotics
image_source: online
arxiv_html: https://arxiv.org/html/2605.21572v1
created: 2026-05-25
---

# 论文笔记：PhysX-Omni: Unified Simulation-Ready Physical 3D Generation

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | S-Lab, Nanyang Technological University; ACE Robotics |
| 日期 | May 2026 |
| 项目主页 | [physx-omni.github.io](https://physx-omni.github.io) |
| 对比基线 | [[PhysXGen]], [[MonoArt]], [[PhysX-Anything]] |
| 链接 | [arXiv](https://arxiv.org/abs/2605.21572) / [Code](https://github.com/physx-omni/PhysX-Omni) / [Dataset](https://huggingface.co/datasets/PhysX-Omni/PhysXVerse) |

---

## 一句话总结

> PhysX-Omni 提出统一框架，通过新颖的 Template-based RLE 几何表示与 VLM，从单张图像生成涵盖刚体、可变形体和关节体的仿真就绪 3D 资产，并构建了首个综合性物理3D生成数据集和基准。

---

## 核心贡献

1. **PhysX-Omni 框架**: 统一生成框架，从单张图像生成具备完整物理属性（几何、质量、材料、运动学）的仿真就绪 3D 资产，支持刚体、可变形体、关节体三类对象
2. **PhysXVerse 数据集**: 首个通用仿真就绪物理 3D 数据集，包含 8,700+ 资产、2,900+ 类别，覆盖室内家具、无人机、机器人、车辆等
3. **PhysX-Bench 基准**: 首个从几何、绝对尺度、材料、可供性、运动学、功能描述六维度系统评估物理 3D 生成的基准

---

## 问题背景

### 要解决的问题

现有 3D 生成方法主要关注外观（几何形状、纹理），忽略了物理属性（材料、质量、关节运动学等），无法直接用于物理仿真器（如 [[MuJoCo]]、[[Isaac Lab]]）进行机器人训练和具身 AI 研究。

### 现有方法的局限

- **外观中心方法**（DreamFusion、Large-vocabulary 3D 等）：仅生成视觉美观资产，缺乏物理属性标注
- **单类别物理方法**：仅针对关节体（MonoArt）或可变形体（单独方法），无法统一处理
- **数据稀缺**：缺乏大规模、多类别、带完整物理标注的 3D 数据集
- **评估缺失**：没有系统性评估物理属性生成质量的基准

### 本文的动机

通过构建大规模多类别物理 3D 数据集（PhysXVerse），并设计适合 [[视觉语言模型|VLM]] 处理 3D 几何的新颖表示（Template-based RLE），实现统一的粗到细、全局到局部的物理 3D 生成范式。

---

## 方法详解

### 模型架构

![Figure 1: PhysX-Omni概览](https://arxiv.org/html/2605.21572v1/x1.png)

**说明**: PhysX-Omni 可从单张图像生成刚体、可变形体、关节体三类仿真就绪 3D 资产，支持下游机器人策略学习和场景生成应用。

PhysX-Omni 采用**粗到细、全局到局部**的生成范式，以 [[Qwen2.5-VL]] 为骨干网络：

- **输入**: 单张完整或部分遮挡的对象图像
- **Backbone**: Qwen2.5-VL-7B-Instruct（[[大型视觉语言模型|VLM]]）
- **核心模块**: [[Template-based RLE]] 几何表示 + 多轮生成过程
- **输出**: 仿真就绪 3D 资产（网格 + 物理属性标注）
- **解码器**: [[TRELLIS]]（体素到网格转换）
- **总参数**: ~7B（Qwen2.5-VL-7B 骨干）

### 核心模块

#### 模块1: 粗到细全局-局部生成范式（Coarse-to-Fine Generation）

![Figure 2: 生成流水线](https://arxiv.org/html/2605.21572v1/x2.png)

**说明**: 给定单张图像，PhysX-Omni 首先推断高层整体信息（绝对尺度、类别、功能），然后通过多轮生成过程产生详细的零件级几何形状和物理属性。

**设计动机**: 仿照人类认知方式，先理解整体结构再细化局部细节，利用 [[自回归语言模型]] 的序列推理能力处理层次化物理属性

**具体实现**:
- **第一轮（全局推理）**: 推断对象整体信息——绝对尺度（实际物理尺寸）、对象类别、功能描述、零件数量
- **第二轮（局部生成）**: 对每个零件独立生成体素几何 + [[运动学|关节运动参数]]（关节类型、轴向、运动范围）
- 多轮对话格式允许模型在零件间保持几何一致性

#### 模块2: Template-based RLE 几何表示

![Figure 3: 几何表示对比](https://arxiv.org/html/2605.21572v1/x3.png)

![Figure 3b: Template-based RLE 详细说明](https://arxiv.org/html/2605.21572v1/x4.png)

**说明**: 将零件级体素与Template-based RLE编码进行对比，展示该方法如何通过模板层共享减少冗余，维持显式几何结构。

**设计动机**: [[自回归语言模型]] 原生处理文本序列，需要将 3D 几何转为紧凑文本表示；传统坐标索引表示 token 数量爆炸，且丢失几何结构信息

**具体实现**:

**步骤1 — 体素化与分解**:
将仿真就绪资产体素化，按标注的对象结构分解为零件级体素网格

**步骤2 — Z轴切片**:
将每个零件级体素沿 z 轴切片为一系列 2D 二值掩码序列

**步骤3 — 2D [[游程编码|RLE]] 编码**:
对每个切片应用紧凑 2D RLE，将占据区域编码为文本 token

$$
\text{slice}_k = \text{RLE}(\text{voxel}_{\text{part}}[:, :, k])
$$

**步骤4 — 模板层优化（核心创新）**:
多个切片共享同一结构模板，仅存储相对变化或残差差异

$$
\text{encoded}_k = \begin{cases} \text{RLE}(\text{slice}_k) & \text{if } k \text{ is template layer} \\ \Delta(\text{slice}_k, \text{template}) & \text{otherwise} \end{cases}
$$

**优势**: 
- 大幅减少 token 冗余，适配 16,384 最大序列长度
- 无需引入额外特殊 token，保持 VLM 词表完整
- 维持显式几何结构，自回归预测鲁棒性强

#### 模块3: 物理属性生成

**设计动机**: 仅有几何形状不足以支撑物理仿真，需要同步生成材料、可供性、运动学等属性

**具体实现**:
- **绝对尺度**: 预测对象实际物理尺寸（米），以对称百分比误差评估
- **材料属性**: 通过自由落体和水滴物理仿真验证生成材料的合理性
- **[[可供性|Affordance]]**: 预测对象可交互区域和交互方式（与人类常识对齐）
- **[[运动学]] 参数**: 对关节体预测各零件的关节类型（转动/平移）、轴向、运动范围

---

## 关键公式

### 公式1: [[对称百分比误差|绝对尺度评估]]

$$
\text{SPE} = \frac{|s_{\text{pred}} - s_{\text{gt}}|}{(s_{\text{pred}} + s_{\text{gt}}) / 2} \times 100\%
$$

**含义**: 预测绝对尺度与真实物理尺寸之间的对称百分比误差，对过大和过小预测同等惩罚

**符号说明**:
- $s_{\text{pred}}$: 预测的绝对尺度（米）
- $s_{\text{gt}}$: 真实物理尺寸（米）

### 公式2: [[Spearman相关系数|人类对齐验证]]

$$
\rho = 1 - \frac{6 \sum d_i^2}{n(n^2 - 1)}
$$

**含义**: 用 Spearman 秩相关系数衡量自动评估指标与人类判断的一致性

**符号说明**:
- $d_i$: 第 $i$ 个样本在自动指标排名和人类排名之间的差值
- $n$: 样本总数
- PhysX-Bench 中绝对尺度、可供性、材料、描述维度均达到 $\rho = 1.0$，几何维度 $\rho = 0.8$

---

## 关键图表

### Figure 4: PhysXVerse 数据集统计

![Figure 4: PhysXVerse统计](https://arxiv.org/html/2605.21572v1/x5.png)

**说明**: PhysXVerse 的规模和类别分布。相比现有仿真就绪物理数据集，PhysXVerse 具有实质性更广的类别覆盖范围，零件数从 1 到 65 不等。

### Figure 5: PhysX-Bench 评估维度

![Figure 5: PhysX-Bench](https://arxiv.org/html/2605.21572v1/x6.png)

**说明**: PhysX-Bench 六维度评估框架，全面覆盖 3D 结构、外观、基础物理属性和语义理解。

### Figure 6: 定性结果对比

![Figure 6: 定性对比](https://arxiv.org/html/2605.21572v1/x7.png)

**说明**: PhysX-Omni 与竞争方法在多类对象上的生成质量对比，PhysX-Omni 在几何细节和物理合理性上显著优于基线。

### Figure 7: 定量结果与人类对齐验证

![Figure 7: 定量结果](https://arxiv.org/html/2605.21572v1/x8.png)

**说明**: 展示 PhysX-Omni 在各指标上的表现及与人类评判的高度一致性（Pearson $r = 0.992$）。

### Figure 8-9: 额外定性结果

![Figure 8: 额外定性结果](https://arxiv.org/html/2605.21572v1/x9.png)

![Figure 9: 可变形体生成](https://arxiv.org/html/2605.21572v1/x10.png)

**说明**: 复杂场景下的生成结果以及可变形体（布料、软体等）的物理仿真可视化。

### Figure 10: 几何表示消融实验

![Figure 10: 消融实验](https://arxiv.org/html/2605.21572v1/x11.png)

**说明**: Template-based RLE 表示相比基线文本坐标索引方法的消融对比，特别对具有复杂拓扑的关节体提升显著。

### Figure 11: 机器人策略学习应用

![Figure 11: 机器人应用](https://arxiv.org/html/2605.21572v1/x12.png)

**说明**: 生成的仿真就绪资产成功部署于物理仿真器中，支持接触丰富的操作任务策略学习。

### Figure 12: 仿真就绪场景生成

![Figure 12: 场景生成](https://arxiv.org/html/2605.21572v1/x13.png)

**说明**: 集成深度估计和 2D 分割，从单张图像构建完整的物理合理仿真就绪场景。

### Table 1: 常规指标对比（PhysXVerse 测试集）

| 方法 | PSNR ↑ | Chamfer Distance ↓ | F-score ↑ | 绝对尺度误差 ↓ | 运动学得分 ↑ |
|------|--------|-------------------|-----------|-------------|------------|
| PhysX-Anything | — | — | — | 298.19 | 0.4191 |
| MonoArt | — | 7.03 | 85.27 | — | — |
| PhysXGen | 19.41 | — | — | — | — |
| **PhysX-Omni (PhysXVerse)** | **21.52** | **2.95** | **91.28** | **2.79** | **0.9185** |

**关键发现**: PhysX-Omni 在所有维度全面超越基线，绝对尺度误差从 298.19 降至 2.79（降低 ~99%），运动学得分从 0.4191 提升至 0.9185（提升 119%）

### Table 2: PhysX-Bench 综合评估

| 方法 | 运动学 ↑ | 可供性 ↑ | 功能描述 ↑ |
|------|---------|---------|----------|
| PhysX-Anything | 65.99 | 59.96 | 26.89 |
| PhysXGen | 69.17 | 66.07 | 22.24 |
| MonoArt | 68.32 | — | — |
| **PhysX-Omni** | **80.72** | **70.57** | **39.02** |

**关键发现**: PhysX-Omni 在运动学（80.72 vs 69.17）、可供性（70.57 vs 66.07）、功能描述（39.02 vs 26.89）三个维度均领先所有基线

### Table 3: 消融实验（几何表示方法对比）

| 配置 | 关节体 F-score | 说明 |
|------|-------------|------|
| 基线（文本坐标索引） | 低 | 复杂拓扑时 token 爆炸 |
| **Template-based RLE（完整）** | **高** | 模板共享大幅降低冗余 |

**关键发现**: Template-based RLE 对具有复杂拓扑结构的关节体提升最为显著，验证了模板层设计的有效性

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| **PhysXVerse（新）** | 8,700+ 资产，2,900+ 类别 | 首个通用多类物理3D数据集 | 训练+测试 |
| PhysXNet | 部分 | 物理标注3D资产 | 预训练 |
| PhysX-Mobility | 部分 | 关节体数据 | 预训练 |
| **组合** | 42,000+ 资产 | 每对象25张多视角渲染图 | 完整训练 |

### 实现细节

- **Backbone**: Alibaba Qwen2.5-VL-7B-Instruct
- **优化器**: AdamW，峰值学习率 $2 \times 10^{-5}$
- **最大序列长度**: 16,384 tokens
- **训练轮数**: 5 epochs
- **硬件**: 64× NVIDIA A100 GPU，训练约 14 天
- **3D 解码器**: [[TRELLIS]]（体素→高质量网格，无需额外分割）

### 可视化结果

- 刚体：从单张图像准确恢复绝对物理尺寸和材料属性
- 关节体：正确预测关节轴向、类型和运动范围，可直接导入仿真器
- 可变形体：生成的资产在物理仿真中表现出合理的形变行为
- 场景级：集成深度+分割后可构建完整可仿真场景

---

## 批判性思考

### 优点

1. **统一框架**: 首次在单一模型中统一处理刚体、可变形体、关节体三类对象，覆盖仿真场景的主要需求
2. **数据贡献**: PhysXVerse 规模和类别多样性显著超越现有数据集，可作为社区基础资源
3. **全面评估**: PhysX-Bench 六维度基准设计合理，人类对齐验证（$\rho = 1.0$）说明指标可靠
4. **实用性强**: 成功集成到机器人策略学习流水线，验证了从生成到应用的完整链路

### 局限性

1. **复杂结构保真度**: 论文明确承认对高度复杂几何结构的保真度有限（如精细零件拓扑）
2. **外观质量次于物理一致性**: 优化目标偏向物理合理性，可能在视觉质量上不如外观中心方法
3. **训练成本高**: 64× A100 × 14天，对学术界复现门槛较高
4. **可变形体细节**: 可变形体生成的评估维度相比关节体和刚体稍欠完整

### 潜在改进方向

1. 引入扩散模型解码器替代 TRELLIS，提升几何细节保真度
2. 支持材料属性的连续参数化（密度、摩擦系数等），而非离散类别
3. 扩展到场景级物理生成，支持多对象交互关系推理

### 可复现性评估

- [ ] 代码开源（GitHub 已创建仓库但内容待更新）
- [ ] 预训练模型（未提及公开发布计划）
- [x] 训练细节完整（论文中有充分描述）
- [x] 数据集可获取（PhysXVerse 发布于 HuggingFace）

---

## 关联笔记

### 基于

- [[Qwen2.5-VL]]: 骨干视觉语言模型
- [[TRELLIS]]: 体素到高质量网格解码器
- [[游程编码|RLE]]: 核心几何压缩技术

### 对比

- [[PhysXGen]]: 先前物理 3D 生成方法，PSNR 指标上被超越
- [[MonoArt]]: 关节体生成专用方法，Chamfer Distance 和 F-score 被超越
- [[PhysX-Anything]]: 物理属性预测方法，绝对尺度误差被大幅超越

### 方法相关

- [[Template-based RLE]]: 核心几何表示创新
- [[视觉语言模型|VLM]]: 统一多类型资产生成的基础架构
- [[运动学]]: 关节体物理属性的核心
- [[可供性]]: 对象交互属性建模

### 硬件/数据相关

- [[Isaac Lab]]: 目标仿真器之一
- [[MuJoCo]]: 目标仿真器之一
- [[PhysXVerse]]: 本文提出的核心数据集

---

## 速查卡片

> [!summary] PhysX-Omni (2026)
> - **核心**: 统一框架从单张图像生成仿真就绪物理 3D 资产（刚体+可变形+关节体）
> - **方法**: Template-based RLE 几何表示 + Qwen2.5-VL VLM + 粗到细生成
> - **结果**: PSNR 21.52 vs 19.41，Chamfer 2.95 vs 7.03，运动学 0.9185 vs 0.4191（全面SOTA）
> - **代码**: https://github.com/physx-omni/PhysX-Omni

---

*笔记创建时间: 2026-05-25*
