---
title: "SketchColour: Channel Concat Guided DiT-based Sketch-to-Colour Pipeline for 2D Animation"
method_name: "SketchColour"
authors: [Bryan Constantine Sadihin, Michael Hua Wang, Shei Pern Chua, Hang Su]
year: 2025
venue: arXiv
tags: [sketch-colorization, 2d-animation, diffusion-transformer, video-generation, lora, channel-concatenation, reference-based-colorization]
zotero_collection: cs.CV
image_source: online
arxiv_html: https://arxiv.org/html/2507.01586
created: 2026-07-04
---

# 论文笔记：SketchColour: Channel Concat Guided DiT-based Sketch-to-Colour Pipeline for 2D Animation

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Tsinghua University |
| 日期 | July 2025 |
| 项目主页 | [bconstantine.github.io/SketchColour](https://bconstantine.github.io/SketchColour/) |
| 对比基线 | [[AniDoc]], [[LVCD]], [[ToonCrafter]] |
| 链接 | [arXiv](https://arxiv.org/abs/2507.01586) / [Code](https://github.com/bconstantine/SketchColour) |

---

## 一句话总结

> SketchColour 是首个基于 [[Diffusion Transformer|DiT]] 骨干的 2D 动画草图上色流水线，用通道拼接 + [[LoRA]] 微调替代 ControlNet，仅 1000 万参数即超越所有基线方法。

---

## 核心贡献

1. **首个 DiT-based 草图上色管道**: 将 [[CogVideoX]] 图生视频模型扩展到参考帧引导的动画上色任务，是首个在此场景使用 [[Diffusion Transformer]] 骨干的工作
2. **轻量级 [[Channel Concatenation Control|通道拼接控制]]**: 用零初始化通道拼接 + [[LoRA]] 微调（rank=192）替代 ControlNet，仅 1000 万参数，显著降低 GPU 内存需求
3. **SOTA 性能**: 仅使用竞争方法一半训练数据（8 万 vs 16 万），在 SAKUGA 数据集上所有指标均优于 AniDoc、LVCD、ToonCrafter

---

## 问题背景

### 要解决的问题

2D 动画制作中，动画师需逐帧手动上色，这是一个极其耗时的流程。现有自动化方案或依赖基于 U-Net 的视频扩散模型，或需要庞大的 ControlNet 参数量，既存在"颜色溢出（colour bleed）"和"结构扭曲"等质量问题，又有参数量巨大的工程难题。

### 现有方法的局限

- **[[ToonCrafter]]**: 基于 U-Net 的参考帧插值方法，不直接接受草图输入，易产生颜色不一致
- **[[LVCD]]**: 采用 ControlNet 注入草图条件，但 ControlNet 与 U-Net 骨干耦合，造成参数爆炸；当参考帧与目标草图结构差异较大时，会出现"潜空间 latent gap"伪影
- **[[AniDoc]]**: 同样基于 ControlNet，面临相同的参数量和伪影问题；需要更多训练数据
- **逐帧图像上色方法**（GAN-based、扩散-based）：独立处理每帧，导致明显的时序闪烁（flickering）

### 本文的动机

[[Diffusion Transformer|DiT]] 架构（尤其是 [[CogVideoX]]）在视频生成上已展现出强大的时序一致性。通过将草图信息和参考帧信息直接通过通道拼接（channel concatenation）融入 [[Causal VAE]] 潜空间，再用零初始化投影权重 + [[LoRA]] 微调，可以在不引入额外大型控制网络的前提下实现精确的草图条件控制，同时规避 ControlNet 的"latent gap"问题。

---

## 方法详解

### 模型架构

SketchColour 采用 **[[Diffusion Transformer]]（DiT）** 架构，以 [[CogVideoX]]-5B-I2V（Image-to-Video）为骨干：

- **输入**:
  - 彩色首帧 $I_\text{start} \in \mathbb{R}^{1 \times H \times W \times 3}$
  - 草图序列 $S \in \mathbb{R}^{T \times H \times W \times 3}$（经 Anime2Sketch 生成）
- **Backbone**: [[CogVideoX]]-5B-I2V（冻结主干 + [[LoRA]] 微调）
- **核心模块**: [[Channel Concatenation Control|通道拼接控制]] 用于草图条件注入，[[Causal VAE]] 用于时空压缩
- **输出**: 彩色视频 $V \in \mathbb{R}^{T \times H \times W \times 3}$
- **总参数（可训练）**: 约 1000 万（[[LoRA]] + 补丁投影权重）

### 核心模块

#### 模块 1: 通道拼接控制（[[Channel Concatenation Control]]）

**设计动机**: 利用 [[Causal VAE]] 的空间结构保持能力，在潜空间层面直接拼接多模态条件，无需额外控制网络

**具体实现**:
1. 用冻结的 3D [[Causal VAE]] 编码器分别编码草图序列和彩色首帧：
   $$Z_\text{sketch},\ Z_{I_\text{start}},\ Z_{V_\text{GT}} \in \mathbb{R}^{T/4 \times H/4 \times W/4 \times 16}$$
2. 将 $Z_\text{sketch}$ 和 $Z_{I_\text{start}}$ 与去噪潜变量通道拼接，总通道数从 16 扩展为 48
3. 扩展的 patch embedding projection 用**零初始化**填充新通道（继承自 ControlNet 的初始化策略）

这种设计消除了 ControlNet 方法中的"latent gap 伪影"——因为草图和彩色帧在同一潜空间内被统一处理。

#### 模块 2: LoRA 微调策略

**设计动机**: 冻结 [[CogVideoX]] 主干参数，仅用低秩适应微调注意力和前馈层，大幅减少可训练参数

**具体实现**:
- [[LoRA]] rank = 192
- 目标层：注意力 QKVO 投影层 + 前馈层
- 可训练参数：约 1000 万（对比完整 ControlNet 架构的数十亿参数）

#### 模块 3: 草图生成（Anime2Sketch）

输入彩色帧序列，通过 Anime2Sketch 工具逐帧转换为草图，并施加二值化（binarization）处理以防止颜色信息通过强度变化泄漏，确保与对比方法使用同质量草图进行公平比较。

---

## 关键公式

### 公式 1: [[Causal VAE|3D VAE 潜空间压缩]]

$$
Z_{I_\text{start}},\ Z_{V_\text{GT}} \in \mathbb{R}^{T/4 \times H/4 \times W/4 \times 16}
$$

$$
Z_\text{sketch} \in \mathbb{R}^{T/4 \times H/4 \times W/4 \times 16}
$$

**含义**: 冻结的 [[Causal VAE]] 编码器将所有输入（首帧、GT 视频、草图序列）压缩到统一的时空潜空间，时间和空间各下采样 4 倍，通道数 16

**符号说明**:
- $T$: 视频帧数（固定 17 帧）
- $H, W$: 空间分辨率（训练时 720×480）
- $Z_{I_\text{start}}$: 彩色首帧的潜表示
- $Z_{V_\text{GT}}$: 真实彩色视频的潜表示（训练时使用）
- $Z_\text{sketch}$: 草图序列的潜表示

### 公式 2: [[Channel Concatenation Control|通道拼接条件化]]

$$
Z_\text{input} = \text{Concat}\!\left[Z_\text{noisy},\ Z_\text{sketch},\ Z_{I_\text{start}}\right] \in \mathbb{R}^{T/4 \times H/4 \times W/4 \times 48}
$$

**含义**: 将去噪潜变量与草图潜变量、首帧潜变量在通道维度拼接，形成 48 通道的输入送入 [[Diffusion Transformer]] 主干

**符号说明**:
- $Z_\text{noisy}$: 加噪后的目标视频潜变量（16 通道）
- $Z_\text{sketch}$: 草图序列潜变量（16 通道）
- $Z_{I_\text{start}}$: 彩色首帧潜变量（16 通道，时间维度广播到 $T/4$）

---

## 关键图表

### Figure 1: 系统概览

![Figure 1 - SketchColour Overview](https://arxiv.org/html/2507.01586/extracted/6589614/figures/CoverFigureCropped.png)

**说明**: SketchColour 接受彩色首帧和整个场景的草图序列作为输入，基于参考颜色为后续每帧生成彩色输出。展示了系统的端到端工作流程。

### Figure 2: 动画师工作流程

![Figure 2 - Animator Workflow](https://arxiv.org/html/2507.01586/extracted/6589614/figures/AnimatorWorkflow.png)

**说明**: 传统 2D 动画制作流程中，动画师需逐帧手动上色。SketchColour 通过自动化后续帧的上色过程，显著减轻动画师的工作负担——动画师只需提供第一帧的颜色参考。

### Figure 3: 模型架构图

![Figure 3 - Model Diagram](https://arxiv.org/html/2507.01586/extracted/6589614/figures/ModelDiagram.png)

**说明**: SketchColour 完整管道架构。冻结的 [[Causal VAE]] 分别编码彩色首帧和草图序列，通道拼接后送入 [[CogVideoX]]。通过零初始化权重扩展 patch embedding projection 的新通道，并用 [[LoRA]] 微调注意力和前馈层。

### Figure 4: VAE 潜变量可视化

![Figure 4 - VAE Latent Visualization](https://arxiv.org/html/2507.01586/extracted/6589614/figures/ShowingVAELatent.png)

**说明**: 通过 PCA 可视化展示冻结 [[Causal VAE]] 对彩色视频和草图视频的编码结果，验证 [[Causal VAE]] 在编码草图时能保留空间结构信息，为通道拼接提供了可行性依据。

### Figure 5: 定性对比

![Figure 5 - Qualitative Comparison](https://arxiv.org/html/2507.01586/extracted/6589614/figures/QualitativeComparison.png)

**说明**: 与 ToonCrafter、LVCD、AniDoc 的可视化对比。SketchColour 在颜色保真度、运动流畅性和伪影抑制方面均优于基线方法，尤其是颜色溢出（colour bleeding）和对象形变问题显著减少。

### Table 1: 定量对比（SAKUGA 数据集）

**14 帧评估**（与支持 14 帧的基线对比）:

| Method | MSCE ↓ | PSNR ↑ | SSIM ↑ | LPIPS ↓ | FVD ↓ |
|--------|--------|--------|--------|---------|-------|
| LVCD | 7937.33 (±4727.72) | 10.00 (±2.75) | 0.57 (±0.16) | 0.45 (±0.12) | 2738.45 (±1555.28) |
| AniDoc | 2612.61 (±4823.98) | 18.04 (±4.99) | 0.73 (±0.12) | 0.30 (±0.13) | 898.19 (±704.30) |
| **SketchColour** | **2214.18 (±3867.13)** | **20.23 (±5.82)** | **0.79 (±0.12)** | **0.24 (±0.14)** | **829.27 (±723.77)** |

**16 帧评估**（与支持 16 帧的基线对比）:

| Method | MSCE ↓ | PSNR ↑ | SSIM ↑ | LPIPS ↓ | FVD ↓ |
|--------|--------|--------|--------|---------|-------|
| ToonCrafter | 4619.98 (±4086.97) | 13.06 (±3.52) | 0.56 (±0.13) | 0.47 (±0.13) | 1464.59 (±1030.63) |
| **SketchColour** | **2403.40 (±4075.10)** | **19.75 (±5.83)** | **0.78 (±0.12)** | **0.25 (±0.14)** | **860.78 (±750.60)** |

**17 帧评估**（SketchColour 最大支持帧数）:

| Method | MSCE ↓ | PSNR ↑ | SSIM ↑ | LPIPS ↓ | FVD ↓ |
|--------|--------|--------|--------|---------|-------|
| **SketchColour** | 2512.78 (±4190.19) | 19.51 (±5.83) | 0.78 (±0.12) | 0.25 (±0.14) | 918.70 (±771.13) |

**关键发现**: SketchColour 在 14 帧和 16 帧评估中全面超越所有基线。随帧数增加（14→16→17），MSCE 略有上升（2214→2403→2513），说明距离参考帧越远的帧越难保持颜色一致性，这与训练固定 17 帧但首帧为参考的设定相符。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| SAKUGA | 80K 训练 / 1K 测试 | 专业动画片段数据集 | 训练 + 测试 |

注：SAKUGA 数据集经过筛选，仅保留高质量动画片段。SketchColour 仅使用 80K 样本，相比 AniDoc 等使用更多数据的方法减少约 50%。

### 实现细节

- **骨干网络**: [[CogVideoX]]-5B-I2V（主干冻结）
- **优化器**: AdamW，学习率 1e-4
- **Batch Size**: 2
- **训练步数**: 40K steps
- **硬件**: 2 × NVIDIA A40 GPU
- **训练时长**: 约 4 天
- **分辨率**: 720 × 480（固定）
- **视频长度**: 17 帧（固定）
- **数据预处理**: Fill-and-crop

### 可视化结果

定性对比表明 SketchColour 在以下方面优势明显：
1. **颜色溢出抑制**: LVCD 和 AniDoc 均存在明显的颜色溢出到相邻区域问题，SketchColour 边界更清晰
2. **对象形状保真**: 与 ToonCrafter 相比，SketchColour 忠实于草图轮廓，不会产生形状扭曲
3. **时序一致性**: [[Diffusion Transformer]] 的全局注意力机制保证了帧间颜色一致性

---

## 批判性思考

### 优点

1. **参数效率极高**: 1000 万可训练参数 vs. ControlNet 的数十亿参数，同等甚至更优的性能
2. **数据效率**: 仅需竞争方法一半的训练数据即可达到 SOTA 性能
3. **清晰的工程决策**: 零初始化通道扩展、冻结 VAE、LoRA 等每个设计选择都有明确动机
4. **实用性强**: 动画师只需提供一帧彩色参考即可自动化整个场景的上色

### 局限性

1. **帧数限制**: 受 [[CogVideoX]] 预训练设定限制，固定输出 17 帧，无法处理更长片段
2. **长序列性能退化**: 距参考帧越远，颜色保真度越低（MSCE 从 14 帧的 2214 升至 17 帧的 2513）
3. **无消融实验**: 论文未提供系统性消融研究，难以评估各设计决策（LoRA rank、零初始化、不同 VAE 等）的独立贡献
4. **单参考帧依赖**: 仅使用首帧作为颜色参考，对于颜色变化丰富的场景（如光线变化、多套服装）可能不足

### 潜在改进方向

1. 支持多参考帧以处理复杂颜色变化场景
2. 滑动窗口推理以支持更长动画序列
3. 添加消融实验验证各组件的贡献
4. 探索更大 LoRA rank 或 QA-LoRA 等变体对质量的影响

### 可复现性评估

- [x] 代码开源（GitHub）
- [ ] 预训练模型（未明确提及）
- [x] 训练细节完整（优化器、学习率、步数、硬件均有说明）
- [x] 数据集可获取（SAKUGA 数据集）

---

## 关联笔记

### 基于

- [[CogVideoX]]: 骨干模型，提供 DiT 架构和 3D Causal VAE
- [[LoRA]]: 参数高效微调方法
- [[ControlNet]]: 本文替代的控制方法（对比参考）
- [[Causal VAE]]: 时空潜空间压缩

### 对比

- [[AniDoc]]: 主要竞争对手，同样接受草图输入，基于 ControlNet
- [[LVCD]]: 早期草图引导视频上色方法，基于 ControlNet
- [[ToonCrafter]]: 基于 U-Net 的动画插值方法

### 方法相关

- [[Diffusion Transformer]]: 核心架构
- [[Channel Concatenation Control]]: 本文提出的轻量级条件化方法
- [[Video Diffusion Model]]: 相关技术路线
- [[FVD]]: 时序质量评估指标
- [[SSIM]]: 结构相似性指标
- [[LPIPS]]: 感知相似性指标
- [[PSNR]]: 峰值信噪比指标

### 硬件/数据相关

- SAKUGA 数据集: 专业动画训练数据
- NVIDIA A40: 训练硬件

---

## 速查卡片

> [!summary] SketchColour
> - **核心**: 首个基于 DiT 的草图引导 2D 动画上色流水线
> - **方法**: CogVideoX-5B-I2V + 通道拼接控制（零初始化）+ LoRA（rank=192）微调，仅 1000 万可训练参数
> - **结果**: SAKUGA 数据集上 PSNR 20.23，SSIM 0.79，FVD 829.27，全面超越 AniDoc/LVCD/ToonCrafter
> - **代码**: [GitHub](https://github.com/bconstantine/SketchColour)

---

*笔记创建时间: 2026-07-04*
