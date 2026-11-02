---
title: "Hy-Nomas: Towards Free-form Interactive World Generation"
method_name: "HyNomas"
authors: [Tencent Hy Visual Generation Team]
year: 2026
venue: arXiv
tags: [world-model, video-generation, interactive-generation, diffusion-model, camera-control]
zotero_collection: ""
image_source: online
arxiv_html: https://arxiv.org/html/2610.36729
created: 2026-11-02
---

# 论文笔记：Hy-Nomas: Towards Free-form Interactive World Generation

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Tencent Hy Visual Generation Team |
| 日期 | October 2026 |
| 项目主页 | [hy-nomas.hy.tencent.com](https://hy-nomas.hy.tencent.com/) |
| 对比基线 | [[Genie 3]]、LingBot-World v2、HappyOyster |
| 链接 | [arXiv](https://arxiv.org/abs/2610.36729) / [Code](https://github.com/Tencent-Hunyuan/Hy-Nomas) / [Weights](https://huggingface.co/tencent/Hy-Nomas) |

---

## 一句话总结

> Hy-Nomas 是一个 8B 参数的[[Video World Model|视频世界模型]]，支持自由文本指令 + 连续相机控制的多轮交互式世界生成，通过 Action-Aware 因果注意力与 VLM 提示增强器实现高质量长时序视频生成。

---

## 核心贡献

1. **Free-form Interactive Generation**: 同时支持自由文本描述和连续相机轨迹的多轮交互式视频生成，无需离散动作空间
2. **Action-Aware Causal Attention**: 将结构化描述型 caption 与动作指令型 caption 分开处理——描述语义跨片段持久保留，动作指令仅作用于当前时间窗口
3. **Vision-Language Prompt Enhancer**: 基于微调的 Qwen3.8-27B 模型将用户自然语言请求与历史交互转化为结构化条件，实现 ~100% 格式遵从率
4. **多阶段课程训练 + 扩散蒸馏**: 5 阶段课程训练后通过两阶段[[Diffusion Distillation|扩散蒸馏]]将每片段推理步数从 100 步压缩到 4 步，支持单张 RTX 4090 上运行

---

## 问题背景

### 要解决的问题

交互式世界模型需要根据用户的文本指令和动作输入，持续生成物理一致、语义连贯的开放世界视频，同时支持多轮交互。这比单次视频生成难得多。

### 现有方法的局限

- **Genie 2/3**：仅支持离散动作空间，不支持自由文本指令
- **LingBot-World v2**：相机控制精度较低
- **HappyOyster**：物理合理性和多轮记忆能力不足
- 通用问题：误差累积、动作时序不对齐、长时空间记忆丢失、长时序计算开销大

### 本文的动机

将预训练视频生成模型（[[HunyuanVideo]] 1.5）适配为交互式世界模型，充分利用其强大的视频生成能力，通过片段级因果注意力设计和 VLM 导演模块来解决上述挑战。

---

## 方法详解

### 模型架构

Hy-Nomas 采用基于 [[DCAE]] + [[MMDiT]] 的 **Latent Diffusion** 架构：

- **输入**: 自由文本指令 $l_t$ + 历史帧（latent 缓存）+ 相机轨迹 $c_t$
- **Latent Encoder**: [[DCAE|Causal DCAE-V]]（32× 空间压缩，4× 时间压缩，32 通道）
- **Text Encoder**: 冻结的 Qwen2.5-VL-7B（宽度 3584，最大序列长度 1280）
- **Backbone**: 54 个 dual-stream [[MMDiT]] blocks（隐层宽度 2048，MLP 宽度 8192，16 头 × 128 维）
- **位置编码**: [[RoPE|3D RoPE]]（分配：时间 16 维 + 高度 56 维 + 宽度 56 维）
- **相机模块**: 64 维相机外参，通过 Camera Cross-Attention 注入（宽度 1024）
- **输出**: 速度预测（32 通道），解码后生成下一片段视频
- **总参数**: ~8B

### 核心模块

#### 模块 1: Causal DCAE-V（因果潜变量编码器）

**设计动机**: 需要对可变长度视频片段进行高压缩率的潜变量编码，同时保持时序因果性。

**具体实现**:
- 空间压缩比 32×，时序压缩比 4×，输出 32 通道 latent
- 时序维度：$\lfloor (F-1)/4 \rfloor + 1$，其中 $F$ 为帧数
- 因果结构确保历史帧 latent 可被缓存复用

#### 模块 2: Segment-wise + Action-Aware Causal Attention（片段级 + 动作感知因果注意力）

**设计动机**: 视频生成的 caption 包含两类语义——描述型（场景外观）需要跨片段持久，动作型（运动指令）只应作用于当前片段。

**具体实现**:
- **Segment-wise Causal Attention**: 每个时序片段内部使用全注意力，片段间用[[Causal Attention|因果注意力]]单向连接（只看历史）
- **Action-Aware Attention**: 结构化 caption 中的描述型 token 可被所有片段看到（双向），动作型 token 仅在对应时间窗口内可见（局部掩码）
- 效果：避免动作指令的时序泄露，同时保留场景记忆

#### 模块 3: Camera Cross-Attention（相机交叉注意力）

**设计动机**: 将连续相机轨迹条件注入 [[MMDiT]] 而不破坏原有预训练权重。

**具体实现**:
- 每帧相机外参编码为 64 维向量
- 视觉 token 对时序对齐的相机 token 做[[Attention Mechanism|交叉注意力]]，以残差方式更新

#### 模块 4: Vision-Language Prompt Enhancer（VLM 提示增强器）

**设计动机**: 用户的自然语言描述往往不结构化，需要转化为结构化的 JSON 条件，且需要跟踪多轮对话历史。

**具体实现**:
- 基于 Qwen3.8-27B 进行 [[SFT|监督微调]]
- 输入：系统提示 + 用户请求 + 历史对话 + 前一片段视频帧
- 输出：结构化 JSON 格式 caption（包含场景描述、运动指令、相机参数等字段）
- 格式遵从率 ~100%（vs. 原始 Qwen3.8-27B 的 ~50%）

#### 模块 5: 两阶段扩散蒸馏

**设计动机**: 原始 100 步推理过于缓慢，不满足交互延迟要求。

**具体实现**:
- **阶段 1**: 100 步 → 中间步数，通过[[Consistency Distillation|一致性蒸馏]]
- **阶段 2**: 中间步数 → 4 步，进一步蒸馏
- 单张 RTX 4090 优化：INT8 量化 + 滑动窗口注意力 + CPU 卸载 + 稀疏注意力

---

## 关键公式

### 公式 1: [[DCAE|DCAE-V Latent 形状]]

$$
z \in \mathbb{R}^{32 \times \left(\lfloor (F-1)/4 \rfloor + 1\right) \times \frac{H}{32} \times \frac{W}{32}}
$$

**含义**: 输入 $F$ 帧、分辨率 $H \times W$ 的视频经 DCAE-V 编码后的 latent 张量形状。

**符号说明**:
- $F$: 输入帧数
- $H, W$: 原始视频高度和宽度
- 32: 通道数
- $\lfloor \cdot \rfloor$: 下取整（因果时序压缩）

### 公式 2: [[Flow Matching|流匹配插值]]

$$
Z_t = (1 - t) \cdot Z + t \cdot \varepsilon, \quad t \sim \mathcal{U}(0, 1)
$$

**含义**: 在干净 latent $Z$ 与高斯噪声 $\varepsilon$ 之间做线性插值，速度学习目标为 $v^* = \varepsilon - Z$。

**符号说明**:
- $Z$: 干净 latent
- $\varepsilon \sim \mathcal{N}(0, I)$: 标准高斯噪声
- $t$: 扩散时间步（0=干净，1=纯噪声）
- $v^* = \varepsilon - Z$: 速度预测目标

### 公式 3: Camera Cross-Attention

$$
x_v \leftarrow x_v + \text{Attention}(Q=x_v,\; K=x_\text{cam},\; V=x_\text{cam})
$$

**含义**: 视觉 token 通过对时序对齐的相机 token 做交叉注意力，以残差方式引入相机条件。

**符号说明**:
- $x_v$: 视觉 token 序列
- $x_\text{cam}$: 相机参数编码（64 维外参向量序列）

### 公式 4: FLOPs 均衡训练打包

$$
F_i = \alpha \cdot n_i + \beta \cdot A_i
$$

**含义**: 估计每个样本的训练计算量，其中线性操作和注意力操作分别加权，用于 FLOPs 均衡的样本打包策略。

**符号说明**:
- $n_i$: 样本 $i$ 的 token 数（linear ops 正比于此）
- $A_i$: 样本 $i$ 的未遮盖 query-key 对数（attention ops 正比于此）
- $\alpha, \beta$: 分别覆盖线性操作和注意力操作的权重系数

### 公式 5: Segment Forcing 的上下文时间步采样

$$
t_\text{ctx} \sim \mathcal{U}(0.1, 0.3)
$$

**含义**: 在训练 Segment Forcing 时，用轻微加噪的历史 latent 作为上下文条件——采样 $t_\text{ctx}$ 范围控制噪声程度，避免训练-推理 gap。

**符号说明**:
- $t_\text{ctx}$: 上下文片段的噪声时间步（接近 0 表示接近干净）

### 公式 6: [[VLM|Prompt Enhancer]] 输出条件

$$
\hat{C}_t = \text{PE}\!\left(\text{sys},\; u_t,\; \{u_i, \hat{C}_i\}_{i<t},\; v_{t-1},\; \text{mode}\right)
$$

**含义**: Prompt Enhancer 在第 $t$ 轮生成结构化条件 $\hat{C}_t$，以系统提示、当前用户输入、历史对话、前一片段视频和改写模式为输入。

**符号说明**:
- $u_t$: 第 $t$ 轮用户输入
- $\hat{C}_i$: 第 $i$ 轮生成的结构化条件
- $v_{t-1}$: 前一片段生成的视频帧
- $\text{mode}$: 改写模式（free-form / structured）

---

## 关键图表

### Figure 1: 交互式生成循环（Teaser）

![Figure 1 - Interactive Generation Loop](https://arxiv.org/html/2610.36729v1/teaser.png)

**说明**: Hy-Nomas 的交互式生成循环。用户通过自由文本和相机控制指令输入，[[World Model|世界模型]]持续生成视频片段并缓存历史，实现多轮无缝扩展。

### Figure 2: 数据处理流水线

**说明**: 数据流水线包含过滤、聚类去重、分类标注、自适应采样、相机轨迹估计（ViPE + 视频逆动力学模型），最终生成多轮结构化 caption。

### Figure 3: 结构化视频 Caption 生成流程

**说明**: 展示如何从原始视频自动生成结构化 JSON caption，区分描述型（外观/场景）和动作型（运动/相机）字段。

### Figure 4: 结构化 Caption JSON 格式示例

**说明**: 结构化 caption 的 JSON 格式，包含 `scene_description`（跨片段共享）和 `action_instruction`（片段局部）两个核心字段。

### Figure 5: 模型架构总览

**说明**: Hy-Nomas 整体架构，展示从用户输入经 Prompt Enhancer → 结构化条件 → [[MMDiT]] backbone + [[Causal Attention|因果注意力]] → DCAE 解码 → 视频输出的完整流程。

### Figure 6: Segment-wise vs. Action-Aware 因果注意力掩码

**说明**: 对比两种注意力掩码：Segment-wise causal 保证片段内全注意力、片段间单向；Action-Aware 进一步区分描述型和动作型 token 的可见范围。

### Figure 7: Segment Forcing 与流式反向传播

**说明**: Segment Forcing 训练策略：用轻微加噪的历史片段 latent 模拟推理时的上下文，避免训练与推理不一致；流式反向传播减小内存开销。

### Figure 8: [[LoRA]] 适配工作流

**说明**: 展示针对电影风格和机器人领域的 LoRA-SFT 适配流程，少量目标域数据即可实现高质量域适配。

### Figure 9: Prompt Enhancer 工作流

**说明**: 基于 Qwen3.8-27B 微调的 PE 接收用户输入和历史对话，输出结构化 JSON 条件；SFT 训练使格式遵从率从 ~50% 提升至 ~100%。

### Figure 10: Hy-Nomas-Bench 评测集总览

![Figure 10 - Hy-Nomas-Bench Overview](https://arxiv.org/html/2610.36729v1/Hy-Nomas-Bench_overview_clear.png)

**说明**: 评测集 Hy-Nomas-Bench 包含 4 个维度（多轮文本交互、相机控制、文本+相机混合、空间记忆），共 280 个场景，覆盖交互式世界模型的核心能力。

### Figure (Bench-4): 室内空间记忆测试场景

![Bench-4 Indoor Spatial Memory](https://arxiv.org/html/2610.36729v1/Hy-Nomas-Bench4_indoor.png)

**说明**: Bench-4 空间记忆基准的室内场景示例，测试模型在长时序多轮后是否能正确记忆并返回之前探索的区域。

### Figure (LoRA): LoRA-SFT 效果对比

![LoRA SFT Comparison](https://arxiv.org/html/2610.36729v1/LoRA_SFT_Comparison.png)

**说明**: 电影风格 LoRA 适配前后对比；机器人 LoRA 定性展示双臂任务的域外泛化能力。

### Table 1: 模型架构参数

| 组件 | 参数 |
|------|------|
| 潜变量编码器 | Causal DCAE-V，32× 空间，4× 时间，32 通道 |
| 文本编码器 | Qwen2.5-VL-7B（冻结），宽度 3584，最大长度 1280 |
| DiT Blocks | 54 个 dual-stream MMDiT blocks |
| 隐层宽度 | 2048（MLP 宽度 8192） |
| 注意力头 | 16 头，每头 128 维 |
| RoPE 分配 | (时间 16, 高度 56, 宽度 56) |
| 相机编码 | 64 维外参，Camera Cross-Attention 宽度 1024 |
| 输出 | 32 通道速度预测 |

### Table 2: 训练阶段课程

| 阶段 | 名称 | Context 长度 | 关键设置 |
|------|------|--------------|----------|
| 1 | Latent 空间适配 | — | DCAE-V 对齐 |
| 2 | 分段因果 + 动作感知 + 相机条件持续预训练 | 180K → 360K tokens | 核心能力注入 |
| 3 | 高质量退火 | 360K tokens | 精细质量提升 |
| 4 | Segment Forcing | 360K tokens | 消除训推不一致 |
| 5 | 两阶段扩散蒸馏 | — | 100 步 → 4 步 |

### Table 3: Bench-1 多轮文本交互（人工评测 Win/Tie/Loss %）

| 对比对象 | Prompt Following | Physical Plausibility |
|----------|-----------------|----------------------|
| vs. HappyOyster-Directing | **65.0** / 25.0 / 10.0 | **45.0** / 43.3 / 11.7 |
| vs. LingBot-World v2 | **75.0** / 16.7 / 8.3 | **68.3** / 26.7 / 5.0 |

### Table 4: Bench-2 相机控制（100 条共享轨迹）

| 对比对象 | Win / Tie / Loss |
|----------|-----------------|
| vs. Genie 3 | 32.0 / **54.0** / 14.0 |
| vs. HappyOyster-Wandering | 36.0 / **54.0** / 10.0 |
| vs. LingBot-World v2 | 29.0 / **61.0** / 10.0 |

### Table 5: Bench-3 文本+相机混合控制（vs. LingBot-World v2，60 场景）

| 指标 | Win / Tie / Loss |
|------|-----------------|
| Prompt Following | **75.0** / 16.7 / 8.3 |
| Physical Plausibility | **73.3** / 25.0 / 1.7 |
| Camera Control | 26.7 / **68.3** / 5.0 |

### Table 6: Bench-4 空间记忆（60 场景）

| 对比对象 | Win / Tie / Loss |
|----------|-----------------|
| vs. Genie 3 | 13.3 / 56.7 / **30.0** |
| vs. HappyOyster | 30.0 / 46.7 / **23.3** |
| vs. LingBot-World v2 | **58.3** / 40.0 / 1.7 |

**关键发现**: 空间记忆能力仍弱于 Genie 3，但显著优于 LingBot-World v2。

### Table 7: Prompt Enhancer 对比

| 模型 | Prompt Following | Physical Plausibility | Format Adherence |
|------|-----------------|----------------------|-----------------|
| Qwen3.8-27B (base) | 低 | 低 | ~50% |
| Kimi-K3 | 中 | 中 | ~70% |
| Hy-Nomas-PE | **高** | **高** | **~100%** |

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| Hy-Nomas 内部数据 | 大规模（未公开） | 多轮结构化 caption，含相机轨迹 | 训练 |
| Hy-Nomas-Bench | 280 场景 | 4 个评测维度，人工标注 | 评测 |

### 实现细节

- **Backbone**: [[HunyuanVideo]] 1.5（改造为 [[MMDiT]] + 因果注意力）
- **序列并行**: Ulysses 8-GPU 并行，最长 360K tokens
- **分布式策略**: [[FSDP|FSDP2]] 混合分片
- **梯度优化**: 激活检查点 + FLOPs 均衡打包 + FFA 注意力核
- **优化器**: 分布式 Muon + 异步检查点
- **推理（消费级）**: INT8 量化 + 滑动窗口注意力 + CPU 卸载 + 稀疏注意力，支持单张 RTX 4090
- **解码器**: 基于 PixelUMM 的 latent-to-pixel 解码器

### 可视化结果

- 文本交互：用户指令"增加一辆红色汽车"等自由文本在多轮后仍能正确执行
- 相机控制：连续轨迹下场景几何一致
- 机器人 LoRA：双臂任务不再被误识别为人手
- 电影 LoRA：生成风格明显更接近电影镜头语言

---

## 批判性思考

### 优点

1. **统一多模态控制**: 同时支持自由文本和连续相机控制，优于仅支持其一的竞品
2. **工程完整性**: 从数据（结构化 caption）到模型（Action-Aware 注意力）到训练（课程+蒸馏）到推理（单卡优化）形成完整闭环
3. **开放资源**: 开源权重、代码和评测集，社区可复现
4. **实用蒸馏**: 4 步推理在消费级 GPU 上可运行，实用价值高

### 局限性

1. **空间记忆弱于 Genie 3**: Bench-4 中对 Genie 3 仅 13.3% 胜率，长时空间记忆仍是核心挑战
2. **相机控制回归**: LoRA 适配后 Bench-2 相机控制有所下降（19/44/37），归因于 SFT 数据的轨迹多样性不足
3. **延迟未量化**: 论文未给出端到端推理延迟的具体数字
4. **评测依赖人工**: 所有主要指标为人工对比评测，缺少客观自动化指标

### 潜在改进方向

1. **长时记忆检索**: 论文 Section 9 提出 KV context 卸载 + 历史检索，尚未实现
2. **机器人动作对齐**: 结合 Open X-Embodiment 数据训练动作预测头（DreamZero 路线）
3. **MoE 扩展**: 参考 MAGI-2，通过 Mixture-of-Experts 继续扩展容量

### 可复现性评估

- [x] 代码开源 (https://github.com/Tencent-Hunyuan/Hy-Nomas)
- [x] 预训练模型 (https://huggingface.co/tencent/Hy-Nomas)
- [x] 评测集公开 (https://huggingface.co/datasets/tencent/Hy-Nomas-Bench)
- [ ] 训练数据未公开

---

## 关联笔记

### 基于

- [[HunyuanVideo]]: Backbone 来自 HunyuanVideo 1.5，继承其强视频生成能力
- [[DCAE]]: 使用改进的 Causal DCAE-V 进行高压缩 latent 编码
- [[FlowMatching]]: 训练目标为流匹配速度预测

### 对比

- [[Genie 3]]: 空间记忆基准上的强对比基线，仅支持离散动作
- LingBot-World v2: 文本交互和相机控制维度的对比基线

### 方法相关

- [[MMDiT]]: 核心 Backbone 架构
- [[Causal Attention]]: 片段级和动作感知因果注意力
- [[Diffusion Distillation]]: 将推理步数从 100 压缩到 4
- [[LoRA]]: 域适配微调策略
- [[Video World Model]]: 所属研究领域

### 硬件/数据相关

- RTX 4090: 消费级推理优化目标硬件

---

## 速查卡片

> [!summary] Hy-Nomas: Towards Free-form Interactive World Generation
> - **核心**: 8B 视频世界模型，支持自由文本 + 连续相机控制的多轮交互生成
> - **方法**: Action-Aware 因果注意力 + VLM Prompt Enhancer + 扩散蒸馏（100→4 步）
> - **结果**: Bench-1 文本交互 +75% win（vs LingBot-World），Bench-4 空间记忆弱于 Genie 3
> - **代码**: https://github.com/Tencent-Hunyuan/Hy-Nomas

---

*笔记创建时间: 2026-11-02*
