---
title: "Cosmos 3: Omnimodal World Models for Physical AI"
method_name: "Cosmos3"
authors: [NVIDIA]
year: 2026
venue: arXiv
tags: [world-model, omnimodal, physical-ai, video-generation, action-generation, robotics, autonomous-driving]
zotero_collection: 3-Robotics/World Model
image_source: online
arxiv_html: https://arxiv.org/abs/2606.02800
created: 2026-06-04
---

# 论文笔记：Cosmos 3: Omnimodal World Models for Physical AI

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | NVIDIA |
| 日期 | June 2026 |
| 项目主页 | [research.nvidia.com/labs/cosmos-lab/cosmos3](https://research.nvidia.com/labs/cosmos-lab/cosmos3/) |
| 对比基线 | [[Cosmos-Predict2]] |
| 链接 | [arXiv](https://arxiv.org/abs/2606.02800) / [Code](https://github.com/NVIDIA/cosmos) / [HuggingFace](https://huggingface.co/collections/nvidia/cosmos3) |

---

## 一句话总结

> Cosmos 3 提出一种统一的 [[Mixture-of-Transformers]] 架构，在单模型内同时处理语言、图像、视频、音频和动作序列，实现"先理解、再生成、再行动"的 Physical AI 全流程。

---

## 核心贡献

1. **Omnimodal 统一架构**: 首个同时支持文本、图像、视频、音频和动作五种模态输入输出的开源 Physical AI 基础模型，[[Mixture-of-Transformers]] 中 AR 子序列负责理解推理，[[Diffusion Model]] 子序列负责生成，通过联合注意力机制交互。
2. **State-of-the-art 性能全覆盖**: 在 Artificial Analysis 文生图/图生视频、Physics-IQ、PAI-Bench、R-Bench 生成基准以及 VANTAGE-Bench、TAR 推理基准、RoboLab/RoboArena 策略基准上均夺冠开源模型第一。
3. **全面开源生态**: 开源模型权重（Nano 16B + Super 64B）、训练脚本、6 个合成数据集、HUE 评估框架，并与 Diffusers / vLLM-Omni 深度集成，推动 Physical AI 研究民主化。

---

## 问题背景

### 要解决的问题

物理 AI（机器人、自动驾驶、智慧空间）的开发需要同时具备世界理解、世界生成和动作预测三种能力，而现有系统需要拼接多个专用模型（VLM + Video Generator + Policy Model），导致部署复杂、跨任务泛化差。

### 现有方法的局限

- [[Cosmos-Predict2]]、[[Open-Sora]] 等视频生成模型缺乏原生推理和动作输出能力
- [[OpenVLA]]、[[pi0]] 等策略模型不具备高质量视频世界模拟能力
- 多模型拼接在信息流传递中存在语义损失，且各子模块需单独维护

### 本文的动机

将 [[Autoregressive Transformer]] 的序列推理能力与 [[Diffusion Transformer]] 的高质量生成能力在同一骨干中融合——理解阶段的 context 直接流入生成阶段，实现"先思考，再生成"的物理感知闭环。

---

## 方法详解

### 模型架构

Cosmos 3 采用 **[[Mixture-of-Transformers]]（MoT）** 架构，将两类 Transformer 子序列融合在同一骨干中：

- **输入**: 文本 $l$ + 图像/视频帧 $v$ + 音频 $a$ + 动作轨迹 $\mathbf{q}$
- **编码器**: [[Vision Transformer]]（ViT）处理视觉理解，[[VAE]] 处理视觉/音频生成，Domain-aware 向量处理动作
- **核心双流**:
  - **AR（自回归）子序列**: 处理推理与理解，基于 next-token prediction
  - **DM（扩散）子序列**: 处理连续模态生成，基于迭代去噪
- **联合注意力**: AR tokens 与 DM tokens 在 Transformer 层内使用独立参数集，但通过 joint attention 互相交互
- **输出**: 文本 / 图像 / 视频 / 音频 / 动作轨迹（多模态全覆盖）
- **总参数**: Nano 16B（8B Reasoner + 8B Generator）/ Super 64B（32B + 32B）

![Cosmos 3 Architecture](https://research.nvidia.com/labs/cosmos-lab/cosmos3/assets/omni-model/transformer.png)

**说明**: Cosmos 3 的 MoT 双流架构示意图。AR（自回归）子序列和 DM（扩散）子序列共享 Transformer 骨干但使用各自的参数集，通过 joint attention 实现模态间信息流动。

![Cosmos 3 Shared Backbone](https://research.nvidia.com/labs/cosmos-lab/cosmos3/assets/omni-model/shared.png)

**说明**: 展示 Cosmos 3 统一主干如何支持多种 Physical AI 应用模式（VLM、Forward/Inverse Dynamics、Policy 等）。

### 核心模块

#### 模块 1: AR 子序列（Reasoner）

**设计动机**: 利用 [[Autoregressive Transformer]] 的 next-token prediction 实现跨模态理解和推理，用于场景理解、物体关系推断、物理因果分析。

**具体实现**:
- 处理文本、图像、视频等离散/连续 token
- 上下文窗口最大 256K tokens，推理输出最大 4,096 tokens
- 基础为 [[Vision Transformer]]（ViT）的视觉编码
- 输出作为条件信号传入 DM 子序列

#### 模块 2: DM 子序列（Generator）

**设计动机**: 利用 [[Diffusion Transformer]] 的迭代去噪过程生成高保真度的连续信号（视频、音频、动作轨迹）。

**具体实现**:
- [[VAE]] 编码视觉/音频内容进入扩散潜空间
- 动作轨迹使用 Domain-aware 向量表征（机器人关节角、夹爪位置、轨迹点等）
- 通过 joint attention 接收来自 AR 子序列的物理推理上下文
- 支持 [[Classifier-Free Guidance]]（CFG）控制生成多样性

#### 模块 3: 动作表征

**支持的机器体具身维度**:

| 机器人类型 | 动作维度 |
|-----------|----------|
| 通用相机运动 | 9D |
| 自动驾驶车辆 | 9D |
| 以自我为中心运动 | 57D |
| 单臂 Franka Panda + RobotiQ | 10D |
| 双臂 Franka Panda + RobotiQ | 20D |
| Agibot | 29D |
| UR 系列 | 10D |
| Google Robot | 10D |
| WidowX 250 | 10D |
| UMI | 9D |

#### 模块 4: 多任务模式切换

Cosmos 3 通过同一模型权重支持多种任务模式，无需结构修改：

| 输入模态 | 输出模态 | 应用模式 |
|---------|---------|---------|
| Text, Image, Video | Video | 世界模型/视频预测 |
| Text, Video | Text | VLM 推理 |
| Action, Image, Text | Video | 正向动力学模型 |
| Text, Video | Action | 逆向动力学模型 |
| Image, Text | Video + Action | 策略模型 |

---

## 关键公式

### 公式 1: [[Diffusion Model|扩散去噪目标]]

$$
\mathcal{L}_{DM} = \mathbb{E}_{x_0, \epsilon, t}\left[\|\epsilon - \epsilon_\theta(x_t, t, c)\|^2\right]
$$

**含义**: 扩散模型训练目标，在时间步 $t$ 预测噪声 $\epsilon$，条件 $c$ 包含来自 AR 子序列的物理推理上下文。

**符号说明**:
- $x_0$: 原始干净样本（视频帧/音频/动作）
- $x_t$: 时间步 $t$ 的带噪样本
- $\epsilon$: 添加的高斯噪声
- $\epsilon_\theta$: 参数化的去噪网络（DM 子序列）
- $t$: 扩散时间步 $t \sim \mathcal{U}(0, T)$
- $c$: 条件信号（AR 输出的推理上下文 + 文本/图像输入）

### 公式 2: [[Autoregressive Transformer|自回归推理损失]]

$$
\mathcal{L}_{AR} = -\sum_{i=1}^{N} \log P_\theta(x_i \mid x_{<i})
$$

**含义**: AR 子序列的自回归语言建模损失，最大化序列中每个 token 的条件对数似然。

**符号说明**:
- $x_i$: 第 $i$ 个 token（文本/视觉离散 token）
- $x_{<i}$: 位置 $i$ 之前的所有 token
- $P_\theta$: 参数化的 AR Transformer

### 公式 3: [[Mixture-of-Transformers|联合训练目标]]

$$
\mathcal{L}_{total} = \mathcal{L}_{AR} + \lambda \cdot \mathcal{L}_{DM}
$$

**含义**: 联合优化 AR 推理损失和 DM 生成损失，权重 $\lambda$ 平衡两个子任务。

**符号说明**:
- $\mathcal{L}_{AR}$: 自回归推理损失
- $\mathcal{L}_{DM}$: 扩散生成损失
- $\lambda$: 损失权衡权重

### 公式 4: [[Classifier-Free Guidance|无分类器引导采样]]

$$
\tilde{\epsilon}_\theta(x_t, c) = \epsilon_\theta(x_t, \varnothing) + s \cdot (\epsilon_\theta(x_t, c) - \epsilon_\theta(x_t, \varnothing))
$$

**含义**: 推理时用引导尺度 $s$ 增强条件信号的影响力，提升生成质量与文本/物理条件的对齐程度。

**符号说明**:
- $s$: 引导尺度（guidance scale，默认 6.0）
- $\epsilon_\theta(x_t, \varnothing)$: 无条件预测
- $\epsilon_\theta(x_t, c)$: 有条件预测
- $c$: 包含 AR 推理上下文的条件信号

---

## 关键图表

### Figure 1: Capabilities Matrix / 能力全景图

**说明**: Cosmos 3 的五大模态能力矩阵。以 [[Mixture-of-Transformers]] 为核心，单模型同时支持：(1) 文生图/文生视频生成、(2) 图生视频，(3) VLM 视觉推理，(4) 正向动力学（给动作→预测未来视频），(5) 逆向动力学（给视频→推断动作轨迹），(6) 策略模型（给当前帧→生成视频+动作）。

### Figure 2: Model Architecture / MoT 双流架构

![MoT Architecture](https://research.nvidia.com/labs/cosmos-lab/cosmos3/assets/omni-model/transformer.png)

**说明**: AR（自回归）子序列与 DM（扩散）子序列在同一 Transformer 骨干中并行，通过 joint attention 互通信息。AR 子序列产出推理上下文，作为条件向量注入 DM 子序列的去噪过程。

### Figure 3: Physical AI 应用场景示例

![Robot Image Generation](https://research.nvidia.com/labs/cosmos-lab/cosmos3/assets/image-generation/phi182.webp)

**说明**: 固定翼测量无人机在山地森林上空飞行——Cosmos 3 文生图示例，展示物理感知图像生成质量。

![Industrial Robot](https://research.nvidia.com/labs/cosmos-lab/cosmos3/assets/image-generation/phi113.webp)

**说明**: 两台工业机器人协作吊装发动机缸体——展示 Cosmos 3 对工业场景物理关系的精准建模。

![Surgical Robot](https://research.nvidia.com/labs/cosmos-lab/cosmos3/assets/image-generation/phi102.webp)

**说明**: 手术机器人持手术刀场景——展示 Cosmos 3 对高精度操作场景的生成能力。

![AGV Warehouse](https://research.nvidia.com/labs/cosmos-lab/cosmos3/assets/image-generation/phi124.webp)

**说明**: 冷库中的自动叉车——展示 Cosmos 3 在仓储自动化场景中的视觉生成能力。

### Figure 4: 动作推理示例（Vision-Language-Action CoT）

![Action CoT Input](https://research.nvidia.com/labs/cosmos-lab/cosmos3/assets/vision-language/action-cot-input.png)

**说明**: 机器人夹爪轨迹推理输入示例。Cosmos 3 的 AR 子序列解析当前场景，生成 chain-of-thought 推理（"移动手臂到花朵 → 抓取花朵 → 移动到红瓶 → 放入"），再由 DM 子序列生成相应动作轨迹。

### Table 1: 模型参数规格

| 模型 | 总参数 | Reasoner | Generator | 目标场景 |
|------|--------|----------|-----------|---------|
| Cosmos 3 Nano | 16B | 8B | 8B | 工作站推理（RTX PRO 6000）、实时机器人 |
| Cosmos 3 Super | 64B | 32B | 32B | 数据中心（Hopper/Blackwell）、SDG |
| Cosmos 3 Edge | TBD | TBD | TBD | 边缘实时推理（即将发布） |

### Table 2: 输入规格（Nano 模型）

| 模态 | 格式 | 最大限制 |
|------|------|---------|
| 文本 | String | 4,096 tokens |
| 图像 | jpg/png/webp | 720p，多宽高比 |
| 视频 | mp4 | 256p-720p，最多 5 帧输入 |
| 音频 | AAC 立体声 | 0.5 秒，48kHz |
| 动作 | JSON 1D 列表 | 16-400 帧，9D-57D |

### Table 3: 输出规格（Nano 模型）

| 模态 | 格式 | 规格 |
|------|------|------|
| 视频 | MP4 | 5-400 帧（默认 189），可变帧率 |
| 音频 | AAC | 48kHz 立体声，与视频合并 |
| 动作 | JSON | 按具身类型输出 |
| 文本 | String | 可变长度 |

### Table 4: Benchmark 性能

| 评测维度 | 基准 | 排名 |
|---------|------|------|
| 文生图 | Artificial Analysis | 开源第一 |
| 图生视频 | Artificial Analysis | 开源第一 |
| 视频物理理解 | Physics-IQ | 开源第一 |
| Physical AI 综合 | PAI-Bench | 开源第一 |
| 机器人视频生成 | R-Bench | 开源第一 |
| 场景理解推理 | VANTAGE-Bench (32B/8B 档) | 开源第一 |
| 交通异常推理 | TAR (AI City Challenge 2026 Track 3) | 开源第一 |
| 仿真策略评估 | RoboLab | 开源第一 |
| 真机策略评估 | RoboArena (DROID 机器人) | 开源第一 |

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| 合计 | 13 亿数据点 | 393 个子数据集，2024-2026 | 训练 |
| OpenImage | 120 万 | 开放图像理解 | 图像推理 |
| Coyo700M | 1 亿 | 大规模图文对 | 图像生成 |
| YouTube Video | 3.4 亿 | 视频理解 | 视频生成 |
| UMI | 450 万 | 机器人操作 | 动作训练 |
| HiDream-I1 synthetic | 1500 万 | 合成高质量图像 | 图像生成 |
| Qwen3-VL synthetic captions | 11.15 亿 | 合成多模态标注 | 多模态对齐 |
| PhysicalAI-SDG 系列（6个）| - | NVIDIA 合成场景数据 | 物理 AI 评估 |

**数据模态分布**:

| 模态 | 推理样本 | 生成样本 |
|------|---------|---------|
| 文本 | 2200 万 | - |
| 图像 | 1900 万 | 7.67 亿 |
| 视频 | 100 万 | 3.48 亿 |
| 音频 | - | 1.39 亿 |
| 动作 | - | 800 万 |

### 实现细节

- **精度**: BF16（唯一测试精度）
- **量化选项**: FP8（支持）、NVFP4（4-bit，最高 2× 加速）
- **推理引擎**: PyTorch、vLLM-Omni、HuggingFace Diffusers
- **视频采样优化**: EVS（Efficient Video Sampling）—— chunk-level token 剪枝，降低推理 token 数量
- **测试硬件**: NVIDIA GB200、H100
- **许可证**: OpenMDW-1.1（Linux Foundation）
- **发布日期**: 2026 年 5 月 31 日

### 关键推理参数（视频生成）

- `num_inference_steps`: 35
- `guidance_scale`: 6.0
- `flow_shift`: 10.0
- `num_frames`: 189（默认）
- `fps`: 24

### NVIDIA HUE 评估框架

NVIDIA Cosmos Human Evaluation（HUE）是本文随模型同步开源的评估框架，将视频评估从主观偏好打分改为客观事实验证：

- 将视频分解为原子粒度的二元问题（是/否）
- 四个评估维度：**语义对齐**、**物理定律**、**几何推理**、**视觉完整性**
- 覆盖七大 Physical AI 领域
- 开源于 HuggingFace：`nvidia/Cosmos-HumanEval-v1`

---

## 批判性思考

### 优点

1. **统一架构的实际价值**: 单模型覆盖理解-生成-动作三个 Physical AI 核心环节，消除多模型拼接的语义损失和工程复杂度
2. **最大规模开源**: 开源权重、代码、6 个合成数据集、评估框架，极大降低 Physical AI 研究门槛
3. **多具身动作支持**: 原生支持 10+ 种机器人具身类型和自动驾驶，覆盖主流 Physical AI 平台

### 局限性

1. **时序一致性**: 长序列视频中相机运动和物体运动存在抖动和闪烁（temporal inconsistency）
2. **物理接触建模**: 缺少显式物理仿真器，接触动力学近似，无真正的 3D 几何和刚体动力学
3. **长程动作漂移**: 长水平线动作序列中状态不一致（action drift），限制复杂操作任务
4. **幻觉问题**: 复杂场景中可能生成不存在的物体，推理中对象状态和空间关系可能出错
5. **计算成本**: 64B 的 Super 版本需要数据中心级 GPU（Hopper/Blackwell），限制边缘部署

### 潜在改进方向

1. 集成轻量级显式物理仿真器约束扩散生成过程，改善物理接触精度
2. 引入递归状态表征（如 [[V-JEPA2]] 风格）改善长程时序一致性
3. 通过 RL 后训练（如 GRPO）增强长程动作稳定性

### 可复现性评估

- [x] 代码开源（GitHub: nvidia/cosmos）
- [x] 预训练模型（HuggingFace: nvidia/Cosmos3-Nano, nvidia/Cosmos3-Super）
- [x] 数据集可获取（HuggingFace: 6 个 PhysicalAI SDG 数据集）
- [ ] 完整训练细节（训练配方部分披露，但完整超参数未全开放）

---

## 关联笔记

### 基于

- [[Cosmos-Predict2]]: Cosmos 系列前代视频预测模型
- [[Cosmos-Predict]]: Cosmos 系列原始版本
- [[Diffusion Transformer]]: DM 子序列的基础架构
- [[Vision Transformer]]: 视觉编码骨干

### 对比

- [[Open-Sora]]: 开源视频生成基线，Cosmos 3 在生成质量上大幅超越
- [[pi0]]: 机器人策略模型，Cosmos 3-Nano-Policy-DROID 在 RoboArena 上超越
- [[OpenVLA]]: VLA 基线，Cosmos 3 额外增加了世界生成能力
- [[V-JEPA2]]: NVIDIA 另一视频预测模型，侧重自监督特征学习

### 方法相关

- [[Mixture-of-Transformers]]: 核心架构创新
- [[Diffusion Model]]: DM 子序列的生成范式
- [[Autoregressive Transformer]]: AR 子序列的推理范式
- [[VAE]]: 视觉/音频生成的潜空间编码
- [[Classifier-Free Guidance]]: 生成质量控制
- [[Vision Transformer]]: 视觉编码器

### 硬件/数据相关

- [[Franka Research 3]]: 支持的机器人具身之一（Franka Panda）
- [[GR00T-N1.6]]: NVIDIA 另一机器人策略基础模型

---

## 速查卡片

> [!summary] Cosmos 3: Omnimodal World Models for Physical AI
> - **核心**: 首个统一理解-生成-动作的开源 Physical AI Omnimodal 基础模型
> - **方法**: [[Mixture-of-Transformers]]（AR 推理子序列 + DM 生成子序列，joint attention 互通）
> - **结果**: 9 个主要基准开源第一（文生图、图生视频、物理理解、机器人策略等）
> - **代码**: [github.com/NVIDIA/cosmos](https://github.com/NVIDIA/cosmos) | 模型：Nano 16B / Super 64B

---

*笔记创建时间: 2026-06-04*
