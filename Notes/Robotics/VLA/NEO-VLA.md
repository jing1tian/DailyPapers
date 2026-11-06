---
title: "From Pixels to Actions: Towards Native Vision-Language-Action Policies"
method_name: "NEO-VLA"
authors: [Weichen Fan, Haiwen Diao, Penghao Wu, Siyuan Hu, Ziqi Huang, Tianqi Liu, Mingze Sun, Ziyi Wang, Zhongang Cai, Hanming Deng, Quan Wang, Lei Yang, Lewei Lu, Fangzhou Hong, Zhaoxi Chen, Dahua Lin, Ziwei Liu]
year: 2026
venue: arXiv
tags: [vla, vision-language-action, encoder-free, flow-matching, robotics-manipulation, native-vla, bimanual-manipulation, embodied-ai]
zotero_collection: Robotics/VLA
image_source: online
arxiv_html: https://arxiv.org/html/2611.06918
created: 2026-11-06
---

# 论文笔记：From Pixels to Actions: Towards Native Vision-Language-Action Policies

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | NTU S-Lab, Shanghai AI Lab, SenseTime Research |
| 日期 | November 2026 |
| 项目主页 | https://github.com/EvolvingLMMs-Lab/NEO |
| 对比基线 | [[SmolVLA]], [[π0.5]], Qwen3-VL-2B, Gemma-3-4B, StarVLA |
| 链接 | [arXiv](https://arxiv.org/abs/2611.06918) / [Code](https://github.com/EvolvingLMMs-Lab/NEO) |

---

## 一句话总结

> NEO-VLA 是一个无编码器（encoder-free）的原生[[Vision-Language-Action Model|VLA]]框架，图像通过轻量卷积块直接 token 化后与语言 token 在整个主干网络中联合演化，动作专家跨层访问视觉表示，在 LIBERO 和 RoboTwin 2.0 上超越基于预训练视觉编码器的对比方法。

---

## 核心贡献

1. **无编码器原生 VLA 架构**：图像通过轻量两阶段卷积 patch 嵌入层（步长 32）直接映射为 token，无 ViT、自编码器或 codebook，视觉信息在整个主干网络中与语言联合计算
2. **跨层流匹配动作专家**：随机初始化的窄动作专家通过层映射关注主干各深度的缓存 KV，实现对指令条件视觉表示的多层利用
3. **受控对比实验**：在相同动作专家和数据条件下对比原生与基于编码器的骨干，定量分析预训练方式对动作迁移的影响
4. **层级激活探测分析**：通过对不同主干层的干预实验，定位动作相关视觉信号在网络中的分布，发现早层（层 0–10）对平移控制最敏感

---

## 问题背景

### 要解决的问题

现有 VLA 模型普遍使用预训练视觉编码器压缩图像后再进行动作推理。操作任务依赖几何、边界、深度和细微状态变化等精细视觉细节，这些信息可能在压缩中丢失。

### 现有方法的局限

- 基于编码器的 VLA（如 OpenVLA、π0）将视觉处理与语言推理分离，编码器输出的固定表示可能缺失动作相关的精细视觉信息
- 喂入视觉编码器中间层特征到 LLM（如 Qwen3-VL）的冻结测试得分仅 64.9，低于端到端原生方案的 90.9
- 简单增大编码器规模或进行全量微调可缩小差距，但无法完全消除

### 本文的动机

若视觉计算在演化过程中始终对动作专家可见，动作相关信号是否会在网络早期就已涌现？这促使作者设计无编码器的"原生"VLA——图像 token 与语言 token 自始至终共同参与主干计算，而非被预先压缩成固定表示。

---

## 方法详解

### 整体架构

NEO-VLA 由三部分组成：**视觉 patch 嵌入**、**视语主干**和**流匹配动作专家**。

#### 视觉 patch 嵌入

轻量两阶段卷积将原始图像映射为每 32×32 像素一个 token，无任何预训练视觉编码器、codebook 或自编码器。每个时间步最多支持三路相机视角，每路最多 88 个视觉 token。

#### 视语主干

- **规模**：2B 参数，40 层
- **注意力**：文本使用因果注意力；视觉 token 在每个图像块内进行双向注意力
- **位置编码**：原生 3D RoPE（时间、高度、宽度）
- **几何信息**：相机内参和外参通过几何 MLP 编码；6 个状态 token 编码双臂和躯体的前后两步状态

#### 流匹配动作专家

- **设计**：随机初始化的窄网络，每层通过映射关注主干对应层的缓存 KV，加上自身的动作 token
- **动作空间**：每臂 26 通道/步（7 关节目标 + 3 平移增量 + 3 旋转增量 + 1 夹爪 + 12 手部关节），躯体 8 通道，时域为 30 步
- **层映射方案**：测试了 first-28（层 0–27）、skip-28（均匀分布 28 层）、end-28（层 12–39）、full-40（40 层专家）四种配置

#### 流匹配训练

训练时对噪声和动作进行插值（τ = 0.999·u，u ~ Beta(1, 1.5)），使用四组动作的加权 MSE（权重 2:1:1:1）；推理时 10 步 Euler 求解，无分类器自由引导。

### 训练流程（三阶段）

| 阶段 | 配置 | 步数 |
|------|------|------|
| Stage 1：主干冻结 | 仅训练动作专家，全局 batch 8192，专家 LR 2×10⁻⁴ | 113,400 步 |
| Stage 2：全量解冻 | 专家 LR 2×10⁻⁴，VLM LR 5×10⁻⁵，交替步辅助 next-token loss（权重 0.1） | 75,000 步 |
| Stage 3：内部数据精调 | 全量参数继续训练 | 75,000 步 |

优化器：AdamW（β₁=0.9，β₂=0.95），2000 步 warmup，cosine 衰减至 10%，bf16，梯度裁剪 1.0。

### 预训练数据

9 个公开数据集家族，共 2,231,583 个 episode，9,927 小时：AgiBot-World（158,100 eps）、Open X-Embodiment、DROID、RH20T、RoboMIND、RoboCOIN、Galaxea Open-World、Bimanual YAM、InternData-A1 仿真（585,953 eps）。数据过滤涵盖关节范围异常值、常量通道、不连续性、动作异常值、损坏 episode 和相机位姿验证。

---

## 关键图表

### Figure 1：原生 vs 编码器架构对比示意图

**说明**：左侧为传统基于编码器的 VLA——图像先经预训练 ViT/编码器压缩为固定表示再送入 LLM；右侧为 NEO-VLA——图像以 patch token 形式与语言 token 在整个主干中联合演化，动作专家可跨层访问视觉计算中间结果。

### 骨干对比实验（LIBERO average，冻结/全量微调）

| 模型 | 冻结 | 全量微调 |
|------|------|---------|
| Qwen3-VL-2B | 79.5 | 90.8 |
| Gemma-3-4B | 84.5 | 91.0 |
| **NEO-VLA-2B** | **90.9** | **95.3** |

**说明**：喂入 Qwen3-VL 中间层特征（代替最终编码器输出）的冻结得分降至 64.9；全量微调可将 Qwen3-VL 提升至接近 NEO-VLA 冻结水平（90.8 vs 90.9），但仍低于 NEO-VLA 微调后的 95.3。

### 层映射消融（LIBERO average）

| 层映射方案 | 冻结 | 全量微调 |
|-----------|------|---------|
| first-28（0–27 层） | **94.9** | 94.6 |
| skip-28（均匀分布） | 90.9 | **95.3** |
| end-28（12–39 层） | 86.7 | 92.5 |
| first-14 | 93.6 | — |
| last-14 | 89.6 | — |

**说明**：冻结时早层最优（first-28 达 94.9），微调后 skip-28 最优（95.3），表明全量微调使中间层和晚层也产生动作相关特征。

### 多模态协训练效果（Table 6）

RoboTwin clean→randomized 成功率：协训练 69.4 vs 仅动作微调 10.7，说明保留理解能力对分布迁移鲁棒性至关重要。多模态理解基准（MMBench、RealWorldQA、AI2D 等）仅下降约 1.5 点。

### 标准基准测试

| 基准 | 成绩 |
|------|------|
| **LIBERO** average | 97.4（Spatial 98.4 / Object 99.4 / Goal 95.6 / Long 96.2） |
| **LIBERO-Plus** total | 89.7（Camera 94.5 / Light 96.2 / Background 96.6 / Layout 82.4 / Robot 72.6） |
| **LIBERO-Pro** average | 47.9（Object 84.3 / Semantic 98.0 / Position 3.0 / Task 6.3） |
| **RoboTwin 2.0 Clean** | 75.78 |
| **RoboTwin 2.0 Clean2Rand** | 69.40 |

**说明**：LIBERO 上接近 EO-1（98.4）和 MMaDA-VLA（98.0）；LIBERO-Plus 位列公开权重模型最高（π0.5 为 84.4）；LIBERO-Pro 中空间配置（Position 3.0）和任务目标（Task 6.3）仍是弱点。

### 真实机器人实验（三任务，各 20 次测试，半数出域场景）

| 模型 | Task 1 | Task 2 | Task 3 |
|------|--------|--------|--------|
| StarVLA (3B) | 1/20 | 2/20 | 0/20 |
| π0.5 (3B) | 10/20 | 14/20 | 13/20 |
| **NEO-VLA (2B)** | **12/20** | **17/20** | **13/20** |

### 推理延迟（RTX 3090，bf16）

- 编译后单次前向（10 步 Euler）：282.5 ms（动作专家 163.4 ms，主干前缀 84.8 ms）
- 完整请求-分块查询：261.9 ms；2 步 Euler 时为 178.5 ms

---

## 实验

### 数据集与评估

- **模拟**：LIBERO（4 个任务套件）、LIBERO-Plus（鲁棒性扰动）、LIBERO-Pro（细粒度控制）、RoboTwin 2.0（双臂操作，2.5K 演示设置）
- **真实机器人**：双臂平台，每任务 50 个演示，20 次测试，半数出域

### 层级激活干预实验

通过将动作专家在不同层段（0–10、10–20、20–30、30–40）的 KV 交换为其他时间步的 KV，测量平移误差和腕部图像敏感性：
- **平移误差峰值**：替换层 0–10 时误差最大，说明早层视觉特征对位置控制最关键
- **腕部图像敏感性峰值**：约层 6–10
- **指令敏感性峰值**：层 0（最早层即携带指令信息）

---

## 批判性思考

### 优点

1. **原生视语融合**：视觉与语言 token 全程联合演化，无信息瓶颈；层级 KV 访问使动作专家按需索取不同抽象度的视觉信号
2. **实证分析充分**：受控对比（相同数据和动作专家条件）、层映射消融、激活干预多维度支撑核心论点
3. **协训练策略**：交替步保留理解能力，显著提升分布外鲁棒性（69.4 vs 10.7），为后续 VLA 训练提供参考

### 局限性

1. **增益归因模糊**：作者坦承无法将性能提升完全归因于无编码器设计，因为预训练数据和模型容量也不同
2. **基准协议差异**：不同论文使用不同模拟器版本和相机设置，作者已警示跨论文直接对比的风险
3. **空间 / 任务扰动弱点**：LIBERO-Pro 中 Position（3.0）和 Task（6.3）成绩极低，说明细粒度空间推理和任务目标理解仍是瓶颈
4. **推理延迟**：RTX 3090 上 10 步 Euler 需 282.5 ms，实时控制需依赖 2 步（178.5 ms）或更激进的加速方案

### 潜在改进方向

1. 结合 [[3DThinkVLA]] 式显式 3D 感知或深度信息，缓解空间配置扰动弱点
2. 探索视觉 token 压缩（如 [[SmolVLA]] 的 token 压缩）在无编码器架构中的适用性，降低主干计算量
3. 进一步研究层映射与任务复杂度的关系，自适应选择不同任务的最优层映射

### 可复现性评估

- [x] 代码开源（GitHub: EvolvingLMMs-Lab/NEO）
- [ ] 预训练权重（待发布）
- [x] 训练细节完整（三阶段配置、超参、数据量均有详细说明）
- [x] 数据集公开（9 个公开数据集家族，来源均有引用）

---

## 关联笔记

### 基于

- [[流匹配|Flow Matching]]: 动作专家的生成框架
- [[Qwen3-VL]]: 受控对比实验中的编码器基线骨干之一
- [[RoPE]]: 位置编码基础（3D RoPE）

### 对比

- [[SmolVLA]]: 轻量 VLA，基于 SmolVLM-2 编码器，对比参照
- [[π0.5]]: 真实机器人实验和 LIBERO-Plus 对比基线
- [[Qwen-RobotManip]]: RoboTwin 2.0 上的强基线（89.58，非开源）

### 方法相关

- [[Vision-Language-Action Model|VLA]]: 整体任务框架
- [[FocusVLA]]: 同类关注精细视觉细节的 VLA 工作
- [[AtlasVLA]]: 双臂操作 VLA，相近任务设定

---

## 速查卡片

> [!summary] NEO-VLA
> - **核心**: 无编码器原生 VLA，图像 patch token 与语言 token 在 2B/40 层主干中联合演化，流匹配动作专家跨层访问视觉 KV
> - **方法**: 轻量卷积 patch 嵌入（步长 32）+ 3D RoPE 视语主干 + 多层映射窄动作专家 + 多模态协训练
> - **结果**: LIBERO 97.4 / LIBERO-Plus 89.7（公开权重最高）/ RoboTwin 2.0 Clean2Rand 69.40；真实机器人超越 π0.5（3B）
> - **代码**: https://github.com/EvolvingLMMs-Lab/NEO

---

*笔记创建时间: 2026-11-06*
