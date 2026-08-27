---
title: "Latent Action as Intention Enables Efficient Future Imagination for World Action Models"
method_name: "LAWA"
authors: [Xiang Li, Yupeng Zheng, Songen Gu, Huailiang Ma, Feng Yu, Xian Nie, Shanshuai Yuan, Yujie Zang, Weize Li, Shuai Tian, Moyang Liu, Ya-Qin Zhang, Wenchao Ding]
year: 2026
venue: arXiv
tags: [world-action-model, latent-action, robot-manipulation, flow-matching, egocentric-pretraining]
zotero_collection: 3-Robotics/World Model
image_source: online
arxiv_html: https://arxiv.org/html/2608.24882v1
created: 2026-08-27
---

# 论文笔记：LAWA — Latent Action as Intention

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | 未注明（推测为中国顶级研究机构） |
| 日期 | August 2026 |
| 项目主页 | — |
| 对比基线 | [[Fast-WAM]], [[Joint-WAM]] |
| 链接 | [arXiv](https://arxiv.org/abs/2608.24882) / Code（即将发布） |

---

## 一句话总结

> LAWA 用紧凑的**离散隐动作序列**替代完整的未来视频预测，在保留 Joint-WAM 性能和泛化性的同时，将推理延迟降低 42.9%。

---

## 核心贡献

1. **隐动作即意图（Latent Action as Intention）**: 将未来意图压缩为离散隐动作 token 序列，推理时无需生成未来视频帧，同时保留 [[世界动作模型（WAM）]] 的世界知识优势。
2. **无动作标注的自中心预训练**: 在大规模机器人+第一视角视频上进行动作无关预训练，通过运动速度对齐与加权重采样，让隐动作 tokenizer 学到通用的时序变化表示。
3. **结构化多模态注意力掩码**: 三路分支（未来视频、隐动作、可执行动作）通过因果掩码控制信息流，推理时直接丢弃未来视频分支，零额外改动即可提速。

---

## 问题背景

### 要解决的问题

[[世界动作模型（WAM）]] 存在两种范式之间的**性能-效率权衡**：
- **Joint-WAM**：同时预测未来视频和动作，泛化性强但推理慢（593ms/chunk）
- **Fast-WAM**：删除未来想象模块，速度快（196ms）但泛化性明显下降

### 现有方法的局限

- Joint-WAM 在测试时必须生成完整的未来视频帧，计算代价高
- Fast-WAM 放弃了世界知识驱动动作生成，在分布外场景（如摄像头视角扰动）表现差 14.4 个百分点
- 现有 [[Latent Action]] 方法多依赖动作标注或仅用于单一场景

### 本文的动机

未来视频的**核心价值不在于像素精度**，而在于它承载的"接下来要做什么"的**意图信息**。只要能将这种意图以紧凑形式（离散 token）传递给动作专家，就不需要生成完整视频。

---

## 方法详解

### 模型架构

LAWA 采用**三分支联合去噪**架构，基于 [[Flow Matching]] 框架：

- **输入**: 语言指令 + 当前观测帧序列 $o_{t-H:t}$
- **Backbone（隐动作 tokenizer）**: 基于 [[DINOv2]] 的时空编码器 + [[Vector Quantization|VQ 码本量化]]
- **三路分支**:
  1. 未来视频分支（仅训练时激活）
  2. **隐动作分支**：预测离散隐意图 token $\hat{l}_{t:t+T}$
  3. 动作分支：以隐动作为条件去噪，输出 [[Action Chunking|动作块]] $a_{t:t+k}$
- **推理时**: 丢弃未来视频分支，保留隐动作 + 动作联合去噪
- **注意力掩码**: 当前观测 token 不可访问未来；隐动作 token 条件化动作但不暴露未来像素

### 核心模块

#### 模块1：隐动作 Tokenizer（ViPRA-based）

**设计动机**: 利用 [[DINOv2]] 强大的视觉表征能力 + [[SAM-2|SAM2]] 自动生成操作区域掩码，使 tokenizer 专注于物体交互区域而非背景

**具体实现**:
- **编码器**: DINOv2 + 空间-时序注意力，对相邻帧对 $(o_t, o_{t+\Delta t})$ 建模时序差异
- **量化**: 将压缩 token $h_{k,p}$ 分配到最近码本项（公式 1）
- **解码器**: 因果前向解码器，重建下一帧（L1 + LPIPS 监督）
- **辅助目标**: [[SAM-2]] 自动生成分割掩码作为 mask-prediction 辅助损失，BCE + Dice + IoU

#### 模块2：无动作自中心预训练（Action-Free Egocentric Pre-training）

**设计动机**: 机器人数据稀缺，利用大量第一视角视频（Ego4D 等）扩充训练数据，同时通过**运动速度对齐**和**加权重采样**（保持约 20% 机器人数据）维持领域锚定

**具体实现**:
- 无需动作标注，纯粹基于视频帧预测
- 视频数据从 10% 扩展到 100%，LAWA 的 few-shot 收益为 +4.0 pt（Fast-WAM 仅 +1.1 pt）

#### 模块3：结构化多模态联合注意力（Multi-Modal Joint Attention）

**设计动机**: 在训练时让三路分支共享 Transformer 上下文，但通过因果掩码保证推理时的可分离性

**具体实现**:
- 结构化掩码矩阵控制信息流：视频 branch → 隐动作 branch → 动作 branch（单向）
- 推理时完全丢弃视频分支，无任何额外计算
- 隐动作 token 携带的意图信号仍然条件化动作去噪

---

## 关键公式

### 公式 1：[[Vector Quantization|码本分配]]

$$
j_{k,p}^* = \arg\min_j \| h_{k,p} - e_j \|_2^2, \quad l_{k,p} = e_{j_{k,p}^*}
$$

**含义**: 将压缩特征 $h_{k,p}$ 映射到码本 $\mathcal{C} = \{e_j\}_{j=1}^K$ 中最近的离散条目

**符号说明**:
- $h_{k,p}$: 第 $k$ 步第 $p$ 个空间位置的压缩 token
- $e_j$: 码本中第 $j$ 个条目（可学习向量）
- $l_{k,p}$: 量化后的隐动作 token

### 公式 2：[[Flow Matching|隐动作 Tokenizer 训练损失]]

$$
\mathcal{L}_{\text{tok}} = \mathcal{L}_1 + \lambda_{\text{perc}} \mathcal{L}_{\text{perc}} + \lambda_{\text{mask}} \mathcal{L}_{\text{mask}}
$$

**含义**: 组合重建损失，同时监督像素精度、感知质量和操作区域分割

**符号说明**:
- $\mathcal{L}_1$: L1 像素重建损失（下一帧预测）
- $\mathcal{L}_{\text{perc}}$: LPIPS 感知损失（特征空间相似性）
- $\mathcal{L}_{\text{mask}}$: 掩码预测损失（BCE + Dice + IoU，掩码由 [[SAM-2]] 生成）
- $\lambda_{\text{perc}}, \lambda_{\text{mask}}$: 权重系数

### 公式 3：[[Flow Matching|每模态 Flow Matching 损失]]

$$
\mathcal{L}_m = \mathbb{E} \left[ w_m(\tau_m) \| \hat{v}_\theta^m - v^m \|_2^2 \right]
$$

**含义**: 对每个模态 $m$（视频 / 隐动作 / 动作）分别用流匹配训练速度场预测器

**符号说明**:
- $\tau_m \sim \mathcal{U}(0,1)$: 模态独立的流时间采样
- $\epsilon^m \sim \mathcal{N}(0, I)$: 独立高斯噪声
- $y_{\tau_m}^m = (1-\tau_m)y^m + \tau_m \epsilon^m$: 噪声插值目标
- $v^m = \epsilon^m - y^m$: 目标速度场
- $\hat{v}_\theta^m$: 网络预测速度
- $w_m(\tau_m)$: 模态特定权重函数

### 公式 4：[[世界动作模型（WAM）|LAWA 联合训练目标]]

$$
\mathcal{L}_{\text{LAWA}} = \lambda_{\text{vid}} \mathcal{L}_{\text{vid}} + \lambda_{\text{lat}} \mathcal{L}_{\text{lat}} + \lambda_{\text{act}} \mathcal{L}_{\text{act}}
$$

**含义**: 三路分支损失的加权和，视频项保留世界建模信号，隐动作项匹配 tokenizer 给出的码本目标，动作项学习可执行控制

**符号说明**:
- $\lambda_{\text{vid}}, \lambda_{\text{lat}}, \lambda_{\text{act}}$: 各分支损失权重
- $\mathcal{L}_{\text{vid}}$: 未来视频重建流匹配损失
- $\mathcal{L}_{\text{lat}}$: 隐动作流匹配损失（目标为冻结 tokenizer 给出的 VQ 码本条目）
- $\mathcal{L}_{\text{act}}$: 可执行动作流匹配损失

---

## 关键图表

### Figure 1：WAM 范式概念对比

![Figure 1](https://arxiv.org/html/2608.24882v1/teaser.png)

**说明**: 三种 WAM 范式的概念图与定量对比。Fast-WAM 最快但泛化差；Joint-WAM 性能最强但推理慢（593ms）；LAWA 以 338.5ms 的延迟匹配 Joint-WAM 性能，同时泛化优于 Fast-WAM 14.4 pt。

### Figure 2：LAWA 整体架构

![Figure 2](https://arxiv.org/html/2608.24882v1/main.png)

**说明**: LAWA 的三分支架构总览。结构化掩码控制三路（未来视频、[[Latent Action|隐动作]]、可执行动作）信息流。推理时丢弃未来视频分支，仅保留隐动作 + 动作联合去噪，实现"有意图的高效生成"。

### Figure 3：Tokenizer 自中心预训练流程

![Figure 3](https://arxiv.org/html/2608.24882v1/lam.png)

**说明**: 无动作标注的 tokenizer 预训练 pipeline。输入相邻帧对，经 [[DINOv2]] 编码 + VQ 量化后，由前向解码器重建下一帧，[[SAM-2]] 自动生成操作区域掩码辅助监督。

### Figure 4：动作-视觉注意力图对比

![Figure 4](https://arxiv.org/html/2608.24882v1/attention.png)

**说明**: 在同一轨迹片段上，LAWA 的动作 token 到视觉 token 的注意力集中在被操作物体和交互区域，而 Fast-WAM 注意力分散于背景，揭示了 LAWA 的"意图引导"机制。

### Figure 5：自中心预训练数据量 vs. 收益

**说明**: 随视频数据量从 10% 增至 100%，LAWA 在 few-shot 场景的收益（+4.0 pt）显著高于 Fast-WAM（+1.1 pt）和 Joint-WAM（+0.5 pt），显示 LAWA 对无动作视频数据的更强利用能力。

### Figure 6：真实世界任务设置

![Figure 6](https://arxiv.org/html/2608.24882v1/setup.png)

**说明**: xArm7 机器人上的四项任务：Gear（精细装配）、Battery（精细装配）、Block（长视野搬运）、Laboratory（长视野多步操作）。蓝色箭头为初始位置，红色为目标位置。

### Table 1：RoboCasa 主结果

| 方法 | 范式 | Few-shot SR (%) | Full SR (%) |
|------|------|-----------------|-------------|
| GR00T N1.6 | VLA | — | 47.6 |
| StarVLA | VLA | — | 48.8 |
| StarVLA-α | VLA | — | 53.8 |
| TwinBrainVLA | VLA | — | 54.6 |
| ABot-M0 | VLA | — | 58.3 |
| RLDX-1 | VLA | — | 58.7 |
| JoyAI-RA | VLA | — | 63.2 |
| DIAL | VLA | 58.3 | 70.2 |
| Being-H0.7 | WAM | — | 49.2 |
| DiT4DiT | WAM | — | 50.8 |
| LDA-1B | WAM | — | 55.4 |
| Fast-WAM† | WAM | 56.0 | 76.3 |
| Joint-WAM† | WAM | 64.1 | 78.8 |
| **LAWA（ours）** | **WAM** | **65.6** | **80.8** |

**说明**: † 表示作者自己实现的对照组。LAWA 在 few-shot 超越 Fast-WAM 9.6 pt，在 full data 超越 Fast-WAM 4.5 pt，与 Joint-WAM 持平或略优。

### Table 2：LIBERO-Plus 零样本迁移

| 方法 | 摄像头 | 机器人 | 语言 | 光照 | 背景 | 噪声 | 布局 | Total |
|------|--------|-------|------|------|------|------|------|-------|
| OpenVLA | 0.8 | 3.5 | 23.0 | 8.1 | 34.8 | 15.2 | 28.5 | 15.6 |
| OpenVLA-OFT | 56.4 | 31.9 | 79.5 | 88.7 | 93.3 | 75.8 | 74.2 | 69.6 |
| UniVLA | 1.8 | 46.2 | 69.6 | 69.0 | 81.0 | 21.2 | 31.9 | 42.9 |
| WorldVLA | 0.1 | 27.9 | 41.6 | 43.7 | 17.1 | 10.9 | 38.0 | 25.0 |
| π₀ | 13.8 | 6.0 | 58.8 | 85.0 | 81.4 | 79.0 | 68.9 | 53.6 |
| π₀-FAST | 65.1 | 21.6 | 61.0 | 73.2 | 73.2 | 74.4 | 68.8 | 61.6 |
| Fast-WAM† | 24.9 | 50.7 | 76.9 | 89.2 | 62.3 | 58.0 | 67.7 | 60.0 |
| Joint-WAM† | 47.2 | 65.3 | 91.8 | 94.8 | 57.6 | 61.0 | 78.9 | 70.4 |
| **LAWA（ours）** | **69.2** | **66.7** | 62.8 | **96.2** | 64.5 | **85.5** | **78.6** | **74.4** |

**关键发现**: LAWA 在摄像头视角扰动（+44.3 pt vs. Fast-WAM）和传感器噪声（+27.5 pt）上提升最大，印证隐动作意图对结构性干扰的鲁棒性；同时在 Total 上超越 Joint-WAM 4.0 pt。

### Table 3：推理延迟对比

| 方法 | 延迟 (ms/chunk) |
|------|----------------|
| Fast-WAM | 196.5 |
| **LAWA** | **338.5** |
| Joint-WAM | 593.1 |

**说明**: 测试硬件 NVIDIA A800 GPU。LAWA 比 Joint-WAM 快 42.9%，同时保持相近性能。

### Table 4：隐动作扰动验证

| 输入 | 无扰动 | 高斯噪声 (σ=1.0) | 时序随机 |
|------|--------|-----------------|---------|
| SR (%) | 80.8 | 52.2 | 56.4 |

**关键发现**: 扰动隐动作序列导致成功率骤降 24-28 pt，证明动作专家真实依赖预测出的隐动作结构，而非仅将其视为冗余。

### Table 5：消融实验

| LA | EP | AL | Few-shot SR (%) | Full SR (%) |
|----|----|----|-----------------|-------------|
| — | — | — | 54.5 | 74.6 |
| ✓ | — | — | 59.7 | 76.3 |
| ✓ | ✓ | — | 64.8 | 79.3 |
| ✓ | ✓ | Flow | 63.5 | 78.6 |
| ✓ | ✓ | Mask | **65.6** | **80.8** |

**说明**: LA=Latent Action，EP=Egocentric Pre-training，AL=Auxiliary Loss。Mask 辅助损失优于 Flow 辅助损失（+2.1 pt few-shot），自中心预训练在 few-shot 场景带来 +5.1 pt 的最大增益。

### Table 6：真实世界实验（xArm7）

| 方法 | 数据量 | Gear | Battery | Block | Laboratory | Average |
|------|--------|------|---------|-------|------------|---------|
| Fast-WAM | 25% | 30.0 | 5.0 | 0.0 | 0.0 | 8.8 |
| LAWA | 25% | 55.0 | 30.0 | 45.0 | 30.0 | **40.0** |
| Fast-WAM | 50% | 50.0 | 20.0 | 10.0 | 0.0 | 20.0 |
| LAWA | 50% | 70.0 | 50.0 | 60.0 | 45.0 | **56.3** |
| Fast-WAM | 100% | 65.0 | 40.0 | 25.0 | 5.0 | 33.8 |
| LAWA | 100% | 85.0 | 65.0 | 70.0 | 50.0 | **67.5** |

**关键发现**: LAWA 用 25% 数据（40.0%）超越 Fast-WAM 用 100% 数据（33.8%），在数据效率上展现出显著优势，尤其在长视野任务（Block 45.0% vs. 0.0%，Laboratory 30.0% vs. 0.0%）。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| [[RoboCasa]] | 24,000 轨迹（full）/ 2,400（few-shot）| 24 个桌面操作任务 | 主要训练+测试 |
| [[LIBERO-Plus]] | 130 个任务 | 6 类系统性扰动（摄像头/机器人/语言/光照/背景/噪声） | 零样本迁移测试 |
| 自中心预训练视频 | 大规模（比例可变） | 机器人+Ego4D 第一视角，无动作标注 | Tokenizer 预训练 |

### 实现细节

- **Tokenizer Backbone**: DINOv2 + ViPRA 风格时空注意力
- **量化**: 可学习码本，噪声替换量化（Noise-Substitution Quantization）
- **策略训练**: 冻结 tokenizer，联合训练三路 Flow Matching 分支
- **硬件**: NVIDIA A800 GPU（延迟测试）
- **预训练数据混合**: 机器人视频约 20%，自中心视频约 80%（运动速度加权重采样）

---

## 批判性思考

### 优点

1. **优雅的效率-性能权衡**: 通过隐动作这个"代理意图"，LAWA 在 Joint-WAM 和 Fast-WAM 之间找到了一条新路，无需牺牲性能即可实现 42.9% 提速
2. **泛化性显著改善**: LIBERO-Plus 零样本测试中全面超越 Fast-WAM（+14.4 pt），尤其在结构性扰动（摄像头/噪声）场景收益巨大
3. **无动作标注预训练**: 无需昂贵的动作标注即可从大量第一视角视频中学习，大幅降低数据获取壁垒
4. **数据效率高**: 真实实验中 25% 数据的 LAWA 超越 100% 数据的 Fast-WAM

### 局限性

1. **仍慢于 Fast-WAM**: 338.5ms vs. 196.5ms，在对实时性要求极高的场景（如动态抓取）仍有差距
2. **无代码发布**: 复现难度较高，且论文未给出详细超参数设置
3. **ViPRA 依赖**: tokenizer 依赖 ViPRA 架构，但相关细节未在论文中充分展开
4. **单机器人平台**: 真实实验仅在 xArm7 上验证，跨平台泛化能力未知

### 潜在改进方向

1. 探索更激进的隐动作压缩（更小码本）以进一步加速
2. 将 tokenizer 预训练扩展到更多样的第三视角视频
3. 结合在线 RL 微调进一步提升长视野任务表现

### 可复现性评估

- [ ] 代码开源（计划发布）
- [ ] 预训练模型（计划发布）
- [ ] 训练细节完整（基本充分）
- [x] 数据集可获取（RoboCasa 和 LIBERO-Plus 均公开）

---

## 关联笔记

### 基于

- [[Fast-WAM]]: 无未来想象的 WAM 基线，LAWA 的主要效率参考
- [[Hierarchical Latent Action]]: 分层隐动作表示，与本文方法高度相关
- [[Flow Matching]]: 训练框架，用于三路分支的去噪预测

### 对比

- [[Fast-WAM]]: 速度优先但泛化差，LAWA 在保速的同时恢复泛化
- [[Diffusion Policy]]: 基于扩散的动作生成 baseline

### 方法相关

- [[Latent Action]]: 核心表示，离散隐动作 token 序列
- [[Vector Quantization]]: 码本量化机制
- [[DINOv2]]: tokenizer 编码器主干
- [[SAM-2]]: 掩码辅助损失的掩码来源
- [[世界动作模型（WAM）]]: 所属大类范式

### 数据集相关

- [[RoboCasa]]: 主要训练与测试 benchmark
- [[LIBERO-Plus]]: 零样本迁移测试 benchmark

---

## 速查卡片

> [!summary] LAWA — Latent Action as Intention
> - **核心**: 用离散隐动作序列代替完整未来视频，作为 WAM 的"轻量意图"
> - **方法**: 三路联合 Flow Matching（视频/隐动作/动作）+ 无动作自中心预训练
> - **结果**: RoboCasa 80.8% SR，LIBERO-Plus 74.4% zero-shot，延迟 338.5ms（-42.9% vs. Joint-WAM）
> - **代码**: 即将发布

---

*笔记创建时间: 2026-08-27*
