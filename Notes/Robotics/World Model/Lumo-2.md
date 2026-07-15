---
title: "Towards Predictive, Aligned, and Scalable Robot Learning"
method_name: "Lumo-2"
authors: [Peijun Tang, Shangjin Xie, Baifu Huang, Binyan Sun, Haotian Yang, Kuncheng Luo, Weiqi Jin, Shilin Fang, Jianan Wang]
year: 2026
venue: arXiv
tags: [world-action-model, latent-world-dynamics, robotic-manipulation, imitation-learning, multi-stage-training, block-wise-autoregression, vision-language-action]
zotero_collection: 3-Robotics/World Model
image_source: online
arxiv_html: https://arxiv.org/html/2607.11270
created: 2026-07-15
---

# 论文笔记：Towards Predictive, Aligned, and Scalable Robot Learning (Lumo-2)

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Astribot Team |
| 日期 | July 2026 |
| 项目主页 | [www.astribot.com/research/Lumo2](https://www.astribot.com/research/Lumo2) |
| 对比基线 | [[Fast-WAM]]、[[pi0.5]] |
| 链接 | [arXiv](https://arxiv.org/abs/2607.11270) / Code: N/A |

---

## 一句话总结

> Lumo-2 通过三阶段渐进对齐策略，将 action 表征与[[Latent World Dynamics|潜在世界动态]]、视觉、语言依次对齐，在 22 个真实机械臂操作任务上全面超越 [[pi0.5]] 和 [[Fast-WAM]] 基线。

---

## 核心贡献

1. **三阶段渐进对齐训练范式**: 依次将 action 表征对齐到潜在世界动态（Stage 1）、视觉-语言语义（Stage 2）、大规模多模态数据（Stage 3），系统性解决 action modality 对齐不足的问题
2. **[[Latent World Dynamics|潜在世界动态]] 预测机制**: 利用[[Inverse Dynamics Model|逆动力学模型（IDM）]]从帧间视觉特征编码紧凑的世界动态表征 $\phi$，显式建模未来状态，为 action 生成提供预测性上下文
3. **[[Block-wise Autoregression|块状自回归（BAR）]]推理加速**: 将 action token 组织为语义块进行并行预测，实现 2.71× 推理加速，同时通过[[Temporal Context Buffer|时序上下文缓冲区]]解决多阶段任务中的[[Perceptual Aliasing|感知混叠]]问题

---

## 问题背景

### 要解决的问题

机器人操作策略学习面临三个核心挑战：
1. **感知混叠（Perceptual Aliasing）**: 多阶段任务中，不同阶段可能呈现相似的视觉观测（如倒水前后的透明水杯），导致策略无法判断当前所处阶段
2. **Action modality 对齐缺失**: 现有方法在重建质量上优化 action tokenization，但忽视了 action 表征与视觉-语言语义的深度对齐，导致控制性能受限
3. **推理效率低下**: 自回归生成 action token 速度慢，难以满足实时控制需求

### 现有方法的局限

- **标准 VLA（如 π₀.₅）**: 直接从观测预测动作，无显式未来状态建模，在时序推理和物理理解任务上能力有限
- **现有 WAM（如 [[Fast-WAM]]）**: 虽建模世界动态，但 action 表征与多模态信息的跨模态对齐尚未充分探索；对视觉输入质量高度敏感
- **[[Lumo-1]]**: 依赖显式结构化文本推理，通过粗到细规划引导 action 生成，没有潜在世界动态建模

### 本文的动机

作者认为，action modality 在跨模态对齐方面长期被忽视，重建导向的 action tokenization 与控制性能之间存在根本性偏差。通过让系统在**潜在空间**中推理世界动态后再生成动作，可以获得更强的时序推理和物理理解能力；而渐进式多阶段对齐可以保持通用多模态理解能力（避免灾难性遗忘）的同时引入 action 监督。

---

## 方法详解

### 模型架构

**Lumo-2** 采用 **[[Autoregressive Policy|自回归]][[Latent World Dynamics|世界-动作]]** 架构，基于 Qwen3.5-4B 语言模型构建：

- **输入**: 语言指令 $\ell$ + 视觉观测 $o_t$（5帧，224×224）+ 时序上下文 $\omega_{t-H':t}$ + 本体感觉状态 $s_t$
- **视觉 Backbone**: 冻结 [[DINOv2]]（提取视觉特征，不参与梯度更新）
- **世界动态模块**: 因果时空 Transformer + [[Vector Quantization|VQ Codebook]] → 潜在世界动态 $\phi$
- **核心语言模型**: Qwen3.5-4B，通过 [[Block-wise Autoregression|BAR]] 因果注意力掩码并行生成动作 token
- **输出**: [[Action Chunking|动作块]] $a_{t:t+H}$（8 个语义动作组，每组 4-8 token）
- **总参数**: ~4B（基于 Qwen3.5-4B）

### 核心模块

#### 模块 1: [[Temporal Context Buffer|时序上下文缓冲区]]

**设计动机**: 解决多阶段任务中的[[Perceptual Aliasing|感知混叠]]问题（如图 1 所示，倒水任务中面板 2 和 5 的视觉观测几乎相同，但所处任务阶段不同）

**具体实现**:
- 维护历史 action token 的紧凑队列 $\omega_{t-H':t}$，而非存储原始观测帧（效率更高）
- 在 Stage 3 训练时，以 0.5 的 dropout 率随机丢弃历史 action，增强鲁棒性
- 使 $\phi$（世界动态）能够访问时序上下文，从而区分视觉相似但任务阶段不同的状态

#### 模块 2: [[Inverse Dynamics Model|逆动力学模型（IDM）]]与潜在世界动态提取

**设计动机**: 从相邻帧的视觉特征差异中提取最小充分的[[Latent World Dynamics|世界动态]]表征，捕捉动作相关的未来演化，同时保持跨具身的语义对齐

**具体实现**（Stage 1）:
- 利用冻结 [[DINOv2]] 提取每帧视觉特征
- 因果时空 Transformer 处理 5 帧时序序列（每 8 帧采样一次，30 FPS）
- 经 [[Vector Quantization|VQ Codebook]] 量化为离散潜在码 $Z_t^V$
- 时序压缩：32→4 帧（Transformer AutoEncoder）
- 损失：视觉分支用 feature-level MSE，动作分支用 L1 重建损失
- 双向约束：$Z_t^V \to A$（引导）且 $A \to Z_t^V$（正则化）

#### 模块 3: 统一模态对齐语义模块

**设计动机**: 将 action 表征从重建导向转变为语义丰富的跨模态对齐表征 $A^s$

**具体实现**（Stage 2，多任务学习）:
- **任务 1**: Action 重建（保持运动轨迹精度）
- **任务 2**: 行为理解（给定 action token，预测语言描述）
- **任务 3**: 视觉-语言引导的 action 生成（V-L → A）
- 仅训练语义模块和特征投影层，保持 backbone 冻结
- 结果：action 表征 $A^s$ 同时具备运动精度和跨模态语义区分能力

#### 模块 4: [[Block-wise Autoregression|块状自回归（BAR）]]

**设计动机**: 打破 token-by-token 自回归的推理瓶颈，在保持因果一致性的同时实现组内并行预测

**具体实现**:
- 将 action token 组织为 4-8 个 token 的语义块
- 块内 token 并行生成（无因果约束），块间保持自回归顺序
- 通过块状因果注意力掩码实现（见图 8a），不同颜色区域对应视觉、上下文、语言、状态、世界动态、动作 token
- 推理加速：**2.71×**（在 NVIDIA RTX 5090 上使用 vLLM 测量）

---

## 关键公式

### 公式 1: [[Imitation Learning|模仿学习]]基础目标

$$
\max_\theta \mathbb{E}_{(a_{t:t+H},\, o_t,\, \ell) \sim \mathcal{D}} \log \pi_\theta(\varphi,\, a_{t:t+H} \mid o_t,\, \ell)
$$

**含义**: 最大化在数据集 $\mathcal{D}$ 上，给定观测和语言指令时，联合生成世界动态 $\varphi$ 和动作序列的对数似然

**符号说明**:
- $\theta$: 模型参数
- $a_{t:t+H}$: 从时刻 $t$ 起、时域 $H$ 内的动作序列
- $o_t$: 时刻 $t$ 的视觉+本体感觉观测
- $\ell$: 自然语言指令
- $\mathcal{D}$: 演示数据集
- $\varphi$: 潜在世界动态表征

### 公式 2: [[Temporal Context Buffer|时序上下文增强]]目标

$$
\max_\theta \mathbb{E}_{(a_{t:t+H},\, o_t,\, \ell,\, \omega_{t-H':t}) \sim \mathcal{D}} \log \pi_\theta(\varphi,\, a_{t:t+H} \mid o_t,\, \ell,\, \omega_{t-H':t})
$$

**含义**: 在基础目标上引入时序上下文缓冲区 $\omega$，使模型能够利用历史动作信息解决多阶段任务中的感知混叠

**符号说明**:
- $\omega_{t-H':t}$: 最近 $H'$ 步的历史动作 token 队列（时序上下文缓冲区）

### 公式 3: [[Latent World Dynamics|世界动态]]因子分解

$$
\pi_\theta(\varphi,\, a_{t:t+H} \mid o_t,\, \ell,\, \omega_{t-H':t}) = \pi_\theta(a_{t:t+H} \mid o_t,\, \varphi)\; \pi_\theta(\varphi \mid o_t,\, \ell,\, \omega_{t-H':t})
$$

**含义**: 将 action 生成分解为两阶段：首先从视觉、语言、历史上下文中推断潜在世界动态 $\varphi$，然后仅以 $\varphi$ 和观测 $o_t$ 为条件生成低级动作序列

**符号说明**:
- 第一项 $\pi_\theta(a_{t:t+H} \mid o_t, \varphi)$: 以世界动态为条件的动作生成分布
- 第二项 $\pi_\theta(\varphi \mid o_t, \ell, \omega_{t-H':t})$: 世界动态推断分布

### 公式 4: [[Inverse Dynamics Model|IDM]] 动作解码（Stage 1 协同训练）

$$
\hat{a} = \mathcal{D}_A(A,\, \varphi)
$$

**含义**: 动作解码器 $\mathcal{D}_A$ 同时接收动作潜在编码 $A$ 和世界动态 $\varphi$，通过特征注入建立感知与控制的双向协同

**符号说明**:
- $A$: 动作潜在编码（由 VQ Codebook 量化）
- $\varphi$: 潜在世界动态（由 IDM 提取）
- $\hat{a}$: 重建的动作序列

### 公式 5: 双向协同关系（Stage 1）

$$
\varphi \rightarrow A \;\text{（引导）}, \quad A \rightarrow \varphi \;\text{（正则化）}
$$

**含义**: 世界动态 $\varphi$ 通过注入全局环境上下文引导动作生成，而动作编码 $A$ 也反过来正则化世界动态的学习，形成双向约束

---

## 关键图表

### Figure 1: 多阶段任务中的感知混叠问题

![Figure 1](https://arxiv.org/html/2607.11270v1/figures/context_differentiation.png)

**说明**: 以"倒水"任务为例，说明[[Perceptual Aliasing|感知混叠]]问题。面板 2（倒水前）与面板 5（倒水后）的视觉观测几乎相同（透明水），面板 4 显示正在倒水的动作。引出[[Temporal Context Buffer|时序上下文缓冲区]]设计的动机。

### Figure 2: Lumo-2 总体框架

![Figure 2](https://arxiv.org/html/2607.11270v1/x1.png)

**说明**: Lumo-2 整体架构概览。展示三部分：(1) 在多模态数据、in-the-wild 视频和机器人动作数据上的协同训练；(2) 全面评估框架（22 个真实操作任务），涵盖时序推理、物理理解、灵巧操作等维度。

### Figure 3: Stage 1 — 潜在世界动态预对齐

![Figure 3](https://arxiv.org/html/2607.11270v1/x2.png)

**说明**: Stage 1 训练流程。[[Inverse Dynamics Model|IDM]] 从冻结 [[DINOv2]] 提取的帧间特征中编码[[Latent World Dynamics|世界动态]] $Z_t^V$，与 action 自编码器（产生 $Z_t^A$）协同训练，通过 feature-level MSE（视觉）+ L1（动作）双损失建立双向约束。

### Figure 4: Stage 2 — 统一模态对齐

![Figure 4](https://arxiv.org/html/2607.11270v1/x3.png)

**说明**: Stage 2 多任务对齐训练。通过三个任务（动作重建、行为理解、V-L 引导动作生成）将 action token 转化为语义丰富的表征 $A^s$，实现[[Latent World Dynamics]]、视觉、语言的统一 V-L-A 对齐。

### Figure 5: 数据混合分布

![Figure 5](https://arxiv.org/html/2607.11270v1/figures/vlm_weights.png)

**说明**: ~5300 万样本的视觉-语言数据集分布。以 FineVision（24.3M 样本）为主，配合定位、规划、认知类数据，支撑具身推理能力构建。

### Figure 6: 视觉-语言数据概览

![Figure 6](https://arxiv.org/html/2607.11270v1/x4.png)

**说明**: 精心策划的 VLM 数据集结构。核心具身推理能力覆盖：通用多模态理解（FineVision 等）、定位（ADE20K、RoboAfford 等）、规划（AgibotWorld 等）、认知（EgoRe-5M、Robo2vlm 等）。

### Figure 7: 视频数据概览

![Figure 7](https://arxiv.org/html/2607.11270v1/x5.png)

**说明**: Stage 3 使用的视频数据来源：通用互联网视频、第一视角（egocentric）视频和自采集/跨具身机器人视频数据集，用于学习广泛的世界动态。

### Figure 8: BAR 因果注意力设计与推理性能

![Figure 8a](https://arxiv.org/html/2607.11270v1/x6.png)
![Figure 8b](https://arxiv.org/html/2607.11270v1/x7.png)

**说明**: (a) [[Block-wise Autoregression|BAR]] 块状因果注意力掩码设计，不同颜色区域分别对应视觉、上下文、语言、状态、世界动态和动作 token；(b) 延迟测量结果（via vLLM on RTX 5090），BAR 实现 **2.71× 推理加速**。

### Figure 9: 潜在世界动态与观测特征可视化

![Figure 9](https://arxiv.org/html/2607.11270v1/x8.png)

**说明**: 将特征表征的前三个主成分投影到 RGB 通道。[[Latent World Dynamics|潜在世界动态]]特征（经 Stage 1 对齐后）展示出不同任务阶段和物体状态的清晰语义聚类，验证了 IDM 提取的表征确实捕捉了动作相关的语义信息。

### Figure 10: 潜在世界动态跨具身语义对齐

![Figure 10](https://arxiv.org/html/2607.11270v1/x9.png)

**说明**: 从聚类后的[[Latent World Dynamics|潜在世界动态]] token 中检索示例，揭示跨不同机器人和人类域的**涌现式语义对齐**：相同的抓取动作语义在 Astribot、Galaxea 机器人和人手数据中形成一致的聚类。

### Figure 11: 潜在世界动态编码紧凑预测性未来语义

![Figure 11](https://arxiv.org/html/2607.11270v1/x10.png)

**说明**: 消融实验：从初始观测 + 潜在世界动态预测任务指令（VLWA）可达到与使用5帧完整观测相当的准确率（~90%），远超无世界动态的基线（~43%）。验证了[[Latent World Dynamics]]的紧凑预测性。

### Figure 12: 通用拾放任务实验结果

![Figure 12](https://arxiv.org/html/2607.11270v1/x11.png)

**说明**: (a)(b) Lumo-2 在所有三个评估类别中持续超越 π₀.₅ 和 Fast-WAM 基线。特别是在需要动态场景理解的类别上优势更为显著。

### Figure 13: 确保一致评估设置

![Figure 13](https://arxiv.org/html/2607.11270v1/x12.png)

**说明**: 为确保跨模型、跨次实验的公平对比，记录场景布局并叠加从机器人头部相机提取的边缘图，确保初始化条件一致性。

### Figure 14: 任务 1-6 代表性策略展开

![Figure 14](https://arxiv.org/html/2607.11270v1/x13.png)

**说明**: 任务 1-6（包含灵巧操作类）的关键帧展示。按 Table 5 中定义的子任务顺序排列，展示 Lumo-2 在复杂操作序列中的实际执行效果。

### Figure 15: 任务 7-8 策略展开

![Figure 15](https://arxiv.org/html/2607.11270v1/x14.png)

**说明**: 任务 7-8（长时域执行类，包含制作咖啡等复杂序列）的时序关键帧，展示 Lumo-2 对长时序相互依赖动作链的持续追踪能力。

### Figure 16: 任务 9-11 策略展开

![Figure 16](https://arxiv.org/html/2607.11270v1/x15.png)

**说明**: 任务 9-11（物理理解类）的关键帧，包含翻鸡蛋等基线完全失败的任务，Lumo-2 通过物理常识推理成功完成。

### Figure 17: 任务 12-20 策略展开

![Figure 17](https://arxiv.org/html/2607.11270v1/x16.png)

**说明**: 任务 12-20（含灵巧操作、记忆类等多类别）的关键帧，覆盖笔帽装配、行李箱打包等高精度操作。

### Figure 18: 任务 21-22 策略展开

![Figure 18](https://arxiv.org/html/2607.11270v1/x17.png)

**说明**: 任务 21-22（动态场景与运动推理类，包含传送带取蛋等动态任务）的关键帧，展示 Lumo-2 对动态目标的时序响应能力。

### Figure 19: 全部 22 个任务成功率

![Figure 19](https://arxiv.org/html/2607.11270v1/x18.png)

**说明**: Lumo-2 在所有 22 个真实操作任务上的完整成功率对比（vs. π₀.₅ 和 Fast-WAM），**Lumo-2 在全部任务上持续领先**，展示综合性能提升。

### Figure 20: 分类成功率

![Figure 20](https://arxiv.org/html/2607.11270v1/x19.png)

**说明**: 按 6 个子类别（动态场景、运动推理、记忆、物理理解、长时域执行、灵巧操作）聚合的成功率。Lumo-2 在时序推理（Dynamic Scene、Motion Reasoning、Memory）和物理理解上优势尤为突出。

### Figure 21: 人类-机器人迁移示意图

![Figure 21](https://arxiv.org/html/2607.11270v1/x20.png)

**说明**: 涌现式人类-机器人迁移框架概览，无需专门迁移机制即可从 EgoTaskQA、EPIC-KITCHENS、Ego4D 等第一视角人类视频数据以及 VisionPro 设备数据中迁移操作技能。

### Table 1: 多模态数据语料库

| 类别 | 代表性数据集 |
|------|-------------|
| 通用理解 | FineVision (24.3M), FineVideo, LLaVA-Video, ShareGPT4Video |
| 定位 | ADE20K, COCOStuff, PACO-LVIS, RoboRefit, RefSpatial, RoboAfford |
| 规划 | AgibotWorld, Galaxea Open-World, sharerobot |
| 认知 | DAM, RefCOCO, VideoRefer-700k, ScanNet, EgoRe-5M, Robo2vlm |

**关键**: 共 ~5300 万样本，强化具身推理能力（定位、认知、规划）同时保留通用多模态理解

### Table 2: VLM 能力评估结果（Q1）

| 能力 | Benchmark | Lumo-2 VLM | Lumo-2 Stage 3 | Lumo-1 Stage 1 | Qwen-3.5 4B |
|------|-----------|-----------|----------------|----------------|-------------|
| 具身理解 | BLINK-Depth | **91.93** | 87.1 | 87.9 | 79.03 |
| | BLINK-Spatial | 80.42 | **86.01** | 77.6 | **86.01** |
| | CV-Bench | **86.92** | 87.45 | 86.36 | 84.84 |
| | VSIBench | **52.34** | 48.56 | 27.61 | 21.73 |
| | MindCube | **66.08** | 54.93 | 25.54 | 39.95 |
| 具身定位 | RefSpatialBench-Mean | **61.52** | 57.45 | 49.96 | 44.09 |
| | ShareRobot-Traj (DFD) ↓ | **0.2259** | 0.2386 | — | 0.428 |

**关键发现**: Lumo-2 VLM 在 VSIBench（52.34 vs 21.73）和 MindCube（66.08 vs 39.95）上大幅超越 Qwen-3.5 基线；Stage 3 协同训练后 VLM 能力基本保持，不发生灾难性遗忘

### Table 3: 潜在世界动态消融（Q2）

| 任务 | VLWA（有世界动态） | VLA（无世界动态） |
|------|------------------|-----------------|
| 传送带取蛋 | **100.00%** | 92.00% |
| 旋转架放积木 | **81.67%** | 74.17% |

**关键发现**: 显式潜在世界动态监督在时序推理任务上带来 +8.0% 和 +7.5% 的一致性提升

### Table 4: 真实任务评估维度

| 维度 | 子类别 | 主要挑战 |
|------|--------|---------|
| 时序推理 | 动态场景（反应式） | 对瞬态状态的时间关键响应 |
| | 运动推理（预测式） | 预判轨迹和接触时机 |
| | 记忆 | 序列任务中的阶段消歧 |
| 物理理解 | 物理理解 | 重力、流体动力学、因果推理 |
| 控制复杂度 | 长时域执行 | 长时序相互依赖动作链 |
| | 灵巧操作 | 高精度、高容差协调 |

### Table 5: 22 个真实操作任务

**灵巧操作（7 个）**: 笔帽装配、拆包装、装书包（衣服/玩具）、提垃圾袋、装行李箱、熨烫挂衣

**长时域执行（2 个）**: 制作咖啡、调制鸡尾酒

**物理理解（4 个）**: 整理行李箱、舀小米、翻鸡蛋、钉钉子

**动态场景（2 个）**: 传送带取蛋、接球

**运动推理（4 个）**: 挂杯子到架、抓鱼、在旋转架上放/叠积木

**记忆（3 个）**: 擦白板、倒水、备鸡蛋

评估指标：单步 0-1 二值评分；多步任务按子任务完成比例；每模型 20 次试验（10 布局 × 2 次）

---

## 实验

### 数据集

| 数据集类型 | 规模 | 特点 | 用途 |
|-----------|------|------|------|
| VLM 数据（FineVision 等） | 53M 样本 | 具身推理强化 | Stage 3 co-training |
| 机器人演示 | 自采集 + 跨具身 | Astribot S1 + 6 种其他机器人 | Stage 1/2/3 |
| 视频数据 | 互联网 + 自我中心 | EgoTaskQA, Ego4D, EPIC-KITCHENS | Stage 3 world dynamics |
| 人类第一视角 | VisionPro 采集 | 人类操作演示 | 人类-机器人迁移 |

### 实现细节

- **Backbone**: Qwen3.5-4B（语言）+ 冻结 DINOv2（视觉）
- **Stage 1**: AdamW, lr=1e-4, wd=1e-2, 30,000 步, cosine decay + 5% warmup, 24/GPU × 64 H100
- **Stage 2**: lr=1e-4（语义模块 5×，对比/判别头 2×）, 12,000 步, 64 H100
- **Stage 3**: lr=1e-5, cosine scheduler, 120,000 步, 160 H100, max 8192 tokens, history action dropout=0.5
- **视觉输入**: 5 帧序列，224×224，每 8 帧采样（30 FPS）
- **Action tokenization**: VQ Codebook，8 个语义动作组，每组 4-8 token，32→4 帧时序压缩
- **推理加速**: vLLM on RTX 5090，BAR 实现 2.71×

### 可视化结果

- 潜在世界动态特征（前 3 主成分 PCA）展示清晰的任务阶段聚类（图 9）
- 跨具身语义对齐（图 10）：相同操作语义在不同机器人和人手数据中形成一致聚类
- 翻鸡蛋等物理任务中，Lumo-2 成功而 Fast-WAM 完全失败

---

## 批判性思考

### 优点

1. **系统性解决对齐问题**: 三阶段训练范式从根本上解决了 action modality 对齐不足的问题，而非 ad-hoc 补丁
2. **强有力的实验验证**: 22 个真实任务的大规模评估，设计了公平对比的评估流程（图 13），覆盖 6 大能力维度
3. **涌现能力**: 跨具身语义对齐和人类-机器人迁移是无需显式设计的涌现能力，体现了框架的泛化潜力
4. **推理效率**: BAR 的 2.71× 加速有实用价值，对实时控制意义重大

### 局限性

1. **代码未开源**: 无代码库，可复现性受限；模型权重和训练数据也未公开
2. **平台特定**: 所有实验均在 Astribot S1 双臂移动操作平台上，跨平台泛化性未充分验证
3. **计算开销大**: Stage 3 需要 160 张 H100，训练 120,000 步，学术界难以复现
4. **人类-机器人迁移是"涌现"**: 即作者未对此进行专门优化，也未深入分析其机制

### 潜在改进方向

1. 探索更轻量的世界动态编码器，降低 Stage 1/2 训练开销
2. 对人类-机器人迁移机制进行显式建模和深入分析
3. 将 BAR 推理加速与 diffusion-based action head 结合，探索更高效的 action 生成
4. 扩展到移动操作场景（导航 + 操作）

### 可复现性评估

- [ ] 代码开源
- [ ] 预训练模型
- [x] 训练细节完整（超参数、数据配方等均在论文中详述）
- [ ] 数据集可获取（部分公开数据集，自采集机器人数据未公开）

---

## 关联笔记

### 基于

- [[Lumo-1]]: 前序工作，依赖显式文本推理，Lumo-2 改为潜在世界动态建模
- [[Fast-WAM]]: 世界动作模型基线，Lumo-2 在其基础上引入渐进对齐策略
- [[DINOv2]]: 冻结视觉特征提取 Backbone
- [[Qwen3.5]]: 基础语言模型

### 对比

- [[pi0.5]]: VLA 基线，无显式世界动态建模
- [[Fast-WAM]]: WAM 基线，无渐进模态对齐

### 方法相关

- [[Latent World Dynamics]]: 核心概念，潜在空间中的世界动态表征
- [[Inverse Dynamics Model]]: IDM，提取潜在世界动态的核心模块
- [[Block-wise Autoregression]]: BAR，推理加速机制
- [[Temporal Context Buffer]]: 时序上下文缓冲，解决感知混叠
- [[Action Chunking]]: 动作块输出形式
- [[Vector Quantization]]: VQ Codebook，离散化潜在表征
- [[Block Causal Attention]]: BAR 的注意力掩码实现基础
- [[Perceptual Aliasing]]: 感知混叠问题，时序上下文缓冲所解决的核心问题

### 硬件/数据相关

- [[Astribot S1]]: 实验平台，双臂移动操作机器人
- [[VisionPro]]: 人类演示数据采集设备

---

## 速查卡片

> [!summary] Lumo-2: Towards Predictive, Aligned, and Scalable Robot Learning
> - **核心**: 三阶段渐进对齐，将 action 表征与潜在世界动态、视觉、语言依次对齐
> - **方法**: IDM 提取世界动态 → 多任务 V-L-A 对齐 → 53M VLM+视频+机器人数据 co-training + BAR 加速
> - **结果**: 22 个真实任务全面超越 π₀.₅ 和 Fast-WAM；BAR 推理 2.71×；跨具身语义对齐涌现
> - **代码**: 未开源（research@astribot.com）

---

*笔记创建时间: 2026-07-15*
