---
title: "GTA-VLA: Guide, Think, Act: Interactive Embodied Reasoning in Vision-Language-Action Models"
method_name: "GTA-VLA"
authors: [Yiran Ling, Qing Lian, Jinghang Li, Qing Jiang, Tianming Zhang, Xiaoke Jiang, Chuanxiu Liu, Jie Liu, Lei Zhang]
year: 2026
venue: arXiv
tags: [vla, chain-of-thought, flow-matching, interactive-robotics, spatial-guidance, robot-manipulation, embodied-ai]
zotero_collection: Robotics/VLA
image_source: online
arxiv_html: https://arxiv.org/html/2605.13632v2
created: 2026-07-23
---

# 论文笔记：GTA-VLA: Guide, Think, Act: Interactive Embodied Reasoning in Vision-Language-Action Models

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Futian Labs / Harbin Institute of Technology |
| 日期 | May 2026（v2 July 2026） |
| 项目主页 | [GitHub](https://github.com/FutianLabs/GTA-VLA) |
| 对比基线 | [[X-VLA]], [[π0.5]], [[GR00T-N1]] |
| 链接 | [arXiv](https://arxiv.org/abs/2605.13632) / [Code](https://github.com/FutianLabs/GTA-VLA) |

---

## 一句话总结

> GTA-VLA 通过"引导-思考-行动"三阶段框架，让用户用点、框、路径等视觉空间 cue 实时修正机器人策略的中间推理，在 SimplerEnv 上达到 SOTA 81.2% 成功率。

---

## 核心贡献

1. **交互式空间引导接口（Guide）**: 支持点（affordance point）、框（bounding box）、路径（trace）三种视觉空间先验，让人类可在运行时向 [[VLA（视觉-语言-动作模型）]] 注入精确空间意图，解决自动 grounding 失败时无从纠正的问题
2. **结构化空间-视觉 Chain-of-Thought（Think）**: 将推理分解为固定三段式 $\mathcal{C} = [\mathcal{C}_{task}, \mathcal{C}_{vision}, \mathcal{C}_{robot}]$——任务语义分解 → 视觉目标定位 → 2D 末端轨迹草图，消融证明结构化 CoT 比自由文本 CoT 高出 +15.6 pts
3. **异步 Flow-Matching 动作头（Act）**: VLM 推理慢速（~2Hz）与 [[Flow-Matching]] 动作生成快速（~10Hz）解耦，缓存 $H_{reasoning}$ 供动作头持续使用，兼顾推理深度和控制频率
4. **Interact-306K 数据集**: 从 [[Open-X-Embodiment]]、[[DROID]]、[[RoboMIND]] 等六个来源自动化构建 306K 条带结构化 CoT 标注和交互增强的操作数据

---

## 问题背景

### 要解决的问题

传统 [[VLA（视觉-语言-动作模型）]] 将观测直接映射到动作，当模型内部 grounding 出错时（如抓取歧义目标、视觉扰动、空间语义不精确），没有可交互纠正的机制，只能重新采集数据或重新训练。

### 现有方法的局限

- **ECoT / Mind2Hand**: 引入 [[Chain-of-Thought Reasoning]] 提升透明度，但推理过程自封闭（self-contained），无法接受外部引导修正
- **SAM / Ferret 等视觉提示方法**: 几何提示主要用于像素级感知，未连接到连续动作生成
- **现有 VLA 基线（OpenVLA、π0、GR00T-N1）**: 对 OOD 视觉变化、空间歧义、语言多样性缺乏鲁棒性，且对用户透明度为零

### 本文的动机

若将人类视觉空间先验注入 VLM 的**中间推理**而非直接指令，则可在不改变任务描述的情况下精确修正模型的 grounding，同时利用结构化 [[Chain-of-Thought Reasoning]] 让推理步骤可解释、可干预。

---

## 方法详解

### 模型架构

GTA-VLA 采用**三阶段异步 VLM + Flow-Matching** 架构：

- **输入**: 语言指令 $L$ + 主视角图像 $\mathcal{I}_t^{main}$ + 可选空间先验 $P_{spatial}$ + 末端状态 $s_t$（本体感知）
- **VLM Backbone**: [[Qwen3-VL]]（Qwen3-VL-2B）
- **核心模块**: 结构化 [[Chain-of-Thought Reasoning]] 推理头 + 异步 [[Flow-Matching]] 动作头
- **输出**: 动作块 $\mathcal{A}_t = [a_t, a_{t+1}, \ldots, a_{t+k-1}]$（ee6d 末端执行器 6 DoF + 夹爪）
- **总参数**: 约 2B（VLM 主干）

### 核心模块

#### 模块 1: Guide — 空间先验接口

**设计动机**: 利用 [[可供性（Affordance）]] 和空间几何约束，让人类视觉意图以统一坐标格式注入 VLM

**三种空间先验类型**:
- **Affordance Point** $P_{point}$: 单个 2D 图像坐标 $(x, y)$，标记目标接触点或抓握位置
- **Bounding Box** $P_{bbox}$: 矩形区域 $(x_{min}, y_{min}, x_{max}, y_{max})$，标定目标物体范围
- **Trace Guide** $P_{trace}$: 有序 2D 航路点序列 $[(x_1,y_1),(x_2,y_2),\ldots,(x_m,y_m)]$，表达运动偏好和障碍回避路径

**实现**: 利用 [[Qwen3-VL]] 原生支持的相对坐标空间直接序列化 $P_{point}$ 和 $P_{bbox}$，无需额外编码模块

#### 模块 2: Think — 结构化空间-视觉 CoT

**设计动机**: 通过固定字段的 [[Chain-of-Thought Reasoning]] 强制推理路径透明化，同时使空间先验能精确条件化每个推理阶段

**三段式推理结构** $\mathcal{C} = [\mathcal{C}_{task}, \mathcal{C}_{vision}, \mathcal{C}_{robot}]$:
- $\mathcal{C}_{task}$（任务 CoT）: 将指令 $L$ 分解为可执行子任务序列，识别相关对象和交互
- $\mathcal{C}_{vision}$（视觉 CoT）: 在图像空间预测视觉 grounding 目标——目标区域和任务相关的 [[可供性（Affordance）]] 位置
- $\mathcal{C}_{robot}$（机器人 CoT）: 基于视觉目标预测末端执行器的粗粒度 2D 运动草图（waypoints 序列）

**潜在状态**: VLM 推理 token 的隐藏状态组成 $H_{reasoning} \in \mathbb{R}^{N \times D}$，传递给下游动作头

#### 模块 3: Act — 异步 Flow-Matching 动作生成

**设计动机**: [[异步推理]] 解耦慢速 VLM 推理（约 2Hz）与快速连续控制（约 10Hz），避免 VLM 成为实时控制瓶颈

**双模块架构**:
- **慢速推理模块**: VLM backbone 以低频率（~2Hz）执行 Guide-Think，产生缓存的 $H_{reasoning}^{latest}$
- **快速动作模块**: [[Flow-Matching]] 动作头以高频率（~10Hz）接收最新推理状态 + 高频控制观测，生成连续动作块（[[Asynchronous Inference|异步解耦]]）

**训练目标**: 对推理序列 $\mathcal{C}$ 施加自回归 token 预测损失；对动作块施加标准 flow-matching 目标

#### 模块 4: Interact-306K 数据集构建

**六个数据来源**: Open X-Embodiment（Bridge、Fractal 等）、DROID、RoboMind（多变体）及内部数据，共约 306K 条轨迹

**自动化标注流水线**:
1. 关键帧提取 + 轨迹任务分解（生成 $\mathcal{C}_{task}$）
2. 开放词汇目标检测 + 时序一致跟踪（生成 $\mathcal{C}_{vision}$）
3. 末端执行器运动投影到主视角生成 2D affordance 点和运动草图（生成 $\mathcal{C}_{robot}$）
4. 对生成的空间标注施加随机噪声扰动，产生合成的点/框/路径训练样本

---

## 关键公式

### 公式 1: [[Action Chunking|动作块定义]]

$$
\mathcal{A}_t = [a_t, a_{t+1}, \ldots, a_{t+k-1}]
$$

**含义**: 策略在时刻 $t$ 预测未来 $k$ 步的连续动作序列，而非单步动作，提升稳定性。

**符号说明**:
- $\mathcal{A}_t$: 时刻 $t$ 的动作块
- $a_t$: 时刻 $t$ 的单步动作（末端执行器 6 DoF + 夹爪开合）
- $k$: 动作块长度

### 公式 2: [[VLA（视觉-语言-动作模型）|基础策略形式]]

$$
\pi: (\mathcal{I}_t, L, s_t) \rightarrow \mathcal{A}_t
$$

**含义**: 传统 VLA 策略将当前图像、语言指令和本体状态直接映射到动作块，无交互接口。

**符号说明**:
- $\mathcal{I}_t$: 时刻 $t$ 的图像观测
- $L$: 自然语言任务指令
- $s_t$: 机器人本体感知状态（proprioception）

### 公式 3: [[可供性（Affordance）|扩展策略（含空间先验）]]

$$
\pi: (\mathcal{I}_t, L, s_t, P_{spatial}) \rightarrow \mathcal{A}_t
$$

**含义**: GTA-VLA 在标准策略输入基础上增加可选空间先验 $P_{spatial}$，支持有/无引导两种模式。

**符号说明**:
- $P_{spatial}$: 可选空间先验，可为 $P_{point}$、$P_{bbox}$ 或 $P_{trace}$ 之一

### 公式 4: [[可供性（Affordance）|路径引导先验定义]]

$$
P_{trace} = [(x_1, y_1), (x_2, y_2), \ldots, (x_m, y_m)]
$$

**含义**: Trace guide 为有序 2D 图像坐标序列，表达用户期望的末端执行器运动方向和路径偏好。

**符号说明**:
- $(x_i, y_i)$: 路径中第 $i$ 个航路点的图像空间坐标（相对坐标系）
- $m$: 路径点数量

### 公式 5: [[Chain-of-Thought Reasoning|结构化 CoT 分解]]

$$
\mathcal{C} = [\mathcal{C}_{task}, \mathcal{C}_{vision}, \mathcal{C}_{robot}]
$$

**含义**: 将 VLM 推理序列强制分解为三个功能明确的片段，每段对应操作推理的一个认知层级。

**符号说明**:
- $\mathcal{C}_{task}$: 语义任务分解（子任务识别和对象指认）
- $\mathcal{C}_{vision}$: 视觉 grounding 目标（目标区域 + affordance 位置）
- $\mathcal{C}_{robot}$: 机器人运动草图（2D 末端执行器 waypoints）

### 公式 6: [[Chain-of-Thought Reasoning|条件化推理分布]]

$$
P(\mathcal{C} \mid \mathcal{I}_t, L, P_{spatial})
$$

**含义**: VLM 生成的推理序列 $\mathcal{C}$ 以图像、语言指令和可选空间先验为联合条件，使引导信号精确影响中间推理。

### 公式 7: [[Flow-Matching|潜在推理状态维度]]

$$
H_{reasoning} \in \mathbb{R}^{N \times D}
$$

**含义**: VLM 推理 token 的隐藏状态矩阵，维度为序列长度 $N$ × 模型宽度 $D$，作为动作头的高层语义输入。

**符号说明**:
- $N$: 推理 token 序列长度（最大 768）
- $D$: 模型隐藏维度（1024）

### 公式 8: [[Flow-Matching|VLM 到推理状态的映射]]

$$
(\mathcal{I}_t^{main}, L, P_{spatial}) \xrightarrow{\text{VLM}} \mathcal{C}, H_{reasoning}
$$

**含义**: VLM backbone 以低频率（~2Hz）处理主视角图像、语言指令和空间先验，输出推理序列和缓存的隐藏状态。

### 公式 9: [[Flow-Matching|异步动作生成向量场]]

$$
v_\theta\!\left(x, \tau \;\middle|\; \mathcal{I}_t^{main},\, \mathcal{I}_t^{wrist},\, s_t,\, H_{reasoning}^{latest}\right)
$$

**含义**: [[Flow-Matching]] 动作头的向量场，条件化于双目图像、本体状态和最新（缓存的）推理状态，以高频率（~10Hz）生成连续动作块。

**符号说明**:
- $x$: flow 过程中的动作变量
- $\tau$: flow 时间参数 $\tau \in [0,1]$
- $\mathcal{I}_t^{wrist}$: 腕部相机图像
- $H_{reasoning}^{latest}$: 最新缓存的 VLM 推理状态（慢速更新，快速复用）

---

## 关键图表

### Figure 1: 动机示意——传统 VLA vs GTA-VLA

![Figure 1](https://arxiv.org/html/2605.13632v2/x1.png)

**说明**: 传统直接映射型 [[VLA（视觉-语言-动作模型）]] 在空间歧义或 grounding 不精确时会失败，且无法纠正。GTA-VLA 通过一次性 [[可供性（Affordance）|空间引导]]（affordance 点、bounding box 或路径轨迹）修正 grounding，实现准确执行。

### Figure 2: GTA-VLA 框架总览

![Figure 2](https://arxiv.org/html/2605.13632v2/x2.png)

**说明**: 三阶段架构示意。**Guide** 接收主视角图像、语言指令和可选 $P_{spatial}$；**Think** 由 VLM 生成结构化 $\mathcal{C}$ 和 $H_{reasoning}$；**Act** 的 [[Flow-Matching]] 动作头异步消耗最新推理状态 + 高频控制观测，实现慢速推理（~2Hz）与快速控制（~10Hz）解耦。

### Figure 3: Interact-306K 数据集构成与自动标注流水线

![Figure 3](https://arxiv.org/html/2605.13632v2/x3.png)

**说明**: **左**：306K 条轨迹来自六个来源（Bridge、Fractal、[[DROID]]、[[RoboMIND]] 等）。**右**：自动标注流水线——关键帧提取 + 任务分解 → 开放词汇 grounding + 时序跟踪 → 生成带时序一致对象标注的结构化子任务指令。

### Figure 4 & 5: 真实机器人部署（AgileX Piper）

![Figure 4](https://arxiv.org/html/2605.13632v2/x4.png)

![Figure 5](https://arxiv.org/html/2605.13632v2/x5.png)

**说明**: 实验设置使用 AgileX Piper 机械臂，配主视角相机和腕部相机，搭载双 RTX 5090。四类抓取任务（已见/未见目标 × 单/多候选物）的定性示例和成功率。推理 CoT 相比无推理基线有显著提升；在未见目标和空间歧义场景中，点引导带来最大收益。

### Figure 6 & 7: SimplerEnv WidowX 和 Google Robot 基准可视化

![Figure 6](https://arxiv.org/html/2605.13632v2/x6.png)

![Figure 7](https://arxiv.org/html/2605.13632v2/x7.png)

**说明**: GTA-VLA 在 [[SimplerEnv]] WidowX 和 Google Robot 两个域的基础 benchmark 中均达到 SOTA，总体平均 81.2%，显著优于所有对比方法。

### Figure 8: 实时 CoT 输出可视化

![Figure 8](https://arxiv.org/html/2605.13632v2/x8.png)

**说明**: 机器人运行时结构化 [[Chain-of-Thought Reasoning]] 输出示例，展示 $\mathcal{C}_{task}$、$\mathcal{C}_{vision}$（视觉定位）、$\mathcal{C}_{robot}$（运动草图）三段推理的可读性和与实际执行的对应关系。

### Figure 9: SimplerEnv-Plus 视觉扰动和对象扰动可视化

![Figure 9](https://arxiv.org/html/2605.13632v2/x9.png)

**说明**: 在 SimplerEnv-Plus 的六类 OOD 扰动下，GTA-VLA 平均 61.4%，相比 X-VLA（52.3%）和 π0.5（7.3%）大幅提升，尤其在未见对象（+22.1 pts vs X-VLA）和机器人状态扰动（+10.5 pts）上优势显著。

### Table 1: 主要结果——LIBERO 和 SimplerEnv

| 方法 | LIBERO Spatial | LIBERO Object | LIBERO Goal | LIBERO Long | **LIBERO Avg** | Spoon | Carrot | Cube | Eggplant | **SimplerEnv Avg** |
|------|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| OpenVLA | 84.7 | 88.4 | 79.2 | 53.7 | 76.5 | 4.2 | 0.0 | 8.3 | 45.8 | 14.6 |
| π0 | 96.8 | 98.8 | 95.8 | 85.2 | 94.1 | 50.0 | 41.7 | 29.2 | 70.8 | 47.9 |
| GR00T-N1 | 94.4 | 97.6 | 93.0 | 90.6 | 93.9 | 64.5 | 65.5 | 5.5 | 93.0 | 57.1 |
| π0.5 | 98.8 | 98.2 | 98.0 | 92.4 | 96.9 | — | — | — | — | — |
| X-VLA | 98.2 | 98.6 | 97.8 | 97.6 | 98.1 | 95.8 | 75.0 | 62.5 | 70.8 | 76.0 |
| ThinkAct | 88.3 | 91.4 | 87.1 | 70.9 | 84.4 | 37.5 | 8.7 | 58.3 | 70.8 | 43.8 |
| CoT-VLA | 87.5 | 91.6 | 87.6 | 69.0 | 81.1 | — | — | — | — | — |
| Uni-VLA | 97.0 | 99.0 | 92.6 | 90.8 | 94.8 | 83.3 | 66.7 | 33.3 | 95.8 | 69.8 |
| **GTA-VLA** | **99.0** | **98.8** | **98.4** | **97.6** | **98.6** | **95.8** | **87.5** | 66.7 | 75.0 | **81.2** |

**说明**: GTA-VLA 在 [[LIBERO]] 四套件平均 98.6%，[[SimplerEnv]] 平均 81.2%（SOTA）。LIBERO Long（最难，需长程规划）97.6% 与 X-VLA 持平，均超出 π0.5 5.2 pts。

### Table 2: OOD 泛化——SimplerEnv-Plus

| 方法 | Sensor | Lighting | Robot State | Language | Unseen Object | Distractor | **Avg** |
|------|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| OpenVLA | 5.2 | 6.3 | 0.0 | 8.3 | 2.1 | 0.0 | 3.7 |
| π0.5 | 9.4 | 10.4 | 9.4 | 8.3 | 6.3 | 0.0 | 7.3 |
| X-VLA | 27.1 | 68.8 | 68.7 | 66.3 | 36.2 | 46.9 | 52.3 |
| **GTA-VLA** | **39.6** | **76.1** | **79.2** | **68.1** | **58.3** | **50.0** | **61.4** |

**说明**: 六类 OOD 扰动下全面超越所有基线，未见对象（+22.1 pts vs X-VLA）和机器人状态扰动（+10.5 pts）提升最显著，验证了结构化 [[Chain-of-Thought Reasoning]] 对 OOD 鲁棒性的贡献。

### Table 3: 视觉引导有效性——歧义场景

| 引导模式 | Unseen Toy | Unseen Fruit | Unseen Tool | Avg (Unseen) | Color Distractor | Pos. Distractor | Avg (Distractor) |
|---------|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| Dense Instruction (π0.5) | 8.3 | 12.5 | 8.3 | 9.7 | 8.3 | 8.3 | 8.3 |
| Dense Instruction (GTA-VLA) | 12.5 | 41.6 | 29.2 | 27.8 | 45.8 | 29.2 | 37.5 |
| + Point Guide | 33.3 | 47.9 | 41.6 | 40.9 | 58.3 | 50.0 | 54.2 |
| + Box Guide | **54.1** | **70.8** | **45.8** | **56.9** | 41.6 | 45.8 | 43.7 |

**说明**: 单次 [[可供性（Affordance）|视觉引导]] 干预在未见对象场景中将平均成功率从 27.8% 提升至最高 56.9%（+29.1 pts）。框引导在未见对象上最优，点引导在干扰物场景上更好，体现两种引导模态的互补性。

### Table 4: 消融——结构化 CoT 各字段贡献

| 配置 | Spoon | Carrot | Cube | Eggplant | **Avg** | Δ |
|------|:-:|:-:|:-:|:-:|:-:|:-:|
| GTA-VLA（完整） | 95.8 | 87.5 | 66.7 | 75.0 | **81.2** | — |
| −$\mathcal{C}_{robot}$ | 95.8 | 87.5 | 54.2 | 75.0 | 78.1 | −3.1 |
| −$\mathcal{C}_{task}$ | 95.8 | 91.7 | 54.2 | 33.3 | 68.8 | −12.4 |
| −$\mathcal{C}_{vision}$ | 95.8 | 79.2 | 41.7 | 50.0 | 66.7 | −14.5 |
| Free-form CoT | 100.0 | 91.7 | 50.0 | 20.8 | 65.6 | −15.6 |

**关键发现**: $\mathcal{C}_{vision}$（视觉 grounding）贡献最大（−14.5 pts），其次是 $\mathcal{C}_{task}$（−12.4 pts）；自由文本 CoT 比无结构更差，说明**结构约束**本身就是重要的归纳偏置。

### Table 5: 交互增强采样配比（Interact-306K）

| 交互模式 | 概率 |
|---------|------|
| none（无引导） | 0.40 |
| pick_box | 0.20 |
| place_box | 0.12 |
| pick_and_place | 0.12 |
| affordance_2d | 0.10 |
| gripper_path_2d | 0.06 |

**说明**: 40% 无引导样本确保模型在纯自主模式下仍有竞争力；点、框、路径按任务语义分布赋权。

### Table 6: 主要超参数

| 超参数 | 预训练 | 微调 |
|--------|--------|------|
| 精度 | bfloat16 | bfloat16 |
| 动作模式 | ee6d | ee6d |
| 模型宽度/层数/头数 | 1024 / 24 / 16 | 1024 / 24 / 16 |
| MLP 比 | 4.0 | 4.0 |
| 最大序列长度 | 1024 | 1024 |
| Projection 层/隐藏/dropout | 2 / 1536 / 0.1 | 2 / 1536 / 0.1 |
| CoT 损失权重 / 最大长度 | 1.0 / 768 | 1.0 / 768 |
| Diffusion 采样数 | 4 | 4 |
| 交互增强比例 | 0.5 | 0.5 |
| 每 GPU Batch | 8 | 16 |
| 学习率 | 1e-4 | 1e-4 |
| 余弦衰减 | ✓ | ✓ |

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| Interact-306K | 306K 条轨迹 | 自动生成 CoT + 空间引导标注，六个来源 | 预训练 |
| [[LIBERO]] | 4 套件 × 50 任务 | 桌面操作 benchmark，含 Spatial/Object/Goal/Long | 主要评估 |
| [[SimplerEnv]] | WidowX + Google Robot 域 | Real-to-sim benchmark | 主要评估 |
| SimplerEnv-Plus | 6 类 OOD 扰动 | 视觉变化/机器人状态/语言多样性/未见对象/干扰物 | OOD 评估 |
| 真实机器人数据 | AgileX Piper | 4 类抓取任务（已见/未见 × 单/多候选物） | 真实世界验证 |

### 实现细节

- **Backbone**: [[Qwen3-VL]]（Qwen3-VL-2B）
- **优化器**: 学习率 1e-4，余弦衰减
- **Batch Size**: 预训练 8/GPU，微调 16/GPU
- **预热步数**: 4000 步，冻结步数（backbone）: 2000 步
- **硬件**: 预训练 48× H800，微调 16× H800，真实机器人部署双 RTX 5090
- **动作模式**: ee6d（末端执行器 6 DoF + 夹爪）
- **真实机器人**: AgileX Piper，主视角 + 腕部相机

### 可视化结果

- 运行时 CoT 输出可读性强，$\mathcal{C}_{task}$ 准确分解子任务，$\mathcal{C}_{vision}$ 生成精确 grounding 框，$\mathcal{C}_{robot}$ 显示合理运动草图（Figure 8）
- 视觉引导在存在颜色相似干扰物时，相比仅语言指令成功率翻倍（37.5% → 54.2%，Box Guide）

---

## 批判性思考

### 优点

1. **可交互纠错的新范式**: 将人类视觉意图注入 VLM 中间推理而非末端动作，从根本上解决了黑盒策略无法在线纠正的问题
2. **结构化 CoT 的鲁棒性收益**: 固定三段式 CoT 比自由文本 CoT 高出 +15.6 pts，且各字段消融清晰，为"推理结构化"的价值提供了扎实的实验证据
3. **异步解耦的工程价值**: ~2Hz 推理 + ~10Hz 控制的分离设计在不依赖投机采样的前提下实现了实时性，对实际部署有直接意义
4. **数据集自动化**: Interact-306K 的全自动标注流水线降低了结构化 CoT 数据的获取门槛，可扩展到新的机器人数据

### 局限性

1. **2D 空间限制**: Guide、Think 和 Act 均在 2D 图像空间操作，缺乏 3D 几何理解，对遮挡、深度估计要求高的任务有天然瓶颈
2. **小 VLM 规模（2B）**: 使用 Qwen3-VL-2B，与 GR00T-N1（更大）等对比有参数量优势，但也意味着可能缺乏足够语义理解能力来处理极度复杂的指令
3. **引导信息依赖人工**: 空间先验 $P_{spatial}$ 在失败恢复场景需要人工标注，实际部署中难以完全自动化，增加了人机交互成本
4. **评估多样性有限**: 真实机器人仅测试了 4 类抓取任务，泛化到更复杂操作（装配、倒液体、双臂）的能力未充分验证

### 潜在改进方向

1. 将推理和引导接口扩展到 3D 空间（如点云、深度图），解决 2D 视角受限问题
2. 引入在线自主引导生成（利用 VLM 自身生成 $P_{spatial}$），降低对人工标注的依赖
3. 探索多轮引导对话，支持细粒度递进式纠错而非单次干预

### 可复现性评估

- [x] 代码开源（https://github.com/FutianLabs/GTA-VLA）
- [ ] 预训练模型权重（待发布）
- [x] 训练细节完整（超参数、数据集构成均已公开）
- [x] 部分数据集可获取（Open X-Embodiment、DROID、LIBERO 公开，RoboMind 部分公开）

---

## 关联笔记

### 基于

- [[Qwen3-VL]]: 使用的 VLM 主干（Qwen3-VL-2B）
- [[Flow-Matching]]: 动作头的生成方式
- [[Open-X-Embodiment]]: Interact-306K 的主要数据来源
- [[DROID]]: Interact-306K 数据来源之一
- [[RoboMIND]]: Interact-306K 数据来源之一

### 对比

- [[X-VLA]]: SimplerEnv SOTA 对比基线（76.0% vs 81.2%）
- [[GR00T-N1]]: NVIDIA 发布的 VLA 基线（57.1% SimplerEnv）
- [[π0.5]]: 强 LIBERO 基线（96.9% vs 98.6%）
- [[CoT-VLA]]: 自由文本 CoT 对比，验证结构化 CoT 的优势
- [[ECoT]]: 早期 CoT VLA 工作，启发了本文的推理结构设计

### 方法相关

- [[Chain-of-Thought Reasoning]]: 核心推理方法，结构化三段分解
- [[可供性（Affordance）]]: Guide 阶段的空间先验设计依据
- [[Flow-Matching]]: Act 阶段的动作生成方法
- [[Action Chunking]]: 输出动作块的基础范式
- [[Asynchronous Inference]]: Act 阶段慢推理/快控制的解耦策略

### 硬件/数据相关

- [[SimplerEnv]]: 主要评估 benchmark（real-to-sim）
- [[LIBERO]]: 主要评估 benchmark（sim-to-sim）

---

## 速查卡片

> [!summary] GTA-VLA: Guide, Think, Act
> - **核心**: 用点/框/路径视觉空间 cue 引导 VLM 中间推理，实现可交互机器人策略
> - **方法**: 结构化三段 CoT（任务分解 → 视觉 grounding → 运动草图）+ 异步 Flow-Matching 动作头
> - **结果**: SimplerEnv 81.2%（SOTA），LIBERO 98.6%，OOD 平均 61.4% 超越所有基线
> - **代码**: https://github.com/FutianLabs/GTA-VLA

---

*笔记创建时间: 2026-07-23*
