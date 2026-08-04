---
title: "FibVLA: An Efficient Temporal Vision-Language-Action Model with Fibonacci Sampling"
method_name: "FibVLA"
authors: [Li Lin, Wujun Xu, Weiwei Meng, Kaiwen Xia, Kang Hao Cheong, Shuai Wang]
year: 2026
venue: arXiv
tags: [vla, temporal-perception, flow-matching, embodied-ai, robot-manipulation, action-chunking]
zotero_collection: Robotics/VLA
image_source: online
arxiv_html: https://arxiv.org/html/2607.29596
created: 2026-08-04
---

# 论文笔记：FibVLA: An Efficient Temporal Vision-Language-Action Model with Fibonacci Sampling

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | 待补充 |
| 日期 | July 2026 |
| 项目主页 | 暂无 |
| 对比基线 | [[π0]], [[CogACT]], [[TraceVLA]] |
| 链接 | [arXiv](https://arxiv.org/abs/2607.29596) / Code 暂未开源 |

---

## 一句话总结

> FibVLA 通过斐波那契序列对齐的对数间隔历史采样、通道时序编码和递归 KV Cache 复用，在不重训视觉编码器的条件下，使 VLA 同时具备长时序感知能力与实时推理效率。

---

## 核心贡献

1. **[[Logarithmic Hindsight Sampling|对数回顾采样]]**: 利用对数间隔 + 斐波那契约束构建历史帧索引，兼顾近期细粒度信息与长程任务上下文，消除离散化冗余
2. **[[Channel-wise Temporal Encoding|通道时序编码 (CTE)]]**: 对采样帧做帧差运动掩码，将近中远三档时序特征分别映射到 RGB 通道，以单帧输入融合多尺度时序信息
3. **[[Fibonacci Recurrent Inference|斐波那契递归推理]]**: 利用斐波那契数学性质使前一步的 [[KV Cache]] token 在下一步精确对齐采样点，实现无需重新编码的特征复用，推理延迟降低 27%

---

## 问题背景

### 要解决的问题

传统 [[Vision-Language-Action Model|VLA]] 仅处理当前帧观测，遵循马尔可夫假设，忽略了机器人操作中的时序依赖关系。在物体遮挡、长时序操作等场景下，缺乏历史上下文导致任务成功率显著下降。

### 现有方法的局限

- **稀疏表征学习**（TraceVLA、4D-VLA）：需要大量离线预处理，部署灵活性差
- **固定频率采样**（MemoryVLA、HiF-VLA）：忽略机器人操作中时序信息密度的非均匀性（近期细粒度变化 vs. 远期任务级上下文），高频区域浪费算力，低频区域信息不足
- **视频编码 VLA**：引入大量视频 token，推理延迟大幅增加，难以满足实时需求

### 本文的动机

机器人操作中的时序信息天然呈非均匀密度：近期帧捕捉精细动作变化，远期帧提供任务级语义背景。对数采样天然对应人类感知的时间感受野特性。进一步地，斐波那契数列的递推性质 $k_i = k_{i-1} + k_{i-2}$ 意味着采样索引在时间步推进 $L$ 步后自动对齐，使得历史特征 token 可以直接复用，无需重新编码。

---

## 方法详解

### 模型架构

FibVLA 基于 **[[π0]]** 的双流 [[Flow Matching]] 架构，在不改动主干网络参数的前提下插入三个关键模块：

- **输入**: 语言指令 $\mathcal{L}$ + 当前帧 $o_t$ + [[Logarithmic Hindsight Sampling|历史帧集合]] $\mathcal{O}_\mathcal{K}$ + 本体感知历史 $\mathcal{S}_\mathcal{K}$
- **视觉 Backbone**: SigLIP 视觉编码器（冻结）
- **语言 Backbone**: PaliGemma (3B) 多模态基础模型
- **核心模块**: [[Logarithmic Hindsight Sampling]] → [[Channel-wise Temporal Encoding|CTE]] → [[Fibonacci Recurrent Inference]]
- **输出**: [[Action Chunking|动作块]] $\mathcal{A}_{t:t+L}$（长度 $L$，通过 flow matching 生成）

![Figure 2: FibVLA 整体框架](https://arxiv.org/html/2607.29596v1/x2.png)

**说明**: FibVLA 整体架构图，展示对数回顾采样、通道时序编码模块、斐波那契递归推理三大组件的数据流。

---

### 核心模块

#### 模块 1: [[Logarithmic Hindsight Sampling|对数回顾采样 (Logarithmic Hindsight Sampling)]]

**设计动机**: 利用对数尺度在有限帧数内同时覆盖近期细粒度动作和远期任务级语境，避免[[固定频率采样]]的均匀信息冗余。

**具体实现**:

- 给定最小采样间隔 $q_{min}$ 和增长率 $r > 1$，生成初始索引：

$$
k_i = \lfloor q_{min} \cdot r^i \rfloor \tag{3}
$$

- 施加斐波那契递归稀疏约束，消除离散化后的索引碰撞：

$$
k_i \geq k_{i-1} + k_{i-2}, \quad \forall i > 2 \tag{4}
$$

- 最终索引集 $\mathcal{K} = \{k_1, \ldots, k_N\}$ 保证严格单调递增，且相邻索引比趋近黄金分割比 $\phi = 1.618$

![Figure 3: 时序采样策略对比](https://arxiv.org/html/2607.29596v1/x3.png)

**说明**: 左侧为均匀采样（近期帧密度过高），右侧为 FibVLA 的对数采样（近密远疏），更高效地利用有限帧数。

---

#### 模块 2: [[Channel-wise Temporal Encoding|通道时序编码 (CTE)]]

**设计动机**: 将稀疏历史帧的时序运动信息压缩为单帧表示，以 RGB 三通道对应近（Near）、中（Mid）、远（Far）三档时间范围，背景静态区域过滤掉以减少视觉噪声干扰。

**具体实现**:

1. **帧差运动图**: 对相邻采样帧做像素级绝对差：

$$
D(\cdot, i) = |I(\cdot, t-k_i) - I(\cdot, t-k_{i+1})| \tag{5}
$$

2. **二值运动掩码**: 以阈值 $\xi$ 过滤背景噪声：

$$
\Psi(\cdot, i) = \begin{cases} 1, & \text{if } D(\cdot, i) > \xi \\ 0, & \text{otherwise} \end{cases} \tag{6}
$$

3. **衰减时序编码**: 通过递归衰减机制生成持久性运动热度图：

$$
H(\cdot, i) = \begin{cases} \tau, & \text{if } \Psi(\cdot, i) = 1 \\ \max(0,\, H(\cdot, i+1) - \delta), & \text{otherwise} \end{cases} \tag{7}
$$

4. **三通道融合**: Near/Mid/Far 三档特征分别映射到 R/G/B 通道，与当前帧拼接为回顾特征 $\hat{o}_t$

![Figure 4: 通道时序编码模块](https://arxiv.org/html/2607.29596v1/x4.png)

**说明**: CTE 通过空间差异识别运动区域（亮色=运动区域），三通道颜色编码对应不同时间尺度的运动历史。

---

#### 模块 3: [[Fibonacci Recurrent Inference|斐波那契递归推理 (Fibonacci Recurrent Inference)]]

**设计动机**: 当 [[Action Chunking|动作块]] 长度 $L = k_{i-2}$ 与斐波那契索引对齐时，前一步骤的历史帧精确对应下一步骤的采样点，历史 [[KV Cache]] token 可直接复用，无需重新通过视觉编码器。

**数学推导**:

设当前时刻 $t$，下一执行时刻 $t' = t + L = t + k_{i-2}$。利用斐波那契性质 $k_i = k_{i-1} + k_{i-2}$：

$$
t' - k_i = (t + k_{i-2}) - k_i = t - (k_i - k_{i-2}) = t - k_{i-1} \tag{8}
$$

即：$t$ 时刻回溯 $k_{i-1}$ 步的帧 = $t'$ 时刻回溯 $k_i$ 步的帧（同一物理帧）。所有高频（近期）更新仅发生在新时间窗口 $[t, t+L]$ 内，历史部分完全可复用。

随着采样深度增加，相邻采样间隔比趋近黄金分割比 $\phi$，自然维持对数分布特性。

![Figure 5: 斐波那契递归推理示意图](https://arxiv.org/html/2607.29596v1/x5.png)

**说明**: 展示 $t$ 时刻与 $t'$ 时刻的采样点对齐关系，蓝色 token 为可复用的历史特征，红色 token 为需要新编码的当前窗口帧。

---

## 关键公式

### 公式 1: [[Vision-Language-Action Model|瞬时观测策略]]

$$
\mathcal{A}_{t:t+L} \sim \pi(\cdot \mid \mathcal{L},\, o_t,\, s_t) \tag{1}
$$

**含义**: 传统 VLA 的马尔可夫策略，仅依赖当前帧观测生成动作块，忽略历史信息。

**符号说明**:
- $\mathcal{A}_{t:t+L}$: 长度为 $L$ 的动作块序列
- $\mathcal{L}$: 自然语言任务指令
- $o_t \in \mathbb{R}^{H \times W \times 3}$: 当前视觉观测（分辨率 $H \times W$）
- $s_t \in \mathbb{R}^{d_s}$: 当前本体感知状态

---

### 公式 2: [[Logarithmic Hindsight Sampling|扩展时序策略]]

$$
\mathcal{A}_{t:t+L} \sim \pi(\cdot \mid \mathcal{L},\, \mathcal{O}_\mathcal{K},\, \mathcal{S}_\mathcal{K}) \tag{2}
$$

**含义**: FibVLA 将策略条件扩展为稀疏历史帧集合 $\mathcal{O}_\mathcal{K}$ 和状态历史 $\mathcal{S}_\mathcal{K}$，实现时序感知决策。

**符号说明**:
- $\mathcal{K} = \{k_1, \ldots, k_N\}$: 稀疏时序索引集合（$N$ 帧）
- $\mathcal{O}_\mathcal{K} = \{o_{t-k_1}, \ldots, o_{t-k_N}\}$: 采样历史帧集合
- $\mathcal{S}_\mathcal{K}$: 对应时刻的本体感知状态历史

---

### 公式 3: [[Logarithmic Hindsight Sampling|对数采样索引生成]]

$$
k_i = \lfloor q_{min} \cdot r^i \rfloor \tag{3}
$$

**含义**: 以指数增长方式生成历史帧时间索引，近期帧密集采样、远期帧稀疏采样。

**符号说明**:
- $k_i$: 第 $i$ 个历史帧相对当前时刻的时间偏移
- $q_{min} > 0$: 最小采样间隔（控制近期帧密度）
- $r > 1$: 增长率（控制采样稀疏化速度）
- $\lfloor \cdot \rfloor$: 向下取整

---

### 公式 4: [[Logarithmic Hindsight Sampling|斐波那契稀疏约束]]

$$
k_i \geq k_{i-1} + k_{i-2}, \quad \forall i > 2 \tag{4}
$$

**含义**: 通过斐波那契递推约束消除对数离散化后的索引碰撞，同时为递归推理特征复用奠定数学基础。

**符号说明**:
- 此约束保证 $k_i > k_{i-1}$ 严格单调，且相邻比值趋近黄金分割比 $\phi \approx 1.618$

---

### 公式 5: [[Channel-wise Temporal Encoding|帧差运动图 + 二值掩码]]

$$
D(\cdot, i) = |I(\cdot, t-k_i) - I(\cdot, t-k_{i+1})| \tag{5}
$$

$$
\Psi(\cdot, i) = \begin{cases} 1, & \text{if } D(\cdot, i) > \xi \\ 0, & \text{otherwise} \end{cases} \tag{6}
$$

**含义**: 通过相邻采样帧的像素差异检测运动区域，阈值 $\xi$ 过滤背景静态噪声。

**符号说明**:
- $I(\cdot, t-k_i)$: 时刻 $t-k_i$ 的图像（空间坐标用 $\cdot$ 简记）
- $\xi$: 运动检测阈值（超参数）
- $\Psi(\cdot, i) \in \{0, 1\}$: 二值运动掩码

---

### 公式 6: [[Channel-wise Temporal Encoding|衰减时序编码]] + [[Fibonacci Recurrent Inference|递归对齐]]

$$
H(\cdot, i) = \begin{cases} \tau, & \text{if } \Psi(\cdot, i) = 1 \\ \max(0,\, H(\cdot, i+1) - \delta), & \text{otherwise} \end{cases} \tag{7}
$$

$$
(t + k_{i-2}) - k_i = t - k_{i-1} \tag{8}
$$

**含义（公式 7）**: 运动区域赋予最大强度 $\tau$，非运动区域以衰减率 $\delta$ 递减，形成时序热度图（历史运动轨迹的持久性表示）。

**含义（公式 8）**: 斐波那契对齐等式，证明时间步推进 $L = k_{i-2}$ 后，前步骤历史帧精确对应当前步骤的采样点，使 [[KV Cache]] 特征直接复用成为可能。

**符号说明**:
- $\tau$: 最大运动强度值（超参数）
- $\delta$: 强度衰减率（超参数）
- $H(\cdot, i)$: 第 $i$ 时间档的时序热度编码

---

## 关键图表

### Figure 1: 动作平滑性与特征可视化

![Figure 1](https://arxiv.org/html/2607.29596v1/x1.png)

**说明**: 左侧展示不同方法生成的动作块平滑度对比（FibVLA 曲线更平滑连续）；右侧展示不同采样间隔下 VLA 提取的视觉特征分布，说明对数采样在各时间尺度均能捕捉有效信息。

---

### Figure 7: 实验平台总览

![Figure 7: 实验设置](https://arxiv.org/html/2607.29596v1/x7.png)

**说明**: 覆盖四个评测环境：LIBERO（仿真桌面操作）、MIKASA-Robo（瞬时视觉目标任务）、SimplerEnv（真实机器人数据泛化评测）、真实机器人 Piper 机械臂平台。

---

### Figure 6: 真实机器人分任务性能

![Figure 6: 真实世界性能对比](https://arxiv.org/html/2607.29596v1/x6.png)

**说明**: FibVLA 在 15 个子任务中平均得分 85.7（满分 10 分/子任务），领先 [[π0]] 达 11.4 分，在长时序挑战任务和错误恢复任务上优势尤为突出。

---

### Figure 8: 真实机器人实验场景

![Figure 8: Piper 机械臂设置](https://arxiv.org/html/2607.29596v1/assets/Figure6.png)

**说明**: Piper 机械臂真实世界实验场景，训练数据规模 60 万帧以上，覆盖 15 个操作任务。

---

### Figure 9: 任务执行关键步骤截图

![Figure 9a](https://arxiv.org/html/2607.29596v1/assets/Figure7.png)
![Figure 9b](https://arxiv.org/html/2607.29596v1/assets/Figure8.png)
![Figure 9c](https://arxiv.org/html/2607.29596v1/assets/Figure9.png)
![Figure 9d](https://arxiv.org/html/2607.29596v1/assets/Figure10.png)
![Figure 9e](https://arxiv.org/html/2607.29596v1/assets/Figure11.png)

**说明**: "Place Bowl (Test)" 任务的关键步骤快照，展示 FibVLA 在遮挡恢复和精细操作阶段的时序感知能力。

---

### Table 1: LIBERO 基准成功率 (%)

| 方法 | 平均 SR | Spatial | Object | Goal | Long |
|------|---------|---------|--------|------|------|
| Octo | 75.1 | 78.9 | 85.7 | 84.6 | 51.1 |
| OpenVLA | 76.5 | 84.7 | 88.4 | 79.2 | 53.7 |
| TraceVLA | 74.8 | 84.6 | 85.2 | 75.1 | 54.1 |
| SpatialVLA | 78.1 | 88.2 | 89.9 | 78.6 | 55.5 |
| 4D-VLA | 88.6 | 88.9 | 95.2 | 90.9 | 79.1 |
| CogACT | 93.5 | 97.2 | 98.0 | 90.2 | 88.8 |
| [[π0]] | 94.2 | 96.8 | 98.8 | 95.8 | 85.2 |
| **FibVLA** | **96.8** | **97.8** | **98.0** | **96.4** | **95.2** |

**关键发现**: LIBERO-Long（长时序任务）上 FibVLA 达到 95.2%，较第二名（[[π0]] 85.2%）高出 7.21%，验证了时序感知对长时序任务的决定性价值。

---

### Table 2: SimplerEnv-Fractal 成功率 (%)

| 方法 | VM 平均 | T1 | T2 | T3 | T4 | VA 平均 | T1 | T2 | T3 | T4 | 总体 |
|------|---------|----|----|----|----|---------|----|----|----|----|------|
| RT-1-X | 42.4 | 56.7 | 31.7 | 59.7 | 21.3 | 30.2 | 49.0 | 32.3 | 29.4 | 10.1 | 36.3 |
| RT-2-X | 46.3 | 78.7 | 77.9 | 25.0 | 3.7 | 54.4 | 82.3 | 79.2 | 35.5 | 20.6 | 50.4 |
| OpenVLA | 34.3 | 18.0 | 56.3 | 63.0 | 0.0 | 39.3 | 60.8 | 67.7 | 28.8 | 0.0 | 36.8 |
| Octo | 11.0 | 17.0 | 4.2 | 22.7 | 0.0 | 1.2 | 0.6 | 3.1 | 1.1 | 0.0 | 6.1 |
| [[π0]] | 69.1 | 88.0 | 80.3 | 56.0 | 52.2 | – | – | – | – | – | – |
| [[CogACT]] | 74.8 | 91.3 | 85.0 | 71.8 | 50.9 | 61.3 | 89.6 | 80.8 | 28.3 | 46.6 | 68.1 |
| [[TraceVLA]] | 45.8 | 45.0 | 63.8 | 63.1 | 11.1 | 49.8 | 64.3 | 60.6 | 61.6 | 12.5 | 47.8 |
| **FibVLA** | **78.6** | **89.0** | **86.7** | **86.5** | **52.2** | **65.5** | **82.1** | **77.9** | **43.6** | **58.3** | **72.1** |

**关键发现**: 总体 72.1%，较 [[CogACT]] 提升 5.87%。Visual Aggregation（VA）压力测试下仍保持 65.5% 均值，说明历史信息增强了视觉鲁棒性。

---

### Table 3: SimplerEnv-Bridge 成功率 (%)

| 方法 | 平均 SR | T1 | T2 | T3 | T4 |
|------|---------|----|----|----|----|
| RT-1-X | 1.1 | 0.0 | 4.2 | 0.0 | 0.0 |
| OpenVLA | 4.2 | 4.2 | 0.0 | 0.0 | 12.5 |
| Octo | 17.5 | 15.8 | 12.5 | 0.0 | 41.7 |
| [[π0]] | 55.7 | 63.3 | 58.8 | 21.3 | 79.2 |
| [[CogACT]] | 51.3 | 71.7 | 50.8 | 15.0 | 67.5 |
| [[TraceVLA]] | 27.7 | 12.5 | 16.6 | 16.6 | 65.0 |
| SpatialVLA | 42.7 | 16.7 | 25.0 | 29.2 | 100 |
| **FibVLA** | **67.3** | **75.0** | **65.4** | **35.2** | **93.7** |

**关键发现**: 平均 SR 67.3%，较 [[CogACT]] 提升 3.12%，较 [[π0]] 提升 20.9%，动态物体交互建模显著改善。

---

### Table 4: MIKASA-Robo 成功率 (%)

| 方法 | 平均 SR | T1 | T2 | T3 | T4 |
|------|---------|----|----|----|----|
| [[π0]] | 33.0 | 33.0 | 42.0 | 31.0 | 26.0 |
| SpatialVLA | 22.0 | 23.0 | 27.0 | 18.0 | 20.0 |
| OpenVLA-OFT | 26.5 | 48.0 | 14.0 | 27.0 | 17.0 |
| **FibVLA** | **46.5** | **78.0** | **37.0** | **36.0** | **35.0** |

**关键发现**: MIKASA-Robo 专门测试瞬时视觉目标理解，FibVLA 以 46.5% 领先 [[π0]] 41%，历史帧保留帮助模型在目标消失后仍能定位。

---

### Table 5: 消融实验

| 变体 | LIBERO-Long 平均 SR (%) | 真实世界平均得分 |
|------|------------------------|----------------|
| **FibVLA (完整)** | **95.2** | **85.7** |
| 去除对数采样 (w/o Sampling) | 88.4 | 78.3 |
| 去除 CTE 模块 (w/o CTE) | 91.2 | 80.0 |

**关键发现**: 对数采样是核心贡献（去掉后 LIBERO-Long 下降 6.8%）；CTE 提供约 4% 额外增益，两者协同缺一不可。

---

### Table 6: 推理效率对比

| 采样方法 | 推理时间 (ms/step) | LIBERO-Long SR (%) |
|----------|-------------------|-------------------|
| **FibVLA** | **177** | **95.2** |
| 对数采样（无递归复用） | 201 | 94.6 |
| Long-Short 采样 | 235 | 95.5 |
| [[TraceVLA]] | 196 | 54.1 |
| HiF-VLA | 243 | 94.4 |

**关键发现**: FibVLA 177ms 推理延迟比 HiF-VLA 降低 27.16%，比 [[TraceVLA]] 降低 9.69%，同时维持最高 LIBERO-Long 成功率，帕累托最优。

---

## 实验

### 数据集与评测平台

| 数据集 / 平台 | 规模 | 特点 | 用途 |
|--------------|------|------|------|
| [[LIBERO]] | 4 个子套件（130 任务） | 桌面操作仿真，含长时序子任务 | 主要仿真评测 |
| [[SimplerEnv]] (Fractal) | Google Fractal 数据 | 真实机器人数据的仿真重放 | 泛化能力评测 |
| [[SimplerEnv]] (Bridge) | BridgeV2 数据 | 动态物体操作 | 跨数据集泛化 |
| MIKASA-Robo | 4 任务 | 瞬时视觉目标，测试时序记忆 | 时序感知专项评测 |
| Piper 真实平台 | 60 万帧以上 | 15 个真实操作任务 | 真实世界验证 |

### 实现细节

- **基础模型**: PaliGemma (3B) + SigLIP 视觉编码器（参数冻结）
- **动作生成**: [[Flow Matching]] (与 [[π0]] 相同)
- **优化器**: AdamW（$\beta_1=0.9, \beta_2=0.95$），峰值学习率 $2.5 \times 10^{-5}$
- **梯度裁剪**: 1.0；学习率调度: Cosine Decay + 1000 步 Warmup
- **训练轮数**: LIBERO 30k 步，MIKASA 70k 步，SimplerEnv-Bridge 100k 步，SimplerEnv-Fractal 70k 步，真实世界 30k 步
- **Batch Size**: 16（大部分任务）/ 32（SimplerEnv-Fractal）

### 可视化结果

真实机器人实验（Figure 8/9）展示 FibVLA 在多步操作（放置碗任务）中的连贯执行能力，历史帧信息在抓取后的放置定位阶段提供关键空间记忆支持。

---

## 批判性思考

### 优点

1. **零冻结主干改动**: 所有改进均以插件形式加入，不需要重新训练 SigLIP / PaliGemma，迁移成本极低
2. **数学优雅性**: 斐波那契递推约束同时解决了离散化冗余（index collision）和推理效率（feature reuse）两个问题，设计统一
3. **全面评测**: 覆盖 4 个仿真基准 + 真实世界，消融实验明确定位各组件贡献

### 局限性

1. **超参数敏感**: $q_{min}$、$r$、$\xi$、$\tau$、$\delta$ 等超参数的选取缺乏理论指导，论文未详细分析敏感性
2. **单臂局限**: 真实世界实验仅使用 Piper 单臂平台，未验证双臂或人形机器人场景
3. **代码未开源**: 无法独立验证实现细节，尤其是 Fibonacci KV Cache 对齐的具体工程实现

### 潜在改进方向

1. **自适应 Fibonacci 参数**: 根据任务复杂度动态调整 $L$（动作块长度）与 $k_{i-2}$ 的绑定关系
2. **多摄像头时序融合**: 将 CTE 扩展至多视角输入（如 wrist + head camera），结合时序特征
3. **与 VLA-RL 结合**: 在线强化学习微调阶段保持 Fibonacci 推理效率约束

### 可复现性评估

- [ ] 代码开源
- [ ] 预训练模型
- [x] 训练细节完整（优化器、学习率、步数均有记录）
- [x] 数据集可获取（LIBERO、SimplerEnv 均为公开数据集）

---

## 关联笔记

### 基于

- [[π0]]: FibVLA 在 π₀ 的 flow matching 双流架构基础上添加时序模块，骨干 PaliGemma + SigLIP 完全沿用
- [[Action Chunking]]: FibVLA 输出为动作块，块长 $L$ 与 Fibonacci 索引绑定是递归推理的核心

### 对比

- [[CogACT]]: 主要仿真基准对比方，无时序历史建模
- [[TraceVLA]]: 时序 VLA 的代表，依赖大量离线预处理，推理效率低
- [[π0]]: 基础架构来源，FibVLA 在其上增加时序感知

### 方法相关

- [[Logarithmic Hindsight Sampling]]: 核心采样策略（本文提出）
- [[Channel-wise Temporal Encoding]]: 时序特征编码模块（本文提出）
- [[Fibonacci Recurrent Inference]]: 推理效率优化策略（本文提出）
- [[Flow Matching]]: 动作生成采用的生成模型框架
- [[KV Cache]]: 斐波那契递归推理的底层机制

### 硬件/数据相关

- [[LIBERO]]: 主要仿真评测基准
- [[SimplerEnv]]: 仿真到真实泛化评测平台

---

## 速查卡片

> [!summary] FibVLA: An Efficient Temporal Vision-Language-Action Model with Fibonacci Sampling
> - **核心**: 利用斐波那契序列对齐实现历史帧 KV Cache 复用，兼顾时序感知与推理效率
> - **方法**: 对数回顾采样 + CTE 通道时序编码 + Fibonacci 递归推理
> - **结果**: LIBERO-Long 95.2%（+7.21% over 2nd）；推理 177ms（-27% over HiF-VLA）；真实机器人 85.7/100
> - **代码**: 暂未开源

---

*笔记创建时间: 2026-08-04*
