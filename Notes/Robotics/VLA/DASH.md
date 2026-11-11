---
title: "Decoupled Asynchronous System for Humanoids (DASH): Latency-Agnostic Scaling of Vision-Language-Action Models for Humanoid Manipulation"
method_name: "DASH"
authors: [Bingzhang Wang, Nathan Zhao, Jialeng Ni, Jiajin Zheng, Yin Zhang, Kehua Chen, Jian Yao]
year: 2026
venue: arXiv
tags: [vla, humanoid-robot, asynchronous-control, flow-matching, vision-language-model, latency-agnostic, decoupled-policy]
zotero_collection: Robotics/VLA
image_source: online
arxiv_html: https://arxiv.org/html/2611.11854v1
created: 2026-11-11
---

# 论文笔记：DASH — Decoupled Asynchronous System for Humanoids

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | XPENG Robotics, University of Washington |
| 日期 | November 2026 |
| 项目主页 | 未公开 |
| 对比基线 | Synchronous VLA inference |
| 链接 | [arXiv](https://arxiv.org/abs/2611.11854) / Code: 未公开 |

---

## 一句话总结

> DASH 将人形机器人策略解耦为慢速 [[VLM]] 语义感知（System 2）和快速 [[DiT]] 动作头（System 1），通过异步并发执行和去同步化训练，实现控制频率与骨干网络规模无关的 [[VLA]] 推理。

---

## 核心贡献

1. **去耦异步架构**: 将 [[VLA]] 策略拆分为慢速 [[VLM]] 进程（~5 Hz，System 2）和快速动作头进程（20 Hz，System 1），通过 [[LCM通信中间件|LCM]] 发布订阅总线非阻塞通信，控制频率完全独立于骨干网络大小。
2. **跨时间尺度融合（Cross-Timescale Fusion）**: 动作头的 [[Cross-Attention]] DiT 块同时关注缓存的 VLM 隐藏状态（陈旧语义上下文）和来自自身 [[ResNet]] 编码器的新鲜视觉 token，以冻结骨干网络 + 专用视觉编码器的配方平衡语义与时效性。
3. **去同步化训练（Desynchronized Training）**: 训练时随机将 VLM 分支的输入帧向前偏移最多 $\Delta_{\max}$ 步，模拟部署时的陈旧观测，无需将延迟作为显式输入即可让模型自适应推理时序漂移。
4. **自适应时序聚合（Adaptive Temporal Aggregation）**: 利用分数偏移量将重叠 [[Action Chunking|动作块]] 对齐并线性插值，消除定时器漂移带来的控制跳变。

---

## 问题背景

### 要解决的问题

在闭环机器人控制中，直接将大型 [[VLA]] 模型置于控制回路上会导致：控制频率随骨干网络规模增大而线性下降（4.1–9.7 Hz），机器人每次执行动作时依赖陈旧观测，制约了缩放到更大模型的实用性。

### 现有方法的局限

同步 [[VLA]] 推理（Synchronous VLA）将 [[VLM]] 骨干强制绑定在控制时钟上。骨干越大，$\delta_{\text{exec}}$ 越长，控制延迟越高，机器人实际执行时面向的是已过时的感知状态。在真实人形机器人实验中，同步基线仅达到 66.7% 的成功率。

### 本文的动机

认知科学的 [[System 1 Thinking|System 1]]/[[System 2 Thinking|System 2]] 二元框架启发了两速并发架构：慢速（System 2）高层语义感知与快速（System 1）低层运动控制无需互相等待，只需通过共享缓冲区传递语义上下文。如此，控制速率由动作头单独决定，与骨干网络大小解耦。

---

## 方法详解

### 模型架构

DASH 采用**双进程异步**架构：

- **输入**: 语言指令 $\ell_t$ + 观测 $o_t$ + 本体感知状态 $s_t$
- **System 2（VLM 进程）**: 冻结的 [[Qwen3-VL]] 骨干（0.8B–8B），产生语义上下文向量 $c_t$，~5 Hz 异步运行
- **System 1（动作头进程）**: 12 层 [[Cross-Attention]] [[DiT]]（隐藏维度 1024，8 注意力头），以 $K=4$ 步 [[Flow Matching]] 在 20 Hz 运行
- **新鲜视觉编码器**: [[ResNet]]-152，与 DiT 一同训练，提供当前时刻的新鲜视觉 token
- **通信总线**: [[LCM通信中间件|LCM]] 发布/订阅，非阻塞，单向缓冲最新 VLM 状态
- **输出**: 动作块 $a_{t:t+H}$

### 核心模块

#### 模块 1: 跨时间尺度融合（Cross-Timescale Fusion）

**设计动机**: 利用 [[Cross-Attention]] 实现陈旧语义上下文与新鲜视觉感知的信息融合，解决 VLM 输出延迟导致的语义过时问题。

**路由规则**: 将 DiT 的 $L_{\text{dit}}=12$ 层均匀映射到 VLM 的 $L_{\text{vlm}}$ 层（每 4 个 DiT 层对应 1 段 VLM 层），层级路由索引为：

$$
\phi(d) = \left\lceil \frac{\left\lfloor 4d / L_{\text{dit}} \right\rfloor + 1}{4}\, L_{\text{vlm}} \right\rceil, \quad d = 0, \dots, L_{\text{dit}}-1 \tag{6}
$$

**具体实现**:
- 每个 DiT 块的 Key 和 Value 由陈旧 VLM 隐藏状态与新鲜视觉 token 拼接而成
- 只有动作头和 ResNet 编码器接受训练，VLM 骨干冻结

#### 模块 2: 去同步化训练（Desynchronized Training）

**设计动机**: 训练与推理时 VLM 分支的输入时序存在系统性偏差（部署时 VLM 状态必然陈旧），通过训练时人工引入随机时序偏移来弥合分布差异。

**具体实现**:
- VLM 分支输入取早于动作分支 $\Delta$ 帧的观测
- $\Delta$ 从均匀分布随机采样，上界由实测部署延迟决定

#### 模块 3: 自适应时序聚合（Adaptive Temporal Aggregation）

**设计动机**: 利用分数偏移累积消除重叠 [[Action Chunking|动作块]] 在对齐时的漂移，替代固定偏移聚合器。

**具体实现**:
- 计算每个新到达动作块的时序分数偏移 $\rho_k$
- 通过累积分数决定整数跳步数 $n_k$ 和线性插值权重 $f_k$
- 对每个动作维度独立插值，实现亚步级时序对齐

---

## 关键公式

### 公式 1: [[VLA|VLA 策略定义]]

$$
a_{t:t+H} = \pi\big(o_t, s_t, \ell_t\big) \tag{1}
$$

**含义**: 完整 VLA 策略以当前观测 $o_t$、状态 $s_t$ 和语言指令 $\ell_t$ 为输入，输出未来 $H$ 步动作块。

**符号说明**:
- $a_{t:t+H}$: 从时刻 $t$ 起的 $H$ 步动作序列
- $o_t$: 当前视觉观测
- $s_t$: 本体感知状态
- $\ell_t$: 语言指令

---

### 公式 2: [[Asynchronous Inference|同步执行延迟]]

$$
\delta_{\text{exec}} = L(N), \quad \frac{\partial L}{\partial N} > 0 \tag{2}
$$

**含义**: 同步推理时，执行延迟 $\delta_{\text{exec}}$ 是骨干参数量 $N$ 的单调递增函数，大模型必然导致高延迟。

**符号说明**:
- $\delta_{\text{exec}}$: 单步推理执行延迟
- $N$: 骨干网络参数量
- $L(\cdot)$: 推理延迟函数

---

### 公式 3: [[Decoupled Policy|DASH 双流策略分解]]

$$
c_t = \pi_{\text{VLM}}\big(o_t, s_t, \ell_t\big), \quad a_{t:t+H} = \pi_{\text{action}}\big(s_t, o_t, c_{t-\Delta_{\text{VLM}}}\big) \tag{3}
$$

**含义**: DASH 将策略拆分为 VLM 语义流（产生上下文向量 $c_t$）和动作头流（使用陈旧语义上下文 $c_{t-\Delta_{\text{VLM}}}$ 产生动作）。

**符号说明**:
- $c_t$: VLM 在时刻 $t$ 产生的语义上下文向量
- $\Delta_{\text{VLM}}$: VLM 推理引入的延迟帧数
- $c_{t-\Delta_{\text{VLM}}}$: 动作头接收到的陈旧 VLM 上下文

---

### 公式 4: [[Asynchronous Inference|控制频率与骨干解耦]]

$$
f_{\text{ctrl}} = f_{\text{action}} \perp N \tag{5}
$$

**含义**: DASH 的控制频率 $f_{\text{ctrl}}$ 等于动作头频率 $f_{\text{action}}$，与骨干参数量 $N$ 完全独立（$\perp$）。

**符号说明**:
- $f_{\text{ctrl}}$: 实际控制频率
- $f_{\text{action}}$: 动作头运行频率（固定 20 Hz）
- $\perp$: 统计独立符号

---

### 公式 5: [[Cross-Timescale Fusion|DiT 层到 VLM 层的路由映射]]

$$
\phi(d) = \left\lceil \frac{\left\lfloor 4d / L_{\text{dit}} \right\rfloor + 1}{4}\, L_{\text{vlm}} \right\rceil, \quad d = 0, \dots, L_{\text{dit}}-1 \tag{6}
$$

**含义**: 将第 $d$ 个 DiT 层映射到应使用的 VLM 层索引 $\phi(d)$，实现均匀的跨层语义路由。

**符号说明**:
- $d$: DiT 层索引，$d \in \{0, \dots, L_{\text{dit}}-1\}$
- $L_{\text{dit}} = 12$: DiT 总层数
- $L_{\text{vlm}}$: VLM 总层数
- $\phi(d)$: 对应的 VLM 层索引

---

### 公式 6: [[Cross-Timescale Fusion|跨时间尺度融合 Key/Value 构造]]

$$
K_d = V_d = \Big[\, \underbrace{W_{\phi(d)} H^{(\phi(d))}_{t-\Delta t}}_{\text{陈旧语义上下文}} \;;\; \underbrace{W_v V^{\text{fresh}}_t}_{\text{新鲜视觉}} \,\Big] \tag{7}
$$

**含义**: 第 $d$ 个 DiT 块的 Key 和 Value 由路由到的 VLM 隐藏状态（陈旧）和 ResNet 编码的新鲜视觉 token（即时）拼接构成。

**符号说明**:
- $H^{(\phi(d))}_{t-\Delta t}$: 时刻 $t-\Delta t$ 的 VLM 第 $\phi(d)$ 层隐藏状态（陈旧）
- $V^{\text{fresh}}_t$: ResNet-152 在时刻 $t$ 编码的新鲜视觉 token
- $W_{\phi(d)}, W_v$: 可学习投影矩阵
- $[\,;\,]$: 序列维度拼接

---

### 公式 7: [[DiT|DiT 跨注意力块前向]]

$$
Z^{\text{sa}}_{d+1} = \mathrm{FFN}\big(\mathrm{CrossAttn}(Q = Z^{\text{sa}}_d,\, K_d,\, V_d)\big) \tag{8}
$$

**含义**: DiT 第 $d$ 层以动作序列隐状态 $Z^{\text{sa}}_d$ 为 Query，以拼接后的陈旧+新鲜上下文为 Key/Value 进行跨注意力，再经 FFN 更新隐状态。

**符号说明**:
- $Z^{\text{sa}}_d$: 第 $d$ 层的动作隐状态
- $Q$: Query（来自动作序列）
- $K_d, V_d$: Key/Value（陈旧 VLM + 新鲜视觉，见公式 7）

---

### 公式 8: [[Desynchronized Training|去同步化训练输入采样]]

$$
\big(o^{\text{vlm}}, s^{\text{vlm}}\big) = \big(o_{t-\Delta}, s_{t-\Delta}\big), \quad \big(o^{\text{act}}, s^{\text{act}}\big) = \big(o_t, s_t\big) \tag{9}
$$

**含义**: 训练时 VLM 分支使用早于动作分支 $\Delta$ 帧的观测，模拟部署时 VLM 输出必然陈旧的情况。

**符号说明**:
- $(o^{\text{vlm}}, s^{\text{vlm}})$: VLM 分支的训练输入（取自早 $\Delta$ 帧）
- $(o^{\text{act}}, s^{\text{act}})$: 动作头的训练输入（取自当前帧 $t$）
- $\Delta$: 人工引入的帧偏移量

---

### 公式 9: [[Desynchronized Training|帧偏移均匀采样]]

$$
\Delta \sim \mathcal{U}\{0, 1, \ldots, \Delta_{\max}\}, \quad \Delta_{\max} = \left\lceil \frac{\Delta^{\max}_{\text{obs}}}{\tau_{\text{data}}} \right\rceil \tag{10}
$$

**含义**: 帧偏移量从均匀离散分布中采样，上界由实测最大部署延迟 $\Delta^{\max}_{\text{obs}}$ 除以数据采集间隔 $\tau_{\text{data}}$ 确定。

**符号说明**:
- $\Delta_{\max}$: 最大允许帧偏移量（帧数）
- $\Delta^{\max}_{\text{obs}}$: 实测最大部署延迟（秒）
- $\tau_{\text{data}}$: 训练数据的采样时间间隔（秒）

---

### 公式 10: [[Adaptive Temporal Aggregation|自适应时序对齐——分数偏移]]

$$
\rho_k = \frac{T_k}{T_{\text{exec}}} \tag{11}
$$

**含义**: 第 $k$ 个动作块到达时，计算其到达间隔 $T_k$ 相对于执行周期 $T_{\text{exec}}$ 的分数偏移 $\rho_k$，用于亚步级对齐。

**符号说明**:
- $T_k$: 第 $k$ 个动作块相对于上一个的到达间隔
- $T_{\text{exec}}$: 单步动作执行周期（$= 1/f_{\text{action}}$，即 50 ms）
- $\rho_k$: 分数偏移（通常接近 1）

---

### 公式 11: [[Adaptive Temporal Aggregation|整数跳步与线性插值系数]]

$$
\begin{aligned}
n_k &= \Big\lfloor \sum_{j \le k} \rho_j \Big\rfloor - \Big\lfloor \sum_{j < k} \rho_j \Big\rfloor \\
f_k &= \Big\{ \sum_{j \le k} \rho_j \Big\}
\end{aligned}
\tag{12}
$$

**含义**: $n_k$ 为第 $k$ 次更新时动作块应跳过的整数步数，$f_k$ 为线性插值权重（分数部分），防止累积漂移。

**符号说明**:
- $n_k$: 整数跳步数（决定使用动作块中的第几步动作）
- $f_k$: 线性插值系数（分数部分，$\in [0, 1)$）
- $\lfloor \cdot \rfloor$: 下取整；$\{\cdot\}$: 取分数部分

---

### 公式 12: [[Adaptive Temporal Aggregation|线性插值执行动作]]

$$
\tilde{a} = (1 - f_k)\, a_i + f_k\, a_{i+1}
$$

**含义**: 对每个动作维度，在连续两步动作 $a_i$ 和 $a_{i+1}$ 之间以 $f_k$ 为权重线性插值，生成平滑过渡的执行动作 $\tilde{a}$。

**符号说明**:
- $\tilde{a}$: 插值后的执行动作
- $a_i, a_{i+1}$: 动作块中第 $i$ 步和第 $i+1$ 步的动作
- $f_k$: 见公式 11

---

## 关键图表

### Figure 1: Teaser / 真实人形机器人执行任务

![Figure 1 Teaser](https://arxiv.org/html/2611.11854v1/figures/fig_teaser.png)

**说明**: DASH 在真实双灵巧手人形机器人上执行语言条件化抓放任务，以等时间间隔展示 6 帧。展示了 [[VLA]] 在真实机器人上的端到端部署效果。

---

### Figure 2: Prediction Uncertainty / 动作块预测不确定性

**说明**: (a) 预测不确定性随 [[Action Chunking|动作块]] 时间步的推移而增大；(b) 频繁更新动作块可将执行保持在可靠的早期部分，减小相邻块之间的分歧。这是 DASH 在高频控制下保持稳定性的理论依据。

> 注：此图在 arXiv HTML 中无可用图片链接，原始图片请参阅 [PDF](https://arxiv.org/pdf/2611.11854)。

---

### Figure 3: DASH Architecture / 系统架构总览

![Figure 3 DASH System Overview](https://arxiv.org/html/2611.11854v1/DASH_System_Overview.png)

**说明**: DASH 的整体架构。慢速 [[VLM]] 进程（[[System 2 Thinking|System 2]]）和快速动作头进程（[[System 1 Thinking|System 1]]）以及环境进程并发运行，通过 [[LCM通信中间件|LCM]] 总线非阻塞通信。动作头的 [[Cross-Attention]] [[DiT]] 同时接收缓存的 VLM 隐藏状态和 [[ResNet]]-152 编码的新鲜视觉 token。

---

### Figure 4: Desynchronized Training / 去同步化训练数据

![Figure 4 Desynchronized Training](https://arxiv.org/html/2611.11854v1/DASH_Async_Training.png)

**说明**: 展示去同步化训练的数据采样方式。VLM 视觉通道和动作头视觉通道取自不同时刻的帧（VLM 取自早 $\Delta$ 帧），目标动作锚定在动作头的时间戳。这使训练分布与部署时的 VLM 状态陈旧性相匹配。

---

### Figure 5: Adaptive Temporal Aggregation / 自适应时序聚合

**说明**: 展示 [[Adaptive Temporal Aggregation|自适应时序聚合]] 的工作原理。分数偏移被拆分为整数跳步数和线性插值权重，残余分数随时间累积，确保长期运行不会产生漂移。对比固定偏移聚合器，自适应方案能更准确地对齐重叠动作块。

> 注：此图在 arXiv HTML 中无可用图片链接，原始图片请参阅 [PDF](https://arxiv.org/pdf/2611.11854)。

---

### Figure 6: Backbone Scaling Results / 骨干缩放成功率对比

**说明**: 对比四种骨干规模（0.8B/2B/4B/8B）下同步推理、无去同步化训练的异步推理（Async w/o Desync）和完整 DASH 三种方案的成功率。可见去同步化训练对 4B 模型带来 +7.2 pp 增益，对 0.8B 仅 +3.7 pp，说明骨干越大、推理延迟越高时，去同步化训练的效果越显著。

> 注：此图在 arXiv HTML 中无可用图片链接，原始图片请参阅 [PDF](https://arxiv.org/pdf/2611.11854)。

---

### Figure 7: Simulation Qualitative / 仿真定性对比

![Figure 7 Simulation Qualitative](https://arxiv.org/html/2611.11854v1/figures/fig_sim_qualitative.png)

**说明**: 在 [[RoboTwin]] 仿真环境中，以等仿真时间间隔展示同步推理（上）和 DASH（下）的匹配场景对比。同步推理因延迟导致动作与当前状态不匹配，DASH 凭借 20 Hz 控制频率保持更流畅、更准确的执行轨迹。

---

### Figure 8: Real Robot / 真实机器人抓放

![Figure 8 Real Robot](https://arxiv.org/html/2611.11854v1/figures/fig_real_robot.png)

**说明**: 展示 DASH 在真实人形机器人（双灵巧手）上执行三条语言指令的抓放任务，每条指令展示 6 帧。真实机器人实验中 DASH 达到 95.8% 成功率（23/24），远超同步基线的 66.7%（16/24）。

---

### Table I: 骨干缩放——同步 vs. DASH（50 任务，250 次试验）

| 骨干规模 | 同步延迟 (ms) | DASH 延迟 (ms) | 同步频率 (Hz) | DASH 频率 (Hz) | 加速比 | 同步 SR (%) | Async w/o Desync SR (%) | DASH SR (%) | ΔDASH (pp) |
|---------|-------------|--------------|------------|--------------|-------|-----------|----------------------|-----------|-----------|
| 0.8B | 103 | 50 | 9.7 | 20 | 2.1× | 64.5 | 61.8 | 65.5 | +1.0 |
| 2B | 132 | 50 | 7.6 | 20 | 2.6× | 66.4 | 65.5 | 69.1 | +2.7 |
| 4B | 183 | 50 | 5.5 | 20 | 3.6× | 63.6 | 62.3 | 69.5 | +5.9 |
| 8B | 243 | 50 | 4.1 | 20 | 4.9× | 62.7 | 62.7 | 65.9 | +3.2 |

**关键发现**: DASH 在所有骨干规模下保持固定 50 ms 动作路径，实现 20 Hz 控制频率；去同步化训练对延迟最大的 8B 骨干（+3.2 pp）和 4B（+5.9 pp）提升最为显著。

---

### Table II: 推理延迟测量（RTX 4090）

| 组件 | 延迟 (ms) |
|------|---------|
| VLM Qwen3.5-0.8B | 53 |
| VLM Qwen3-VL-2B | 82 |
| VLM Qwen3-VL-4B | 133 |
| VLM Qwen3-VL-8B | 193 |
| 图像预处理 | 15 |
| 动作头（ResNet-152 + 4步 Flow-Matching DiT）| 35 |

**关键发现**: 动作头固定占用 35 ms（加图像预处理共 50 ms），与骨干规模无关；VLM 延迟从 53 ms 到 193 ms 随规模线性增长，同步推理时这将直接降低控制频率。

---

### Table III: 真实机器人抓放（每配置 24 次试验）

| 方法 | 成功次数 | 成功率 (%) |
|------|---------|----------|
| 同步推理 + 固定偏移聚合器 | 16/24 | 66.7 |
| **DASH + 自适应聚合器（本文）** | **23/24** | **95.8** |

**关键发现**: DASH 相比同步基线提升 29.1 pp 成功率，[[Adaptive Temporal Aggregation|自适应时序聚合]] 对于真实机器人的精确抓放至关重要。

---

### Table IV: 配方消融实验（4B 骨干，异步推理，50 任务，250 次试验）

| 配方 | 成功率 (%) | 说明 |
|-----|---------|------|
| 全量微调，无额外编码器 | 62.6 | 基线：整个 VLM 参与训练，仅用 VLM 视觉 |
| 冻结 VLM，无额外编码器 | 55.0 | 语义冻结但无新鲜视觉：性能最差 |
| **冻结 VLM + ResNet-152（DASH）** | **69.5** | 冻结语义 + 新鲜视觉：性能最佳 |

**关键发现**: 新鲜视觉编码器对冻结骨干配方至关重要（+14.5 pp vs. 冻结无编码器）；且冻结+编码器方案超越全量微调（+6.9 pp），说明保留 VLM 预训练特征有利于泛化。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| [[RoboTwin]] | 50 任务 × 250 轮 | [[ALOHA]] 机身，仿真器评测 | 仿真训练与测试 |
| 真实人形机器人 | 24 次试验/配置 | 双灵巧手，语言条件化抓放 | 真实部署评测 |

### 实现细节

- **骨干**: [[Qwen3-VL]]-2B/4B/8B（冻结），Qwen3.5-0.8B（冻结）
- **动作头**: 12 层 [[Cross-Attention]] [[DiT]]，$D=1024$，8 注意力头，$K=4$ 步 [[Flow Matching]] 推理
- **视觉编码器**: [[ResNet]]-152，与 DiT 一同训练
- **通信框架**: [[LCM通信中间件|LCM]] 发布/订阅，三进程并发（环境、VLM 服务器、动作服务器）
- **控制频率**: 20 Hz（动作路径 50 ms）；VLM 异步运行，频率取决于骨干大小（4.1–19 Hz）
- **硬件**: RTX 4090，480×640 分辨率
- **仿真**: [[RoboTwin]]，[[ALOHA]] 机身，延迟在仿真时间内计算
- **真实机器人**: 双灵巧手人形机器人，24 次试验/配置
- 训练超参数（优化器、学习率、batch size、训练步数）未在论文中明确说明

### 可视化结果

DASH 在仿真对比中（Fig 7）可见同步推理执行路径抖动明显，DASH 执行轨迹更平滑连续；真实机器人（Fig 8）中 DASH 展示了对三种不同语言指令的准确抓放执行能力。

---

## 批判性思考

### 优点

1. **解耦思路清晰**: 借鉴认知科学 [[System 1 Thinking|System 1]]/[[System 2 Thinking|System 2]] 框架，将控制频率与骨干规模解耦，思路自然且易于扩展。
2. **去同步化训练技巧实用**: 仅需在训练时随机偏移 VLM 分支输入帧，无需修改模型结构或引入额外输入，工程实现简单。
3. **自适应聚合解决实际问题**: 相比固定偏移聚合器，自适应聚合器对真实硬件定时抖动具有较强鲁棒性，真实机器人成功率提升 29.1 pp。
4. **全面的消融实验**: 表 IV 明确揭示了冻结骨干 + 新鲜视觉编码器的配方优于全量微调，提供了有价值的工程经验。

### 局限性

1. **任务仅为准静态**: 论文实验任务均为准静态抓放，未涉及高速动态操作或长时序任务，实际价值有待验证。
2. **训练细节不充分**: 优化器、学习率、batch size、数据集大小、训练步数等关键实现细节未披露，可复现性受限。
3. **硬件依赖强**: 仅在 RTX 4090 上验证，不同 GPU 和嵌入式平台的性能未知，$\Delta_{\max}$ 的设置与硬件强相关。
4. **无代码/数据集开源**: 缺乏开源实现，限制了社区复现和基准对比。

### 潜在改进方向

1. 扩展到高速动态任务（如投掷、接球）和长时序任务，验证 DASH 在更大时序跨度下的稳健性。
2. 探索 VLM 分支自适应频率控制（如感知变化大时主动加速）以进一步降低语义陈旧度。
3. 结合强化学习在线微调动作头，利用真实机器人交互数据持续改进。

### 可复现性评估

- [ ] 代码开源
- [ ] 预训练模型
- [ ] 训练细节完整（优化器/学习率/数据集等未披露）
- [ ] 数据集可获取（依赖内部数据集）

---

## 关联笔记

### 基于

- [[VLA]]: DASH 是对 VLA 推理延迟问题的系统性解决方案
- [[Action Chunking]]: DASH 采用动作块输出，并通过自适应聚合处理块间对齐
- [[Flow Matching]]: 动作头使用 4 步 flow-matching DiT 生成动作
- [[System 1 Thinking]] / [[System 2 Thinking]]: DASH 双速架构的认知科学灵感来源

### 对比

- Synchronous VLA 推理: DASH 的直接对比基线，成功率低 29.1 pp（真实机器人）
- [[Asynchronous Inference]]: DASH 属于异步推理的具体实现，增加了去同步化训练和自适应聚合

### 方法相关

- [[DiT]]: 动作头的核心架构组件（12 层 Cross-Attention DiT）
- [[ResNet]]: 动作头的新鲜视觉编码器（ResNet-152）
- [[Cross-Attention]]: DiT 中陈旧语义与新鲜视觉融合的核心操作
- [[Qwen3-VL]]: DASH 使用的 VLM 骨干系列

### 硬件/数据相关

- [[ALOHA]]: 仿真实验使用的机器人机身（RoboTwin 环境）
- [[RoboTwin]]: 仿真实验平台（50 任务，250 轮/配置）

---

## 速查卡片

> [!summary] DASH: Latency-Agnostic Scaling of VLA for Humanoids
> - **核心**: 解耦慢速 VLM（System 2）和快速 DiT 动作头（System 1），控制频率固定 20 Hz 与骨干规模无关
> - **方法**: Cross-Timescale Fusion + 去同步化训练 + 自适应时序聚合，LCM 进程间通信
> - **结果**: 真实人形机器人抓放成功率 95.8%（vs 同步基线 66.7%），仿真最大 +5.9 pp（4B backbone）
> - **代码**: 未开源

---

*笔记创建时间: 2026-11-11*
