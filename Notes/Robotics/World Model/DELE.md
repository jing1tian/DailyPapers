---
title: "Inferring Action from Future Latent State for Robotic Manipulation"
method_name: "DELE"
authors: [Fenghao Lei, Zhixiong Huang, Long Yang, Jiabao Chen, Peilin Huang, Han Fu, Zhuo Li, Xiaoxue Ren]
year: 2026
venue: arXiv
tags: [world-model, robotic-manipulation, flow-matching, dual-stream, latent-prediction, VLA, action-chunking]
zotero_collection: Robotics/World Model
image_source: online
arxiv_html: https://arxiv.org/html/2608.22067v2
created: 2026-08-26
---

# 论文笔记：Inferring Action from Future Latent State for Robotic Manipulation

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Astribot |
| 日期 | August 2026 |
| 项目主页 | N/A |
| 对比基线 | [[GigaWorldPolicy0.5]]、[[pi0.5]]、GigaWorld-Policy-0.5、Xiaomi Robotics-0、Spirit-v1.5、GR00T-N1.7 |
| 链接 | [arXiv](https://arxiv.org/abs/2608.22067) / Code: N/A |

---

## 一句话总结

> DELE 是一个 [[World Action Model|World-Action Model]]，通过预测未来**紧凑潜变量状态**而非完整视频帧来推断机器人动作，在480次真实机器人试验中取得62.5%的全任务成功率，比最强基线高出32.5个百分点。

---

## 核心贡献

1. **潜变量未来状态预测**: 用单步未来潜变量 $O_{t+\Delta t}$ 替代连续视频生成，聚焦物理状态转移而非视觉重建，大幅降低冗余计算。
2. **双流 [[Flow Matching]] 架构**: 条件流与噪声流通过 Masked Attention 耦合，实现动作块与未来潜状态的联合预测，并共享同一[[Flow Matching]]目标。
3. **真实机器人基准评测**: 在 Astribot S1 双臂机器人上设计了4类长视界操纵任务（开门、取百事可乐、加冰、微波炉操作），提供细粒度阶段性进度指标。

---

## 问题背景

### 要解决的问题

[[Robotic Manipulation|机器人操纵]]中，策略需要从当前观测预测动作序列。长视界任务（如从冰箱取饮料再关门）要求模型理解物体的物理状态变化，仅靠当前帧无法捕捉充分的因果信息。

### 现有方法的局限

- **[[VLA|视觉-语言-动作模型 (VLA)]]** 直接从当前观测推断动作 $p_\theta(A_t \mid O_t, q_t, L)$，缺乏对未来物理状态的显式建模，难以泛化到长视界场景。
- **[[World Action Model|World-Action Model (WAM)]]** 通过预测完整视频轨迹 $p_\theta(A_t, O_{t:t+\Delta t} \mid O_t, q_t, L)$ 引入物理先验，但：
  - 生成 dense 视频帧带来巨大计算开销；
  - 中间帧重建引入无关于动作的冗余误差；
  - 视频生成质量并不直接对应操纵成功率。

### 本文的动机

机器人控制无需重建整个视觉演化过程——只需关注与后续动作因果相关的**物理状态**。DELE 将未来预测定义为**状态转移**问题，只预测单步关键未来潜变量 $O_{t+\Delta t}$，在保持物理约束的同时避免密集视频生成的代价。

---

## 方法详解

### 模型架构概览

DELE 的条件推断形式为：

$$
p_\theta(A_t, O_{t+\Delta t} \mid O_t, q_t, L)
$$

对比三类范式：

| 模型类型 | 推断形式 |
|---------|---------|
| VLA | $p_\theta(A_t \mid O_t, q_t, L)$ |
| WAM（视频生成） | $p_\theta(A_t, O_{t:t+\Delta t} \mid O_t, q_t, L)$ |
| **DELE（本文）** | $p_\theta(A_t, O_{t+\Delta t} \mid O_t, q_t, L)$ |

DELE 采用**双流（Dual-Stream）**架构，包含：
- **条件流（Condition Stream）**：处理语言指令与当前观测，使用单流注意力 + [[AdaLN]] 调制
- **噪声流（Noise Stream）**：处理带噪动作块与未来潜状态，使用 [[Flow Matching]] 目标
- 两流通过 **Masked Attention** 耦合，实现联合预测

**关键组件**:
- **视觉编码器**: [[DINO|DINO-v3]] 用于空间几何特征提取
- **语言编码器**: Qwen3 tokenizer
- **动作空间**: 7-DoF 单臂（3位置 + 3旋转 + 1夹爪），双臂14-DoF
- **动作块长度**: $H = 60$ 步

### 核心模块

#### 模块1：条件流（Condition Stream）

**设计动机**: 将语言指令和当前观测编码为条件上下文，通过 [[AdaLN]] 将扩散时间步 $\tau$ 注入特征，为噪声流提供稳定的条件信号。

**具体实现**:

语言与视觉特征编码并投影：

$$
\hat{\mathbf{l}} = \texttt{TextEncoder}(\mathbf{L}), \quad \hat{\mathbf{z}}_t = \texttt{VisionEncoder}(\mathbf{O}_t)
$$

$$
\mathbf{l} = \texttt{Projection}(\hat{\mathbf{l}}), \quad \mathbf{z}_t = \texttt{Projection}(\hat{\mathbf{z}}_t)
$$

拼接后进行单流注意力：

$$
\mathbf{Z}^c_t = \texttt{cat}(\mathbf{l}, \mathbf{z}_t), \quad \hat{\mathbf{Z}}^c_t = \texttt{Attention}(\mathbf{Z}^c_t)
$$

时间步嵌入（正弦编码）：

$$
\tilde{\mathbf{t}}[j] = \begin{cases}
\cos\!\left(\tau \cdot 10000^{-j/k}\right) & 0 \leq j < k \\
\sin\!\left(\tau \cdot 10000^{-(j-k)/k}\right) & k \leq j < 2k \\
0 & j = d-1 \text{ 且 } d \text{ 为奇数}
\end{cases}, \quad \mathbf{t} = \texttt{FNN}(\tilde{\mathbf{t}})
$$

AdaLN 缩放因子：

$$
\boldsymbol{\gamma}_c = 1 + \mathbf{W}_{\text{AdaLN}}\mathbf{t}_c + \mathbf{b}_{\text{AdaLN}}
$$

条件流 QKV 计算（附 QK-Norm 和 [[RMSNorm]]）：

$$
\mathbf{Q}_c = \text{QKNorm}\left\{\mathbf{W}_q(\text{RMSNorm}(\hat{\mathbf{Z}}^c_t) \odot \boldsymbol{\gamma}_c)\right\}
$$

$$
\mathbf{K}_c = \text{QKNorm}\left\{\mathbf{W}_k(\text{RMSNorm}(\hat{\mathbf{Z}}^c_t) \odot \boldsymbol{\gamma}_c)\right\}
$$

$$
\mathbf{V}_c = \mathbf{W}_v(\text{RMSNorm}(\hat{\mathbf{Z}}^c_t) \odot \boldsymbol{\gamma}_c)
$$

#### 模块2：噪声流（Noise Stream）

**设计动机**: 同时预测带噪动作块与未来潜状态，将两个预测任务耦合为更难的代理问题，迫使模型学习物理因果关系。

**具体实现**:

训练时对动作和未来视觉编码分别加噪：

$$
\mathbf{A}^{\tau}_t = \tau \mathbf{A}_t + (1 - \tau)\boldsymbol{\varepsilon}_a
$$

$$
\hat{\mathbf{z}}_{t+\Delta t} = \texttt{VisionEncoder}(\mathbf{O}_{t+\Delta t}), \quad \mathbf{z}^{\tau}_{t+\Delta t} = \tau \hat{\mathbf{z}}_{t+\Delta t} + (1 - \tau)\boldsymbol{\varepsilon}_o
$$

其中 $\tau \sim \mathcal{U}[0, 1]$，$\boldsymbol{\varepsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$。

两流通过 Masked Attention 联合预测：

$$
\mathbf{Q} = \texttt{cat}(\mathbf{Q}_c, \mathbf{Q}_n), \quad
\mathbf{K} = \texttt{cat}(\mathbf{K}_c, \mathbf{K}_n), \quad
\mathbf{V} = \texttt{cat}(\mathbf{V}_c, \mathbf{V}_n)
$$

$$
\hat{\mathbf{A}}^{\tau}_t,\ \hat{\mathbf{z}}^{\tau}_{t+\Delta t} = \texttt{Projection}\!\left(\texttt{MaskedAttention}(\mathbf{Q}, \mathbf{K}, \mathbf{V})\right)
$$

---

## 关键公式

### 公式1：[[Flow Matching]] 动作损失

$$
\mathcal{L}_a = \mathbb{E}_{\tau \sim \mathcal{U}[0,1],\, \boldsymbol{\varepsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})}
\left[\left\|\mathbf{u}_a\!\left(\mathbf{A}^{\tau}_t, \mathbf{z}^{\tau}_{t+\Delta t}, \mathbf{q}_t, \mathbf{l}\right) - \left(\mathbf{A}_t - \boldsymbol{\varepsilon}\right)\right\|^2_2\right]
$$

**含义**: 监督噪声流中的动作预测头，学习从带噪动作 $\mathbf{A}^{\tau}_t$ 还原真实速度场（即 $\mathbf{A}_t - \boldsymbol{\varepsilon}$）。

**符号说明**:
- $\tau \sim \mathcal{U}[0,1]$: 随机噪声比例（插值系数）
- $\mathbf{u}_a(\cdot)$: 动作速度场预测函数
- $\mathbf{A}_t$: 真实动作块
- $\boldsymbol{\varepsilon}$: 标准高斯噪声

### 公式2：未来潜状态损失

$$
\mathcal{L}_o = \mathbb{E}_{\tau \sim \mathcal{U}[0,1],\, \boldsymbol{\varepsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})}
\left[\left\|\mathbf{o}_a\!\left(\mathbf{A}^{\tau}_t, \mathbf{z}^{\tau}_{t+\Delta t}, \mathbf{q}_t, \mathbf{l}\right) - \left(\hat{\mathbf{z}}_{t+\Delta t} - \boldsymbol{\varepsilon}\right)\right\|^2_2\right]
$$

**含义**: 监督噪声流中的观测预测头，学习从带噪未来潜变量 $\mathbf{z}^{\tau}_{t+\Delta t}$ 还原真实未来潜变量速度场。

### 公式3：总训练目标

$$
\mathcal{L} = \mathcal{L}_a + \lambda \mathcal{L}_o
$$

**含义**: 联合优化动作预测与未来状态预测，$\lambda$ 平衡两个任务的权重。

**符号说明**:
- $\lambda$: 未来状态损失权重（论文中使用 $\lambda = 0.5$，因此命名为 DELE-w0.5）
- $\mathcal{L}_a$: 动作 flow matching 损失
- $\mathcal{L}_o$: 观测潜变量 flow matching 损失

---

## 关键图表

### Figure 1: 三种模型范式对比

![[DELE_fig1_paradigm_comparison.png]]

**说明**: (a) VLA 仅从当前观测直接推断动作；(b) WAM 预测完整视频轨迹再推断动作；(c) DELE 仅预测单步未来潜变量状态，去除冗余视觉演化，保留物理因果约束。

### Figure 2: DELE 双流架构全景

![[DELE_fig2_architecture.png]]

**说明**: 条件流（上）处理语言和当前图像特征，使用 [[AdaLN]] 将时间步嵌入注入；噪声流（下）同时处理带噪动作块和带噪未来潜变量；两流通过 Masked Attention 耦合，并行输出动作预测头和观测预测头。

### Figure 3: 训练与推理的注意力掩码

![[DELE_fig3_attention_masks.png]]

**说明**: 训练时，掩码阻止预测目标（未来潜变量、动作）之间的互相信息泄露；推理时，未来潜变量由噪声流逐步去噪生成，动作块同步解码。

### Figure 4: 四类操纵任务场景

![[DELE_fig4_task_scenes.png]]

**说明**: 实验平台 Astribot S1 双臂机器人上的四类任务关键阶段：(1) 开门（4阶段），(2) 取百事可乐（5阶段），(3) 加冰（6阶段），(4) 微波炉操作（5阶段）。

### Figure 5: 性能与效率综合对比

![[DELE_fig5_performance_efficiency.png]]

**说明**: (a) 进度 vs. 成功率气泡图，DELE 在右上角明显领先；(b) 成功率 vs. 推理延迟，DELE 以 87.5ms 延迟实现最高成功率；(c-f) 各任务进度与成功率柱状图详情。

### Figure 6: 阶段性长视界分析热力图

![[DELE_fig6_stage_analysis.png]]

**说明**: 每个任务各阶段到达概率热力图，展示模型在各子阶段的瓶颈分布。

### Table 1: 任务定义与评估标准

| 任务 | 阶段数 | 成功条件 |
|------|--------|---------|
| 开门 | 4 | 门开角 ≥45° 保持 ≥3秒 |
| 取百事可乐 | 5 | 取出饮料罐，冰箱门关闭 |
| 加冰 | 6 | 冰进入杯中，场景复原 |
| 微波炉操作 | 5 | 加热可见激活 |

### Table 2: 真实机器人性能对比（640 trials）

| 方法 | 开门 P/S | 取百事可乐 P/S | 加冰 P/S | 微波炉 P/S | 宏平均进度 | 全任务成功 |
|------|---------|-------------|--------|----------|----------|---------|
| **DELE-w0.5** | **95.0/80.0** | **82.0/65.0** | **63.3/45.0** | **85.0/60.0** | **81.3** | **62.5** |
| GigaWorld-Policy-0.5 | 80.0/45.0 | 60.0/25.0 | 50.0/20.0 | 55.0/15.0 | 61.3 | 26.3 |
| Xiaomi Robotics-0 | 77.5/45.0 | 67.0/30.0 | 41.7/25.0 | 52.0/20.0 | 59.5 | 30.0 |
| π₀.₅ | 80.0/45.0 | 42.0/5.0 | 32.5/0.0 | 48.0/10.0 | 50.6 | 15.0 |
| LingBot-VLA2 | ≤60.0 | ≤50.0 | ≤37.5 | ≤48.0 | ≤46.0 | ≤6.3 |
| Hy-Embodied-0.5-VLA | — | — | — | — | — | ≤20.0 |
| Spirit-v1.5 | — | — | — | — | — | ≤20.0 |
| GR00T-N1.7 | — | — | — | — | — | ≤20.0 |

*P = 归一化阶段进度(%)，S = 全任务成功率(%)；每任务各20次试验*

**关键发现**: DELE-w0.5 比第二名（Xiaomi Robotics-0）高出32.5 pp 成功率，比最强进度基线（GigaWorld-Policy-0.5）高出20.1 pp 宏平均进度。

---

## 实验

### 数据集 / 评估基准

| 评估项 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| 自建操纵基准 | 640 trials（4任务×20次×8对比方法） | 长视界、多阶段、细粒度进度 | 主要评测 |
| Astribot S1 | 双臂14-DoF机器人 | 工业级灵巧操作硬件 | 测试平台 |

### 实现细节

- **视觉编码器**: [[DINO|DINO-v3]]，提供空间几何表征
- **语言编码器**: Qwen3 tokenizer
- **优化器**: 未明确说明
- **动作块**: $H = 60$ 步前瞻
- **推理延迟**: 中位数 87.5ms（NVIDIA RTX 4090）
- **$\lambda$ 设置**: $\lambda = 0.5$（即 DELE-w0.5）
- **动作空间**: 7-DoF 单臂，14-DoF 双臂

### 消融与敏感性分析

- **$\lambda$ 命名约定**: 模型名称直接编码 $\lambda$ 值，DELE-w0.5 表示 $\lambda=0.5$，说明未来状态损失权重对性能存在影响。
- **掩码设计**: Masked Attention 防止训练时预测目标间的信息泄露，是双流耦合机制的关键。

### 可视化结果

- **涌现行为（Emergent Behaviors）**: 在微波炉任务中观察到模型自主重开已关闭微波炉门以继续完成任务的行为——此行为未在训练演示中出现，表明早期**目标条件行为重组**（goal-conditioned behavioral recomposition）的迹象。
- **失败分析主要瓶颈**:
  - 开门：旋转-推压的同步协调
  - 取百事可乐：右手到左手的交接及冰箱关门
  - 加冰：冰块获取与倾倒
  - 微波炉：插入、关门、启动加热

---

## 批判性思考

### 优点

1. **概念简洁优雅**: 用单步潜变量替代完整视频序列，理论动机清晰，实现简单但效果显著。
2. **效率与性能兼得**: 87.5ms 推理延迟在所有对比方法中具竞争力，同时取得最高操纵成功率。
3. **细粒度评估协议**: 阶段进度指标比单纯成功率更能揭示模型在长视界任务中的能力瓶颈，方法论贡献明确。
4. **涌现行为文档化**: 记录了非演示行为，为 [[World Action Model]] 的泛化能力提供了初步证据。

### 局限性

1. **未来状态预测的有效性未深入验证**: 未单独展示预测的 $\hat{\mathbf{z}}_{t+\Delta t}$ 质量如何，潜变量预测与动作质量之间的相关性缺乏直接分析。
2. **$\lambda$ 敏感性未系统消融**: 仅报告 $\lambda=0.5$ 的结果，不同 $\lambda$ 值对各任务的影响尚不清晰。
3. **单一硬件平台**: 所有实验在 Astribot S1 上进行，泛化到其他机器人平台的能力未验证。
4. **训练数据规模与来源未披露**: 未说明训练数据量、采集方式，复现存在障碍。

### 潜在改进方向

1. **多步潜变量预测**: 探索预测 $k > 1$ 步未来潜变量是否进一步提升长视界任务性能。
2. **自适应 $\lambda$**: 根据任务复杂度或训练进度动态调整观测损失权重。
3. **更丰富的潜变量空间**: 结合 3D/深度信息或多视角特征，增强未来状态预测的物理约束。

### 可复现性评估

- [ ] 代码开源
- [ ] 预训练模型
- [x] 训练细节（部分）完整
- [ ] 数据集可获取

---

## 关联笔记

### 基于

- [[Flow Matching]]: 核心生成框架，用于动作和观测的联合 flow matching 训练
- [[DINO]]: 视觉编码器，提供 DINO-v3 空间几何特征
- [[Action Chunking]]: 动作块预测范式，$H=60$ 步前瞻

### 对比

- [[GigaWorldPolicy0.5]]: 最强进度基线，宏平均进度 61.3% vs. DELE 81.3%
- [[pi0.5]]: 知名 VLA 基线，成功率仅 15.0%，远低于 DELE 的 62.5%

### 方法相关

- [[World Action Model]]: DELE 所属的模型范式
- [[VLA]]: 对比范式，缺乏未来状态建模
- [[AdaLN]]: 时间步嵌入注入机制
- [[Masked Attention]]: 双流耦合与信息防泄露机制

### 硬件/数据相关

- [[Astribot S1]]: 实验平台，双臂14-DoF灵巧操作机器人

---

## 速查卡片

> [!summary] DELE: Inferring Action from Future Latent State
> - **核心**: 预测单步未来潜变量状态而非完整视频，推断机器人动作
> - **方法**: 双流 Flow Matching（条件流 + 噪声流），Masked Attention 耦合，联合损失 $\mathcal{L} = \mathcal{L}_a + 0.5\mathcal{L}_o$
> - **结果**: 4类真实操纵任务，全任务成功率62.5%，比最强基线高32.5 pp，推理延迟87.5ms
> - **代码**: 未开源

---

*笔记创建时间: 2026-08-26*
