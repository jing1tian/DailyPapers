---
title: "SmolVLA: A Vision-Language-Action Model for Affordable and Efficient Robotics"
method_name: "SmolVLA"
authors: [Mustafa Shukor, Dana Aubakirova, Francesco Capuano, Pepijn Kooijmans, Steven Palma, Adil Zouitine, Michel Aractingi, Caroline Pascal, Martino Russi, Andres Marafioti, Simon Alibert, Matthieu Cord, Thomas Wolf, Remi Cadene]
year: 2025
venue: arXiv
tags: [vla, vision-language-action, flow-matching, asynchronous-inference, efficient-robot-learning, imitation-learning, community-driven]
zotero_collection: Robotics/VLA
image_source: online
arxiv_html: https://arxiv.org/html/2506.01844
created: 2026-07-19
---

# 论文笔记：SmolVLA: A Vision-Language-Action Model for Affordable and Efficient Robotics

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Hugging Face, Sorbonne University, valeo.ai, École Normale Supérieure Paris-Saclay |
| 日期 | June 2025 |
| 项目主页 | https://github.com/huggingface/lerobot |
| 对比基线 | [[π₀]], [[OpenVLA]], [[ACT]], [[TinyVLA]] |
| 链接 | [arXiv](https://arxiv.org/abs/2506.01844) / [Code](https://github.com/huggingface/lerobot) |

---

## 一句话总结

> SmolVLA 是一个仅 0.45B 参数的轻量 [[Vision-Language-Action Model|VLA]] 模型，通过架构效率优化与异步推理栈，以 7-10 倍更小的体量匹敌甚至超越大型 VLA 的性能。

---

## 核心贡献

1. **轻量 VLA 架构**: 利用 [[SmolVLM2|SmolVLM-2]] 骨干、视觉 token 压缩（每帧 64 个）、VLM 层跳过（仅用前 L/2 层）和交错注意力机制，将模型压缩至 0.45B 参数，同时保持竞争力性能
2. **社区数据驱动预训练**: 在 481 个公开数据集的约 23k 个 episode 上预训练，数据量比 [[OpenVLA]] 少一个数量级，展示社区数据的高效利用
3. **异步推理栈**: 将感知与动作执行解耦为 PolicyServer-RobotClient 架构，实现约 30% 更快的任务完成速度和 2× 的任务吞吐量

---

## 问题背景

### 要解决的问题

大型 [[Vision-Language-Action Model|VLA]] 模型（如 7B 参数的 [[OpenVLA]]）训练和推理成本高，难以在消费级 GPU 上训练或实时部署，限制了社区参与和机器人落地。

### 现有方法的局限

- [[π₀]]（3.3B）、[[OpenVLA]]（7B）等需要数百至数千 GPU 小时训练
- 同步推理导致感知-计算-执行串行，存在延迟瓶颈
- 预训练数据多为私有大规模数据集，社区难以复现

### 本文的动机

若能通过精心的架构设计和推理优化，以极小的模型匹敌大型 VLA 的性能，则可大幅降低机器人 AI 的门槛，推动社区参与和快速迭代。

---

## 方法详解

### 模型架构

SmolVLA 采用 **VLM Backbone + Action Expert** 双组件架构：
- **输入**: 语言指令 $l$ + 多视角 RGB 图像 $o_t$（每帧压缩为 64 个视觉 token）+ 机器人关节状态 $s_t$（注入 VLM）
- **Backbone**: [[SmolVLM2|SmolVLM-2]]（[[SigLIP]] 视觉编码器 + [[SmolLM2]] 语言解码器），仅使用前 $N = L/2$ 层（层跳过）
- **核心模块**: [[Action Expert]] — [[Flow Matching|流匹配]]变换器，交错 [[Cross-Attention|交叉注意力]] 与因果自注意力
- **输出**: [[Action Chunking|动作块]] $\mathbf{A}_t \in \mathbb{R}^{n \times d_a}$（$n=50$ 步）
- **总参数**: 约 0.45B（VLM 骨干 ~350M + Action Expert ~100M）

### 核心模块

#### 模块 1: VLM Backbone 效率优化

**设计动机**: 在保留语义理解能力的前提下最大化计算效率

**具体实现**:
- **视觉 Token 压缩**: 使用 Pixel Shuffling（像素重排）将每帧图像压缩至 **64 个 token**，大幅减少 VLM 输入长度
- **层跳过（Layer Skipping）**: 仅使用 [[SmolVLM2|SmolVLM-2]] 的前 $N = L/2$ 层，丢弃后半部分层（消融实验证明 VLM 高层信息对动作预测贡献有限）
- **机器人状态注入**: 将 $s_t$（关节角度等本体感知）作为 token 输入 VLM，而非输入 [[Action Expert]]（消融：注入 VLM 比注入 Action Expert 提升 +7%）

#### 模块 2: Flow Matching Action Expert

**设计动机**: 利用 [[Flow Matching|流匹配]] 的强表达力，取代 [[Diffusion Policy|扩散策略]] 的多步去噪，实现高质量动作生成

**具体实现**:
- 交错 [[Cross-Attention|交叉注意力（CA）]] + 因果自注意力（SA）层：每两层 CA 之间插入一层 SA，既利用 VLM 观测特征，又建模动作序列内部依赖
- 因果掩码防止未来动作信息泄漏（消融：因果 SA vs 双向 SA = 74.5% vs 67.5%）
- 隐藏维度 $d_{expert} = 0.75 \times d_{VLM}$（略微压缩，约 100M 参数）
- 动作块大小 $n = 50$（消融最优范围 $n \in [10, 50]$）

#### 模块 3: 异步推理栈（[[AsyncInfer|Async Inference]]）

**设计动机**: 打破同步推理中「感知-预测-执行」的串行瓶颈，提升实际控制频率和任务吞吐量

**具体实现**:
- **PolicyServer**: 在 GPU（可远程）上运行，接收观测，输出动作块
- **RobotClient**: 在机器人端（可 CPU）上运行，维护动作队列，持续执行动作
- **阈值触发机制**: 动作队列剩余量降至比例阈值 $g \in [0, 1]$ 时向 PolicyServer 发起新的观测请求
- **关节空间相似度过滤**: 检测机器人状态相似度，避免状态未变化时的冗余 server 调用

---

## 关键公式

### 公式 1: [[Flow Matching|流匹配训练损失]]

$$
\mathcal{L}^{\tau}(\theta) = \mathbb{E}_{p(\mathbf{A}_t|\mathbf{o}_t),\, q(\mathbf{A}_t^{\tau}|\mathbf{A}_t)} \left[ \left\| \mathbf{v}_\theta(\mathbf{A}_t^{\tau}, \mathbf{o}_t) - \mathbf{u}(\mathbf{A}_t^{\tau}|\mathbf{A}_t) \right\|^2 \right]
$$

**含义**: 训练动作专家的速度场网络 $\mathbf{v}_\theta$，使其在带噪动作 $\mathbf{A}_t^{\tau}$ 处预测的速度接近真实条件速度场 $\mathbf{u}$

**符号说明**:
- $\mathbf{A}_t^{\tau}$: 在噪声时间步 $\tau$ 处的插值动作
- $\mathbf{v}_\theta$: 动作专家（参数化为 $\theta$）预测的速度场
- $\mathbf{u}(\mathbf{A}_t^{\tau}|\mathbf{A}_t)$: 条件速度场目标（见公式 3）
- $\mathbf{o}_t$: 当前观测经 VLM 编码后的特征

### 公式 2: [[Flow Matching|噪声动作插值]]

$$
\mathbf{A}_t^{\tau} = \tau \mathbf{A}_t + (1 - \tau)\boldsymbol{\varepsilon}, \quad \boldsymbol{\varepsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}), \quad \tau \sim \text{Beta}(a, b)
$$

**含义**: 在真实动作 $\mathbf{A}_t$ 与标准高斯噪声 $\boldsymbol{\varepsilon}$ 之间按比例 $\tau$ 线性插值生成训练样本；$\tau$ 从 Beta 分布采样以平衡简单/困难去噪任务

**符号说明**:
- $\tau \in [0, 1]$: 流匹配时间步（$\tau=1$ 为纯真实动作，$\tau=0$ 为纯噪声）
- $\boldsymbol{\varepsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$: 标准高斯噪声
- $\text{Beta}(a, b)$: 控制 $\tau$ 采样分布形状的超参数

### 公式 3: [[Flow Matching|条件速度场目标]]

$$
\mathbf{u}(\mathbf{A}_t^{\tau}|\mathbf{A}_t) = \boldsymbol{\varepsilon} - \mathbf{A}_t
$$

**含义**: 在插值点 $\mathbf{A}_t^{\tau}$ 处的条件流方向，即从真实动作 $\mathbf{A}_t$ 指向噪声 $\boldsymbol{\varepsilon}$ 的方向（推理时反向流动，从噪声恢复动作）

**符号说明**:
- $\boldsymbol{\varepsilon}$: 高斯噪声端点
- $\mathbf{A}_t$: 真实动作端点

### 公式 4: [[AsyncInfer|异步推理无空闲条件]]

$$
g \geq \frac{\mathbb{E}[\ell_S]}{\Delta t \cdot n}
$$

**含义**: 当动作队列触发阈值 $g$ 满足此不等式时，PolicyServer 推理完成前 RobotClient 的动作队列不会耗尽，从而消除机器人等待空闲

**符号说明**:
- $g \in [0, 1]$: 触发新观测请求时的队列剩余比例阈值
- $\mathbb{E}[\ell_S]$: PolicyServer 的平均推理延迟（期望）
- $\Delta t$: 控制周期（两次动作执行间隔）
- $n$: 动作块大小（chunk size）

---

## 关键图表

### Figure 1: SmolVLA 整体架构

![Figure 1](https://arxiv.org/html/2506.01844/x16.png)

**说明**: SmolVLA 的整体架构图。输入三类信息（语言指令、RGB 图像序列、机器人关节状态）经压缩后送入仅保留前 $L/2$ 层的 [[SmolVLM2|SmolVLM-2]] Backbone，VLM 输出的特征序列通过 [[Cross-Attention|交叉注意力]] 与 [[Action Expert]] 的交错注意力块相结合，最终输出 $n=50$ 步的动作块。

### Figure 2: 异步推理栈架构

![Figure 2](https://arxiv.org/html/2506.01844/x17.png)

**说明**: PolicyServer（GPU 端）与 RobotClient（机器人端）的解耦架构图。RobotClient 持续执行动作队列中的动作，当队列长度低于阈值 $g$ 时向 PolicyServer 发起新的推理请求，两者并行运行消除串行等待。

### Figure 3: 不同阈值下的动作队列演化

![Figure 3](https://arxiv.org/html/2506.01844/x18.png)

**说明**: 在不同 $g$ 值（0、0.7、1）下动作队列大小随时间的变化曲线，以及（A）无关节空间过滤 vs（B）启用过滤的对比。展示阈值和过滤策略对推理触发频率与队列稳定性的影响。

### Figure 4: 真实场景任务演示

![Figure 4](https://arxiv.org/html/2506.01844/x19.png)

**说明**: SmolVLA 在 [[SO-100]] 和 [[SO-101]] 机械臂上执行 4 种任务（Pick-Place、Stacking、Sorting、Pick-Place-Lego）的初始帧与终止帧对比，展示在真实硬件上的泛化能力。

### Figure 5: 同步 vs 异步推理性能对比

**说明**: SmolVLA 在三项真实任务上同步/异步推理的三维对比：(a) 成功率相近（异步略优）、(b) 异步任务完成时间缩短约 30%、(c) 固定 60 秒内异步可完成约 2× 的任务量。超参数在 Pick-Place 上优化后直接复用于其他任务。

### Table 1: LIBERO 仿真基准结果

| Method | Params | LIBERO-Spatial | LIBERO-Object | LIBERO-Goal | LIBERO-Long | Avg |
|--------|--------|---------------|--------------|------------|------------|-----|
| OpenVLA | 7B | — | — | — | — | 76.5% |
| π₀ | 3.3B | — | — | — | — | 86.0% |
| **SmolVLA** | **0.45B** | — | — | — | — | **87.3%** |

**关键发现**: SmolVLA 以 1/7 的参数量超越 [[π₀]]（+1.3%），以 1/15 的参数量超越 [[OpenVLA]]（+10.8%），证明紧凑架构设计能在仿真基准上达到 SOTA 级别性能。

### Table 2: Meta-World 仿真基准结果（50 任务）

| Method | Params | Easy | Medium | Hard | Very Hard | Avg |
|--------|--------|------|--------|------|-----------|-----|
| TinyVLA | 1B | — | — | — | — | 31.6% |
| π₀ | 3.5B | — | — | — | — | 47.9% |
| **SmolVLA** | **0.45B** | — | — | — | — | **57.3%** |

**关键发现**: SmolVLA 在跨 50 个任务的 [[Meta-World]] 基准上超越所有对比方法，比更大的 [[π₀]] 高 9.4 个百分点，比同为轻量化设计的 [[TinyVLA]] 高 25.7 个百分点。

### Table 3: 真实场景任务成功率（SO-100 机械臂）

| Task | SmolVLA | π₀ | ACT |
|------|---------|----|----|
| Pick-Place | 75% | **100%** | 70% |
| Stacking | **90%** | 40% | 50% |
| Sorting | **70%** | 45% | 25% |
| **Average** | **78.3%** | 61.7% | 48.3% |

**关键发现**: SmolVLA 平均成功率（78.3%）领先 [[π₀]]（61.7%）16.6 个百分点，在 Stacking 和 Sorting 任务上大幅超越；Pick-Place 单项因 π₀ 的机器人专用预训练而稍逊。

### Table 4: 异步 vs 同步推理对比（Pick-Place 任务）

| Mode | Success Rate | Avg Time | Tasks/60s |
|------|--------------|----------|-----------|
| Synchronous | 75% | 13.75s | 9 |
| **Asynchronous** | **80%** | **9.70s** | **19** |

**关键发现**: 异步推理在保持成功率的同时将平均完成时间缩短 29%（13.75s → 9.70s），固定时间内任务吞吐量翻倍（9 → 19 tasks/60s）。

### Table 5: 消融实验汇总（LIBERO 基准）

| 配置 | 成功率 | 对比说明 |
|------|--------|----------|
| **CA + SA 交错（最优）** | **85.5%** | 最终方案 |
| 仅 CA | 79.0% | 缺少动作序列内部建模 |
| 仅 SA（因果） | 74.5% | 缺少 VLM 观测注入 |
| 仅 SA（双向） | 67.5% | 额外的未来信息泄漏 |
| Standard hidden dim | **82.3%** | 标准隐层维度 |
| 0.75× hidden dim | 77.5% | 过度压缩损失容量 |
| Flow Matching 目标 | **80.25%** | 最终方案 |
| L1 Regression 目标 | 75.25% | 单峰假设限制表达力 |
| 状态注入 VLM | **80.3%** | 最终方案 |
| 状态注入 Action Expert | 73.3% | 与 VLM 特征解耦 |

**关键发现**: 交错 CA+SA 是最优注意力配置（+6.5% vs 仅 CA）；Flow Matching 显著优于 L1 回归（+5%）；机器人状态应注入 VLM 而非 Action Expert（+7%）。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| [[LIBERO]] | 40 任务 × 50 demos | 仿真，4 类任务子集 | 仿真评测 |
| [[Meta-World]] | 50 任务 | 仿真，4 级难度 | 仿真评测 |
| Community Datasets | 481 数据集，22,900 episodes，10.6M 帧 | HuggingFace 公开数据，VLM 自动标注 | 预训练 |
| SO100 Pick-Place | 自采集 | 真实 [[SO-100]] 机械臂 | 微调 + 测试 |
| SO100 Stacking | 自采集 | 真实 [[SO-100]] 机械臂 | 微调 + 测试 |
| SO100 Sorting | 自采集 | 真实 [[SO-100]] 机械臂 | 微调 + 测试 |
| SO101 Pick-Place-Lego | 自采集 | 真实 [[SO-101]] 机械臂 | 跨机器人泛化测试 |

### 数据预处理亮点

- **任务描述标注**: 使用 Qwen2.5-VL-3B 对社区数据集中缺失或嘈杂的任务描述进行 VLM 自动标注，统一语言格式
- **相机视角归一化**: 手动将各数据集的相机视角映射为标准化类型（top / wrist / side）

### 实现细节

- **Backbone**: [[SmolVLM2|SmolVLM-2]]（[[SigLIP]] 视觉编码器 + [[SmolLM2]] 语言解码器）
- **Action Expert**: 流匹配变换器，约 100M 参数，$d_{expert} = 0.75 \times d_{VLM}$
- **视觉 token 数**: 每帧 64 个（Pixel Shuffling 压缩）
- **VLM 层数**: 仅保留前 $N = L/2$ 层
- **动作块大小**: $n = 50$
- **预训练硬件**: 4 GPU，约 30k GPU 小时（比 [[π₀]] 快 40%）
- **微调**: 单 GPU 可微调

### 跨机器人泛化结果（SO-101）

| 场景 | SmolVLA | ACT |
|------|---------|-----|
| 分布内（In-distribution） | **90%** | 70% |
| 分布外（Out-of-distribution） | **50%** | 40% |

---

## 批判性思考

### 优点

1. **极高效率**: 0.45B 参数 + 单 GPU 可微调，大幅降低社区参与门槛
2. **异步推理设计新颖**: PolicyServer-RobotClient 解耦架构在实际机器人部署中具有重要工程价值，理论分析（队列无空闲条件）为超参选择提供指导
3. **全面消融**: 对注意力机制、状态注入位置、训练目标、动作块大小等关键设计决策均有详细消融验证
4. **全开源**: 代码、数据、权重全部发布至 [[LeRobot]]，可复现性强

### 局限性

1. **单机器人预训练**: 预训练数据主要来自 SO-100 形态，跨机器人泛化受形态差异限制
2. **数据规模有限**: ~23k episodes 远少于 [[OpenVLA]] 的 ~1M episodes，分布内泛化边界尚不明确
3. **VLM 骨干非机器人优化**: [[SmolVLM2|SmolVLM-2]] 主要为文档阅读/OCR 设计，缺乏机器人交互的视觉先验
4. **短时程任务为主**: 实验任务相对简单（Pick-Place/Stacking/Sorting），未验证长时程复杂任务
5. **纯模仿学习**: 未结合强化学习，分布外场景的鲁棒性提升空间有限

### 潜在改进方向

1. 在更多异构机器人数据上预训练，拓展跨机器人泛化能力
2. 结合在线 RL 微调（类 [[VLA-RL]] 方向）提升分布外鲁棒性
3. 使用机器人交互专用的 VLM 骨干（如 SpatialVLA 类方法）替换通用 SmolVLM-2

### 可复现性评估

- [x] 代码开源（LeRobot GitHub）
- [x] 预训练模型（Hugging Face Hub）
- [x] 训练细节完整（论文 + 代码均提供）
- [x] 数据集可获取（HuggingFace Hub 发布 4 个数据集）

---

## 关联笔记

### 基于
- [[π₀]]: 流匹配 VLA 的直接先驱，SmolVLA 在其基础上追求极致效率
- [[SmolVLM2|SmolVLM-2]]: VLM 骨干，提供视觉-语言理解能力
- [[Flow Matching]]: 核心训练目标，取代扩散策略的多步去噪

### 对比
- [[OpenVLA]]: 7B 参数基线，SmolVLA 以 1/15 参数超越
- [[π₀]]: 3.3B 基线，SmolVLA 以 1/7 参数匹敌
- [[TinyVLA]]: 1B 轻量 VLA 基线，SmolVLA 以更小参数显著超越
- [[ACT]]: 经典 [[模仿学习|imitation learning]] 基线

### 方法相关
- [[Action Expert]]: 动作专家模块设计
- [[Flow Matching]]: 流匹配训练目标
- [[Action Chunking]]: 动作块预测机制
- [[Cross-Attention]]: Action Expert 中的交叉注意力设计
- [[AsyncInfer|异步推理]]: 解耦感知与执行的核心推理框架

### 硬件/数据相关
- [[SO-100]]: 主要实验硬件（SO100 机械臂）
- [[SO-101]]: 跨机器人泛化实验硬件
- [[LeRobot]]: 开源机器人学习框架，SmolVLA 发布于此
- [[LIBERO]]: 仿真评测基准
- [[Meta-World]]: 仿真评测基准（50 任务）

---

## 速查卡片

> [!summary] SmolVLA
> - **核心**: 0.45B 参数的轻量 VLA，以 7-10× 更小体量媲美大型 VLA
> - **方法**: SmolVLM-2 骨干（前 L/2 层）+ 流匹配 Action Expert（交错 CA+SA）+ 异步推理栈
> - **结果**: LIBERO 87.3%（超越 π₀）、Meta-World 57.3%（超越 π₀ +9.4%）、真实场景 78.3%、异步推理 2× 吞吐量
> - **代码**: https://github.com/huggingface/lerobot

---

*笔记创建时间: 2026-07-19*
