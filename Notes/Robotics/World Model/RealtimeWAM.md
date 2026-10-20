---
title: "RealtimeWAM: One-Step Asynchronous World Action Models"
method_name: "RealtimeWAM"
authors: [Chengtao Lv, Jinyang Du, Shuyi Feng, Yang Yong, Shiqiao Gu, Shunzi Yang, Ruihao Gong, Shen Ren, Tianwei Zhang, Wenya Wang]
year: 2026
venue: arXiv
tags: [world-action-model, inference-efficiency, consistency-distillation, robot-manipulation, one-step-generation, asynchronous-inference, diffusion-policy]
zotero_collection: Robotics/World Model
image_source: mixed
arxiv_html: https://arxiv.org/html/2610.06617v2
created: 2026-10-20
---

# 论文笔记：RealtimeWAM: One-Step Asynchronous World Action Models

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Nanyang Technological University, Beihang University, SenseTime, Continental Automotive Singapore |
| 日期 | October 2026 |
| 项目主页 | [LightX2V / RealtimeWAM](https://github.com/ModelTC/LightX2V/tree/main/scripts/realtimewam) |
| 对比基线 | [[Fast-WAM]], [[Faster-WAM]], [[Flash-WAM]] |
| 链接 | [arXiv](https://arxiv.org/abs/2610.06617) / [Code](https://github.com/ModelTC/LightX2V/tree/main/scripts/realtimewam) |

---

## 一句话总结

> RealtimeWAM 通过 **Teacher-Anchored Consistency Distillation（TACD）** 实现一步动作生成，并通过 **Cross-Expert Wavefront Pipelining（CEWP）** 将视频专家与动作专家的执行流水线化，在 H100 上实现约 25× 推理加速（12.2 ms），同时仅损失不到 1% 的成功率。

---

## 核心贡献

1. **Teacher-Anchored Consistency Distillation（TACD）**: 在标准[[Consistency Distillation|一致性蒸馏（CD）]]基础上引入"教师端点锚定损失"$\mathcal{L}_{TA}$，以多步教师展开的 K 步均匀速度作为监督信号，解决 CD 局部误差与全局误差之间的"脱钩"问题，使单步生成质量接近多步基线。
2. **Cross-Expert Wavefront Pipelining（CEWP）**: 将[[Mixture-of-Transformers|MoT]] 架构中视频专家和动作专家的按块执行改为异步波前流水线，仅在动作注意力需要视频 KV 时才同步，大幅压缩两专家串行等待开销，且不改变任何参数或输出，无需重新训练。
3. **近乎无损的极致加速**: 在 LIBERO、LIBERO-Plus、RoboTwin 2.0 三大基准上，以 Fast-WAM 和 Faster-WAM 为基础模型，实现 H100 单次推理延迟降至 12.2 ms（约 25×），成功率损失 < 1%，超越 Flash-WAM 等所有现有一步蒸馏方案。

---

## 问题背景

### 要解决的问题

[[World Action Model|WAM（世界动作模型）]] 联合建模未来视频观测和机器人动作，天然具有扩散去噪的多步推理开销。即使采用[[Mixture-of-Transformers|MoT]] 设计的高效 WAM（如 Fast-WAM、Faster-WAM）将视频专家计算复用，每次推理仍需在 H100 上超过 300 ms（Fast-WAM: 299.7 ms），远不能满足实时控制的频率需求。

### 现有方法的局限

- **共享骨干 WAM**（DreamZero、Cosmos）：视频和动作 token 同步去噪，延迟极高（秒级）。
- **MoT-based WAM**（Fast-WAM、Faster-WAM）：视频专家只运行一次，但动作专家仍需多步去噪（10 步），且两个专家串行执行，视频结束后动作才开始。
- **Flash-WAM**：尝试一步 / 两步蒸馏，但在一步时成功率明显下降（RoboTwin: 81.41%，vs. Fast-WAM 的 91.51%），说明朴素 CD 存在全局误差累积问题。

### 本文的动机

作者识别出 MoT-based WAM 中两个独立瓶颈：

1. **Intra-expert iteration（专家内迭代）**：动作专家多步去噪 → 用一步蒸馏解决。
2. **Inter-expert waiting（专家间等待）**：视频专家运行完毕后动作专家才能启动 → 用异步流水线解决。

两者相互独立，可组合叠加。

---

## 方法详解

### 模型架构

RealtimeWAM 以 MoT-based WAM（如 [[Fast-WAM]] 或 [[Faster-WAM]]）为基础，核心架构不变：

- **Video Expert**: 对当前观测 $o$ 和语言指令 $l$ 进行一次前向，产生逐块 KV 缓存 $\{(K_i^v, V_i^v)\}_{i=1}^{N}$，供动作专家跨注意力使用。
- **Action Expert**: 消费视频 KV 对动作 Chunk 执行[[Flow Matching|流匹配]]去噪。
- **训练策略**: 冻结视频专家，仅对动作专家施加 [[LoRA]]（rank 128）微调。

### 核心模块

#### 模块1: Teacher-Anchored Consistency Distillation（TACD）

**设计动机**: 标准[[Consistency Distillation]]的目标是最小化局部一致性误差（相邻噪声级别之间），但这并不保证全局误差（到真实干净动作的距离）减小。论文以实验证明（Fig. 3）：CD 训练过程中，局部 CD 损失下降，而学生-教师平均速度距离（全局误差代理）几乎不下降。

**具体实现**:
1. 从相同的噪声动作出发，运行冻结的教师（K 步，默认 K=10），得到教师端点 $a_0^T$。
2. 计算教师均匀速度 $u_T$（从噪声动作到 $a_0^T$ 的直线速率）。
3. 以 $u_T$ 作为额外监督，驱使学生速度场与教师端点对齐。
4. 三个动作专家（学生、教师、EMA 目标）共享同一冻结的视频 KV 缓存，节省内存。

**注意**: K 步教师展开引入训练开销（5h 21min → 8h 3min on 16×H100），但推理时完全不需要教师，且峰值显存从 65.98 GiB 降至 24.73 GiB（视频专家冻结带来的收益）。

#### 模块2: Cross-Expert Wavefront Pipelining（CEWP）

**设计动机**: 在 MoT-based WAM 中，每个 Transformer Block 可分解为三个算子：QKV 投影（P）、注意力（Attn）、其余计算（R）。动作注意力（Attn_a）是唯一需要等待视频 KV 的操作，而动作投影（P_a）完全独立于视频。

**具体实现**:
- 视频流（CUDA Stream_v）：执行 $P_v^{(i)}$，记录 CUDA Event $e_i$，再执行 $\text{Attn}_v^{(i)}$ 和 $R_v^{(i)}$。
- 动作流（CUDA Stream_a）：立即开始 $P_a^{(i)}$，仅在 $\text{Attn}_a^{(i)}$ 之前 wait on $e_i$。
- 两条流在模型深度方向形成"波前"，P_a 与上层视频计算重叠，消除串行等待。

**等价性保证**: CEWP 仅重排执行顺序，使用完全相同的参数、输入、掩码和算子，数学输出与顺序执行完全一致，无需重新训练（Appendix D 严格证明）。

---

## 关键公式

### 公式1: [[Consistency Distillation|一致性蒸馏学生映射]]

$$
f_\theta(a_t, t) = a_t - t \cdot v_\theta(a_t, t)
$$

**含义**: 学生模型将任意噪声级别 $t$ 下的动作 $a_t$ 直接映射到干净动作端点，$v_\theta$ 为学习的速度场。

**符号说明**:
- $a_t$: $t$ 噪声级别下的含噪动作
- $t \sim \mathcal{U}(0, 1)$: 均匀采样的噪声时间步
- $v_\theta$: 学生动作专家速度场

### 公式2: [[Consistency Distillation|教师端点与均匀速度]]

$$
u_T(a_t, t) = \frac{a_t - a_0^T}{t}
$$

**含义**: 从噪声动作 $a_t$ 到教师 K 步展开端点 $a_0^T$ 的平均（直线）速率，作为学生速度场的全局锚定目标。

**符号说明**:
- $a_0^T$: 冻结教师从 $a_t$ 出发跑 K 步（默认 K=10）后得到的端点
- $u_T$: 教师均匀速度，代表"直奔终点"的理想方向

### 公式3: [[TACD|Teacher-Anchored 损失 $\mathcal{L}_{TA}$]]

$$
\mathcal{L}_{TA} = \mathbb{E} \left[ \left\| v_S(a_t, t) - \text{sg}[u_T(a_t, t)] \right\|^2 \right]
$$

**含义**: 驱使学生速度场 $v_S$ 与教师均匀速度 $u_T$ 对齐，直接约束全局误差，其中 $\text{sg}[\cdot]$ 表示停止梯度。

**符号说明**:
- $v_S$: 学生速度场
- $\text{sg}[\cdot]$: stop-gradient 操作，不对教师计算图回传梯度

### 公式4: [[TACD|总训练损失]]

$$
\mathcal{L} = \mathcal{L}_{CD} + \lambda \cdot \mathcal{L}_{TA}, \quad \lambda = 0.2
$$

**含义**: 标准一致性蒸馏损失与教师锚定损失的加权组合，$\lambda=0.2$ 为平衡超参数。

**符号说明**:
- $\mathcal{L}_{CD}$: 标准一致性蒸馏损失（局部一致性约束）
- $\lambda$: 权重系数，默认 0.2

### 公式5: [[误差分解|全局误差分解]]

$$
e_{\text{total}} = e_{\text{local}} + e_{\text{global}} = -t \cdot [v_S - u_T] + \delta
$$

$$
\mathbb{E}\|e_{\text{total}}\|^2 \leq 2 \cdot \mathcal{L}_{TA} + 2 \cdot \mathbb{E}\|\delta\|^2
$$

**含义**: 一步推理的总误差分解为局部误差（学生与教师速度的偏差，由 $\mathcal{L}_{TA}$ 控制）和全局误差（教师数值积分误差 $\delta$，随 K 增大以 $O(1/K^2)$ 收缩）。

**符号说明**:
- $e_{\text{local}} = -t[v_S - u_T]$: 学生速度场与教师均匀速度的偏差引起的局部误差
- $e_{\text{global}} = \delta$: 教师 K 步数值积分的截断误差
- $K$: 教师展开步数，默认 10

### 公式6: [[Video Expert|视频专家 KV 缓存]]

$$
z(o, l) = \{(K_i^v, V_i^v)\}_{i=1}^{N}
$$

**含义**: 视频专家对观测 $o$ 和语言 $l$ 一次前向，产生 N 个 Transformer 块的 Key-Value 缓存，供动作专家跨注意力复用。

**符号说明**:
- $o$: 当前视觉观测
- $l$: 语言指令
- $N$: Transformer 块数
- $(K_i^v, V_i^v)$: 第 $i$ 块的视频键值对

---

## 关键图表

### Figure 1: 推理延迟与组件分解

![[RealtimeWAM_fig1.png|600]]

**说明**: Fast-WAM 和 Faster-WAM 在 H100、RTX 4090D、RTX 5090 上的推理延迟及各组件（VAE、视频专家、动作专家）耗时分解。动作专家占动作生成延迟约 90%，是主要瓶颈。Fast-WAM 在 H100 上基线为 299.7 ms。

### Figure 2: TACD 方法概览

![Figure 2](https://arxiv.org/html/2610.06617v2/method1_new.png)

**说明**: [[TACD]] 的整体流程。冻结的视频专家提供共享 KV 缓存给三个动作专家（学生 S、教师 T、EMA 目标 T̄）。学生同时受局部 $\mathcal{L}_{CD}$（与 EMA 目标对齐）和全局 $\mathcal{L}_{TA}$（与教师端点均匀速度对齐）的双重约束。

### Figure 3: CD 全局-局部误差脱钩现象

![Figure 3](https://arxiv.org/html/2610.06617v2/loss_curves3.png)

**说明**: 标准 CD 训练曲线（左轴，CD 损失）与学生-教师平均速度距离（右轴，全局误差代理）的对比。CD 损失持续下降，但全局速度距离几乎不变，验证了"局部最优 ≠ 全局最优"的核心动机。

### Figure 4: 专家执行调度

**说明**: (a) 传统串行执行：视频专家全部完成 → 动作专家启动；(b) CEWP 波前执行：动作专家的投影（P_a）立即与视频专家并行，仅在注意力时同步，形成深度方向的波前流水线。

### Figure 5: RealtimeWAM 推理延迟

**说明**: TACD + CEWP + CUDA Graph 各组件对延迟的贡献拆解（H100 单卡）：Fast-WAM 基线 299.7 ms → TACD 后 65.8 ms → CEWP+CUDA Graph 后 17.4 ms → 全部优化后 12.2 ms（约 24.55×）；Faster-WAM 类似路径压缩至 16.1 ms（约 13.56×）。

### Table 1: 主要基准成功率对比（%）

| 方法 | NFE（视频/动作）| RoboTwin Clean | RoboTwin Random | RoboTwin Overall | LIBERO Overall |
|------|:-:|:-:|:-:|:-:|:-:|
| Motus | 10/10 | 88.66 | 87.02 | 87.84 | 97.7 |
| LingBot-VA | 25/50 | 92.90 | 91.50 | 92.20 | 98.5 |
| Fast-WAM | 1/10 | 91.82 | 91.19 | 91.51 | 97.0 |
| Faster-WAM | 1/10 | 93.20 | 92.66 | 92.93 | 98.9 |
| Flash-WAM | 1/2 | 88.42 | 82.66 | 85.54 | 96.1 |
| Flash-WAM | 1/1 | 82.56 | 80.26 | 81.41 | 95.1 |
| Light-WAM | 1/1 | 76.40 | 76.30 | 76.40 | 97.2 |
| Fast-WAM (1-step, baseline) | 1/1 | 70.42 | 68.12 | 69.27 | 95.0 |
| MeanFlow* | 1/1 | 85.74 | 85.72 | 85.73 | 96.9 |
| Consistency Distillation* | 1/1 | 89.88 | 89.38 | 89.63 | 96.9 |
| DMD* | 1/1 | 88.64 | 87.80 | 88.22 | 96.1 |
| **RealtimeWAM*** | **1/1** | **91.96** | **89.72** | **90.84** | **97.0** |
| **RealtimeWAM†** | **1/1** | **92.98** | **92.30** | **92.64** | **99.0** |

*基于 Fast-WAM；†基于 Faster-WAM

**关键发现**: RealtimeWAM† 仅比 10 步 Faster-WAM 低 0.29 个百分点（RoboTwin），在 LIBERO 上反超（99.0 vs 98.9）。RealtimeWAM* 仅比 10 步 Fast-WAM 低 0.67 个百分点。

### Table 2: LIBERO-Plus OOD 成功率（%）

| 方法 | 总体 | Camera | Noise | Layout | ... |
|------|:-:|:-:|:-:|:-:|:-:|
| Fast-WAM | 49.1 | — | — | — | — |
| Faster-WAM | 73.6 | 53.8 | 63.6 | 79.1 | — |
| **RealtimeWAM†** | **73.0** | **54.2** | **64.6** | **79.3** | — |

**关键发现**: RealtimeWAM† 在 Camera、Noise、Layout 三个 OOD 子集上略微超越 Faster-WAM（基础模型），说明 TACD 蒸馏不仅保持而且略微提升了 OOD 泛化。

### Table 3: 延迟分解（H100 单卡，ms）

| 优化阶段 | Fast-WAM | Faster-WAM | 加速比（FW） |
|----------|:-:|:-:|:-:|
| 基线（10 步） | 299.7 | 218.9 | 1× |
| + TACD（1 步） | 65.8 | 57.1 | ~4.6× |
| + CEWP + CUDA Graph | 17.4 | 22.4 | ~17.2× |
| + 全部优化 | **12.2** | **16.1** | **~24.6×** |

### Table 4: TACD 消融——损失与专家微调（RoboTwin Overall, %）

| 配置 | 成功率 |
|------|:-:|
| CD（视频专家也微调） | 89.63 |
| CD（视频专家冻结） | 89.85 |
| **TACD（视频专家冻结）** | **90.84** |

**关键发现**: 冻结视频专家本身带来 +0.22 提升，TACD 进一步带来 +0.99 提升。

### Table 5: 教师展开步数 K 消融（RoboTwin Overall, %）

| K | 成功率 |
|:-:|:-:|
| 5 | 90.39 |
| **10（默认）** | **90.84** |
| 20 | 90.77 |

**关键发现**: K=10 达到最优，K=20 几乎没有增益，验证教师误差已充分小。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| LIBERO | 4 子集（Spatial/Object/Goal/Long）| 桌面操作，标准仿真基准 | 训练+测试 |
| LIBERO-Plus | 7 OOD 扰动子集 | Camera / Noise / Layout 等分布外 | 测试（泛化） |
| RoboTwin 2.0 | 50 任务，Clean + Randomized | 双臂仿真，多样任务 | 训练+测试 |
| Real-world Garment Folding | 真实场景 | 双臂衣物折叠 | 定性验证（Appendix F.6）|

### 实现细节

- **Backbone**: Fast-WAM 或 Faster-WAM（视频专家冻结）
- **LoRA**: 仅动作专家，rank=128
- **优化器**: AdamW，学习率 1e-4，weight decay 1e-2
- **训练轮数**: 30,000 iterations
- **EMA decay**: 0.995（EMA 目标用于 CD 损失）
- **硬件**: 16 × H100（TACD 训练 8h 3min）
- **延迟测量**: 包含 VAE 编码，不含文本编码

### 可视化结果

- 在真实双臂衣物折叠任务（Appendix F.6）上，RealtimeWAM† 有效完成多步接触丰富任务，验证了实时性在真实场景的可行性。
- 在 RTX 4090D 上延迟约 12.2 ms 量级（具体见 Appendix F.5 Table F.3），RTX 5090 更低。

---

## 批判性思考

### 优点

1. **系统级思维**: 同时识别并解决"专家内迭代"和"专家间等待"两个独立瓶颈，叠加效果超过各自单独优化。
2. **理论扎实**: 对 TACD 给出误差分解和界（局部 + 全局），对 CEWP 给出等价性证明，方法有理论保障。
3. **基础模型通用**: TACD 和 CEWP 设计为后训练方法，理论上适用于任意 MoT-based WAM，扩展性好。
4. **显存节约**: 冻结视频专家 + LoRA 使峰值显存从 65.98 GiB 降至 24.73 GiB，实用价值高。

### 局限性

1. **仅适用于 MoT 架构**: CEWP 依赖视频/动作双专家分离的结构，不能用于共享骨干 WAM（DreamZero、Cosmos 等）。
2. **TACD 增加训练时长**: K 步教师展开使训练从 5h 21min 增至 8h 3min（约 +50%），规模扩大时开销更明显。
3. **教师质量依赖基础模型**: TACD 的效果上界由教师（10 步基础模型）决定，若基础模型本身误差大，蒸馏效果受限。
4. **实验范围**: 主要验证于仿真环境，真实世界仅有一个定性的衣物折叠任务，实机泛化性有待大规模验证。

### 潜在改进方向

1. 将 CEWP 思路推广至共享骨干（非 MoT）架构，探索更细粒度的跨层异步。
2. 减少 TACD 教师展开的训练开销，如通过缓存教师轨迹或在线蒸馏策略。
3. 在更多真实机器人平台（人形机器人、移动操作）上的大规模验证。

### 可复现性评估

- [x] 代码开源（LightX2V 仓库）
- [x] 预训练模型（checkpoints 在仓库中）
- [x] 训练细节完整（LoRA rank, lr, iterations, EMA 等均公开）
- [x] 数据集可获取（LIBERO、RoboTwin 均公开）

---

## 关联笔记

### 基于

- [[Fast-WAM]]: 主要基础模型之一，MoT-based WAM 代表
- [[Faster-WAM]]: 另一基础模型，性能更强的 MoT WAM
- [[Consistency Distillation]]: TACD 的核心蒸馏框架
- [[Flow Matching]]: 动作生成的概率框架

### 对比

- [[Flash-WAM]]: 最接近的竞争工作，也做 WAM 一步蒸馏，但全局误差问题未解决
- [[DMD]]: Distribution Matching Distillation，另一单步蒸馏基线
- [[MeanFlow]]: 均匀速度流匹配，作为蒸馏基线对比

### 方法相关

- [[Mixture-of-Transformers]]: MoT 架构，本文优化对象
- [[TACD]]: Teacher-Anchored Consistency Distillation，本文核心创新1
- [[CEWP]]: Cross-Expert Wavefront Pipelining，本文核心创新2
- [[LoRA]]: 用于仅微调动作专家
- [[CUDA Stream]]: CEWP 实现的底层并行原语
- [[Action Chunking]]: 动作块预测，WAM 输出形式

### 硬件/数据相关

- [[RoboTwin 2.0]]: 50 任务双臂仿真基准
- [[LIBERO]]: 桌面操作仿真基准

---

## 速查卡片

> [!summary] RealtimeWAM: One-Step Asynchronous World Action Models
> - **核心**: 两步加速 MoT-WAM：TACD（1步蒸馏）+ CEWP（异步流水线）
> - **方法**: Teacher-Anchored CD 解决全局误差；CUDA 流级波前并行消除跨专家等待
> - **结果**: H100 上 12.2 ms，约 25× 加速，RoboTwin 损失 < 1%，LIBERO 反超基线
> - **代码**: [LightX2V/RealtimeWAM](https://github.com/ModelTC/LightX2V/tree/main/scripts/realtimewam)

---

*笔记创建时间: 2026-10-20*
