---
title: "Riemann-1.0: An Embodied World Action Model for Physical AI"
method_name: "Riemann-1.0"
authors: [Haofeng Sun, Jiangbo Pei, Fei Kang, Zexiang Liu, Yaokun Li, Boyi Jiang, Hua Xue, Cindy Zhou, Wei Li, Yichen Wei, Mengyin An, Fanliang Zhao, Biao Jiang, Zile Wang, Yang Liu, Yangguang Li]
year: 2026
venue: arXiv
tags: [world-action-model, embodied-AI, causal-autoregressive, flow-matching, progressive-pretraining, latent-action-model, multi-embodiment, physical-AI]
zotero_collection: Robotics/World Model
image_source: online
arxiv_html: https://arxiv.org/html/2608.27033
created: 2026-08-29
---

# 论文笔记：Riemann-1.0: An Embodied World Action Model for Physical AI

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | （多机构合作，对应一线工业研究团队） |
| 日期 | August 2026 |
| 项目主页 | — |
| 对比基线 | [[ABot-M0.5]], [[Flash-WAM]], [[LingBot-VA]], [[GigaWorldPolicy0.5]], [[DreamWAM]] |
| 链接 | [arXiv](https://arxiv.org/abs/2608.27033) |

---

## 一句话总结

> Riemann-1.0 提出一个**全因果自回归世界动作模型（Fully Causal Autoregressive WAM）**，通过统一处理视觉观测、机器人状态和动作，以严格的因果顺序（动作先于视觉后果）联合建模可执行策略与动作条件视觉仿真，基于 230K+ 小时具身数据的渐进式预训练在 RoboTwin 2.0（94.3%）、LIBERO（99.0%）和 RoboCasa-365（62.6%）上实现全面 SOTA。

---

## 核心贡献

1. **全因果自回归 WAM 公式化（Fully Causal Autoregressive WAM）**: 严格遵循"动作先预测、视觉后生成"的因果序，避免 DreamZero 式的联合去噪造成的模态耦合问题，训练与推理分布高度一致。
2. **渐进式具身预训练框架（Progressive Embodied Pretraining）**: 三阶段课程——人类视频（LAM 伪动作）→ 手持夹爪演示（具身桥接）→ 机器人轨迹（高质量策略强化），$\lambda$ 从 0.1 渐进至 0.9，逐步从世界建模迁移到动作预测。
3. **多体态单一模型（Multi-Embodiment Single Model）**: 共享 Transformer 主干 + 体态专用线性投影（按体态 ID 索引）+ 每体态归一化，无需强制统一物理动作空间，同时支持人形、双臂、单臂、灵巧手等平台的策略执行与视觉仿真。
4. **大规模具身数据基础设施**: 构建 230K+ 小时具身经验数据集（200K+ 小时自我中心人类视频 + 12K+ 小时手持夹爪演示 + 20K+ 小时机器人轨迹），六阶段数据处理流水线含 3D 手部轨迹重建和语义感知平衡。

---

## 问题背景

### 要解决的问题

将[[WAM|世界动作模型（World Action Model）]]推广到物理 AI（Physical AI）面临两大核心挑战：

1. **策略与仿真的统一建模**: 现有方法要么只做策略（忽略世界建模带来的数据杠杆），要么做视频生成后再单独预测动作（解耦导致推理延迟和分布偏差）。
2. **大规模多体态预训练**: 不同机器人体态的动作空间、时间分辨率、坐标系差异巨大，如何在共享主干中高效利用异质具身数据是开放问题。

### 现有方法的局限

| 范式 | 代表方法 | 问题 |
|------|----------|------|
| 联合去噪（Joint Denoising） | DreamZero-Style | 动作与视觉维度/时序分辨率不匹配，模态耦合造成优化冲突 |
| 视频优先（Video-First） | LingBot-VA | 先生成完整视频再预测动作，推理延迟高、无法在线交互 |
| 解耦视频-动作 DiT | FastWAM | 两个独立 DiT 导致因果一致性缺失，动作-视觉条件关系是被动而非主动的 |
| 通用视频预训练迁移 | GIGA-World-Policy | 以通用视频生成目标塑造动态先验，与机器人具身动作监督不一致 |

### 本文的动机

真实机器人交互的因果顺序是：**执行动作 → 观测后果**，而不是反过来。Riemann-1.0 将这一物理约束内化到模型结构和训练目标中，通过严格的因果自回归公式使训练-推理过程天然一致，再通过渐进式预训练充分利用大规模人类视频的世界知识，逐步迁移到高精度机器人控制。

---

## 方法详解

### 模型架构

Riemann-1.0 以**全因果自回归序列**统一处理所有具身信号：

$$
[\text{context latent},\; \text{clean latent},\; \text{noisy latent},\; \text{state},\; \text{clean action},\; \text{noisy action}]
$$

- **视觉编码（Visual Encoding）**: 多视角帧经 Wan VAE 编码为潜在表示，再通过 3D patch embedding 转换为 Transformer tokens
- **动作接口（Action Interface）**: 体态专用标准化表示，通过每体态统计量归一化，以体态 ID 索引的线性投影输入/输出
- **状态处理（State Handling）**: 机器人状态嵌入为 token 但**不由模型生成**，直接作为条件信号注入（observed/injected）
- **主干（Backbone）**: 共享 Transformer + 体态专用输入/输出投影
- **Flow-Matching 头（Prediction Heads）**: 视觉潜变量和动作各有独立的 flow-matching 预测头

**注意力掩码（Attention Mask）**: 采用结构化注意力掩码而非无限制自注意力，维护训练和推理阶段的因果性。Clean tokens 只关注前序 clean tokens（因果掩码）；Noisy tokens 关注前序步骤的 clean tokens 以及同一帧内的 noisy tokens，防止信息泄漏同时保持 teacher forcing。

### 因果自回归公式

模型的核心公式化（Eq. 1）将联合分布分解为严格的因果乘积：

$$
p(a_{1:T},\, z_{1:T} \mid z_0, s_0, c) = \prod_{t=1}^{T} \underbrace{p(a_t \mid z_{<t},\, s_{<t},\, a_{<t},\, c)}_{\text{动作优先}} \cdot \underbrace{p(z_t \mid z_{<t},\, s_{<t},\, a_{\leq t},\, c)}_{\text{视觉后生成}}
$$

这一公式编码了**动作预测先于视觉潜变量生成**的基本因果序，与真实机器人交互中"执行动作才产生可观测后果"的物理现实严格对齐。

### Flow-Matching 目标

对于连续向量场，从噪声 $\epsilon \sim \mathcal{N}(0, I)$ 向数据样本输运，插值样本和目标速度定义为：

$$
x_\sigma = (1-\sigma)x + \sigma\epsilon, \quad v^\star = \epsilon - x
$$

**视觉潜变量损失（Eq. 2）:**

$$
\mathcal{L}_z = \mathbb{E}_{z,\epsilon,\sigma}\left[\left\|v_\theta^z\!\left(z_\sigma,\, \sigma,\, z_{<t},\, s_{<t},\, a_{\leq t},\, c\right) - (\epsilon_z - z)\right\|_2^2\right]
$$

**动作损失（Eq. 3）:**

$$
\mathcal{L}_a = \mathbb{E}_{a,\epsilon,\sigma}\left[\left\|v_\theta^a\!\left(a_\sigma,\, \sigma,\, z_{<t},\, s_{<t},\, a_{<t},\, c\right) - (\epsilon_a - a)\right\|_2^2\right]
$$

**联合目标（Eq. 4）:**

$$
\mathcal{L} = (1-\lambda)\mathcal{L}_z + \lambda\mathcal{L}_a
$$

其中 $\lambda$ 在训练各阶段渐进增大：$0.1 \to 0.5 \to 0.9$，逐步将重心从视觉动态建模迁移到动作预测精度。

### 潜在动作模型（LAM）

Stage I 使用冻结的 LAM 从无标注人类视频中提取伪动作，LAM 的训练目标为：

$$
\mathcal{L}_{LAM} = \left\|\hat{x}_{t+1} - x_{t+1}\right\|_2^2 + \beta\, D_{KL}\!\left(q_\phi(z_t \mid x_t, x_{t+1}) \;\|\; \mathcal{N}(0, I)\right)
$$

LAM 采用时空 Transformer 架构，具有 32 维潜在动作空间，利用下一帧重建损失和弱 KL 正则化自监督训练，无需任何动作标注。

### 渐进式具身预训练三阶段

| 阶段 | 数据 | $\lambda$ 权重 | 目的 |
|------|------|---------------|------|
| **Stage I: LAM-Action Bootstrap** | 200K+ 小时无标注自我中心人类视频（冻结 LAM 生成伪动作） | 0.1（强调视觉动态） | 初始化视觉动态主干；建立动作-运动-视觉关联 |
| **Stage II: Trajectory-Grounded Alignment** | UMI 手持夹爪演示（12K+ 小时）+ 机器人轨迹（20K+ 小时）+ 带 3D 手部位姿标注的人类视频 | 0.5（平衡动作与视觉） | 将动作-视频接口落地到真实轨迹监督；跨体态对齐 |
| **Stage III: Robot Policy Enhancement** | 仅高质量机器人演示（排除人类视频、LAM 伪动作、UMI、3D 标注人类数据） | 0.9（优先动作预测） | 精化可执行动作分布；减少训练-部署分布差距 |

### 数据处理六阶段流水线

1. 视觉归一化与语义标注
2. 基于 VLM 的层级任务/动作片段分解
3. 基于 MANO 建模 + VGGT-Ω 相机位姿估计的 3D 手部轨迹重建
4. 时序精炼与质量过滤
5. 动作标定与体态专用过滤
6. 跨多维度的语义感知数据均衡

### 在线自回归推理

1. 将首帧多视角帧编码为 $z_0$；嵌入初始状态 $s_0$
2. 在步骤 $t$：采样高斯噪声；利用 flow 调度器对 cached clean 历史 + 文本条件去噪
3. 执行预测的动作 chunk，持续预定的控制步数
4. 环境返回观测 → 编码为 $z_t$；追加到 cache 与 $s_t$
5. **策略部署模式**: 真实观测替换预测的潜变量
6. **仿真模式**: 预测的潜变量生成下一视频帧，递归反馈

Cache 维护固定窗口；满时重置，以最新观测帧作为新的 context frame。

---

## 关键公式

### 公式 1：因果自回归核心分解

$$
p(a_{1:T},\, z_{1:T} \mid z_0, s_0, c) = \prod_{t=1}^{T} p(a_t \mid z_{<t},\, s_{<t},\, a_{<t},\, c)\cdot p(z_t \mid z_{<t},\, s_{<t},\, a_{\leq t},\, c)
$$

**含义**: 将联合分布分解为严格因果乘积，动作预测先于视觉潜变量生成，与机器人物理交互的因果序完全一致。

**符号说明**:
- $a_{1:T}$：长度 $T$ 的动作序列
- $z_{1:T}$：长度 $T$ 的视觉潜变量序列
- $z_0, s_0$：初始观测和状态
- $c$：条件信息（语言指令等）

### 公式 2：视觉潜变量 Flow-Matching 损失

$$
\mathcal{L}_z = \mathbb{E}_{z,\epsilon,\sigma}\left[\left\|v_\theta^z\!\left(z_\sigma,\, \sigma,\, z_{<t},\, s_{<t},\, a_{\leq t},\, c\right) - (\epsilon_z - z)\right\|_2^2\right]
$$

**含义**: 以当前步动作 $a_{\leq t}$ 为条件预测视觉潜变量的速度场（即已执行的动作产生什么视觉后果）。

**符号说明**:
- $v_\theta^z$：视觉流速度场预测网络
- $z_\sigma = (1-\sigma)z + \sigma\epsilon$：噪声插值样本
- $\sigma \in [0,1]$：flow 时间步
- $\epsilon_z \sim \mathcal{N}(0, I)$：高斯噪声

### 公式 3：动作 Flow-Matching 损失

$$
\mathcal{L}_a = \mathbb{E}_{a,\epsilon,\sigma}\left[\left\|v_\theta^a\!\left(a_\sigma,\, \sigma,\, z_{<t},\, s_{<t},\, a_{<t},\, c\right) - (\epsilon_a - a)\right\|_2^2\right]
$$

**含义**: 仅以**历史**视觉观测和状态为条件预测当前动作速度场，不依赖未来视觉（因果约束）。

**符号说明**:
- $v_\theta^a$：动作流速度场预测网络
- $a_\sigma = (1-\sigma)a + \sigma\epsilon$：含噪动作

### 公式 4：渐进加权联合目标

$$
\mathcal{L} = (1-\lambda)\mathcal{L}_z + \lambda\mathcal{L}_a, \quad \lambda: 0.1 \to 0.5 \to 0.9
$$

**含义**: $\lambda$ 随训练阶段渐进增大，Stage I 重视觉动态学习，Stage III 重动作执行精度，实现知识的渐进迁移。

### 公式 5：LAM 自监督目标

$$
\mathcal{L}_{LAM} = \left\|\hat{x}_{t+1} - x_{t+1}\right\|_2^2 + \beta\, D_{KL}\!\left(q_\phi(z_t \mid x_t, x_{t+1}) \;\|\; \mathcal{N}(0, I)\right)
$$

**含义**: 通过下一帧重建（左项）和弱 KL 正则化（右项）无监督训练，从视觉帧对提取 32 维伪动作表示，无需任何动作标注。

**符号说明**:
- $\hat{x}_{t+1}$：基于 $z_t$ 重建的下一帧
- $z_t = q_\phi(z_t \mid x_t, x_{t+1})$：帧对编码的潜在动作
- $\beta$：KL 权重（弱正则化）

### 公式 6：Flow-Matching 插值与速度目标

$$
x_\sigma = (1-\sigma)x + \sigma\epsilon, \quad v^\star = \epsilon - x
$$

**含义**: 线性插值路径将数据 $x$ 与噪声 $\epsilon$ 联接，目标速度场为从 $x$ 指向 $\epsilon$ 的方向，flow-matching 拟合此速度场实现连续生成。

---

## 关键图表

### Figure 1: Riemann-1.0 系统概览

![Figure 1: Overview](https://arxiv.org/html/2608.27033/x1.png)

**说明**: 整体框架图，展示统一具身数据基础设施、渐进式具身预训练（Progressive Embodied Pretraining）三阶段和全因果世界动作模型（Fully Causal WAM）的整合关系。

### Figure 2: 数据处理六阶段流水线

![Figure 2: Data Pipeline](https://arxiv.org/html/2608.27033/x2.png)

**说明**: 从视觉预处理、语义标注、质量过滤、3D 手部重建、几何过滤到语义感知数据均衡的完整六阶段处理流程图。

### Figure 3: WAM 范式对比

![Figure 3: WAM Paradigm Comparison](https://arxiv.org/html/2608.27033/x3.png)

**说明**: 联合去噪、视频优先、视频-动作解耦 DiT、动作优先四种范式的可视化对比，以及各自权衡（trade-off）的示意图，直观呈现 Riemann-1.0 选择动作优先因果序的理由。

### Figure 4: 全因果视频-动作架构

![Figure 4: Architecture](https://arxiv.org/html/2608.27033/x4.png)

**说明**: Riemann-1.0 核心架构图，展示 Transformer 联合建模动作预测和视觉潜变量生成，以任务条件、机器人状态和历史观测为条件的结构化注意力机制。

### Figure 5: Teacher Forcing 注意力掩码

![Figure 5: Attention Mask](https://arxiv.org/html/2608.27033/x5.png)

**说明**: 结构化注意力掩码详解，显示帧 ID、时间 ID 和 clean/noisy 标志的组织方式，确保因果性同时支持 teacher forcing 训练。

### Figure 6: 自回归在线推理流程

![Figure 6: Online Inference](https://arxiv.org/html/2608.27033/x6.png)

**说明**: 在线自回归推理步骤示意图——高斯噪声采样、flow 去噪、cache 管理和与环境的交互循环，策略模式与仿真模式的切换。

### Figure 10: 多体态视觉 Rollout 可视化

![Figure 10: Multi-Embodiment Rollouts](https://arxiv.org/html/2608.27033/x10.png)

**说明**: 单一 Riemann-1.0 模型在多种机器人体态（人形、双臂、单臂、灵巧手）上生成动作条件视觉 rollout 的定性结果，展示统一世界动态先验的跨体态泛化能力。

---

## 实验结果

### 真实机器人操纵（Table 1）

| 任务 | 成功率 SR (%) | 过程成功率 PSR (%) |
|------|-------------|-----------------|
| Ordered Cube Stacking（有序立方块堆叠） | 85.0 | 91.6 |
| Kitchen Organization（厨房整理） | 90.0 | 98.4 |
| Clothes Folding（衣物折叠） | 85.0 | 92.5 |
| Desk Organization（桌面整理） | 80.0 | 95.2 |
| **平均** | **85.0** | **94.43** |

相比最强基线 G0.5，成功率提升 **+15%**。

### RoboCasa-365 仿真基准（Table 3）

| 设置 | Riemann-1.0 |
|------|-------------|
| Atomic-Seen | 74.2% |
| Composite-Seen | 56.0% |
| Composite-Unseen | 56.3% |
| **平均** | **62.6%** |

相比 ABot-M0.5 提升 **+8.4pp**，组合泛化（Composite-Unseen 56.3%）尤为突出。

### RoboTwin 2.0 仿真基准（Table 4）

| 环境 | 成功率 (%) |
|------|-----------|
| Clean（标准） | 94.6 |
| Randomized（随机化） | 94.0 |
| **平均** | **94.3** |

### LIBERO 仿真基准（Table 5）

| 子集 | 成功率 (%) |
|------|-----------|
| L-Spatial | 99.6 |
| L-Object | 100.0 |
| L-Goal | 97.6 |
| L-Long | 98.6 |
| **平均** | **99.0** |

### 组合泛化与 OOD（Table 2）

| 场景 | Riemann-1.0 |
|------|-------------|
| 立方块到碗/盘放置 | 80.0% |
| + 颜色-顺序约束 | 50.0% |
| 魔方到收纳盒 | 100.0% |
| 毛巾到盆放置 | 70.0% |
| **总体平均** | **75.0%** |

### 实验对比基准总览

| 基准 | 类型 | 核心指标 | Riemann-1.0 成绩 |
|------|------|---------|-----------------|
| 真实机器人（4 任务） | 物理机器人操纵 | SR / PSR | 85.0% / 94.43% |
| RoboCasa-365 | 移动操纵仿真 | 三类成功率平均 | 62.6% |
| RoboTwin 2.0 | 桌面操纵仿真 | Clean / Randomized | 94.3% |
| LIBERO | 桌面操纵仿真 | 四子集平均 | 99.0% |
| OOD 组合泛化 | 分布外泛化 | 多场景平均 | 75.0% |

---

## 批判性思考

### 优点

1. **因果性原则一致**: 全因果公式直接建立在物理交互的时序因果关系上，训练与推理过程天然对齐，避免了 DreamZero 式联合去噪的模态耦合和 LingBot-VA 式视频优先的推理延迟。
2. **数据杠杆极强**: 200K+ 小时人类视频通过 LAM 伪动作引导，实现大规模无标注数据复用，三阶段渐进式课程是工业级数据利用的典范。
3. **单模型多功能**: 同一参数在策略执行和视觉仿真之间无缝切换，具备显著工程价值；多体态支持无需单独训练体态专用模型。
4. **评估覆盖全面**: 真实机器人（4 任务）+ 三大仿真基准（RoboTwin/LIBERO/RoboCasa）+ OOD 泛化，结论鲁棒。

### 局限性

1. **组合泛化仍有缺口**: 加颜色-顺序约束后 OOD 性能从 80% 骤降至 50%，说明高层次规划推理（符号组合）能力仍然不足。
2. **LAM 伪动作质量上限**: Stage I 的伪动作来自冻结 LAM，其质量直接决定视觉动态初始化的上限，LAM 本身的泛化性未被深入分析。
3. **推理延迟未报告**: 相比解耦方法（FastWAM 等），因果自回归在线推理时每步需要完整去噪，频率受限问题未被定量讨论。
4. **多体态归一化复杂**: 体态专用归一化和投影在大量体态时参数量线性增加，可扩展性（100+ 体态）的分析缺失。

### 潜在改进方向

1. 将渐进 $\lambda$ 策略替换为自适应动态调权（如基于当前动作预测误差反馈），可能更快收敛。
2. 引入符号推理或层次任务表示缓解颜色-顺序等组合约束下的 OOD 性能下降。
3. 探索更轻量的体态适配机制（如 LoRA-style 适配层）以支持大量体态而不线性增加参数量。

### 可复现性评估

- [ ] 代码开源（未公开）
- [ ] 预训练模型（未公开）
- [x] 训练细节（论文中有完整描述）
- [x] 数据集（部分公开数据集：UMI、OXE 等）

---

## 关联笔记

### 基于

- [[FlowMatching]]: 全阶段统一的 flow-matching 训练目标
- [[HunyuanVideo]]: Wan VAE 视觉编码基础
- [[IRASim]]: 早期机器人视频世界模型工作

### 对比

- [[ABot-M0.5]]: RoboCasa-365 前 SOTA（54.2% 目标协议），Riemann-1.0 超越 +8.4pp
- [[Flash-WAM]]: 同期 WAM 方法，侧重推理加速
- [[LingBot-VA]]: 视频优先范式代表
- [[DreamWAM]]: 联合去噪范式代表

### 方法相关

- [[CFM]]: Conditional Flow Matching 训练目标
- [[LAM]]: Latent Action Model，Stage I 伪动作来源
- [[DiffusionForcing]]: 与因果自回归的对比基线训练策略
- [[teacher forcing]]: 结构化注意力掩码与 teacher forcing 的关系
- [[Wan 2.2]]: 视频扩散骨干参考

### 数据集相关

- [[RoboCasa365]]: 主要移动操纵评估基准
- [[RoboTwin 2.0]]: 桌面操纵仿真基准
- [[LIBERO-10]]: 四子集桌面操纵基准

---

## 速查卡片

> [!summary] Riemann-1.0: An Embodied World Action Model for Physical AI
> - **核心**: 全因果自回归 WAM，动作先预测、视觉后生成，训练-推理天然一致
> - **方法**: 渐进式三阶段预训练（$\lambda$: 0.1→0.5→0.9）+ 多体态共享主干 + 体态专用投影
> - **数据**: 230K+ 小时（200K 人类视频 + 12K UMI + 20K 机器人轨迹）
> - **结果**: RoboTwin 94.3%、LIBERO 99.0%、RoboCasa-365 62.6%、真实机器人 85.0%

---

*笔记创建时间: 2026-08-29*
