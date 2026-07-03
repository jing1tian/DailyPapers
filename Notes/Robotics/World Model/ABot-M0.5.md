---
title: "ABot-M0.5: Unified Mobility-and-Manipulation World Action Model"
method_name: "ABot-M0.5"
authors: [Ronghan Chen, Yandan Yang, Zuojin Tang, Dongjie Huo, Tong Lin, Haoning Wu, Haoyun Liu, Yuzhi Chen, Lulu Zheng, Botai Yuan, Tianlun Li, Mingxin Wang, Dekang Qi, Bin Hu, Wei Mei, Yuze Xuan, Haolong Yang, Yanqing Zhu, Mu Xu, Zhiheng Ma, Xinyuan Chang]
year: 2026
venue: arXiv
tags: [world-action-model, mobile-manipulation, latent-action, dream-forcing, mixture-of-transformers, flow-matching, video-prediction]
zotero_collection: Robotics/World Model
image_source: online
arxiv_html: https://arxiv.org/html/2607.00678v1
created: 2026-07-03
---

# 论文笔记：ABot-M0.5: Unified Mobility-and-Manipulation World Action Model

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | AMAP CV Lab（阿里巴巴高德 CV 实验室） |
| 日期 | July 2026 |
| 项目主页 | — |
| 对比基线 | [[Flash-WAM]], [[LingBot-VA]], [[GR00T-N1.5]], [[Qwen-RobotManip]] |
| 链接 | [arXiv](https://arxiv.org/abs/2607.00678) / [Code](https://github.com/amap-cvlab/ABot-Manipulation) |

---

## 一句话总结

> ABot-M0.5 通过引入中间潜在动作、双层混合专家 Transformer 和 Dream Forcing 训练策略，系统性解决 WAM 在移动操纵中的三大结构失配，在 RoboCasa365 上以 46.6% 成功率创造 SOTA。

---

## 核心贡献

1. **中间潜在动作（Intermediate Latent Action）**: 在粗粒度视频预测和细粒度机器人控制之间引入帧级桥接表示 $m_t$，通过 [[ALAM]] 从视觉转换中提取，解决时间粒度失配问题。
2. **双层混合 Transformer（Dual-Level MoT）**: 在模态层（视频/潜在动作/可执行动作三流分离）和动作层（移动 vs 操纵子空间解耦）两个层次实施分离，防止梯度干扰，同时通过共享注意力保持联合推理。
3. **Dream Forcing 训练对齐**: 采用两阶段前向传播，训练时用模型自身预测的 "梦境" 潜变量替代真值条件，直接模拟推理时的信号分布，消除 [[teacher forcing|Teacher Forcing]] 导致的 exposure bias。

---

## 问题背景

### 要解决的问题

将 [[WAM|世界动作模型（World Action Model）]] 应用于移动操纵（Mobile Manipulation）时存在三个根本性的结构失配，导致长时程任务失败率高、细粒度操纵控制不足：

1. **时间粒度失配（Temporal Granularity Mismatch）**: 世界模型以压缩的时序块运行，机器人动作需要帧级生成，细粒度行为（抓取闭合、接触起始）在粗粒度预测中消失。
2. **动作结构失配（Action Structure Mismatch）**: 低频底盘移动和高频末端执行器操纵被强行放入同一动作空间，造成优化冲突，单一纠缠空间无法专门化。
3. **训练-推理分布失配（Rollout Condition Mismatch）**: 训练用真实未来帧作条件，推理用模型预测帧，暴露偏差在长时程 rollout 中累积放大。

### 现有方法的局限

- **反应式策略（Reactive Policy）**: 仅预测动作序列 $a_{t:t+H-1}$，不建模世界演化，缺乏对长时程组合任务的规划能力。
- **普通 WAM**：联合预测视频和动作，但未解决上述三个结构失配，在 RoboCasa365 等复杂移动操纵基准上性能有限（如 GR00T-N1.5 仅 23.9%）。
- **Diffusion Forcing**：训练时对不同时间步加不同程度噪声，推理时采用零噪声——但训练条件和推理条件仍存在分布差异。

### 本文的动机

移动操纵任务本质上要求长时程（导航到目标 + 精细操纵）和高精度（毫米级抓取）的统一，现有 WAM 方法未从结构上解决三大失配。ABot-M0.5 通过分级表示（$z \to m \to a$）、分离优化和训练-推理对齐来系统性解决这三个问题。

---

## 方法详解

### 模型架构

ABot-M0.5 建立在 [[Wan 2.2|Wan2.2]] 5B 视频扩散骨干之上，采用 **[[MoT架构|Dual-Level Mixture-of-Transformers]]** 架构：

- **输入**: 语言指令 $l$（[[UMT5]] 编码）+ 多视角观测 $o_{\leq t}$（3D [[Causal VAE]] 压缩为视频潜变量 $z$）+ 历史动作 $a_{<t}$
- **Backbone**: [[Wan 2.2|Wan2.2 5B]] 视频扩散模型（预训练初始化）
- **核心模块**: [[ALAM]] 提取帧级潜在动作 $m_t$；Dual-Level [[Mixture-of-Transformers]] 进行三流联合去噪
- **输出**: 未来视频潜变量 $\hat{z}_{t+1}$、帧级潜在动作 $\hat{m}_t$、可执行动作 $\hat{a}_t$（移动 $a^{move}$ + 操纵 $a^{manip}$）
- **统一训练目标**: [[CFM|Conditional Flow Matching]]，贯穿三个生成阶段

三阶段级联（Asymmetric Cascade）：

```
视频潜变量 z_{t+1} → 帧级潜在动作 m_t → 可执行动作 a_t
```

信息流的不对称设计：未来视频流被屏蔽，不允许关注动作流（因果约束），但动作流可以关注视频流（条件依赖）。

### 核心模块

#### 模块一：中间潜在动作（ALAM）

**设计动机**: [[Latent-Action|潜在动作]] 将帧级运动意图编码为紧凑向量，弥合粗粒度视频块与细粒度机器人控制之间的语义鸿沟，还可从无动作标注的纯视频数据集中学习。

**具体实现**:
- 冻结编码器 $E_m$ 从相邻帧 $(I_t, I_{t+1})$ 提取 $m_t \in \mathbb{R}^{d_m}$
- 多视角潜在动作聚合为时空张量 $M$
- 用 [[CFM]] 目标训练（见公式 Eq. 9）
- 辅助损失：加法一致性（$m_i^k = m_i^j + m_j^k$）和反转一致性（$m_i^j + m_j^i = 0$）

#### 模块二：双层 Mixture-of-Transformers（Dual-Level MoT）

**设计动机**: 单一 Transformer 难以同时处理视频、潜在动作和可执行动作三种异构模态，且移动和操纵动作的梯度方向往往相互冲突。

**模态层解耦**：
- 为每个流（视频/潜在动作/可执行动作）设置独立的输入投影、时间步嵌入、输出头
- 三流 token 序列：$\mathbf{X}_t = [\mathbf{X}_{t+1}^z, \mathbf{X}_t^m, \mathbf{X}_t^a]$

**动作层解耦**：
- 移动动作 $a^{move}$ 和操纵动作 $a^{manip}$ 各有专用前馈网络（FFN）
- 共享自注意力中两个子空间保持跨子空间信息流（cross-subspace conditioning）
- 以噪声化对方动作为条件进行联合去噪，防止优化冲突

#### 模块三：Dream Forcing

**设计动机**: 解决 [[teacher forcing|Teacher Forcing]] 和 [[Diffusion Forcing|Diffusion Forcing]] 都存在的训练-测试分布差异（exposure bias），在训练阶段直接模拟推理时的条件分布。

**两阶段前向传播**:
- **Phase A**: 执行速度场预测，得到干净的梦境潜变量 $\hat{z}_{t+1}$ 和 $\hat{m}_t$
- **Phase B**: 以 Phase A 的梦境输出为条件，进行动作预测第二次前向传播

---

## 关键公式

### 公式 1-2：[[WAM|反应式策略 vs 世界动作模型]]

$$
a_{t:t+H-1} \sim \pi(\cdot \mid o_{\leq t}, a_{<t}, l) \quad \text{（反应式策略）}
$$

$$
(z_{t+1:t+H}, a_{t:t+H-1}) \sim p(\cdot \mid o_{\leq t}, a_{<t}, l) \quad \text{（世界动作模型）}
$$

**含义**: WAM 联合预测未来视觉动态和动作序列，相比反应式策略具有更强的规划能力。

**符号说明**:
- $a_{t:t+H-1}$：预测视野 $H$ 内的动作序列
- $z_{t+1:t+H}$：未来 $H$ 步视频潜变量
- $l$：语言指令
- $o_{\leq t}$：当前时刻及历史多视角观测

### 公式 3-4：[[Hierarchical Latent Action|三阶段级联生成]]

$$
z_{t+1} \to m_t \to a_t
$$

**含义**: 从视频潜变量到帧级潜在动作再到可执行动作的分级生成，每一级都条件化于上一级的输出。

### 公式 5：[[CFM|视频预测 CFM 损失]]

$$
\mathcal{L}_z = \mathbb{E}_{z_{t+1}, \epsilon, \tau}\left[\left\|v_\theta^z\!\left(z_{t+1}^\tau;\, z_{<t+1}, m_{<t}, a_{<t}, \tau, l\right) - (z_{t+1} - \epsilon)\right\|_2^2\right]
$$

其中 $z_{t+1}^\tau = \tau z_{t+1} + (1-\tau)\epsilon$

**含义**: 预测 $z_{t+1}$ 的速度场，以历史视频潜变量、历史潜在动作和历史可执行动作为条件。

**符号说明**:
- $v_\theta^z$：视频流速度场预测网络
- $\tau \in [0, 1]$：扩散时间步
- $\epsilon \sim \mathcal{N}(0, I)$：标准高斯噪声
- $z_{t+1}^\tau$：$\tau$ 时刻的含噪视频潜变量

### 公式 6：[[MoT架构|三流 Token 序列]]

$$
\mathbf{X}_t = [\mathbf{X}_{t+1}^z,\; \mathbf{X}_t^m,\; \mathbf{X}_t^a]
$$

**含义**: 视频潜变量流、帧级潜在动作流、可执行动作流组成联合 token 序列，输入 Dual-Level MoT。

### 公式 8：[[ALAM|潜在动作提取]]

$$
m_t = E_m(I_t, I_{t+1}) \in \mathbb{R}^{d_m}
$$

**含义**: 冻结编码器 $E_m$ 从相邻原始帧对中提取帧级运动意图向量。

**符号说明**:
- $E_m$：冻结的潜在动作编码器（ALAM 预训练）
- $I_t, I_{t+1}$：相邻原始图像帧
- $d_m$：潜在动作维度

### 公式 9：[[ALAM|潜在动作 CFM 损失]]

$$
\mathcal{L}_m = \mathbb{E}_{m_t, \epsilon, \tau}\left[\left\|v_\theta\!\left(m_t^\tau;\, z_{\leq t+1}, m_{<t}, a_{<t}, \tau, l\right) - (m_t - \epsilon)\right\|_2^2\right]
$$

其中 $m_t^\tau = \tau m_t + (1-\tau)\epsilon$

**含义**: 以视频潜变量为条件，预测帧级潜在动作的速度场，使模型能从视频预测中推断精细运动意图。

### 公式 10-12：[[Mixture-of-Transformers|子空间感知 CFM 损失]]

含噪动作构造：

$$
a_t^{move,\tau} = \tau a_t^{move} + (1-\tau)\epsilon^{move}, \quad a_t^{manip,\tau} = \tau a_t^{manip} + (1-\tau)\epsilon^{manip}
$$

移动动作损失：

$$
\mathcal{L}_a^{move} = \mathbb{E}\left[\left\|v_\theta^{move}\!\left(a_t^{move,\tau};\, z_{\leq t+1}, m_{\leq t}, a_{<t}, a_t^{manip,\tau}, \tau, l\right) - (a_t^{move} - \epsilon^{move})\right\|_2^2\right]
$$

操纵动作损失：

$$
\mathcal{L}_a^{manip} = \mathbb{E}\left[\left\|v_\theta^{manip}\!\left(a_t^{manip,\tau};\, z_{\leq t+1}, m_{\leq t}, a_{<t}, a_t^{move,\tau}, \tau, l\right) - (a_t^{manip} - \epsilon^{manip})\right\|_2^2\right]
$$

### 公式 13：联合动作损失

$$
\mathcal{L}_a = \lambda_{move}\mathcal{L}_a^{move} + \lambda_{manip}\mathcal{L}_a^{manip}
$$

**含义**: 移动和操纵子空间损失加权求和，$\lambda$ 系数控制各子空间权重。

### 公式 14-15：[[teacher forcing|Teacher Forcing]] vs [[Dream-Forcing|Dream Forcing]]

Teacher Forcing（训练时条件化于真实未来）：

$$
a_t \sim p_a(\cdot \mid z_{\leq t+1}, m_{\leq t}, a_{<t}, l)
$$

Dream Forcing（训练时条件化于模型预测的梦境）：

$$
a_t \sim p_a(\cdot \mid \hat{z}_{t+1}, z_{\leq t}, \hat{m}_t, m_{<t}, a_{<t}, l)
$$

**含义**: Dream Forcing 将真实未来潜变量 $z_{t+1}, m_t$ 替换为模型自身生成的 $\hat{z}_{t+1}, \hat{m}_t$，训练分布与推理分布对齐。

### 公式 16：世界模型预训练损失

$$
\mathcal{L}_z^{pretrain} = \mathbb{E}_{z_t, \epsilon, \tau}\left[\left\|v_\theta^z(z_t^\tau;\, z_{<t}, \tau, l) - (z_t - \epsilon)\right\|_2^2\right]
$$

**含义**: 动作无关的未来视频预测，使用大规模机器人操纵视频数据预训练视频生成主干。

### 公式 17-19：[[ALAM]] 自监督一致性损失

加法一致性：

$$
\mathcal{L}_{add} = \left\|m_i^k - (m_i^j + m_j^k)\right\|_2^2
$$

反转一致性：

$$
\mathcal{L}_{rev} = \left\|m_i^j + m_j^i\right\|_2^2
$$

ALAM 总损失：

$$
\mathcal{L}_{LAM} = \lambda_{vq}\mathcal{L}_{vq} + \lambda_{rec}\mathcal{L}_{rec} + \lambda_{perc}\mathcal{L}_{perc} + \lambda_{add}\mathcal{L}_{add} + \lambda_{rev}\mathcal{L}_{rev}
$$

**含义**: 加法一致性要求从帧 $i$ 到帧 $k$ 的潜在动作等于经过帧 $j$ 的两步潜在动作之和；反转一致性要求正向和逆向动作相互抵消。

**符号说明**:
- $m_i^j$：从帧 $i$ 到帧 $j$ 的潜在动作
- $\mathcal{L}_{vq}$：向量量化损失
- $\mathcal{L}_{rec}$：重建损失
- $\mathcal{L}_{perc}$：感知损失

### 公式 20-23：SFT Stage I（联合监督微调损失）

$$
z_{t+1} \sim p_z(\cdot \mid z_{\leq t}, m_{<t}, a_{<t}, l)
$$

$$
m_t \sim p_m(\cdot \mid z_{\leq t+1}, m_{<t}, a_{<t}, l)
$$

$$
a_t \sim p_a(\cdot \mid z_{\leq t+1}, m_{\leq t}, a_{<t}, l)
$$

$$
\mathcal{L}_{SFT1} = \lambda_z\mathcal{L}_z + \lambda_m\mathcal{L}_m + \lambda_a\mathcal{L}_a
$$

**含义**: 第一阶段用真实标注作为条件，稳定学习后再切换到梦境条件，避免早期训练不稳定。

### 公式 24-28：SFT Stage II（Dream Forcing 微调损失）

$$
a_t \sim p_a(\cdot \mid \hat{z}_{t+1}, z_{\leq t}, \hat{m}_t, m_{<t}, a_{<t}, l)
$$

$$
\tilde{\mathcal{L}}_a^{move} = \mathbb{E}\left[\left\|v_\theta^{move}(a_t^{move,\tau};\, \hat{z}_{\leq t+1}, \hat{m}_{\leq t}, a_{<t}, a_t^{manip,\tau}, \tau, l) - (a_t^{move} - \epsilon)\right\|_2^2\right]
$$

$$
\tilde{\mathcal{L}}_a^{manip} = \mathbb{E}\left[\left\|v_\theta^{manip}(a_t^{manip,\tau};\, \hat{z}_{\leq t+1}, \hat{m}_{\leq t}, a_{<t}, a_t^{move,\tau}, \tau, l) - (a_t^{manip} - \epsilon)\right\|_2^2\right]
$$

$$
\mathcal{L}_{SFT2} = \lambda_z\mathcal{L}_z + \lambda_m\mathcal{L}_m + \lambda_a\tilde{\mathcal{L}}_a
$$

$$
= \lambda_z\mathcal{L}_z + \lambda_m\mathcal{L}_m + \lambda_a(\lambda_a^{move}\tilde{\mathcal{L}}_a^{move} + \lambda_a^{manip}\tilde{\mathcal{L}}_a^{manip})
$$

**含义**: 第二阶段将真实视频/潜在动作条件替换为梦境预测，强制动作预测器适应推理时的信号分布。

---

## 关键图表

### Figure 1: ABot-M0.5 系统概览

![Figure 1: Overview](https://arxiv.org/html/2607.00678v1/x1.png)

**说明**: ABot-M0.5 整体框架——一个对齐粒度（granularity-aligned）、解耦动作结构（action-disentangled）、训练-测试一致（train-test-consistent）的移动操纵 [[WAM|世界动作模型]]。展示三大失配问题及对应解决方案的对应关系。

### Figure 2: 总体架构

![Figure 2: Architecture](https://arxiv.org/html/2607.00678v1/x2.png)

**说明**: ABot-M0.5 总体架构。模型通过 [[MoT架构|Dual-Level MoT]] 的结构化非对称级联设计，联合预测未来视频潜变量（$z$）、帧级潜在动作（$m$）和可执行动作（$a$）三个并行 token 流。

### Figure 3: Dual-Level Mixture-of-Transformers

![Figure 3: Dual-Level MoT](https://arxiv.org/html/2607.00678v1/x3.png)

**说明**: [[MoT架构|Dual-Level MoT]] 详细结构。通过模态专用投影/嵌入/输出头实现模态层解耦；通过移动和操纵各自专用 FFN 实现动作层解耦；共享自注意力机制在解耦的同时保持跨子空间的协调推理。

### Figure 4: 训练范式对比

![Figure 4: Training Paradigms](https://arxiv.org/html/2607.00678v1/x4.png)

**说明**: 三种 WAM 训练范式的对比。(a) [[teacher forcing|Teacher Forcing]]：训练用真实未来，推理用预测值，存在分布差异；(b) [[Diffusion Forcing|Diffusion Forcing]]：混合噪声级别，但训练与推理条件仍不完全匹配；(c) **Dream Forcing**（本文方法）：训练时用自梦境视频条件化动作预测，彻底对齐训练和推理分布。

### Figure 5: Dream Forcing 两阶段训练策略

![Figure 5: Dream Forcing Two-Phase](https://arxiv.org/html/2607.00678v1/x5.png)

**说明**: [[Dream-Forcing|Dream Forcing]] 两阶段前向传播详解。Phase A 执行速度场预测得到干净梦境潜变量 $\hat{z}_{t+1}$；Phase B 以 Phase A 的梦境输出为条件再次前向传播进行动作预测，直接模拟部署时的条件分布。

### Figure 6: ALAM 模型结构

![Figure 6: ALAM Structure](https://arxiv.org/html/2607.00678v1/x6.png)

**说明**: [[ALAM]] 的详细结构。从视觉帧对 $(I_t, I_{t+1})$ 提取 $m_t \in \mathbb{R}^{d_m}$，通过加法一致性（Eq. 17）和反转一致性（Eq. 18）约束确保潜在动作空间的几何一致性，支持从无动作标注视频中自监督训练。

### Figure 7: 不同训练阶段的注意力掩码

![Figure 7: Attention Masks](https://arxiv.org/html/2607.00678v1/x7.png)

**说明**: 预训练（世界模型预训练）、SFT Stage I 和 SFT Stage II 各阶段的注意力掩码设计。不对称信息流：视频流被屏蔽不关注动作流（因果一致），动作流可关注视频流（条件依赖），实现高效结构化注意力，通过可变长 FlashAttention 实现 5× 加速。

### Figure 8: RoboCasa365 可视化结果

![Figure 8: Visualization](https://arxiv.org/html/2607.00678v1/x8.png)

**说明**: ABot-M0.5 在 [[RoboCasa365]] 上的部分可视化结果。黄色和绿色分别区分移动阶段和操纵阶段，展示了长时程组合任务（如 "Arrange Flower"、"Find Toaster"）中导航→操纵的平滑过渡能力。

### Table 1: 主要符号说明

| 符号 | 含义 | 符号 | 含义 |
|------|------|------|------|
| $t$ | 时间步 | $o_t$ | 原始多视角观测 |
| $I_t$ | 原始帧 | $z_{t+1}$ | 视频潜变量 |
| $m_t$ | 帧级潜在动作 | $a_t$ | 可执行机器人动作 |
| $a_t^{move}$ | 移动动作 | $a_t^{manip}$ | 操纵动作 |
| $H$ | 预测视野 | $N_c$ | 相机数量 |
| $\hat{z}_{t+1}, \hat{m}_t$ | 梦境潜变量 | $\tilde{z}_{t+1}, \tilde{m}_t$ | 含噪潜变量 |
| $\tau$ | 扩散时间步（$\tau \in [0,1]$） | $l$ | 语言指令 |
| $p_z, p_m, p_a$ | 各流的潜变量分布 | $\mathbf{X}_t$ | Token 序列 |

### Table 2：RoboCasa365 预训练协议结果

| 方法 | 平均 | Atomic-Seen | Composite-Seen | Composite-Unseen |
|------|------|-------------|----------------|-----------------|
| Diffusion Policy | 6.1% | 15.7% | 0.2% | 1.3% |
| π₀ | 14.8% | 34.6% | 6.1% | 1.1% |
| π₀.₅ | 16.9% | 39.6% | 7.1% | 1.2% |
| GR00T-N1.5 | 23.9% | 50.7% | 14.8% | 2.7% |
| GR00T-N1.6 | 21.9% | 51.1% | 9.4% | 1.7% |
| GigaWorld-Policy 0.1 | 20.7% | 44.4% | 11.8% | 2.9% |
| RLDX-1 | 33.2% | 63.0% | 27.5% | 5.4% |
| Qwen-RobotManip | 35.9% | 68.6% | 20.1% | 14.9% |
| Qwen-RobotManip-Context | 33.8% | 63.9% | 22.6% | 11.2% |
| **ABot-M0.5（Ours）** | **40.4%** | **75.9%** | **38.3%** | **2.7%** |
| **ABot-M0.5 (+Condensed Memory）** | **46.6%** | **79.4%** | **48.3%** | **7.9%** |

**说明**: ABot-M0.5 在 Atomic-Seen（75.9%）和 Composite-Seen（38.3%）上大幅领先，加入 Condensed Memory 后 Composite-Unseen（7.9%）提升明显，表明长时程组合任务的泛化能力增强。

### Table 3：RoboCasa365 目标协议结果

| 方法 | Atomic-S | Composite-S | Composite-U | 平均 |
|------|----------|-------------|-------------|------|
| **Target 100%** | | | | |
| GR00T-N1.5 | 60.6% | 35.0% | 33.3% | 43.7% |
| Fast-WAM | 59.1% | 36.4% | 33.2% | 43.5% |
| Lingbot-VA | 63.5% | 37.3% | 32.1% | 45.1% |
| **ABot-M0.5（Ours）** | **70.6%** | **44.3%** | **45.6%** | **54.2%** |
| **Target 10%** | | | | |
| GR00T-N1.5 | 38.7% | 11.0% | 11.2% | 21.0% |
| **ABot-M0.5（Ours）** | **49.0%** | **23.4%** | **15.4%** | **30.1%** |

**说明**: 在目标协议（in-distribution fine-tuning）下，ABot-M0.5 的 Target 100% 平均达 54.2%，Target 10% 低数据场景下也达 30.1%，数据效率显著优于 GR00T-N1.5（21.0%）。

### Table 4：RoboTwin 2.0 基准结果

| 模型 | Clean (Easy) | Randomized (Hard) | 平均 |
|------|-------------|-------------------|------|
| X-VLA | 72.80 | 72.84 | 72.82 |
| π₀.₅ | 82.70 | 76.80 | 79.75 |
| ABot-M0 | 86.06 | 85.08 | 85.57 |
| Qwen-VLA | 86.10 | 87.20 | 86.65 |
| Fast-WAM | 91.90 | 91.80 | 91.85 |
| Lingbot-VA | 92.93 | 91.55 | 92.24 |
| HoloBrain-0 | 91.90 | 92.30 | 92.10 |
| AttenA+ | 93.10 | 91.90 | 92.50 |
| G0.5 | 93.70 | 92.80 | 93.30 |
| Qwen-RobotManip | 93.70 | 94.00 | 93.85 |
| **ABot-M0.5（Ours）** | **94.00** | **94.20** | **94.10** |

**说明**: RoboTwin 2.0 上 ABot-M0.5 以 94.10% 平均成功率居首，相较前代 ABot-M0（85.57%）提升约 8.5%，验证了方法在标准操纵 benchmark 上的泛化性。

### Table 5：LIBERO 基准结果

| 方法 | L-Spatial | L-Object | L-Goal | L-Long | 平均 |
|------|-----------|----------|--------|--------|------|
| Diffusion Policy | 78.5 | 87.5 | 73.5 | 64.8 | 76.1 |
| OpenVLA | 84.7 | 88.4 | 79.2 | 53.7 | 76.5 |
| π₀-Fast | 96.4 | 96.8 | 88.6 | 60.2 | 85.5 |
| GR00T-N1 | 94.4 | 97.6 | 93.0 | 90.6 | 93.9 |
| π₀ | 98.0 | 96.8 | 94.4 | 88.4 | 94.4 |
| π₀.₅ | 98.8 | 98.2 | 98.0 | 92.4 | 96.9 |
| GR00T-N1.6 | 97.7 | 98.5 | 97.5 | 94.4 | 97.0 |
| Fast-WAM | 98.2 | 100.0 | 97.0 | 95.2 | 97.6 |
| ImageWAM | 97.2 | 99.2 | 98.8 | 98.4 | 98.4 |
| Lingbot-VA | 98.5 | 99.6 | 97.2 | 98.5 | 98.5 |
| ABot-M0 | 98.8 | 99.8 | 99.0 | 96.6 | 98.6 |
| Being-H0.5 | 99.2 | 99.6 | 99.4 | 97.4 | 98.9 |
| CORAL | 99.6 | 99.8 | 99.0 | 98.8 | 99.3 |
| **ABot-M0.5（Ours）** | **100.0** | **99.8** | **99.4** | **98.4** | **99.4** |

**说明**: LIBERO 平均 99.4%，L-Spatial 首次达到 100%，L-Long 98.4% 也体现出强长时程任务能力。总体超越 CORAL（99.3%）和 Being-H0.5（98.9%）。

### Table 6：LIBERO-Plus 零样本泛化结果

| 方法 | Camera | Robot | Language | Light | Background | Noise | Layout | Total |
|------|--------|-------|----------|-------|------------|-------|--------|-------|
| OpenVLA-OFT | 56.4 | 31.9 | 79.5 | 88.7 | 93.3 | 75.8 | 74.2 | 69.6 |
| π₀ | 13.8 | 6.0 | 58.8 | 85.0 | 81.4 | 79.0 | 68.9 | 53.6 |
| π₀-Fast | 65.1 | 21.6 | 61.0 | 73.2 | 73.2 | 74.4 | 68.8 | 61.6 |
| Fast-WAM | 76.8 | 81.4 | 92.6 | 94.2 | 96.8 | 92.4 | 89.6 | 89.1 |
| Lingbot-VA | 85.2 | 89.4 | 93.8 | 95.6 | 97.4 | 94.2 | 91.8 | 92.3 |
| **ABot-M0.5（Ours）** | **92.4** | **94.8** | **96.2** | **97.4** | **98.6** | **96.8** | **94.2** | **94.8%** |

**说明**: LIBERO-Plus 零样本泛化评估多维度鲁棒性（相机变换、机器人换型、语言多样性、光照/背景/噪声/布局扰动）。ABot-M0.5 在所有维度均领先，Camera（92.4%）和 Robot（94.8%）两个困难维度尤为突出，说明 WAM 框架通过视频预测对外观变化的鲁棒性更强。

---

## 实验

### 基准与评估协议

| 基准 | 类型 | 协议 | 核心指标 |
|------|------|------|---------|
| [[RoboCasa365]] | 移动操纵（仿真） | 预训练（baseline 权重） / 目标（微调） | 任务成功率（Atomic-S / Composite-S / Composite-U） |
| [[RoboTwin 2.0]] | 桌面操纵（仿真） | 标准 | 成功率（Easy / Hard） |
| [[LIBERO-10]] | 桌面操纵（仿真） | 标准 | 四子集平均成功率 |
| LIBERO-Plus | 桌面操纵（仿真，零样本） | 7 维扰动 | 各扰动维度成功率 |
| 真实机器人 | 移动操纵（物理机器人） | 5 任务 | 成功率 + 过程分数 |

### 预训练数据集

| 数据集 | 类型 |
|--------|------|
| OXE | 多体态多任务操纵 |
| OXE-AugE | OXE 增强版 |
| Agibot-Beta | 双臂移动机器人 |
| RoboCOIN | 大规模室内操纵 |
| RoboMind | 多任务机器人 |
| Galaxea | 移动操纵 |
| InternData-A1 | 人形机器人 |
| RoboNet | 多机器人多场景 |
| BridgeData V2 | 厨房操纵 |
| DROID | 多样化操纵 |

### 实现细节

- **主干模型**: [[Wan 2.2|Wan2.2]] 5B 视频扩散（预训练初始化）
- **文本编码器**: [[UMT5]]
- **视频编码器**: 3D [[Causal VAE]]
- **潜在动作编码器**: 冻结的 [[ALAM]] 预训练权重
- **注意力实现**: 变长 FlashAttention（5× 速度提升 vs FlexAttention 基准）
- **训练目标**: [[CFM|Conditional Flow Matching]]（全阶段统一）

### 消融实验

Section 5.4 在 RoboCasa365 和 LIBERO 基准上对四个核心组件进行消融：

- **中间潜在动作的效果**: 去掉 $m_t$ 桥接表示时，长时程组合任务（Composite-Unseen）性能下降最为显著，说明帧级运动意图对跨阶段协调最关键。
- **动作解耦 MoT 的效果**: 去掉移动/操纵子空间解耦，替换为单一动作空间时，精细操纵控制（L-Long 等子集）性能下降，验证梯度干扰的影响。
- **Dream Forcing 的效果**: 替换为 Teacher Forcing 时鲁棒性明显下降（LIBERO-Plus Camera 和 Robot 维度），证明训练-推理对齐的重要性。
- **预训练和 SFT 的效果**: 去掉大规模视频预训练或两阶段 SFT 时性能均有明显退化，验证渐进式训练范式的必要性。

### 真实机器人实验

在物理移动机器人上评估 5 项任务（同时包含导航和操纵阶段），汇报成功率和过程分数，验证 ABot-M0.5 在真实世界移动操纵任务中的可部署性。

---

## 批判性思考

### 优点
1. **问题识别系统化**: 明确提出三大结构失配，解决方案设计与问题直接对应，论文逻辑结构清晰。
2. **效率工程**: Efficient Structured Attention（5× 加速）和 Offset-Based Latent Augmentation 体现了工业落地导向。
3. **全面评估**: 覆盖移动操纵（RoboCasa365）、桌面操纵（RoboTwin/LIBERO）和零样本泛化（LIBERO-Plus），以及真实机器人验证。
4. **前代超越明显**: 相比 [[ABot-M0]]（RoboTwin 85.57%），M0.5 以 94.10% 提升约 8.5%，进步显著。

### 局限性
1. **消融细节不充分**: 论文 HTML 版本中消融表格数值不完整，难以精确量化各组件的独立贡献。
2. **Dream Forcing 计算代价**: 两阶段前向传播（Phase A + Phase B）推理开销更大，实时性能受影响，文中未明确报告额外延迟。
3. **Composite-Unseen 偏低**: 预训练协议下仅 7.9%（含 CM），远低于 Atomic-Seen（79.4%），跨分布泛化仍是挑战。
4. **Condensed Memory 机制不透明**: "+Condensed Memory" 变体在 RoboCasa365 上带来大幅提升（+6.2%），但技术细节在主论文中披露不足。

### 潜在改进方向
1. 将 Dream Forcing 延伸至更长时程的递归 rollout，而非仅当前 chunk 的一步梦境。
2. 探索动作子空间数量自适应（现在仅 move/manip 两类），扩展至多指、工具使用等更多子空间。
3. ALAM 潜在动作向量维度和几何约束的更系统性的设计空间探索。

### 可复现性评估
- [x] 代码开源（amap-cvlab/ABot-Manipulation）
- [ ] 预训练模型（尚未公开）
- [x] 训练细节（论文中有）
- [x] 数据集（使用公开数据集 OXE/LIBERO/RoboTwin 等）

---

## 关联笔记

### 基于
- [[ABot-M0]]: 前代模型，ABot-M0.5 在其基础上增加 WAM 框架和三大对齐机制
- [[Wan 2.2]]: 视频扩散主干网络，提供预训练的世界建模能力
- [[Dream-Forcing]]: 核心训练对齐技术，消除训练-推理 exposure bias

### 对比
- [[Flash-WAM]]: 同期 WAM 方法，侧重推理加速（23× 蒸馏）而非训练对齐
- [[LingBot-VA]]: RoboTwin 2.0 竞争者（92.24%），LIBERO-Plus Total 92.3%
- [[Qwen-RobotManip]]: RoboCasa365 预训练协议亚军（35.9%）

### 方法相关
- [[CFM]]: 全阶段统一的训练目标
- [[MoT架构]]: Dual-Level Mixture-of-Transformers 结构基础
- [[Latent-Action]]: 潜在动作建模方法论
- [[ALAM]]: 本文提出的潜在动作提取模块
- [[LAM]]: 潜在动作模型的广义框架
- [[teacher forcing]]: 与 Dream Forcing 形成对比的经典训练策略

### 数据集相关
- [[RoboCasa365]]: 主要移动操纵评估基准
- [[RoboTwin 2.0]]: 操纵评估基准
- [[LIBERO-10]]: 操纵评估基准

---

## 速查卡片

> [!summary] ABot-M0.5: Unified Mobility-and-Manipulation World Action Model
> - **核心**: 解决 WAM 在移动操纵中的时间粒度/动作结构/训练推理三大结构失配
> - **方法**: 中间潜在动作（ALAM）+ 双层 MoT + Dream Forcing 两阶段训练
> - **结果**: RoboCasa365 46.6%（SOTA），RoboTwin 94.10%，LIBERO 99.4%
> - **代码**: [amap-cvlab/ABot-Manipulation](https://github.com/amap-cvlab/ABot-Manipulation)

---

*笔记创建时间: 2026-07-03*
