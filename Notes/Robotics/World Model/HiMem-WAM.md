---
title: "HiMem-WAM: Hierarchical Memory-Gated World Action Models for Robotic Manipulation"
method_name: "HiMem-WAM"
authors: [Xiaoquan Sun, Ruijian Zhang, Chen Cao, Yihan Sun, Jiahui Chen, Zetian Xu, Bo Chen, Haijier Chen, Zhen Yang, Jiarun Zhu, Yijun Hong, JingZhe Xu, Jingrui Pang, Mingqi Yuan, Jiayu Chen]
year: 2026
venue: arXiv
tags: [world-action-model, hierarchical-learning, memory-mechanism, robotic-manipulation, vla, skill-learning, long-horizon-planning]
zotero_collection: Robotics/World Model
image_source: local
arxiv_html: https://arxiv.org/html/2606.10363
created: 2026-06-11
---

# 论文笔记：HiMem-WAM: Hierarchical Memory-Gated World Action Models for Robotic Manipulation

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | 未明确披露 |
| 日期 | June 2026 |
| 项目主页 | N/A |
| 对比基线 | [[Fast-WAM]], [[WorldVLA]], [[pi0.5\|π₀.₅]] |
| 链接 | [arXiv](https://arxiv.org/abs/2606.10363) / Code N/A |

---

## 一句话总结

> HiMem-WAM 通过分层潜在动作框架与边界触发的记忆门控机制，使机器人在长视野操作任务中实现了结构化时序抽象与因果历史状态推理。

---

## 核心贡献

1. **分层潜在动作框架（Hierarchical Latent Action Framework）**: 联合学习低层运动潜变量（low-level motion latents）和高层技能潜变量（high-level skill latents），提供结构化时序抽象，弥合短期执行与长视野任务分解之间的鸿沟。
2. **记忆门控模块（Memory-Gated Module）**: 仅在预测的技能边界处写入紧凑任务状态，通过可学习的读门与写门实现因果推理，无需测试时生成未来视频。
3. **三阶段训练流水线（Three-Stage Training Pipeline）**: 低级 tokenizer 预训练 → 分层技能发现 → 记忆条件化策略微调，每阶段专注不同层次的表示学习，各阶段信号互相监督。

---

## 问题背景

### 要解决的问题

现有 [[World Action Model|WAM]] 在长视野机器人操作任务中面临两个核心挑战：(1) 缺乏对低层运动的层次化抽象，难以将短时运动片段组织为可复用的高层技能；(2) 缺乏跨技能边界的显式历史记忆，导致在需要多步骤记忆跟踪的任务中性能不稳定。

### 现有方法的局限

- **[[WorldVLA]]、[[Fast-WAM]]** 等 WAM 方法直接从视频/观测中预测动作，缺乏层次化的时序抽象
- 现有记忆方法（如 [[MemoryVLA]]、[[SAM2Act]]）依赖稠密历史聚合或固定记忆窗口，缺乏选择性写入机制
- 基于扩散（[[Diffusion Policy]]）的方法缺乏显式的技能边界检测与长视野规划能力
- 现有 WAM 推理时需要生成未来视频，计算开销大，且不具备真正的因果性

### 本文的动机

通过同时学习低层运动潜变量和高层技能潜变量，并在技能边界处触发记忆写入，实现 (1) 结构化的时序抽象，使策略能够在不同粒度上进行规划；(2) 高效且因果的历史状态保留，无需稠密存储全部历史观测。

---

## 方法详解

### 模型架构

![[HiMem-WAM_fig1_overview.png]]

**Figure 1 说明**: HiMem-WAM 的三阶段总体框架。Stage I 从演示数据中提取低层动作 token 和高层技能潜变量；Stage II 学习从视频和语言输入预测潜在动作；Stage III 引入门控记忆模块进行历史感知动作预测。底部面板展示真实世界和仿真评测结果。

HiMem-WAM 采用**三阶段层次化**架构：

- **输入**: 语言指令 $\ell$ + 多视角 RGB 观测 $o_t$ + 本体感知 $p_t$ + 外部记忆 $M_t$
- **Backbone**: [[Qwen3-VL]] (Qwen3-VL-4B-Instruct) 作为高层规划器
- **核心模块**: [[Hierarchical Latent Action|分层潜在动作框架]] 用于时序抽象，[[Memory Gating|记忆门控]] 用于历史状态保留
- **输出**: [[Action Chunking|动作块]] $a_{t:t+K-1}$
- **光流估计**: [[DPFlow]] 用于低层运动表示

### 核心模块

#### 模块1: Stage I — 低层 Tokenizer（Low-Level Tokenizer）

**设计动机**: 利用 [[Optical Flow|光流]] 作为运动先验，通过 [[Variational Autoencoder|变分自编码器]] 学习紧凑的运动潜变量，使低层表示捕获任务完成所需的运动模式而非视觉过拟合。

**具体实现**:
- 输入短时上下文：多视角观测 $o_t$、本体感知 $p_t$、语言指令 $\ell$
- 流编码器 $E_{\text{flow}}$ 处理 [[DPFlow]] 估计的多视角光流 $\Phi_t$，视觉编码器 $E_{\text{vis}}$ 处理 RGB 图像
- 通过 [[Reparameterization Trick|重参数化技巧]] 采样低层潜变量 $z^l_t$
- 解码器重建光流，可选对齐真实动作标签（由指示变量 $\mathbb{I}^{\text{act}}_t$ 控制）
- 训练完成后 tokenizer 参数**冻结**，供 Stage II/III 复用

#### 模块2: Stage II — 分层技能发现（Hierarchical Skill Discovery）

**设计动机**: 利用 [[Attention Pooling|注意力池化]] 和可学习的边界检测，将低层潜变量序列逐步压缩为高层技能表示，实现多粒度时序抽象。

**具体实现**:
- 对冻结 tokenizer 输出的潜变量序列进行分层分块（hierarchical chunking）
- 每层通过不相似度评分（dissimilarity scoring）检测技能边界
- 多阶段渐进压缩：$Z^{(s+1)} = \text{Chunk}_s(E_s(Z^{(s)}); b^{(s)})$
- 最终高层技能表示 $Z^h = Z^{(H)}$
- 同时预测下一个潜变量（next-latent prediction）和运动重建损失

#### 模块3: Stage III — 记忆条件化策略（Memory-Conditioned Policy）

**设计动机**: 通过可学习的读门和写门，在技能边界处选择性地更新紧凑记忆，实现因果历史感知推理，无需未来视频生成。

**具体实现**:
- [[Qwen3-VL]] 规划器接收当前观测和记忆上下文，预测高层技能 $\hat{z}^h_t$ 和边界分数 $\hat{b}_t$
- **读门**（Read Gate）：通过交叉注意力检索历史相关信息，$\tilde{x}_t = x_t + \alpha^r_t W_m c^m_t$
- **写门**（Write Gate）：当写门置信度超过阈值 $\eta$ 时触发记忆更新
- 执行器（Executor）将技能潜变量展开为动作块，解码器转换为机器人控制信号
- 全因果推理：仅需当前观测和记忆，无需测试时光流或视频生成

![[HiMem-WAM_fig2_wamvsours.png]]

**Figure 2 说明**: 从 WAM 到 HiMem-WAM 的对比。HiMem-WAM 在统一世界动作建模基础上引入记忆专家，使动作预测能够同时条件化于当前观测和任务历史。

---

## 关键公式

### 公式1: [[Hierarchical Policy Factorization|分层策略分解]]

$$
\pi_\theta(a_{t:t+K-1} | o_t, p_t, \ell, M_t) = \int p_\theta(a_{t:t+K-1} | Z^l_{t:t+K-1}, o_t, p_t) \cdot p_\theta(Z^l_{t:t+K-1} | z^h_t, o_t, p_t, M_t) \cdot p_\theta(z^h_t | o_t, p_t, \ell, M_t) \, dZ^l \, dz^h
$$

**含义**: 将策略分解为三个条件分布的乘积——高层技能预测、低层潜变量展开、以及最终动作生成，实现层次化规划与执行。

**符号说明**:
- $a_{t:t+K-1}$: 长度为 $K$ 的动作块
- $Z^l_{t:t+K-1}$: 低层潜变量序列
- $z^h_t$: 高层技能潜变量
- $M_t$: 外部记忆库
- $\ell$: 语言指令
- $o_t, p_t$: 当前观测与本体感知

### 公式2: [[Variational Autoencoder|低层潜变量后验]]

$$
q_\phi(z^l_t | c_t) = \mathcal{N}(\mu_t, \text{diag}(\sigma^2_t)), \quad z^l_t = \mu_t + \sigma_t \odot \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, I)
$$

**含义**: 用变分自编码器对低层动作进行概率建模，通过重参数化技巧实现可微分采样。

**符号说明**:
- $c_t$: 短时上下文（观测、本体感知、语言）
- $\mu_t, \sigma^2_t$: 后验均值与方差
- $\odot$: 逐元素乘法
- $\varepsilon$: 标准正态噪声

### 公式3: [[Tokenizer Loss|低层 Tokenizer 损失]]

$$
\mathcal{L}_l = \|\hat{\Phi}_t - \Phi_t\|_1 + \lambda_a \mathbb{I}^{\text{act}}_t \|\hat{a}_t - a_t\|^2_2 + \beta \, D_{\text{KL}}\!\left(q_\phi(z^l_t | c_t) \,\|\, \mathcal{N}(0, I)\right)
$$

**含义**: 低层 tokenizer 的训练目标，包含光流重建损失、可选动作对齐损失和 KL 散度正则化。

**符号说明**:
- $\hat{\Phi}_t, \Phi_t$: 预测与真实光流
- $\mathbb{I}^{\text{act}}_t$: 是否有动作标签的指示变量
- $\lambda_a$: 动作对齐权重
- $\beta$: KL 散度权重

### 公式4: [[Hierarchical Chunking|分层分块操作]]

$$
Z^{(s+1)} = \text{Chunk}_s(E_s(Z^{(s)}); \, b^{(s)}), \quad Z^h = Z^{(H)}
$$

**含义**: 将第 $s$ 层的潜变量序列经编码和边界引导的分块操作压缩为第 $s+1$ 层表示，逐层提升时间抽象粒度。

**符号说明**:
- $E_s$: 第 $s$ 层编码器
- $b^{(s)}$: 第 $s$ 层边界标签
- $H$: 层次总数
- $Z^h$: 最终高层技能表示

### 公式5: [[Memory Gating|记忆读取门]]

$$
c^m_t = \text{Attn}(W_q x_t, \, W_k M_t, \, W_v M_t), \quad \tilde{x}_t = x_t + \alpha^r_t W_m c^m_t, \quad \alpha^r_t = \sigma(G_r(x_t, c^m_t))
$$

**含义**: 通过可学习的读门控制从记忆库中检索相关历史信息，并以门控方式融入当前特征。

**符号说明**:
- $c^m_t$: 记忆检索向量
- $\alpha^r_t$: 读门权重（由 sigmoid 激活）
- $G_r$: 读门网络
- $W_q, W_k, W_v, W_m$: 可学习投影矩阵

### 公式6: [[Memory Gating|记忆写入门]]

$$
\alpha^w_t = \sigma(G_w(\tilde{x}_t, \hat{z}^h_t, \hat{b}_t)), \quad M_{t+1} = \begin{cases} U_\psi(M_t, \gamma_t) & \text{if } \alpha^w_t > \eta \\ M_t & \text{otherwise} \end{cases}
$$

**含义**: 当写门置信度超过阈值 $\eta$ 时触发记忆更新，将当前任务状态写入记忆库，防止记忆无限增长。

**符号说明**:
- $\alpha^w_t$: 写门权重
- $G_w$: 写门网络
- $\eta$: 写入阈值
- $U_\psi$: 记忆更新函数
- $\gamma_t$: 新记忆 token

### 公式7: [[Stage II Loss|Stage II 潜变量策略损失]]

$$
\mathcal{L}_{\text{latent}} = \lambda_h \, \text{MSE}(\hat{z}^h_t, \bar{z}^h_t) + \lambda_l \, \text{MSE}(\hat{Z}^l_{t:t+K-1}, Z^l_{t:t+K-1}) + \lambda_b \, \text{BCE}(\hat{b}_t, \bar{b}_t)
$$

**含义**: Stage II 的联合训练目标，同时监督高层技能预测、低层潜变量预测和技能边界检测。

**符号说明**:
- $\bar{z}^h_t, \bar{b}_t$: 来自冻结 tokenizer 的高层技能和边界标签
- $\lambda_h, \lambda_l, \lambda_b$: 各损失权重
- BCE: 二元交叉熵损失

### 公式8: [[Stage III Loss|Stage III 微调损失]]

$$
\mathcal{L}_{\text{ft}} = \mathcal{L}_{\text{act}} + \alpha_h \mathcal{L}_{\text{plan}} + \alpha_l \mathcal{L}_{\text{exec}} + \alpha_b \mathcal{L}_{\text{bd}} + \alpha_m \mathcal{L}_{\text{gate}}
$$

**含义**: Stage III 的综合训练目标，包含动作损失、高层规划损失、低层执行损失、边界检测损失和记忆门控损失。

**符号说明**:
- $\mathcal{L}_{\text{act}}$: 最终动作预测损失
- $\mathcal{L}_{\text{plan}}$: 高层技能规划损失
- $\mathcal{L}_{\text{exec}}$: 低层执行损失
- $\mathcal{L}_{\text{bd}}$: 边界检测损失
- $\mathcal{L}_{\text{gate}}$: 记忆门控损失
- $\alpha_h, \alpha_l, \alpha_b, \alpha_m$: 各项权重

### 公式9: [[Memory Gate Loss|记忆门控正则化损失]]

$$
\mathcal{L}_{\text{gate}} = \text{BCE}(\alpha^w_t, \bar{b}_t) + \lambda_r \|\alpha^r_t\|_1 + \lambda_w \|\alpha^w_t\|_1
$$

**含义**: 记忆门控的训练目标，使写门与技能边界对齐，同时通过稀疏正则化控制读写频率。

**符号说明**:
- $\bar{b}_t$: 边界标签（作为写门的监督信号）
- $\lambda_r, \lambda_w$: 读门和写门的 L1 稀疏正则权重

---

## 关键图表

### Figure 3: 真实世界评测结果

![[HiMem-WAM_fig3_real_world.png]]

**说明**: 在 10 个真实世界任务上的评测结果，包含标准设置（ST）和泛化设置（GE）。(a)-(c) 分别报告三类任务的成功率，(d) 展示 GE 设置下的评测变体，(e) 展示双臂硬件平台。

### Figure 4: 10 个真实世界任务可视化

![[HiMem-WAM_fig4_visual.png]]

**说明**: 第一行为简单任务（叠碗、挂杯、放水果进篮、按按钮）；第二行为中等任务（叠三个碗、折叠毛巾、放盘子、按两个按钮）；第三行为困难任务（放两个盘子、制作早餐）。

### Figure 5: RMBench 任务展示与 DPFlow 可视化

![[HiMem-WAM_fig5_rmbench.png]]

**说明**: RMBench 任务的执行展示以及 [[DPFlow]] 光流可视化，展示低层 tokenizer 学到的运动表示。

### Figure 6: LIBERO-PLUS 任务鲁棒性可视化

![[HiMem-WAM_fig6_libero_plus.png]]

**说明**: LIBERO-PLUS benchmark 中七种扰动类型的可视化，用于评测零样本鲁棒性，包括相机视角、初始化位置、语言描述、光照、背景、噪声和布局变化。

### Table 1: RMBench 评测结果

| 任务 | 类型 | DP | ACT | π₀.₅ | X-VLA | HiMem-WAM |
|------|------|----|-----|------|-------|-----------|
| Observe and Pick Up | M(1) | 1% | 1% | 9% | 9% | **28%** |
| Rearrange Blocks | M(1) | 0% | 29% | 13% | 13% | **33%** |
| Put Back Block | M(1) | 0% | 0% | 11% | 18% | **32%** |
| Swap Blocks | M(1) | 11% | 2% | 24% | 16% | **38%** |
| Swap T | M(1) | 2% | 2% | 15% | 3% | **27%** |
| **M(1) 平均** | | 2.8% | 6.8% | 14.4% | 11.8% | **31.6%** |
| Battery Try | M(n) | 10% | 19% | 16% | 26% | **28%** |
| Blocks Ranking Try | M(n) | 10% | 0% | 6% | 1% | **24%** |
| Cover Blocks | M(n) | 0% | 0% | 2% | 0% | **19%** |
| Press Button | M(n) | 0% | 0% | 1% | 2% | **8%** |
| **M(n) 平均** | | 5.0% | 4.8% | 6.3% | 7.3% | **19.8%** |
| **总平均** | | 3.8% | 5.9% | 10.8% | 9.8% | **26.3%** |

**说明**: M(1) 表示单次记忆需求任务，M(n) 表示多次试验场景。HiMem-WAM 在所有类别中大幅领先，M(1) 任务尤为突出。

### Table 2: LIBERO Benchmark 结果

| 方法 | Spatial | Object | Goal | Long | 平均 |
|------|---------|--------|------|------|------|
| OpenVLA | 84.7 | 88.4 | 79.2 | 53.7 | 76.5 |
| SpatialVLA | 88.2 | 89.9 | 78.6 | 55.5 | 78.1 |
| π₀ | 96.8 | 98.8 | 95.8 | 85.2 | 94.2 |
| NORA-1.5 | 97.3 | 96.4 | 94.5 | 89.6 | 94.5 |
| AtomVLA | 96.4 | 99.6 | 97.6 | 94.4 | 97.0 |
| WorldVLA | 87.6 | 96.2 | 83.4 | 60.0 | 81.8 |
| Fast-WAM | 98.2 | 100.0 | 97.0 | 95.2 | 97.6 |
| w/o Stage II | 96.0 | 99.6 | 97.1 | 93.8 | 96.6 |
| **HiMem-WAM** | **98.2** | **99.8** | **98.4** | **94.5** | **97.7** |

**说明**: HiMem-WAM 在 LIBERO 四个子套件上平均达到 97.7%，与 Fast-WAM 持平，在 Goal 子集上更优。Stage II 预训练对 Long 任务贡献明显（93.8% → 94.5%）。

### Table 3: LIBERO-PLUS 零样本鲁棒性结果

| 方法 | Cam. | Init | Lang. | Light | BG. | Noise | Layout | 平均 |
|------|------|------|-------|-------|-----|-------|--------|------|
| OpenVLA | 0.8 | 3.5 | 23.0 | 8.1 | 34.8 | 15.2 | 28.5 | 16.3 |
| OpenVLA-OFT | 56.4 | 31.9 | 79.5 | 88.7 | 93.3 | 75.8 | 74.2 | 71.4 |
| π₀ | 13.8 | 6.0 | 58.8 | 85.0 | 81.4 | 79.0 | 68.9 | 56.1 |
| UniVLA | 1.8 | 46.2 | 69.6 | 69.0 | 81.0 | 21.2 | 31.9 | 45.8 |
| RIPT-VLA | 55.2 | 31.2 | 77.6 | 88.4 | 91.6 | 73.5 | 74.2 | 70.2 |
| WorldVLA | 0.1 | 27.9 | 41.6 | 43.7 | 17.1 | 10.9 | 38.0 | 25.6 |
| HoloBrain-0 | 65.5 | 58.2 | 78.7 | 88.1 | 90.3 | 66.9 | 79.5 | 75.3 |
| w/o Stage II | 77.2 | 37.9 | 71.7 | 91.1 | 83.6 | 73.2 | 70.7 | 72.2 |
| **HiMem-WAM** | **78.2** | **38.1** | **76.6** | **92.2** | **91.0** | **80.7** | **74.9** | **76.0%** |

**说明**: HiMem-WAM 在 LIBERO-PLUS 零样本评测中达到 76.0%，超越 HoloBrain-0（75.3%）并大幅领先 WorldVLA（25.6%）。Stage II 预训练在相机变化（77.2→78.2）和噪声扰动（73.2→80.7）上的提升最明显。

### Table 4: 真实世界动作表示消融

| 类别 | Joint Pos. (w/o Stage II) | Joint Pos. (w/ Stage II) | EE Pose (w/o Stage II) | EE Pose (w/ Stage II) |
|------|--------------------------|--------------------------|------------------------|----------------------|
| Easy | 100.0 | 100.0 | 90.0 | 100.0 |
| Medium | 80.0 | 82.5 | 67.5 | 75.0 |
| Hard | 15.0 | **35.0** | 10.0 | **30.0** |

**关键发现**: Stage II 预训练对困难任务的提升最为显著（关节位置：15%→35%，末端姿态：10%→30%），说明层次化潜变量学习为长视野操作提供了有效的运动先验。

### Table 5: 各阶段训练与推理所用信号

| 信号 | Stage I | Stage II | Stage III | 推理 |
|------|---------|----------|-----------|------|
| RGB 观测 | ✓ | ✓ | ✓ | ✓ |
| 本体感知 | ✓ | ✓ | ✓ | ✓ |
| 语言指令 | ✓ | ✓ | ✓ | ✓ |
| 光流 | ✓ | 仅监督 | — | — |
| 动作标注 | 可选 | — | ✓ | — |
| 低层潜变量 | 输出 | 监督 | 监督 | 预测 |
| 高层技能 | — | 监督 | 辅助 | 预测 |
| 边界标签 | — | 监督 | 门控监督 | 预测 |
| 外部记忆 | — | 禁用 | ✓ | ✓ |

**说明**: 光流仅在训练时作为监督信号，推理时无需估计，保证了系统的因果性和推理效率。

---

## 实验

### 数据集

| 数据集 | 特点 | 用途 |
|--------|------|------|
| LIBERO | 四套任务套件（Spatial/Object/Goal/Long） | 标准操作基准测试 |
| LIBERO-PLUS | 7种部署扰动（相机/初始位置/语言/光照/背景/噪声/布局） | 零样本鲁棒性评测 |
| RMBench | 9个需要记忆的操作任务，分 M(1)/M(n) 两类 | 记忆依赖任务评测 |
| 真实世界（自建） | 10个双臂操作任务，按难度分三级 | 真实机器人评测 |

### 实现细节

- **Backbone**: Qwen3-VL-4B-Instruct（高层规划器）
- **光流估计**: DPFlow（多视角光流预处理）
- **真实世界训练**: 每任务 400 条演示，5 轮 SFT
- **推理**: 全因果，无需测试时光流估计或视频生成
- **硬件平台**: 双臂机器人，配置多视角摄像头

### 可视化结果

真实世界困难任务（叠两盘、做早餐）在标准设置和泛化设置下均比 π₀.₅ 基线提升 25-30%；Stage II 预训练在两种动作表示（关节位置和末端姿态）上均一致性地改善了困难任务性能。

---

## 批判性思考

### 优点
1. **层次化时序抽象**: 同时学习低层运动和高层技能的双层表示，比单一粒度建模更具表达力
2. **边界触发记忆**: 选择性写入机制既避免了记忆无限增长，又保证了关键任务状态的保留
3. **因果推理**: 测试时不依赖光流或未来视频，工程实用性强
4. **全面评测**: 覆盖仿真标准基准（LIBERO）、鲁棒性基准（LIBERO-PLUS）、记忆专项基准（RMBench）和真实机器人，评测体系较为完整

### 局限性
1. **三阶段流水线开销大**: 光流提取、潜变量学习、技能发现和记忆训练的计算成本叠加，训练复杂度高
2. **工程依赖性强**: 层次化设计的性能高度依赖潜变量质量和边界检测精度，调参困难
3. **真实世界验证范围有限**: 仅在一个双臂平台上测试 10 个任务，跨硬件和更大数据规模的泛化性未验证
4. **M(n) 任务性能仍有差距**: 多次试验记忆场景下仅 19.8%，未超越专门针对记忆的 Mem-0 基线

### 潜在改进方向
1. 端到端联合训练三个阶段，减少分阶段训练引入的误差累积
2. 在更多机器人平台（单臂、移动操作）上验证方法的通用性
3. 研究更强的 M(n) 记忆规划策略，解决重复试验场景的挑战

### 可复现性评估
- [ ] 代码开源（未公开）
- [ ] 预训练模型（未提供）
- [x] 训练细节相对完整（有三阶段描述、损失函数、关键超参数说明）
- [x] 标准数据集可获取（LIBERO、LIBERO-PLUS、RMBench 均为公开基准）

---

## 关联笔记

### 基于
- [[Fast-WAM]]: 直接前置工作，HiMem-WAM 在其基础上增加层次结构和记忆机制
- [[WorldVLA]]: WAM 框架的代表性工作，作为主要对比基线
- [[Qwen3-VL]]: 使用其 4B-Instruct 版本作为高层规划器骨干

### 对比
- [[pi0.5|π₀.₅]]: 真实世界实验的主要基线，HiMem-WAM 困难任务提升 20-25%
- [[Diffusion Policy]]: 扩散策略基线，用于 RMBench 对比
- [[ACT]]: Transformer 策略基线
- [[MemoryVLA]]: 记忆感知 VLA 对比方法
- [[SAM2Act]]: 基于 SAM2 的记忆操作方法

### 方法相关
- [[World Action Model|WAM]]: 核心框架基础
- [[Optical Flow]]: 低层运动表示的监督信号来源
- [[DPFlow]]: 光流估计模块
- [[Variational Autoencoder|VAE]]: 低层 tokenizer 的建模方式
- [[Action Chunking]]: 动作块输出形式
- [[Attention Pooling]]: 分层技能发现中的池化机制
- [[Memory Gating]]: 核心记忆机制设计
- [[Hierarchical Latent Action]]: 本文提出的核心表示

### 硬件/数据相关
- [[LIBERO]]: 主要仿真基准数据集
- [[RMBench]]: 记忆依赖操作任务基准
- [[Dual-Arm Robot|双臂机器人]]: 真实世界实验平台

---

## 速查卡片

> [!summary] HiMem-WAM
> - **核心**: 分层潜变量框架 + 边界触发记忆门控，实现因果长视野操作
> - **方法**: Stage I 光流 VAE tokenizer → Stage II 分层技能发现 → Stage III 记忆条件化策略微调
> - **结果**: LIBERO 97.7% / LIBERO-PLUS 76.0% / RMBench 26.3% / 真实困难任务 +20-25% vs π₀.₅
> - **代码**: 未公开

---

*笔记创建时间: 2026-06-11*
