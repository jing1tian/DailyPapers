---
title: "LaLa-World: Scaling Latent World Models with Natural Language Actions"
method_name: "LaLa-World"
authors: [Junha Song, Geonmo Gu, Jeongeun Park, Heejeong Nam, Lucas Maes, Byeongho Heo, Dongyoon Han, Sangdoo Yun]
year: 2026
venue: arXiv
tags: [world-model, latent-world-model, natural-language-actions, jepa, mpc-planning, robot-learning, policy-learning]
zotero_collection: Robotics/World Model
image_source: online
arxiv_html: https://arxiv.org/html/2610.27851v2
created: 2026-10-22
---

# 论文笔记：LaLa-World: Scaling Latent World Models with Natural Language Actions

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | NAVER AI Lab; Georgia Institute of Technology; Brown University; Mila & Université de Montréal |
| 日期 | October 2026 |
| 项目主页 | [june-page.github.io/lala-world/](https://june-page.github.io/lala-world/) |
| 对比基线 | [[DINO-WM]], [[DreamerV3]], [[IRIS]], [[LeWorldModel\|LeWM]], [[Cosmos-Policy\|Cosmos Policy]] |
| 链接 | [arXiv](https://arxiv.org/abs/2610.27851) / [Code](https://github.com/junha1125/LaLa-World) |

---

## 一句话总结

> LaLa-World 是一个以[[语言动作接口|自然语言为统一动作接口]]的 [[JEPA|JEPA 式]]潜在世界模型，通过将数值动作转化为语言文本，同时融合仿真、机器人和视频数据联合训练，在 MPC 规划和策略学习上达到当前最优性能，同时具备约 6× 更低训练成本和 5.5× 更快推理速度。

---

## 核心贡献

1. **[[语言动作接口|语言统一动作接口]]**: 将来自仿真器和机器人的数值动作转化为自然语言描述，以文本作为异构数据来源的统一桥梁，使世界模型能够同时从仿真数据、真实机器人数据和互联网视频数据中学习。
2. **LaLa-World 架构**: 基于 [[JEPA]] 范式构建的潜在世界模型，使用冻结的 [[DINO|DINOv2/v3]] 视觉编码器 + Qwen3.5-0.8B 语言模型作为预测器，在潜空间做转移预测，无需像素重建。
3. **LaLa-Policy**: 在 LaLa-World 预训练基础上微调得到的机器人操作策略，集成[[动作分块|动作块]]预测和扩散动作头，在 [[LIBERO]] 和 [[Robocasa|RoboCasa]] 上达到与 [[Cosmos-Policy|Cosmos Policy]] 相当的性能，推理速度快 5.5×。

---

## 问题背景

### 要解决的问题

现有潜在世界模型（[[JEPA]] 风格）在单一领域内表现良好，但难以扩展到多源异构数据（仿真 + 真实机器人 + 视频）——因为不同数据源的动作空间完全不同（低维关节角度 vs. 末端执行器位置 vs. 视频帧级语言描述），无法共享一个统一的动作条件接口。

### 现有方法的局限

- **生成式世界模型**（如 [[Cosmos-Policy]] 基于的 [[Cosmos Predict2|Cosmos Predict2]]）：支持语言条件，可利用视频数据，但需要像素级重建，训练成本极高，推理速度慢。
- **动作条件潜在世界模型**（如 [[DINO-WM]]、[[JEPA]] 系列）：训练高效，但动作空间 $c_t \in \mathbb{R}^k$ 是数值化的，无法跨领域共享，不支持视频数据联合训练。
- **单独训练**：每个领域分别训练世界模型，无法利用数据多样性带来的泛化能力。

### 本文的动机

自然语言是一种天然的统一接口——任何领域的数值动作都可以用模板化文本表达，而语言模型（LLM）天然擅长理解文本，同时互联网上大量视频数据也带有语言描述。因此，将 [[JEPA]] 预测器替换为以语言为条件的 LLM，就可以一举统一所有数据源。

---

## 方法详解

### 模型架构

LaLa-World 采用 **[[JEPA|JEPA 式潜在预测]]** 架构：

- **输入**: 语言动作文本 $\ell_t$（模板化转换自数值动作）+ 视觉观测 $x_t$
- **视觉编码器 $E_\theta$**: 冻结的 [[DINO|DINOv2-S/14]] 或 [[DINO|DINOv3-L/16]]（主实验用 DINOv3-L/16）
- **预测器 $P_\phi$**: Qwen3.5-0.8B（LLM），直接读取语言动作 token，通过两层 MLP 投影对齐到视觉潜空间
- **位置编码**: 3D 位置嵌入（空间 H×W + 时间帧维度）
- **输出**: 预测的下一帧潜在表示 $\hat{e}_{t+1}$，通过 learned query readout 聚合 patch
- **总参数**: 约 ~0.8B（预测器）+ 冻结编码器

### 核心模块

#### 模块 1: 语言动作转换（Language-Conditioned Transition Data）

**设计动机**: 利用[[语言动作接口|语言作为统一动作接口]]，将数值动作转化为可被 LLM 理解的文本，同时允许接入视频数据（视频描述自然是语言）。

**具体实现**:
- 使用预定义模板将数值动作转化为文字，例如：
  - 关节角速度 → `"Move joint [j] at velocity [v]"`
  - 末端执行器位移 → `"Move gripper [Δx, Δy, Δz] with [open/close]"`
- 视频数据（OpenVid-1M）：使用图像描述语言；图像编辑对（OmniEdit、ShareGPT4o-Image）：使用编辑指令
- 消融实验表明，数值型语言动作明显优于纯定性描述（如"向左"），因为数值精度对精细操作至关重要

#### 模块 2: 潜在世界预测器（JEPA Predictor with LLM）

**设计动机**: 利用[[JEPA|JEPA 范式]]在潜空间做预测，不重建像素，大幅降低训练成本；同时用 LLM 作为预测器骨干以充分利用语言动作。

**具体实现**:
- Qwen3.5-0.8B 接受语言动作 token 序列（经 tokenizer）
- 视觉 patch token 经 MLP 投影后与语言 token 一起输入 LLM（concatenation）
- 使用窗口 $w = 1$~$3$ 帧上下文，均匀采样
- 3D positional embedding：空间 grid + 时间步索引

#### 模块 3: LaLa-Policy（策略微调头）

**设计动机**: 将 LaLa-World 的世界预测能力迁移到机器人策略，通过联合预测未来潜在表示和动作，实现高效策略学习。

**具体实现**:
- 添加 N 个[[动作分块|动作查询（action queries）]]（LIBERO: 16，RoboCasa: 32）
- 动作生成器 $A_\psi$：16 层 [[扩散策略|扩散 DiT]]（Isaac-GR00T 风格），宽度 1,536，使用[[流匹配|流匹配]]训练
- 视觉编码器升级为 SigLIP2-So400M/14（224×224）以获得更强的语义表示
- 联合损失：世界预测损失 + 动作预测损失（系数均为 1）

---

## 关键公式

### 公式 1: [[Action-Conditioned World Model|动作条件潜在世界模型]]

$$
e_t = E_\theta(x_t), \quad \hat{e}_{t+1} = P_\phi(e_{t-w+1:t},\ c_{t-w+1:t}), \quad c_t \in \mathbb{R}^k
$$

**含义**: 基础 LWM 形式——编码器将观测映射为潜在表示，预测器以窗口内历史潜在 + 数值动作条件预测下一帧潜在。

**符号说明**:
- $E_\theta$: 视觉编码器（如 DINOv2）
- $P_\phi$: 潜在预测器
- $c_t \in \mathbb{R}^k$: 数值动作向量
- $w$: 上下文窗口大小（1-3 帧）

### 公式 2: [[JEPA|JEPA 训练损失]]

$$
\mathcal{L} = \|\hat{e}_{t+1} - e_{t+1}\|^2
$$

**含义**: 潜空间内的 L2 预测损失，不重建像素，只对齐潜在表示。训练时 stop-gradient 应用于目标 $e_{t+1}$。

**符号说明**:
- $\hat{e}_{t+1}$: 预测的下一帧潜在表示
- $e_{t+1}$: 目标（真实）潜在表示（stop-gradient）

### 公式 3: [[MPC|MPC 规划代价函数]]

$$
\mathcal{C} = \|\hat{e}_{t+H} - e_{\text{goal}}\|^2, \quad e_{\text{goal}} = E_\theta(x_{\text{goal}})
$$

**含义**: 规划目标是最小化 H 步后的预测潜在与目标图像潜在之间的 L2 距离，通过 [[CEM|CEM（Cross-Entropy Method）]]迭代优化语言动作序列。

**符号说明**:
- $H$: 规划时域（planning horizon）
- $e_{\text{goal}}$: 目标图像的编码潜在表示
- $x_{\text{goal}}$: 目标观测图像

### 公式 4: [[LaLa-Policy]] 联合预测

$$
(\hat{e}_{t+N},\ h_{t:t+N-1}) = P_\phi(E_\theta(x_t),\ \ell), \quad \hat{a}_{t:t+N-1} = A_\psi(\epsilon;\ h_{t:t+N-1}), \quad \epsilon \sim \mathcal{N}(0, I)
$$

**含义**: LaLa-Policy 的预测器同时输出未来潜在表示 $\hat{e}_{t+N}$ 和动作条件隐变量 $h$，动作生成器基于 $h$ 用扩散过程解码动作块。

**符号说明**:
- $N$: 动作块长度（chunk size）
- $h_{t:t+N-1}$: 动作查询的隐变量序列（来自预测器各时间步输出）
- $A_\psi$: 扩散动作生成器（DiT）
- $\epsilon$: 标准高斯噪声

### 公式 5: [[语言动作接口|语言敏感度]]

$$
S(z) = 1 - \frac{\|\hat{e}_{t+1}(z) - e_{t+1}\|_2}{\|e_t - e_{t+1}\|_2}
$$

**含义**: 衡量语言动作 $z$ 对预测准确性的贡献——敏感度接近 1 说明语言描述对预测有显著帮助，接近 0 说明模型忽略了语言输入。

**符号说明**:
- $z$: 语言动作输入
- $S(z)$: 语言敏感度（1 为最好）

### 公式 5（附录）: DROID Action Score

$$
\text{Action Score} = \max(0,\ 800 \cdot (0.1 - \text{error}_{xyz}))
$$

**含义**: 真实机器人 DROID 评测中的连续奖励信号，当末端执行器 XYZ 位置误差小于 0.1m 时给予正向得分，越精确得分越高（最高 80 分）。

**符号说明**:
- $\text{error}_{xyz}$: 末端执行器三维位置误差（米）
- 阈值 0.1m：超过此误差则得分为 0

### 公式 6: [[流匹配|Flow Matching 插值]]（LaLa-Policy 训练）

$$
e^{(\tau)}_{t+1} = (1 - \tau)e_{t+1} + \tau\epsilon, \quad \epsilon \sim \mathcal{N}(0, I), \quad \tau \sim \mathcal{U}(0, 1)
$$

**含义**: 在潜在目标和高斯噪声之间线性插值，用于训练动作生成器的流匹配目标。

**符号说明**:
- $\tau$: 噪声时间步（均匀采样自 [0,1]）
- $e^{(\tau)}_{t+1}$: 加噪后的目标潜在
- $\epsilon$: 标准高斯噪声

---

## 关键图表

### Figure 1: LaLa-World 系统概览

![Figure 1](https://arxiv.org/html/2610.27851v2/teaser.png)

**说明**: LaLa-World 的整体设计——以自然语言作为统一接口，连接仿真控制数据（Wall, PointMaze, Push-T, MetaWorld 等）、真实机器人数据（DROID）和互联网视频数据（OpenVid-1M, OmniEdit, ShareGPT4o）。所有数据通过语言动作描述共享同一个 [[JEPA|JEPA 式]]预测器（Qwen3.5-0.8B），联合训练提升泛化性。

### Figure 2: 语言动作 MPC 规划流程

![Figure 2](https://arxiv.org/html/2610.27851v2/lang-planning.png)

**说明**: 展示 LaLa-World 进行 [[MPC|MPC 规划]]的完整流程。[[CEM|CEM]] 优化器在语言动作空间上迭代采样候选序列，LaLa-World 滚动预测未来潜在状态，选择最小化目标距离的动作序列执行。

### Figure 3: 语言条件转移数据示例

![Figure 3](https://arxiv.org/html/2610.27851v2/data-examples.png)

**说明**: 不同来源数据的语言动作表示示例：(a) 仿真器动作 → 模板化文字（如"Move joint 3 at velocity -0.42"）；(b) 真实机器人末端执行器位移 → 语言描述；(c) 互联网视频 → 图像描述或编辑指令。展示了[[语言动作接口|语言作为统一接口]]如何桥接异构数据。

### Figure 4: 潜在 vs. 生成式世界模型对比

![Figure 4](https://arxiv.org/html/2610.27851v2/latent-vs-pixel.png)

**说明**: 在相同训练预算下，LaLa-World（潜在方法）vs. RAEv2（生成式方法，基于 [[Cosmos Predict2|RAE]]）的规划成功率随训练时长的变化曲线。LaLa-World 在约 2 小时内达到 RAEv2 12 小时的性能，最终以 59.4% 超越 RAEv2 的 48.0%（5步推理），且无需像素重建目标。

### Figure 5: DROID 想象力可视化

![Figure 5](https://arxiv.org/html/2610.27851v2/imagination-visualization.png)

**说明**: LaLa-World 在 [[DROID]] 真实机器人数据上的潜在滚动展示。将预测的潜在表示通过单独训练的解码器重建为像素图像，验证世界模型对真实操作场景的理解能力。

### Figure 6: LaLa-Policy 预测未来可视化

![Figure 6](https://arxiv.org/html/2610.27851v2/policy-visualization.png)

**说明**: LaLa-Policy 在 [[LIBERO]] 和 [[Robocasa|RoboCasa]] 任务上的策略执行与预测未来对比。展示了策略在执行动作同时预测未来状态的能力，验证了世界模型预训练对策略学习的迁移价值。

---

### Table 1: MPC 规划性能对比

**仿真任务（5 个任务平均成功率 %）：**

| 方法 | 联合训练 | Wall | PointMaze | Push-T | MetaWorld-R | MetaWorld-RW | Rope | Granular | 平均 |
|------|---------|------|-----------|--------|-------------|--------------|------|----------|------|
| [[DreamerV3]] | ✗ | — | — | — | — | — | — | — | 30.2 |
| TD-MPC2 | ✗ | — | — | — | — | — | — | — | 17.7 |
| [[IRIS]] | ✗ | — | — | — | — | — | — | — | 26.0 |
| [[LeWorldModel\|LeWM]] | ✗ | — | — | — | — | — | — | — | 28.1 |
| [[DINO-WM]] | ✗ | — | — | — | — | — | — | — | 62.3 |
| JEPA-WMs | ✗ | — | — | — | — | — | — | — | 64.6 |
| **LaLa-World** | **✓** | **95.8** | **56.2** | **81.2** | **75.0** | **52.1** | — | — | **72.1** |

**真实机器人 DROID + 零样本 RoboCasa（平均 action score / 成功率）：**

| 方法 | DROID (Action Score) | RoboCasa-R (SR%) | RoboCasa-P (SR%) | 平均 |
|------|---------------------|-----------------|-----------------|------|
| [[DINO-WM]] | 36.8 | — | — | — |
| [[V-JEPA|V-JEPA 2-AC]] | 42.9 | — | — | — |
| JEPA-WMs | 44.8 | 31.2 | 43.8 | 39.9 |
| **LaLa-World** | **40.2** | **40.6** | **46.9** | **50.5** |

**关键发现**: 联合训练（✓）相比独立训练平均提升 7.5%；LaLa-World 在所有指标上超过所有独立训练的 JEPA 系列基线。

### Table 2: 机器人操作策略性能对比

**LIBERO（平均成功率 %，4 个子套件）：**

| 方法 | 预训练 | LIBERO-Spatial | LIBERO-Object | LIBERO-Goal | LIBERO-Long | 平均 |
|------|--------|---------------|--------------|------------|------------|------|
| OpenVLA-OFT | 语言 VLA | — | — | — | — | 97.1 |
| π₀.₅ | 生成式 WAM | — | — | — | — | 96.9 |
| [[Cosmos-Policy\|Cosmos Policy]] | 生成式 WM | — | — | — | — | 97.3 |
| **LaLa-Policy** | **LaLa-World** | — | — | — | — | **97.7** |

**RoboCasa（成功率 %，24 任务，50 演示）：**

| 方法 | SR % |
|------|------|
| [[Cosmos-Policy\|Cosmos Policy]] | 56.0 |
| **LaLa-Policy** | **57.3** |

**推理速度（每轨迹秒）：**

| 任务 | LaLa-Policy | Cosmos Policy | 加速倍数 |
|------|-------------|--------------|---------|
| LIBERO | **0.86s** | 4.45s | **5.2×** |
| RoboCasa | **2.12s** | 11.71s | **5.5×** |

### Table A: ImageNet 变换检索 MRR（可交流世界模型评估）

| 方法 | MRR |
|------|-----|
| IWM (default) | 0.79 |
| IWM (strong jitter) | 0.85 |
| LaLa-World, source latent | 0.024 |
| LaLa-World, predicted, empty language | 0.026 |
| LaLa-World, predicted, language action | **0.856** |

**关键发现**: 只有在提供语言动作条件时，LaLa-World 的预测潜在才能准确匹配目标变换，说明语言动作编码了真实的状态转移信息。

### Table B: LLM 规划器 vs. CEM（可交流世界建模）

| 规划器 | Wall (SR%) | PointMaze (SR%) | RoboCasa-R (SR%) | DROID | 每轨迹采样数 |
|--------|-----------|----------------|-----------------|-------|------------|
| CEM | 95.8 | 56.2 | 40.6 | 40.2 | 9,000–54,000 |
| Claude Opus 5 | **100.0** | 31.2 | **53.1** | 36.2 | ~90 |
| GPT-6 Astra | 91.3 | 26.1 | 17.4 | **44.7** | ~8–96 |

**关键发现**: LLM 规划器可以大幅减少采样次数（100× 减少），在简单任务（Wall）上表现更好，但在需要精细数值控制的任务（Push-T）上不稳定。

### Table C: 预测器初始化和训练范围消融

| 初始化 | 训练方式 | Wall | PointMaze | Push-T | DROID | RoboCasa-R | RoboCasa-P | 平均 |
|--------|---------|------|-----------|--------|-------|-----------|-----------|------|
| 预训练 LLM | 单域 | 97.9 | 66.7 | 70.8 | 38.3 | 21.9 | 31.2 | 54.5 |
| 随机初始化 | 联合 | 97.9 | 50.0 | 79.2 | 31.6 | 31.2 | 43.8 | 55.6 |
| **预训练 LLM** | **联合** | **95.8** | **56.2** | **81.2** | **40.2** | **40.6** | **46.9** | **60.2** |

**关键发现**: 预训练 LLM 初始化 + 联合训练的组合是最优的，验证了两者都有独立贡献。

### Table D: 视觉编码器消融（10 个 benchmark 平均）

| 编码器 | JEPA-WMs | LaLa-World |
|--------|---------|-----------|
| DINOv2-S/14 | 44.6 | 47.7 |
| DINOv3-L/16 | 49.2 | 53.7 |
| SigLIP2-So400m/14 | — | 37.3 |
| **DINOv2 + DINOv3** | — | **63.0** |

**关键发现**: DINOv3-L/16 > DINOv2-S/14，双编码器组合（参考设置）进一步提升至 63.0%，超过主表 61.3%。

### Table E: 数值 vs. 定性语言动作消融

| 动作类型 | Wall | PointMaze | Push-T | MetaWorld-R | MetaWorld-RW | RoboCasa-R | RoboCasa-P |
|---------|------|-----------|--------|-------------|--------------|-----------|-----------|
| 定性描述 | 83.3 | 47.9 | 4.2 | 70.8 | 60.4 | 28.1 | 25.0 |
| **数值化语言** | **95.8** | **56.2** | **81.2** | **75.0** | **52.1** | **40.6** | **46.9** |

**关键发现**: 数值化语言动作在精细任务（Push-T: 81.2 vs. 4.2）上有决定性优势，证明精确数值信息对精细操作至关重要。

---

## 实验

### 数据集

| 数据集 | 规模 | 类型 | 用途 |
|--------|------|------|------|
| Wall | 86,296 | 2D 仿真控制 | 训练 |
| PointMaze | 186,675 | 2D 仿真控制 | 训练 |
| Push-T | 2,199,488 | 2D 仿真操作 | 训练 |
| MetaWorld | 1,162,310 | 3D 仿真操作 | 训练 |
| Rope | 18,525 | 软体仿真 | 训练 |
| Granular | 18,582 | 颗粒仿真 | 训练 |
| [[DROID]] | 1,500,000 | 真实机器人 | 训练 |
| OpenVid-1M | 1,023,363 | 互联网视频 | 训练 |
| OmniEdit | 295,353 | 图像编辑 | 训练 |
| ShareGPT4o-Image | 45,613 | 图像描述 | 训练 |
| [[Robocasa\|RoboCasa]] | — | 仿真 + 真实 | 零样本评测 |

### 实现细节

- **视觉编码器**: 冻结 DINOv3-L/16（主实验）；DINOv2-S/14 用于消融
- **预测器**: Qwen3.5-0.8B，全参数更新
- **优化器**: AdamW，$\beta = (0.9, 0.98)$，weight decay 0.05
- **学习率调度**: 线性 warmup 从 $1 \times 10^{-5}$ 到 $2 \times 10^{-4}$（前 1,000 步），cosine decay 到 $1 \times 10^{-6}$
- **Batch Size**: 512（有效）
- **训练轮数**: 20 epochs（约每 epoch 100 万样本）
- **硬件**: 8 × H100 80GB GPU，bfloat16 混合精度
- **梯度裁剪**: 1.0

**LaLa-Policy 微调细节**:
- LIBERO: 8 个 epoch；RoboCasa (50 demos): 12 个 epoch；RoboCasa (300 demos): 3 个 epoch
- SigLIP2-So400M/14（224×224）替代 DINOv3 作为视觉编码器
- 动作块大小：16（LIBERO），32（RoboCasa）
- 评测：每任务 50 轮，LIBERO 步数预算 220/280/300/520

### MPC 规划设置

| Benchmark | Horizon H | CEM 迭代×样本×精英 | 成功判据 |
|-----------|-----------|-------------------|---------|
| Wall | 6 | 30×300×10 | 位置误差 < 4.5 |
| PointMaze | 6 | 30×300×10 | 位置误差 < 0.5 |
| Push-T | 6 | 30×300×10 | 位置 < 20 且 角度 < 20° |
| MetaWorld | 6 | 15×300×10 | 任意步成功 |
| DROID | 3 | 15×300×10 | Action Score |
| RoboCasa | 3 | 15×300×10 | 子任务最终成功 |

---

## 批判性思考

### 优点
1. **架构优雅**: 用语言接口解决异构数据融合问题，思路简洁且有效，避免了复杂的多头动作映射。
2. **训练高效**: 潜空间预测（无像素重建）使训练成本仅为生成式方法的约 1/6，推理速度快 5.5×，更适合实际部署。
3. **零样本泛化**: 在未参与 WM 训练的 RoboCasa 任务上仍表现良好，展示了联合训练带来的真实泛化能力。

### 局限性
1. **固定动作模板**: 语言动作转换依赖预设模板，对于动作空间差异极大的新型机器人平台需要人工设计新模板，缺乏自动化。
2. **DROID 性能略低于预期**: 真实机器人数据上 LaLa-World 的 action score（40.2）低于 JEPA-WMs 在 DROID 子集上的表现（44.8），可能源于视频数据的"语言噪声"干扰。
3. **无本体感知（Proprioception）**: 当前架构不使用关节角度等本体感知信息，可能限制精细操作性能。
4. **LLM Actor 不稳定**: Claude Opus 5 和 GPT-6 Astra 作为规划器在某些任务上表现不一致，LLM 规划器的可靠性还需提升。

### 潜在改进方向
1. 探索自动化语言动作模板生成（利用 LLM 自动从动作空间描述中生成模板）
2. 融合本体感知信息（关节角度、末端执行器状态）到 LaLa-World 的预测输入中
3. 研究更长视野（horizon）规划和层次化规划策略

### 可复现性评估
- [x] 代码开源（github.com/junha1125/LaLa-World）
- [ ] 预训练模型（未提及）
- [x] 训练细节完整（附录 D 详细列出所有超参数）
- [x] 数据集可获取（DROID、OpenVid-1M 等均为公开数据集）

---

## 关联笔记

### 基于
- [[JEPA]]: LaLa-World 的核心范式——潜空间预测，不重建像素
- [[DINO]]: 使用 DINOv2/v3 作为冻结视觉编码器
- [[V-JEPA]]: JEPA 在视频预测上的应用，是 LaLa-World 的直接前驱
- [[DINO-WM]]: 另一个 JEPA 式 LWM 基线，LaLa-World 在其基础上扩展了多源数据和语言接口

### 对比
- [[DreamerV3]]: 生成式世界模型，在 MPC 规划上大幅落后于 LaLa-World（30.2 vs. 72.1）
- [[Cosmos-Policy\|Cosmos Policy]]: 生成式方法代表，LaLa-Policy 达到相当性能但快 5.5×
- [[LeWorldModel\|LeWM]]: 同期潜在世界模型，联合训练前的 LaLa-World 已超越（72.1 vs. 28.1）

### 方法相关
- [[MPC\|Model Predictive Control]]: 规划阶段使用的控制框架
- [[CEM\|Cross-Entropy Method]]: MPC 中的动作优化器
- [[流匹配\|Flow Matching]]: LaLa-Policy 动作生成器的训练目标
- [[扩散策略\|Diffusion Policy]]: LaLa-Policy 动作头的基础
- [[动作分块\|Action Chunking]]: LaLa-Policy 的动作预测形式

### 硬件/数据相关
- [[DROID]]: 真实机器人数据集，用于世界模型训练和评测
- [[LIBERO]]: 操作策略 benchmark，LaLa-Policy 主要评测场景
- [[Robocasa\|RoboCasa]]: 多场景操作 benchmark，用于零样本策略迁移评测
- [[MimicGen]]: RoboCasa 300 demos 演示数据来源

---

## 速查卡片

> [!summary] LaLa-World (2026)
> - **核心**: 以自然语言作为统一动作接口的 JEPA 式潜在世界模型，融合仿真+机器人+视频数据联合训练
> - **方法**: 冻结 DINOv3-L/16 编码器 + Qwen3.5-0.8B 语言模型预测器；数值动作 → 语言模板；CEM 规划
> - **结果**: MPC 规划 72.1%（vs. DINO-WM 62.3%）；LaLa-Policy 达 Cosmos Policy 水平但快 5.5×；训练成本仅 1/6
> - **代码**: [github.com/junha1125/LaLa-World](https://github.com/junha1125/LaLa-World)

---

*笔记创建时间: 2026-10-22*
