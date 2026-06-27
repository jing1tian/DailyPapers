---
title: "Hallucination in World Models is Predictable and Preventable"
method_name: "MMBench2"
authors: [Nicklas Hansen, Xiaolong Wang]
year: 2026
venue: arXiv
tags: [world-model, hallucination, dataset, flow-matching, data-coverage, curiosity-driven-exploration, model-based-rl]
zotero_collection: Robotics/World Model
image_source: online
arxiv_html: https://arxiv.org/html/2606.27326v1
created: 2026-06-27
---

# 论文笔记：Hallucination in World Models is Predictable and Preventable

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | UC San Diego |
| 日期 | June 2026 |
| 项目主页 | https://nicklashansen.com/mmbench2 |
| 对比基线 | [[Dreamer-v4]] (Dreamer 4), SD-VAE-MSE, Wan 2.1 VAE |
| 链接 | [arXiv](https://arxiv.org/abs/2606.27326) / Code: 项目主页发布数据集+代码+权重 |

---

## 一句话总结

> 世界模型的幻觉本质上是**数据覆盖问题**：作者提出三个无需标签即可在运行时预测幻觉的信号，并用它们指导训练采样与在线数据采集，仅用 50 条真实轨迹就能让 350M 世界模型适配全新未见环境。

---

## 核心贡献

1. **MMBench2 数据集**: 一个 427 小时、210 个任务、10 个领域的视觉世界建模数据集，具备真实动作/奖励标签、可交互的在线仿真环境，以及人类游玩等行为多样的数据
2. **幻觉的分阶段刻画**: 将[[世界模型]]的幻觉现象拆解为与流水线三个阶段一一对应的三种失效模式——[[Visual Tokenizer|tokenizer]] 的感知幻觉、动态模型的动作边缘化、多步 rollout 的场景发散
3. **三个幻觉预测信号**: 提出 tokenizer 往返残差 $u_r$、流不稳定性 $u_f$、跨种子方差 $u_s$ 三个无需标签和额外训练即可预测幻觉的信号，与 rollout 误差的 Spearman 相关系数达 $\rho \approx 0.80$
4. **覆盖感知训练方案**: 仅需将默认的按帧均匀采样改为按任务均匀采样，即可同时降低三种幻觉信号且不增加训练成本
5. **目标导向数据采集框架**: 将幻觉预测信号作为[[Curiosity-driven Exploration|好奇心奖励]]驱动在线数据采集，仅用 50 条真实轨迹便能让预训练好的 350M 世界模型适配完全未见过的环境，效果接近人类专家数据采集

---

## 问题背景

### 要解决的问题

现代生成式[[世界模型]]（如基于 [[Diffusion Transformer (DiT)]] / [[Flow Matching]] 的视频世界模型）能够渲染出越来越逼真、可由动作控制的未来画面，但它们经常**幻觉（hallucinate）**：rollout 视觉上保持流畅、表面上合理，却悄悄偏离了真实的物理动态。这一现象借用自语言模型文献中的"幻觉"概念，但在世界模型场景下后果更严重——幻觉轨迹会被直接喂给下游的规划器和策略（如 [[MPC]]），导致控制阶段的决策悄无声息地出错。

### 现有方法的局限

主流观点认为幻觉是一个**架构问题**，应该用更大的骨干网络和更多训练算力来解决（scaling law 式思路）。但这种思路缺乏对幻觉发生的位置（where）和原因（why）的理解——现有工作很少系统性地拆解幻觉究竟发生在 tokenizer、动作条件化还是多步 rollout 哪个阶段，也缺乏在运行时不依赖人工标签就能预测幻觉的轻量信号。

### 本文的动机

作者假设：**幻觉本质上集中在状态-动作空间的低覆盖区域**。如果这个假设成立，那么幻觉应当同时具备两个性质——**可预测**（能从运行时可得的信号中预测出来，类比深度集成等不确定性量化思路）和**可预防**（通过调整训练数据分布而非改变架构来缓解）。这意味着无需更大模型，仅通过数据层面的干预即可系统性降低幻觉。

---

## 方法详解

### 模型架构

本文复现并训练了一个遵循 [[Dreamer-v4|Dreamer 4]]（Hafner et al., 2025）整体架构与训练范式的 **350M 参数动作条件生成式世界模型**，采用两阶段训练范式：

- **输入**: 语言任务指令（通过 [[CLIP]]-ViT/B 编码）+ $224\times224$ RGB 观测 $o_t$ + 动作 $a_t$
- **Stage 1 — Tokenizer**: 对称编码器-解码器 Transformer，通过[[VideoMAE|masked autoencoding]] 训练
- **Stage 2 — Dynamics Model**: 250M 参数的 [[Block Causal Attention]] block-causal Transformer，在冻结 tokenizer 产生的空间潜变量 token 上，通过 [[Shortcut Flow-Matching]] 训练
- **输出**: 下一帧的潜在表示 $\hat{z}$，解码后得到预测帧
- **总参数**: 350M（编码器 50M + 解码器 50M + 动态模型 250M）

注意：本文所引用的"Dreamer 4"（Hafner et al., 2025, *Training agents inside of scalable world models*）是 tokenizer + block-causal Transformer + flow matching 的新一代架构，**并非**早期基于 [[RSSM]] 的 Dreamer 系列（[[Dreamer-v4]] 概念库笔记目前记录的是后者，二者架构有显著差异，引用时需注意区分）。

### 核心模块

#### 模块1: Video Tokenizer（感知阶段）

**设计动机**: 将高维像素观测压缩为低维潜在编码，同时保留足够的视觉细节以支持后续动态建模与人类可读性。

**具体实现**:
- 编码器以 stride 14 将 $224\times224$ 帧"patchify"为 256 个 patch token
- 前置 64 个可学习的 latent query，将潜在流投影到 64 维瓶颈，并用 $\tanh$ 激活约束到 $z \in [-1, 1]^{64\times64}$
- 解码器仅基于潜在编码重建图像
- 训练目标为 [[VideoMAE|masked reconstruction]]：每帧按 $\mathcal{U}(0, 0.9)$ 采样的比例随机遮盖 patch，用可学习 mask token 替换，损失（[[PSNR]] pixel MSE + [[LPIPS]]）只在被遮盖位置计算
- 各损失项按自身的 running RMS 归一化，降低对超参数的敏感度
- 编码器和解码器各 50M 参数

#### 模块2: Dynamics Model（动作条件化与 rollout 阶段）

**设计动机**: 在冻结 tokenizer 的潜空间上建模动作条件下的时序动态，同时支持快速、低延迟的多步 rollout 采样以服务闭环规划。

**具体实现**:
- 每个时间步的输入 token 序列包含：1 个动作 token（16 维 padded 动作经 2 层 MLP 编码）、1 个 shortcut 条件 token（编码噪声水平 $\sigma$ 和步长 $d$）、32 个打包的空间潜在 token、4 个 [[Vision Transformer]] register token，以及用于 reward head 和 [[BC]] policy 读出的可选 agent token
- 注意力同时包含帧内的空间自注意力和跨帧的 [[Block Causal Attention]] 因果时间自注意力，并使用 [[3D RoPE]] RoPE、[[QK Normalization]]、[[RMSNorm]] pre-norm
- 用 [[Shortcut Flow-Matching]] 目标训练，使推理时仅需 4~8 个 Euler 子步即可完成下一帧采样（相比传统扩散的几十至上百步大幅降低延迟）
- reward head 与 BC policy 均以每任务的 [[CLIP]] 语言指令嵌入为条件

---

## 关键公式

### 公式1: [[Visual Tokenizer]] 往返残差 $u_r$（感知幻觉预测器）

$$
u_r = \left\| \hat{z} - \mathrm{Encode}(\mathrm{Decode}(\hat{z})) \right\|
$$

**含义**: 把动态模型预测出的下一潜变量 $\hat{z}$ 解码成图像、再重新编码回潜空间，比较往返前后的差异。若 $\hat{z}$ 解码出的画面（例如错乱的场景布局或被"幻想"出来的物体）落在 tokenizer 学到的数据流形之外，就无法在重新编码后还原回原始 $\hat{z}$，产生较大的 $u_r$；这恰好捕捉了感知幻觉的本质症状。

**符号说明**:
- $\hat{z}$: 动态模型预测的下一帧潜变量
- $\mathrm{Decode}(\cdot)$ / $\mathrm{Encode}(\cdot)$: tokenizer 的解码器/编码器
- $\|\cdot\|$: 潜空间中的残差范数

### 公式2: [[Shortcut Flow-Matching]] 流不稳定性 $u_f$（动作边缘化预测器）

**含义**: $u_f$ 衡量在给定 $(\text{context}, \text{action})$ 对下，去噪器对干净目标的预测 $\hat{x}_1$ 在连续 Euler 积分子步之间的移动幅度（取后半段子步的平均）。一个条件信号充分、收敛良好的动态头会快速收敛到稳定的 $\hat{x}_1$（低 $u_f$）；而当动作条件提供的信息不足以约束预测时，$\hat{x}_1$ 会在各子步间持续震荡（高 $u_f$）——这正是"预测结果看似合理但忽略了真实输入动作"（动作边缘化）的信号。

**符号说明**:
- $\hat{x}_1$: 去噪器在当前 Euler 子步对"干净"目标的预测
- $(\text{context}, \text{action})$: 动态模型的条件输入（历史上下文 + 当前动作）

### 公式3: 跨种子方差 $u_s$（场景发散幻觉预测器）

**含义**: 在固定历史和动作的条件下，用 $N$ 个独立的去噪轨迹（不同随机种子）各自预测下一潜变量，计算这些预测之间的方差。种子之间出现分歧的区域，正是模型对该状态存在认知不确定性（epistemic uncertainty）的区域——多步 rollout 在这些区域会"发散"，对应场景发散型幻觉。

**符号说明**:
- $N$: 独立去噪轨迹（噪声种子）数量
- "inter-seed variance": 固定 context/action 下 $N$ 次独立采样所得潜变量预测的方差

### 公式4: 动态归一化（Dynamism Normalization）预测器

$$
u^{\text{norm}} \doteq u / m
$$

**含义**: 三个原始信号 $u_r, u_f, u_s$ 都会被场景活动量（高运动量的转移天然会放大 tokenizer 残差、流不稳定性和跨种子方差）所混淆。作者用每步潜表示变化的 RMS 量 $m$（按任务在数据集上取平均，或在线采集时用滚动估计）对原始信号归一化，使每个预测器衡量的是"相对于场景中正在发生多少变化"的模型不确定性。归一化后的 $u_r^{\text{norm}}, u_f^{\text{norm}}, u_s^{\text{norm}}$ 是论文后续所有分析采用的主预测器，并实验证明其 AUROC 始终优于未归一化的原始信号。

**符号说明**:
- $u$: 原始预测信号（$u_r$、$u_f$ 或 $u_s$）
- $m$: 潜表示逐步变化的 RMS（场景"动态程度"的代理指标）

### 公式5: Shortcut Euler 采样

$$
b = \frac{\hat{x}_1 - z}{1 - \sigma}, \qquad z \leftarrow z + b \cdot d
$$

**含义**: 推理阶段从 $z \sim \mathcal{N}(0, I)$ 出发，用 [[Shortcut Flow-Matching]] 训练得到的去噪器以步长 $d = 1/2^{\text{step}}$ 做 $K=8$ 个子步的 Euler 积分，逐步将噪声潜变量推向干净目标，生成下一帧的潜在编码。

**符号说明**:
- $\sigma$: 离散化噪声水平
- $d$: 当前 Euler 子步的步长
- $b$: 由当前去噪预测反推出的速度场估计

### 公式6: Rollout $\Delta$PSNR（场景发散幻觉的真值标签）

**含义**: 闭环 rollout 的 [[PSNR]] 相对"重复上一帧"基线（frame-repeating baseline）的差值。该指标 $\le 0$ 被用作"场景发散"幻觉的二元标签，因为此时模型的多步预测甚至不如简单地重复上一帧准确。

**符号说明**:
- Rollout $\Delta$PSNR $\le 0$: scene-divergent 幻觉标签
- Action shuffle ratio $\le 1.1$: action-ignored（动作边缘化）幻觉标签——衡量把动作打乱后预测变化的比例，越接近 1 说明模型越不依赖真实动作

---

## 关键图表

### Figure 1: Hallucination / 幻觉的三种失效模式

![Figure 1a - perceptual down](https://arxiv.org/html/2606.27326v1/visualizations/hallucination/hallucination-maniskill-action-down2.png)
![Figure 1a - perceptual down frame](https://arxiv.org/html/2606.27326v1/visualizations/hallucination/hallucination-maniskill-action-down-frame2.png)
![Figure 1b - action marginalization up](https://arxiv.org/html/2606.27326v1/visualizations/hallucination/hallucination-maniskill-action-up2.png)
![Figure 1b - action marginalization up frame](https://arxiv.org/html/2606.27326v1/visualizations/hallucination/hallucination-maniskill-action-up-frame2.png)
![Figure 1c - scene divergence t34](https://arxiv.org/html/2606.27326v1/visualizations/hallucination/pygame-pong_ep001_t0034_wm.png)
![Figure 1c - scene divergence t37](https://arxiv.org/html/2606.27326v1/visualizations/hallucination/pygame-pong_ep001_t0037_wm.png)

**说明**: 论文将幻觉分为三种失效模式：(i) **感知（Perceptual）**——对 OOD 场景的重建质量差，但结构上仍"合理"，未见过的场景布局被重建成相似的已见布局；(ii) **动作边缘化（Action marginalization）**——预测的下一帧视觉上合理却忽略了用户实际输入的动作；(iii) **场景发散（Scene divergence）**——多步 rollout 中误差累积导致动态变得不合理（例如球突然"传送"回赛场）。每种模式分别对应流水线的 tokenizer、动态模型条件化、多步 rollout 三个不同阶段。

### Figure 2: MMBench2 tasks / 任务多样性概览

![Figure 2a](https://arxiv.org/html/2606.27326v1/visualizations/tasks/pygame-bird-attack-2.png)
![Figure 2b](https://arxiv.org/html/2606.27326v1/visualizations/tasks/rd-open-drawer-1.png)
![Figure 2c](https://arxiv.org/html/2606.27326v1/visualizations/tasks/pygame-whirlpool-1.png)
![Figure 2d](https://arxiv.org/html/2606.27326v1/visualizations/domains/maniskill/ms-anymal-reach.png)
![Figure 2e](https://arxiv.org/html/2606.27326v1/visualizations/domains/pygame/pygame-point-maze-var1.png)
![Figure 2f](https://arxiv.org/html/2606.27326v1/visualizations/domains/box2d/bipedal-walker-rugged.png)
![Figure 2g](https://arxiv.org/html/2606.27326v1/visualizations/domains/pygame/pygame-air-hockey.png)
![Figure 2h](https://arxiv.org/html/2606.27326v1/visualizations/domains/pygame/pygame-coconut-dodge.png)
![Figure 2i](https://arxiv.org/html/2606.27326v1/visualizations/tasks/atari-double-dunk-5.png)
![Figure 2j](https://arxiv.org/html/2606.27326v1/visualizations/domains/mujoco/mujoco-walker.png)
![Figure 2k](https://arxiv.org/html/2606.27326v1/visualizations/domains/dmcontrol-extended/jumper-jump.png)

**说明**: 展示 MMBench2 中 210 个任务里抽样的 36 个，体现语料库在视觉和形态上的多样性。所有观测均为 $224 \times 224$ RGB 帧，涵盖 Atari、ManiSkill 机械臂操作、PyGame 2D 物理游戏、Box2D、MuJoCo、DMControl 扩展任务等 10 个领域。

### Figure 3: Dataset composition / 数据集组成

![Figure 3](https://arxiv.org/html/2606.27326v1/x1.png)

**说明**: 训练语料库（2000 万帧）中 210 个任务的每任务帧数统计，按降序排列并按领域着色。分布呈重尾：排名前 20 的任务占总帧数的 26%（以 Atari 为主），而排名后 20 的任务仅贡献 0.7%（多为短时序操作任务）。虚线标出每任务帧数的中位数（6.5 万帧）。这种长尾分布正是后续"覆盖感知训练"要解决的核心问题。

### Figure 4: Data coverage and hallucinations / 数据覆盖与幻觉的关系

![Figure 4a](https://arxiv.org/html/2606.27326v1/visualizations/panels/og-point-maze_hallu_u_r.png)
![Figure 4b](https://arxiv.org/html/2606.27326v1/visualizations/panels/pygame-rocket-collect_hallu_u_r.png)
![Figure 4c](https://arxiv.org/html/2606.27326v1/visualizations/panels/cup-catch_hallu_u_r.png)
![Figure 4d](https://arxiv.org/html/2606.27326v1/visualizations/panels/lunarlander-takeoff_hallu_u_r.png)

**说明**: 从左到右分别展示样例帧、关键 agent/物体位置的状态密度，以及世界模型的 tokenizer 往返残差 $u_r$。深蓝色表示高状态密度，深紫色表示高 $u_r$（高幻觉风险）。幻觉集中在低覆盖区域，尤其是已访问状态分布的边缘地带——这是支撑"幻觉是数据覆盖问题"这一核心论点的关键可视化证据。

### Figure 5: 三个幻觉预测器追踪同一真实 rollout 误差

![Figure 5](https://arxiv.org/html/2606.27326v1/x2.png)

**说明**: 每个点对应 9000 个来自 200 个训练任务的 held-out 24 帧序列之一（用预训练 base model 计算），紫色曲线为 8 个分箱的中位数。三个预测器（$u_r^{\text{norm}}, u_f^{\text{norm}}, u_s^{\text{norm}}$）均与 rollout $\Delta$PSNR 呈现强负相关（$\rho \approx 0.80$），说明它们追踪的是同一种底层误差。

### Figure 6: Data coverage by collection method / 不同数据采集策略的覆盖范围

**说明**: 展示 expert、curiosity（用 $u_r^{\text{norm}}$ 作为好奇心奖励）、human 三种在线数据采集策略在未见任务集中两个任务上的状态密度对比。Curiosity 策略在不依赖专家策略或人工演示的前提下，能主动探索覆盖低密度（高幻觉风险）区域。

### Figure 7 & 8: Visualization of task domains (appendix)

**说明**: 附录图展示 10 个任务领域中的样例任务可视化（分两部分，Figure 7 为 1/2，Figure 8 为 2/2），与正文 Figure 2 互为补充，进一步体现 MMBench2 的视觉多样性。

### Table 1: Coverage-aware training, by stage（覆盖感知训练的分阶段效果）

| Metric | Tok ft | Dyn ft | **Both** |
|--------|--------|--------|----------|
| Recon PSNR (dB) ↑ | **+0.46** | -0.01 | +0.44 |
| Action-shuffle ratio ↑ | +0.02 | **+0.27** | +0.29 |
| Rollout ΔPSNR (dB) ↑ | +0.42 | +0.68 | **+0.88** |
| $u_r^{\text{norm}}$ ↓ | -0.07 | -0.16 | **-0.20** |
| $u_f^{\text{norm}}$ ↓ | -0.03 | -0.06 | **-0.07** |
| $u_s^{\text{norm}}$ ↓ | -0.06 | -0.13 | **-0.14** |

**说明**: 数值为相对 base model 在 200 个任务 held-out 轨迹上的均值变化。"Tok ft"先用覆盖感知采样微调 tokenizer 30k 步、再用默认采样微调动态模型 30k 步；"Dyn ft"顺序相反；"Both"两阶段都用覆盖感知采样各微调 30k 步。**关键发现**: tokenizer 和动态模型都能从覆盖感知训练中获益，且两阶段同时采用覆盖感知采样时综合效果最佳——仅改变采样策略（按任务均匀而非按帧均匀），不增加任何训练成本，就能同时降低三种幻觉信号并提升 rollout 质量。

### Table 2: Targeted data collection for finetuning on 10 unseen tasks（目标导向数据采集微调结果）

| Method | Tok FT | Dyn FT | Recon PSNR ↑ | Rollout ΔPSNR ↑ | Action shuf. ↑ | $u_r^{\text{norm}}$ ↓ | Task perf. (MPC) ↑ |
|--------|--------|--------|------|------|------|------|------|
| Random policy | — | — | — | — | — | — | 0.118 |
| Base | ✗ | ✗ | 17.37 | -12.44 | 1.12 | 3.860 | — |
| Coverage-aware | ✗ | ✗ | 17.21 | -12.52 | 1.29 | 3.769 | 0.276 |
| No-op actions | ✗ | ✓ | 17.21 | -11.66 | 1.41 | 4.175 | — |
| No-op actions | ✓ | ✓ | 34.74 | +0.66 | 1.55 | 1.486 | 0.163 |
| Random policy | ✗ | ✓ | 17.21 | -11.29 | 1.73 | 4.167 | — |
| Random policy | ✓ | ✓ | 35.81 | +2.66 | 2.00 | 1.201 | 0.228 |
| Expert policy | ✓ | ✓ | 35.86 | +2.84 | 2.04 | 1.131 | 0.362 |
| Human play | ✓ | ✓ | 37.11 | +3.89 | **2.42** | 1.002 | 0.362 |
| Curiosity ($u_r^{\text{norm}}$) | ✓ | ✓ | 36.05 | +3.00 | 2.00 | 1.144 | 0.325 |
| **All (combined)** | ✓ | ✓ | **37.91** | **+4.02** | 2.34 | **0.975** | **0.390** |

**说明**: 在 10 个未见任务上，每种数据源各采集 50 条轨迹用于微调 tokenizer（50k 步）和动态模型（30k 步）。Task perf. 通过闭环 [[MPC]]（[[CEM]] 规划）评估，归一化到 $[0,1]$。**关键发现**: 仅靠预训练零样本迁移已能达到随机策略基线的 2.3 倍（0.276 vs. 0.118）；用 $u_r^{\text{norm}}$ 作为好奇心奖励、不依赖任何专家先验，仅用 50 条轨迹采集的数据微调后即可达到 0.325，接近专家/人类 oracle（0.362）效果的约 90%；综合所有数据源效果最佳（0.390）。

### Table 3: Comparison to off-the-shelf tokenizers（与现成 tokenizer 的对比）

| Tokenizer | Params | Latent/frame | PSNR Seen ↑ | PSNR Unseen ↑ | $\Delta_{S-U}$ | LPIPS Seen ↓ | LPIPS Unseen ↓ |
|-----------|--------|------|------|------|------|------|------|
| Ours-Base | 102M | 4096 | 38.29 | 17.34 | +20.95 | 0.011 | 0.389 |
| Ours-Coverage-aware | 102M | 4096 | 38.93 | 17.12 | +21.81 | 0.008 | 0.348 |
| Ours-Post-FT | 102M | 4096 | **39.66** | **38.04** | **+1.62** | **0.007** | **0.010** |
| SD-VAE-MSE | 84M | 3136 | 33.32 | 32.39 | +0.93 | 0.031 | 0.030 |
| Cosmos-CV8x8x8 (1.0) | 106M | 2048 | 32.80 | 32.72 | +0.08 | 0.050 | 0.042 |
| Wan 2.1 VAE | 127M | 4096 | 36.45 | 36.62 | -0.17 | 0.010 | 0.010 |
| DC-AE-f32c32 | 323M | 2048 | 31.49 | 32.15 | -0.66 | 0.035 | 0.031 |

**说明**: 在微调实验所用的 10 个"seen"和"unseen"任务集上评估重建 [[PSNR]] 和 [[LPIPS]]。本文自训 tokenizer 在训练集任务上大幅领先最强的现成 tokenizer（Wan 2.1 VAE），但在未微调情况下对新任务泛化较差（PSNR 从 38.29 跌至 17.34）；微调后（Post-FT）反超所有现成 tokenizer。**关键发现**: 现成大规模预训练 tokenizer 确实是缓解感知幻觉的有效途径，但领域内微调仍有明确价值。

### Table 4: MMBench2 vs. existing datasets（数据集对比）

**说明**: 附录表格，对比 MMBench2 与现有[[世界模型]]数据集（如 [[Open X-Embodiment]]、Procgen、RoboDesk 等）。MMBench2 的特点是混合质量数据、覆盖更多任务和领域、具备真值动作/奖励标签，并提供可在线交互的活体仿真环境，而非仅有离线录制视频。

### Table 5: Overview of task domains（任务领域总览）

**说明**: 附录表格，列出数据集覆盖的任务类型、状态/动作维度、时间跨度、奖励形式等，体现 10 个领域在任务设计上的广泛性（表格引用自 Hansen et al. [2026] 的 MMBench 数据）。

### Table 6: Task list for finetuning experiments（微调实验任务列表）

**说明**: 附录表格，列出用于目标导向数据采集实验（Table 2）的 10 个 seen + 10 个 unseen 任务的具体名单，unseen 任务是专门为 MMBench2 新设计的。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| MMBench2（训练集） | 200 任务，约 2000 万帧，$224\times224$ | 10 领域，长尾任务分布，真值动作+奖励 | 预训练 tokenizer + 动态模型 |
| MMBench2（测试集） | 300 万帧 | 与训练集同分布的 held-out 轨迹 | 离线评估 |
| MMBench2（微调任务集） | 10 seen + 10 unseen 任务，各 50 条轨迹/数据源 | unseen 任务专为本文新设计 | 目标导向数据采集实验（Table 2/8/9/10） |
| MMBench2（全集） | 427 小时，210 任务，10 领域 | 含人类游玩数据（通过专用网页交互界面采集 1,400 条轨迹） | 完整数据集发布 |

### 实现细节

- **Backbone**: tokenizer 编码器/解码器各 50M 参数；动态模型（含预测头）250M 参数；总计 350M
- **训练目标**: tokenizer 用 [[VideoMAE|masked autoencoding]]（pixel MSE + [[LPIPS]]）；动态模型用 [[Shortcut Flow-Matching]]
- **优化细节**: 各损失项按 running RMS 归一化后再加权，解耦损失权重与数据集/分辨率/骨干网络规模的耦合
- **训练步数**: tokenizer 预训练 300k 步（14 GPU-days）；动态模型预训练 180k 步（24 GPU-days）；最终 checkpoint（含微调）分别训练 380k / 210k 步，总计 58 GPU-days，上下文长度 $T=24$
- **采样**: 推理时用步长 $d=0.125$（即 $K=8$ 子步）的 shortcut Euler 积分器
- **硬件**: 8× NVIDIA H100 GPU
- **语言条件**: reward head 与 BC policy 均以 [[CLIP]]-ViT/B 的任务语言指令嵌入为条件
- **下游评估**: 闭环 [[MPC]]，[[CEM]] 规划器，规划时域 $H=32$，每 16 步重新规划一次，报告归一化到 $[0,1]$ 的任务得分（因为不同任务的原始 reward 量级相差数个数量级）

### 可视化结果

通过状态密度可视化（Figure 4、6），作者直观展示了幻觉与数据覆盖之间的空间对应关系：低密度区域（任务空间的边缘/罕见状态）正是 $u_r$ 等预测信号给出高值、同时真实 rollout 误差也最大的区域。这一可视化是支撑全文核心论点（幻觉=覆盖问题）最直接的定性证据，也解释了为什么仅靠按任务重新加权采样（而不改变模型容量）就能系统性改善 rollout 质量。

---

## 批判性思考

### 优点

1. **完整的因果链条**: 从"假设幻觉是覆盖问题" → "设计三个无监督预测信号验证假设" → "用同样信号设计训练/采集干预" → "在 unseen 任务上验证干预有效"，形成了一个自洽、可验证的完整故事，而不是单点的 trick 堆叠
2. **方法几乎与架构无关**: 三个预测信号（tokenizer 残差、流不稳定性、跨种子方差）的设计原理具有一般性，理论上可迁移到任何"tokenizer + 自回归/扩散动态模型"范式的现代生成式世界模型，而不仅限于本文复现的 Dreamer 4 架构
3. **诚实的开源与基础设施投入**: 完整数据集、代码、模型权重、可交互浏览器界面均开源，且专门构建了人类游玩数据采集网页界面（Figure 9），体现了对可复现性的重视

### 局限性

1. **规模限制**: 研究在 350M 参数、210 个仿真任务规模下完成，论文自陈"是否能推广到十亿参数模型或带传感器噪声/部分可观测性的真实机器人数据仍是未解决的实证问题"
2. **仿真环境局限**: 所有任务均为仿真器（ManiSkill、PyGame、Box2D、MuJoCo、DMControl 等），没有真实机器人实验，sim-to-real 的幻觉行为是否一致尚未验证
3. **好奇心采集的目标特异性**: Curiosity 数据采集虽接近专家/人类水平（约 90%），但仍未超越，说明纯探索信号无法完全替代任务相关的行为先验，"用幻觉信号采集数据"和"采集对下游任务真正有用的数据"之间仍有差距（论文 Discussion 部分也坦承了这点：闭环 MPC 评估主要衡量的是任务相关、较窄状态-动作子空间内的模型精度）

### 潜在改进方向

1. 将三个预测信号结合进行多步规划时的不确定性感知 MPC（即在线规划阶段主动规避高 $u^{\text{norm}}$ 区域），而非仅用于离线数据采集
2. 探索预测信号与现成大规模预训练 tokenizer（如 Wan 2.1 VAE）结合的可能性，弥补本文 in-domain tokenizer 对未见任务泛化差的问题（Table 3）
3. 扩展到真实机器人数据，验证 $u_r, u_f, u_s$ 在存在传感器噪声和部分可观测性时是否依然有效

### 可复现性评估
- [x] 代码开源
- [x] 预训练模型
- [x] 训练细节完整
- [x] 数据集可获取

---

## 关联笔记

### 基于
- [[Dreamer-v4]]（Dreamer 4, Hafner et al., 2025）: 本文复现并分析的 tokenizer+block-causal Transformer+flow matching 世界模型架构（注意与 Concepts 库中现有 Dreamer-v4 笔记记录的 [[RSSM]] 版本架构不同，二者实为不同代际的同名方法）
- [[Shortcut Flow-Matching]]: 动态模型的训练目标，支持少步 Euler 采样
- [[VideoMAE]]: tokenizer 的 masked autoencoding 训练范式

### 对比
- [[Open X-Embodiment]]: 真实机器人数据集，作为 MMBench2（仿真+真值动作奖励）的对比对象
- Wan 2.1 VAE / SD-VAE-MSE / Cosmos-CV8x8x8 / DC-AE-f32c32: 四个现成 tokenizer baseline，用于验证 in-domain tokenizer 训练的价值

### 方法相关
- [[世界模型]]: 核心研究对象
- [[Curiosity-driven Exploration]]（Sekar et al., 2020）: 本文将幻觉预测信号改造为好奇心奖励，用于目标导向数据采集
- [[MPC]] / [[CEM]]: 下游任务性能评估所用的闭环规划框架
- [[PSNR]] / [[LPIPS]]: 重建质量与感知质量评估指标
- [[Spearman相关系数]]: 验证预测信号与真实 rollout 误差相关性的统计工具
- [[Block Causal Attention]]: 动态模型 Transformer 的注意力掩码策略

### 硬件/数据相关
- MMBench2: 本文发布的 427 小时、210 任务、10 领域数据集（扩展自 Hansen et al. 2026 的 MMBench）

---

## 速查卡片

> [!summary] Hallucination in World Models is Predictable and Preventable (MMBench2)
> - **核心**: 世界模型幻觉 = 数据覆盖问题，可用无监督信号预测、可用采样/采集策略缓解
> - **方法**: 三预测器（tokenizer 残差 $u_r$ / 流不稳定性 $u_f$ / 跨种子方差 $u_s$，均做动态归一化）+ 覆盖感知训练采样 + 好奇心驱动目标数据采集
> - **结果**: 预测器与 rollout $\Delta$PSNR 相关性 $\rho\approx0.80$；仅用 50 条好奇心采集轨迹微调，未见任务表现达专家/人类水平的 ~90%（0.325 vs 0.362）
> - **代码**: [项目主页](https://nicklashansen.com/mmbench2) / [arXiv](https://arxiv.org/abs/2606.27326)

---

*笔记创建时间: 2026-06-27*
