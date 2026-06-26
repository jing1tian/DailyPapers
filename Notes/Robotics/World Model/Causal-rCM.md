---
title: "Causal-rCM: A Unified Teacher-Forcing and Self-Forcing Open Recipe for Autoregressive Diffusion Distillation in Streaming Video Generation and Interactive World Models"
method_name: "Causal-rCM"
authors: [Kaiwen Zheng, Guande He, Min Zhao, Jintao Zhang, Huayu Chen, Jianfei Chen, Chen-Hsuan Lin, Ming-Yu Liu, Jun Zhu, Qianli Ma]
year: 2026
venue: arXiv
tags: [diffusion-distillation, autoregressive-video-generation, consistency-model, self-forcing, world-model, streaming-video-generation]
zotero_collection: 3-Robotics/World Model
image_source: online
arxiv_html: https://arxiv.org/html/2606.25473
created: 2026-06-26
---

# 论文笔记：Causal-rCM: A Unified Teacher-Forcing and Self-Forcing Open Recipe for Autoregressive Diffusion Distillation in Streaming Video Generation and Interactive World Models

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | 清华大学（朱军组）、NVIDIA |
| 日期 | June 2026 |
| 项目主页 | 无独立项目主页，代码基于 [[rCM]] 仓库扩展 |
| 对比基线 | [[Self-Forcing]]、[[Causal Forcing]]、LongLive、AnyFlow |
| 链接 | [arXiv](https://arxiv.org/abs/2606.25473) / [HTML](https://arxiv.org/html/2606.25473) / [Code (rCM repo)](https://github.com/NVlabs/rcm) |

---

## 一句话总结

> 将 [[rCM]] 的"前向散度一致性模型 + 反向散度分布匹配蒸馏"互补思想搬到自回归视频扩散场景，用 [[teacher forcing|Teacher-Forcing]] 一致性模型作为 [[Self-Forcing]] [[Distribution Matching Distillation|DMD]] 的初始化，1-2 步采样即达到 84.63 的 VBench-T2V SOTA。

---

## 核心贡献

1. **TF-CM 是 SF-DMD 最佳初始化策略**：通过系统性对比 DF、TF、DF-KD、TF-KD、TF-dCM、TF-sCM 六种初始化方式，证明 Teacher-Forcing 一致性模型（TF-CM）在最终 [[Self-Forcing]] [[Distribution Matching Distillation|SF-DMD]] 精调后综合表现最稳健，纹理质量明显优于 DF/TF 基线。
2. **首个连续时间 Teacher-Forcing 一致性模型实现**：基于自研的 custom-mask FlashAttention-2 JVP 内核，首次把 [[sCM]] / [[MeanFlow]] 这类连续时间一致性目标用于自回归视频扩散的 Teacher-Forcing 训练，相比离散时间 [[一致性蒸馏|dCM]] 收敛速度提升 10 倍。
3. **统一算法-基础设施开放配方 Causal-rCM**：把 [[rCM]] 的"CM 处理离线轨迹 + DMD 处理在线 rollout"原则系统迁移到因果自回归设定，提出三阶段训练流水线（TF 转换 → TF-CM 蒸馏 → SF-DMD 精调），并配套 FSDP2、上下文并行、KV 缓存等基础设施兼容方案。
4. **frame-wise 和 chunk-wise 双设定下的 SOTA 流式视频生成**：仅用合成数据训练，蒸馏后的 2 步因果 Wan2.1-1.3B 模型在 VBench-T2V 上取得 84.63 分；并将该配方应用于 NVIDIA Cosmos 3，构建出支持动作条件的交互式世界模型。

---

## 问题背景

### 要解决的问题

自回归（AR）视频扩散用因果注意力的 Diffusion Transformer 逐帧/逐块生成视频，是流式长视频生成、交互式世界模型、闭环机器人控制等应用的关键范式。但常见的因果训练方式——[[teacher forcing|Teacher-Forcing]]（TF）和 [[Diffusion Forcing]]（DF）——在推理时会因为模型只能看到自己生成的历史帧（而非 ground-truth）而产生误差累积和质量退化，即 exposure bias（曝光偏差）问题。

### 现有方法的局限

近年的 [[Self-Forcing]] 范式通过 on-policy 训练（模型在 rollout 自身生成的历史上训练）来缓解 exposure bias，并配合 [[Distribution Matching Distillation|DMD]] 或 GAN 损失做少步蒸馏。但是：
- DMD/GAN 这类基于反向 KL 散度的目标对**初始化非常敏感**，容易出现 mode collapse；
- 现有系统在 Self-Forcing 前各自引入了不同的初始化策略（ODE-pair 回归蒸馏、DF 式因果改造、TF/DF 混合初始化等），但**初始化策略、因果训练范式、蒸馏损失三者之间的联系缺乏系统研究**；
- 已有工作如 APT2 用 TF 一致性蒸馏做 Self-Forcing 初始化，但 Self-Forcing 阶段仍依赖较繁琐的 GAN 目标，且只实现了离散时间一致性模型（dCM），未涉及更高效的连续时间一致性模型。

### 本文的动机

[[rCM]]（score-regularized Continuous-time consistency Model）在双向扩散蒸馏中揭示了一个关键洞察：[[一致性蒸馏|一致性模型]]（CM）是**前向散度、保持轨迹/多样性**的目标，[[Distribution Matching Distillation|DMD]] 是**反向散度、追求分布匹配质量**的目标，两者互补可以同时获得高质量和高多样性。作者认为这一前向-反向互补原则可以自然迁移到自回归设定：[[teacher forcing|Teacher-Forcing]] 提供离线、保持模式覆盖的因果训练信号，[[Self-Forcing]] 提供在线、按策略精调的信号，因此用 TF-CM 初始化 SF-DMD 应该是该原则在因果场景下的对应实现。

---

## 方法详解

### 整体流程

Causal-rCM 把 [[rCM]] 的两个蒸馏目标（CM、DMD）分别与两种因果训练范式（[[teacher forcing|TF]]、[[Self-Forcing|SF]]）配对：TF-CM 提供离线的前向型一致性目标，SF-DMD 提供在线的反向型分布匹配目标。与 rCM 联合训练 CM 和 DMD 的方式不同，Causal-rCM 采用**三阶段串行流水线**：

1. **阶段 1（TF 转换）**：把双向扩散模型转换为自回归扩散模型，同时作为因果教师（causal teacher）和后续 TF-CM 阶段的学生初始化；
2. **阶段 2（TF-CM 蒸馏）**：把因果教师蒸馏为少步因果学生，作为后续 SF-DMD 阶段的学生初始化；
3. **阶段 3（SF-DMD 精调）**：在学生自身的自回归 rollout 上做进一步优化，缩小训练-推理差距，降低 exposure bias。

### 模块 1：Teacher-Forcing 与 Packed Causal Forward

[[teacher forcing|TF]] 训练的核心操作是把标准的单状态前向替换为"拼接干净上下文 + 含噪目标"的打包因果前向。对于速度预测器 $\bm{v}_\theta$，不再直接计算 $\bm{v}_\theta(\bm{x}_t,t)$，而是计算：

$$
\left[\bm{v}_{\theta}\left([\bm{x}_{0}^{\mathrm{clean}},\bm{x}_{t}^{\mathrm{noisy}}],[\bm{0}^{\mathrm{clean}},\bm{t}^{\mathrm{noisy}}];\bm{M}_{\mathrm{TF}}\right)\right]_{\mathrm{noisy}}
$$

其中 $\bm{M}_{\mathrm{TF}}$ 是 TF 专用 attention mask，干净部分提供时间步 0 的真实因果上下文，loss 只作用在含噪部分。这种带掩码的注意力可以用 [[FlexAttention]] 或 MagiAttention 等自定义掩码算子实现；论文另提出一种两段式实现作为替代方案，但该方案需要把干净 token 的 KV 缓存保留在计算图中，与[[激活检查点|activation checkpointing]]兼容性差、显存占用更高。

### 模块 2：Teacher-Forcing dCM

固定干净上下文，让含噪部分沿因果教师 PF-ODE 轨迹移动，得到离散时间一致性蒸馏目标 TF-dCM（公式 10），即标准 [[一致性蒸馏|dCM]] 在因果 packed-forward 设定下的直接推广。

### 模块 3：JVP 驱动的连续时间 Teacher-Forcing sCM/MeanFlow

TF-sCM 沿用与 TF/TF-dCM 相同的打包因果前向，但用连续时间切线目标替代有限步一致性目标：在 [[Rectified Flow|RF]] 调度下，定义因果教师在含噪分支上的速度 $\bm{v}_{\mathrm{teacher}}^{\mathrm{TF}}$（公式 15），以及含噪分支上的 RF 一致性映射 $[\bm{f}_\theta^{\mathrm{TF\text{-}RF}}]_{\mathrm{noisy}}$（公式 16）。其沿因果教师轨迹的连续时间切线 $\bm{h}_{\mathrm{TF\text{-}sCM}}$（公式 17）通过 [[JVP]]（Jacobian-Vector Product，前向模式自动微分）高效计算，由自研的 **custom-mask FlashAttention-2 JVP 内核**支撑。

一个重要设计细节：论文使用 [[Rectified Flow|RF]] 原生形式的 sCM，而**不是**像双向设定下的 [[rCM]] 那样把 RF 速度模型包装进 TrigFlow 坐标系再套用 TrigFlow-sCM 目标。作者发现尽管不同噪声调度（TrigFlow、RF）在理论上可以通过时间相关的缩放相互转换，但二者会导出不同的归一化 MSE 目标（详见附录 A）；在双向设定下 TrigFlow 包装对 [[rCM]] 训练稳定性有益，但在本文的因果 TF 设定下，TrigFlow 包装的 TF-sCM 反而导致生成质量下降，RF 原生形式的 TF-sCM 输出更平滑。

### 模块 4：Self-Forcing DMD（SF-DMD）

SF-DMD 在 TF-CM 之后应用。学生先用 [[KV 缓存]] 做时序自回归 rollout：第 $i$ 个 chunk 在前序已生成 chunk 的缓存状态上生成当前干净 chunk（公式 11），生成后再做一次缓存更新前向把其干净 token 的 KV 状态追加进缓存（公式 12，注意更新时对干净 chunk 做 [[Stop-Gradient]]）。chunk 内部由 $\bm{G}_\theta$ 实现从纯噪声出发的少步自 rollout 去噪（公式 13），每个训练迭代里去噪步数 $N$ 从 $[1, N_{\max}]$ 中随机采样，每步转移可用 [[一致性蒸馏|CM]] 式的"反向去噪 + 前向加噪"实现（公式 14）。为了让 SF-DMD 在显存上可行，仅最后一步去噪（$t_1 \to 0$）保持可微，中间步骤和历史 chunk 的 KV 缓存均做 Stop-Gradient 截断（梯度截断技术）。

### 模块 5：噪声上下文（Noisy Context）与自定义步长调度

与语言模型不同，AR 视频扩散需要维护"去噪时刻感知"的 KV 缓存：标准的干净上下文推理在每个 chunk 的去噪步之后还需要一次额外的干净上下文编码前向，使得 $N$ 步因果扩散模型实际花费 $N+1$ 次前向（NFE）。**噪声上下文**通过复用最后一次去噪步的 KV 状态作为后续 chunk 的上下文，把有效 NFE 从 $N+1$ 降到 $N$；残余噪声还能起到低通滤波的作用，抑制长程累积的高频伪影。**自定义步长调度**允许不同 chunk 使用不同去噪步数（公式 20），例如首个 chunk（需要建立全局场景/布局）用更多步、后续 chunk 用更少步；SF-DMD 训练时按训练迭代循环调整 rollout 长度，使不同去噪区间都能轮流成为"最终可微步"，避免只监督最大步数采样器的最后一段。

### 基础设施：算法-基础设施联合配方

为同时支持双向和因果训练，Causal-rCM 在基础设施层面实现了：
- **custom-mask FlashAttention-2 JVP 内核**：支持带自定义掩码的 JVP 计算（详见 Algorithm 1）；
- **FSDP2 与上下文并行（Context Parallel / Sequence Parallel）兼容**：JVP 计算与 [[FSDP2]]、Ulysses CP 均可协同；
- **selective activation checkpointing（SAC）**：与 [[FlexAttention]]、Self-Forcing 训练路径兼容；
- **KV 缓存与 Ulysses CP 协同**：支持 pre-RoPE 和 post-RoPE 两种缓存方式；
- **replayed back-propagation**：避免在自 rollout 过程中存储完整计算图。

详见 Table 2 对比 Self-Forcing、FastVideo、FastGen 等代码库的基础设施支持差异。

---

## 关键公式

### 公式 1: [[一致性蒸馏|离散时间一致性模型]] (dCM) 目标

$$
\mathcal{L}_{\text{dCM}}(\theta)=\mathbb{E}_{\bm{x}_{0}\sim p_{\text{data}},\bm{\epsilon},t}\left[w(t)\,d\left(\bm{f}_{\theta}(\bm{x}_{t},t),\bm{f}_{\theta^{-}}(\hat{\bm{x}}_{t-\Delta t},t-\Delta t)\right)\right]
$$

**含义**：约束学生在沿教师 PF-ODE 轨迹相邻两个时间步 $t$ 和 $t-\Delta t$ 上的一致性函数输出相同。

**符号说明**：
- $\bm{f}_\theta$: 一致性函数，把 $(\bm{x}_t, t)$ 映射到 $\bm{x}_0$
- $\theta^-$: $\theta$ 的 [[Stop-Gradient]] 版本（EMA 或直接 detach）
- $\hat{\bm{x}}_{t-\Delta t}$: 用数值求解器沿教师 PF-ODE 从 $(\bm{x}_t,t)$ 解到 $t-\Delta t$ 得到
- $w(t)$: 正权重函数；$d(\cdot,\cdot)$: 距离度量

### 公式 2: 连续时间一致性模型 ([[sCM]]) 目标

$$
\mathcal{L}_{\text{sCM}}(\theta)=\mathbb{E}_{\bm{x}_{0}\sim p_{\text{data}},\bm{\epsilon},t\sim p_{G}}\left[\left\|\bm{F}_{\theta}(\bm{x}_{t},t)-\bm{F}_{\theta^{-}}(\bm{x}_{t},t)-\frac{\bm{g}}{\|\bm{g}\|_{2}^{2}+c}\right\|_{2}^{2}\right]
$$

$$
\bm{g}=w(t)\frac{\mathrm{d}\bm{f}_{\theta^{-}}(\bm{x}_{t},t)}{\mathrm{d}t}
$$

**含义**：dCM 取 $\Delta t \to 0$ 极限后的连续时间版本，把一致性约束转化为对教师轨迹切线方向的 MSE 回归，切线 $\bm{g}$ 经 [[JVP]] 高效计算并做归一化以稳定训练。

**符号说明**：
- $\bm{F}_\theta$: 自由形式学生网络（对应速度预测器 $\bm{v}_\theta$）
- $\bm{g}$: 归一化前的教师轨迹切线方向
- $c$: 防止除零的稳定项常数

### 公式 3: 连续时间一致性轨迹映射（CTM）目标

$$
\begin{aligned}
\mathcal{L}_{\text{sCTM}}(\theta) &=\mathbb{E}_{\bm{x}_{0}\sim p_{\text{data}},\bm{\epsilon},t,s}\left[\left\|\bm{F}_{\theta}(\bm{x}_{t},t,s)-\bm{F}_{\theta^{-}}(\bm{x}_{t},t,s)-\frac{\bm{g}}{\|\bm{g}\|_{2}^{2}+c}\right\|_{2}^{2}\right] \\
&=\mathbb{E}_{\bm{x}_{0}\sim p_{\text{data}},\bm{\epsilon},t,s}\left[\frac{\|\Delta_{\theta}\|_{2}^{2}}{\|\Delta_{\theta^{-}}\|_{2}^{2}+c}\right]
\end{aligned}
$$

**含义**：把 sCM 的"终点一致性"推广为"任意区间 $t \to s$ 的一致性轨迹映射"，等价地可写成 [[MeanFlow]] 式的相对误差比形式。

**符号说明**：
- $s$: 目标时间步（CTM 中可以是任意中间时刻，而非仅 0）
- $\Delta_\theta = \bm{F}_\theta(\bm{x}_t,t,s) - \bm{F}_{\theta^-}(\bm{x}_t,t,s) - \mathrm{d}\bm{f}_{\theta^-}(\bm{x}_t,t,s)/\mathrm{d}t$

### 公式 4: [[MeanFlow]] 目标中的切线差

$$
\Delta_{\theta}=\bm{v}_{\theta}(\bm{x}_{t},t,s)-\bm{v}+(t-s)\,\texttt{JVP}(\bm{v}_{\theta^{-}},(\bm{x}_{t},t,s),(\bm{v},1,0))
$$

**含义**：MeanFlow 用平均速度场参数化 CTM，$\Delta_\theta$ 衡量瞬时速度与平均速度场及其切线修正项之间的残差。

**符号说明**：
- $\bm{v}$: 瞬时速度目标（如 RF 下的 $\bm{\epsilon}-\bm{x}_0$）
- $\texttt{JVP}(\cdot,\cdot,\cdot)$: Jacobian-向量积算子，第三个参数是切向方向

### 公式 5: DMD 原始 KL 散度目标

$$
\mathcal{L}_{\mathrm{DMD}\text{-raw}}(\theta)=\mathbb{E}_{t}\left[D_{\mathrm{KL}}(p_{\theta}^{t}\;\|\;p_{\text{teacher}}^{t})\right],\quad \bm{x}_{t}\sim q_{t|0}(\bm{x}_{t}|\bm{x}_{0}^{\theta})
$$

**含义**：[[Distribution Matching Distillation|DMD]] 的理论目标是让学生生成分布在每个噪声水平 $t$ 上与教师分布的 KL 散度最小化（反向散度、追求分布匹配）。

**符号说明**：
- $p_\theta^t$: 学生在噪声水平 $t$ 下的边缘分布；$p_{\text{teacher}}^t$: 教师对应分布

### 公式 6: DMD 梯度（Score 差分形式）

$$
\nabla_{\theta}\mathcal{L}_{\mathrm{DMD}\text{-raw}}(\theta)=\mathbb{E}_{\bm{z},\bm{\epsilon},t}\left[w(t)\left(\nabla_{\bm{x}_{t}}\log p_{\theta}^{t}(\bm{x}_{t})-\nabla_{\bm{x}_{t}}\log p_{\text{teacher}}^{t}(\bm{x}_{t})\right)^{\top}\frac{\mathrm{d}\bm{x}_{t}}{\mathrm{d}\theta}\right]
$$

**含义**：KL 散度梯度可写成"假分布 Score 与教师 Score 之差"沿生成路径对参数的梯度，这是 DMD 实际可计算、可优化的形式。

**符号说明**：
- $\nabla_{\bm{x}_t}\log p_\theta^t$: 学生（fake-score 网络）估计的 Score；$\nabla_{\bm{x}_t}\log p_{\text{teacher}}^t$: 教师（real-score）估计的 Score

### 公式 7: DMD 实用化（归一化）目标

$$
\mathcal{L}_{\mathrm{DMD}}(\theta)=\mathbb{E}_{\bm{x}_{0}^{\theta}\sim p_{\theta},\bm{\epsilon},t\sim p_{D}}\left[\left\|\bm{x}_{0}^{\theta}-\texttt{sg}\left[\bm{x}_{0}^{\theta}-\frac{\bm{f}_{\mathrm{fake}}(\bm{x}_{t},t)-\bm{f}_{\mathrm{teacher}}(\bm{x}_{t},t)}{\texttt{mean}(\texttt{abs}(\bm{x}_{0}^{\theta}-\bm{f}_{\mathrm{teacher}}(\bm{x}_{t},t)))}\right]\right\|_{2}^{2}\right]
$$

**含义**：把公式 6 的 Score 差分梯度等价地写成一个带 [[Stop-Gradient]] 目标的 MSE 回归形式，便于直接用标准反向传播实现，分母做归一化以稳定训练尺度。

**符号说明**：
- $\bm{f}_{\mathrm{fake}}$: 在学生生成样本上训练的 fake-score（一致性）网络；$\bm{f}_{\mathrm{teacher}}$: 冻结教师一致性网络
- $\texttt{sg}[\cdot]$: Stop-gradient 算子；$\texttt{mean}(\texttt{abs}(\cdot))$: 归一化分母，稳定梯度尺度

### 公式 8: Teacher-Forcing 打包因果前向

$$
\left[\bm{v}_{\theta}\left([\bm{x}_{0}^{\mathrm{clean}},\bm{x}_{t}^{\mathrm{noisy}}],[\bm{0}^{\mathrm{clean}},\bm{t}^{\mathrm{noisy}}];\bm{M}_{\mathrm{TF}}\right)\right]_{\mathrm{noisy}}
$$

**含义**：把干净历史上下文（时间步恒为 0）和含噪目标拼接成单次前向输入，配合 TF 专用掩码 $\bm{M}_{\mathrm{TF}}$，只在含噪部分计算 loss。

**符号说明**：
- $\bm{M}_{\mathrm{TF}}$: TF 注意力掩码，使每个含噪块只能看到其允许的干净历史与自身含噪 token

### 公式 9: 标准 [[teacher forcing|TF]] 目标

$$
\mathcal{L}_{\mathrm{TF}}(\theta)=\mathbb{E}_{\bm{x}_{0},\bm{\epsilon},t}\left[w(t)\left\|\left[\bm{v}_{\theta}\left([\bm{x}_{0}^{\mathrm{clean}},\bm{x}_{t}^{\mathrm{noisy}}],[\bm{0}^{\mathrm{clean}},\bm{t}^{\mathrm{noisy}}];\bm{M}_{\mathrm{TF}}\right)\right]_{\mathrm{noisy}}-\bm{v}\right\|_{2}^{2}\right]
$$

**含义**：在 [[Rectified Flow|RF]] 调度 $\bm{v}=\bm{\epsilon}-\bm{x}_0$ 下，对打包因果前向的含噪分支输出做标准流匹配回归，得到一个完整步数的因果扩散模型（用于阶段 1 的 TF 转换）。

### 公式 10: TF-dCM 目标

$$
\begin{aligned}
\mathcal{L}_{\mathrm{TF\text{-}dCM}}(\theta)=\mathbb{E}_{\bm{x}_{0},\bm{\epsilon},t}\Bigg[w(t)\,d\Bigg(
&\left[\bm{f}_{\theta}\left([\bm{x}_{0}^{\mathrm{clean}},\bm{x}_{t}^{\mathrm{noisy}}],[\bm{0}^{\mathrm{clean}},\bm{t}^{\mathrm{noisy}}];\bm{M}_{\mathrm{TF}}\right)\right]_{\mathrm{noisy}}, \\
&\left[\bm{f}_{\theta^{-}}\left([\bm{x}_{0}^{\mathrm{clean}},\hat{\bm{x}}_{t-\Delta t}^{\mathrm{noisy}}],[\bm{0}^{\mathrm{clean}},(\bm{t-\Delta t})^{\mathrm{noisy}}];\bm{M}_{\mathrm{TF}}\right)\right]_{\mathrm{noisy}}\Bigg)\Bigg]
\end{aligned}
$$

**含义**：把 dCM（公式 1）的一致性约束嵌入打包因果前向，固定干净上下文，让含噪部分沿因果教师 ODE 轨迹移动，得到因果场景下的离散时间一致性蒸馏目标。

**符号说明**：
- $\hat{\bm{x}}_{t-\Delta t}^{\mathrm{noisy}}$: 在相同 TF 掩码下，沿因果教师 ODE 从 $\bm{x}_t^{\mathrm{noisy}}$ 解到 $t-\Delta t$ 得到

### 公式 11: 自回归 rollout 生成

$$
\tilde{\bm{x}}_{0}^{i}=\bm{G}_{\theta}(\bm{z}^{i}\mid\mathrm{KV}^{<i}),\qquad\mathrm{KV}^{<i}=\mathrm{KV}(\tilde{\bm{x}}_{0}^{<i})
$$

**含义**：第 $i$ 个 chunk 的生成条件于此前所有已生成 chunk 的 [[KV 缓存]]。

**符号说明**：
- $\bm{G}_\theta$: 学生的 chunk 级生成器；$\bm{z}^i$: 第 $i$ 个 chunk 的初始噪声

### 公式 12: KV 缓存更新

$$
\mathrm{KV}^{\leq i}=\mathrm{Append}\left(\mathrm{KV}^{<i},\mathrm{KV}_{\theta^{-}}(\texttt{sg}[\tilde{\bm{x}}_{0}^{i}])\right)
$$

**含义**：生成的干净 chunk 经一次缓存更新前向（对生成结果做 [[Stop-Gradient]]）后追加进 KV 缓存，供后续 chunk 使用。

### 公式 13: 自 rollout 去噪链

$$
\tilde{\bm{x}}^{i}_{t_{N}}=\bm{z}^{i}\xrightarrow{\ \theta^{-}\ }\tilde{\bm{x}}^{i}_{t_{N-1}}\xrightarrow{\ \theta^{-}\ }\cdots\xrightarrow{\ \theta^{-}\ }\tilde{\bm{x}}^{i}_{t_{1}}\xrightarrow{\ \theta\ }\tilde{\bm{x}}^{i}_{0},\quad 0<t_{1}<t_{2}<\cdots<t_{N}=1
$$

**含义**：chunk 内部从纯噪声 $\bm{z}^i$ 经 $N$ 步去噪到干净样本，除最后一步外全部做 stop-gradient（梯度截断），只让最终一步可微。

### 公式 14: RF 调度下的去噪转移

$$
\tilde{\bm{x}}_{t_{n-1}}^{i}=(1-t_{n-1})\bm{f}_{\theta}\left(\tilde{\bm{x}}_{t_{n}}^{i},t_{n}\mid\mathrm{KV}^{<i}\right)+t_{n-1}\bm{\epsilon}_{n},\quad\bm{\epsilon}_{n}\sim\mathcal{N}(\bm{0},\bm{I})
$$

**含义**：每步去噪转移先用一致性函数预测干净样本，再按 [[Rectified Flow|RF]] 调度重新加噪到下一个噪声水平，即"反向去噪 + 前向加噪"。

### 公式 15: 因果教师切线速度

$$
\bm{v}_{\mathrm{teacher}}^{\mathrm{TF}}=\left[\bm{v}_{\mathrm{teacher}}\left([\bm{x}_{0}^{\mathrm{clean}},\bm{x}_{t}^{\mathrm{noisy}}],[\bm{0}^{\mathrm{clean}},\bm{t}^{\mathrm{noisy}}];\bm{M}_{\mathrm{TF}}\right)\right]_{\mathrm{noisy}}
$$

**含义**：因果教师在打包前向含噪分支上的速度预测，作为 TF-sCM 的切线监督目标来源。

### 公式 16: TF 设定下的 [[Rectified Flow|RF]] 一致性映射

$$
\left[\bm{f}_{\theta}^{\mathrm{TF\text{-}RF}}\right]_{\mathrm{noisy}}=\bm{x}_{t}^{\mathrm{noisy}}-t\left[\bm{v}_{\theta}\left([\bm{x}_{0}^{\mathrm{clean}},\bm{x}_{t}^{\mathrm{noisy}}],[\bm{0}^{\mathrm{clean}},\bm{t}^{\mathrm{noisy}}];\bm{M}_{\mathrm{TF}}\right)\right]_{\mathrm{noisy}}
$$

**含义**：在 RF 调度 $\bm{f}_\theta(\bm{x},t)=\bm{x}-t\bm{F}_\theta(\bm{x},t)$ 的因果打包前向版本，是 TF-sCM 连续时间目标的一致性函数参数化基础。

### 公式 17: TF-sCM 切线目标 $\bm{h}_{\mathrm{TF\text{-}sCM}}$

$$
\begin{aligned}
\bm{h}_{\mathrm{TF\text{-}sCM}} &=\bm{v}_{\mathrm{teacher}}^{\mathrm{TF}}-\left[\bm{v}_{\theta^{-}}\left([\bm{x}_{0}^{\mathrm{clean}},\bm{x}_{t}^{\mathrm{noisy}}],[\bm{0}^{\mathrm{clean}},\bm{t}^{\mathrm{noisy}}];\bm{M}_{\mathrm{TF}}\right)\right]_{\mathrm{noisy}} \\
&\quad-t\left[\texttt{JVP}\left(\bm{v}_{\theta^{-}},\left([\bm{x}_{0}^{\mathrm{clean}},\bm{x}_{t}^{\mathrm{noisy}}],[\bm{0}^{\mathrm{clean}},\bm{t}^{\mathrm{noisy}}]\right),\left([\bm{0}^{\mathrm{clean}},\bm{v}_{\mathrm{teacher}}^{\mathrm{TF}}],[\bm{0}^{\mathrm{clean}},\bm{1}^{\mathrm{noisy}}]\right);\bm{M}_{\mathrm{TF}}\right)\right]_{\mathrm{noisy}}
\end{aligned}
$$

**含义**：沿因果教师轨迹方向计算的连续时间切线目标，[[JVP]] 通过同样的 TF 掩码打包前向计算；干净上下文的切线为零，只有含噪分支沿教师速度方向演化。

**符号说明**：
- $\texttt{JVP}(\bm{v}_{\theta^-}, \cdot, \cdot)$: 在 $\bm{v}_{\theta^-}$ 处沿给定切向方向求 Jacobian-向量积

### 公式 18: TF-sCM 目标

$$
\mathcal{L}_{\mathrm{TF\text{-}sCM}}(\theta)=\mathbb{E}_{\bm{x}_{0},\bm{\epsilon},t}\left[\left\|\Delta\bm{v}_{\theta}^{\mathrm{TF}}-\frac{w(t)\bm{h}_{\mathrm{TF\text{-}sCM}}}{w^{2}(t)\|\bm{h}_{\mathrm{TF\text{-}sCM}}\|_{2}^{2}+c}\right\|_{2}^{2}\right]
$$

**含义**：把因果切线目标（公式 17）做归一化后，约束学生在打包前向下的速度差与归一化切线一致，是 TF-sCM 最终可优化的归一化 MSE 损失。

### 公式 19: TF 速度差 $\Delta\bm{v}_\theta^{\mathrm{TF}}$

$$
\Delta\bm{v}_{\theta}^{\mathrm{TF}}=\left[\bm{v}_{\theta}\left([\bm{x}_{0}^{\mathrm{clean}},\bm{x}_{t}^{\mathrm{noisy}}],[\bm{0}^{\mathrm{clean}},\bm{t}^{\mathrm{noisy}}];\bm{M}_{\mathrm{TF}}\right)-\bm{v}_{\theta^{-}}\left([\bm{x}_{0}^{\mathrm{clean}},\bm{x}_{t}^{\mathrm{noisy}}],[\bm{0}^{\mathrm{clean}},\bm{t}^{\mathrm{noisy}}];\bm{M}_{\mathrm{TF}}\right)\right]_{\mathrm{noisy}}
$$

**含义**：学生网络与其 stop-gradient（EMA）版本在打包前向含噪分支上的速度差，是公式 18 损失的拟合对象。

### 公式 20: 自定义步长调度

$$
[N_{1},N_{2},\ldots,N_{N_{\mathrm{chunk}}}]
$$

**含义**：为每个 chunk 单独指定去噪步数，首 chunk 通常分配更多步数以建立全局场景结构，后续 chunk 可用更少步数加速。

---

## 关键图表

### Figure 1: 流式视频生成 SOTA 性能总览

![Figure 1](https://arxiv.org/html/2606.25473v1/x1.png)

**说明**：Causal-rCM 在 1 步、2 步、4 步生成、frame-wise 和 chunk-wise 两种自回归设定下均取得领先的 VBench-T2V 分数（1 步达到 84.63）。

### Figure 2: rCM 与 Causal-rCM 的统一散度视角

![Figure 2](https://arxiv.org/html/2606.25473v1/x2.png)

**说明**：对比 [[rCM]]（双向场景：CM 处理离线轨迹数据 + DMD 处理在线学生样本）与 Causal-rCM（因果场景：TF-CM 处理离线因果上下文 + SF-DMD 处理在线自回归 rollout），强调两者共享"前向-反向散度互补"的统一原则。

### Figure 3: 因果训练范式示意图

![Figure 3](https://arxiv.org/html/2606.25473v1/x3.png)

**说明**：改编自 [[Self-Forcing]] 论文，对比 [[teacher forcing|Teacher-Forcing]]、[[Diffusion Forcing]]、[[Self-Forcing]] 三种因果训练范式在历史上下文来源（真实历史 vs. 自生成历史）上的差异。

### Figure 4: Causal-rCM 与其他方法的对比

![Figure 4](https://arxiv.org/html/2606.25473v1/x4.png)

**说明**：对比 Causal-rCM 与依赖 ODE-pair 知识蒸馏（KD）或 GAN 后训练的现有方案，Causal-rCM 用 TF-CM→SF-DMD 的简洁两阶段配方避免了这些繁琐组件，同时引入新的 TF-sCM 实现。

### Figure 5: 噪声上下文与自定义步长调度示意图

![Figure 5](https://arxiv.org/html/2606.25473v1/x5.png)

**说明**：图示噪声上下文如何复用最后一次去噪步的 KV 状态把每 chunk 的有效 NFE 从 $N+1$ 降到 $N$，以及自定义步长调度如何为不同 chunk 分配不同去噪步数。

### Figure 6: TF-dCM 与 TF-sCM 训练曲线对比

![Figure 6](https://arxiv.org/html/2606.25473v1/x6.png)

**说明**：TF-sCM 收敛速度比 TF-dCM 快约 10 倍，验证了连续时间一致性目标相比离散时间版本的训练效率优势。

### Figure 7: 不同初始化策略下的 SF-DMD 训练曲线

![Figure 7(a) Frame-wise](https://arxiv.org/html/2606.25473v1/x7.png)
![Figure 7(b) Chunk-wise](https://arxiv.org/html/2606.25473v1/x8.png)

**说明**：对比 DF、TF、DF-KD、TF-KD、TF-dCM、TF-sCM 六种初始化策略下 SF-DMD 的训练曲线（Table 5 的可视化版本）。frame-wise 设定下 TF-dCM 更稳定、支持更长精调并达到更高峰值分数；chunk-wise 设定下 DF/TF 初始化分数略高但纹理质量更差。

### Figure 8: chunk-wise SF-DMD 不同初始化下的可视化对比

![Figure 8(a) DF + SF-DMD](https://arxiv.org/html/2606.25473v1/assets/boat_df.jpg)
![Figure 8(b) TF + SF-DMD](https://arxiv.org/html/2606.25473v1/assets/boat_tf.jpg)
![Figure 8(c) TF-dCM + SF-DMD](https://arxiv.org/html/2606.25473v1/assets/boat_dcm.jpg)
![Figure 8(d) TF-sCM + SF-DMD](https://arxiv.org/html/2606.25473v1/assets/boat_scm.jpg)

**说明**：DF/TF 初始化的模型 VBench 分数更高（接近 85），但水面、头发、树叶等纹理出现明显过度平滑和过饱和、细节缺失；综合 VBench 分数和主观视觉质量判断，TF-CM（TF-dCM 或 TF-sCM）初始化是更可靠的选择，TF-sCM 略优于 TF-dCM 且所需 SF-DMD 迭代更少。

### Figure 9: Cosmos 3 到交互式 Cosmos 3 的架构改造

![Figure 9](https://arxiv.org/html/2606.25473v1/x9.png)

**说明**：[[Cosmos3|Cosmos 3]] 原始架构对 UND（理解）token 用因果自注意力、GEN（生成）token 对 UND 的跨注意力是全连接、GEN 内部自注意力是双向的。交互式 Cosmos 3 保留 UND-GEN 注意力结构，但把 GEN 自注意力替换为在"latent-frame 超 token"层级的时序因果注意力：每个潜在视频帧被视为一个视觉超 token $V_i$，动作超 token $A_i$ 控制从 $V_i$ 到 $V_{i+1}$ 的转移，并在 $V_0$ 之前插入空动作 token 以保持统一的 token 排布。

### Figure 10: Cosmos 3 在自动驾驶场景下的交互式生成

![Figure 10](https://arxiv.org/html/2606.25473v1/x10.png)

**说明**：在给定相同初始场景下，交互式 Cosmos 3 根据车辆自身运动（左转/右转/直行）的动作条件生成不同的未来轨迹，展示了流式可控生成能力。

### Table 1: 前向-反向目标互补性对照

| Method | Domain | Forward Component (Pretrain / Offline) | Reverse Component (Posttrain / On-policy) | Effect / Takeaway |
|--------|--------|------------------------------------------|----------------------------------------------|---------------------|
| DDO | diffusion / AR mid-training | diffusion loss on real data | anti-likelihood diffusion loss on self-generated negatives | new record FIDs on ImageNet without auxiliary data/model |
| DiffusionNFT | diffusion RL | forward-process diffusion objective | reward-ranked positive/negative generated samples | 25× efficiency |
| DDRL | diffusion RL | forward-KL / diffusion-loss regularization to offline data | GRPO-style reward optimization on generated rollouts | alleviating reward hacking and diversity collapse |
| [[rCM]] | diffusion distillation | (s)CM loss on data/teacher trajectories | DMD loss on student-generated samples | alleviating mode collapse |
| Causal-rCM | AR diffusion distillation | teacher-forcing CM on offline causal contexts | self-forcing DMD on autoregressive student rollouts | TF-CM initializes SF with causal structure and mode coverage |

**说明**：把 Causal-rCM 放进一个更大的"前向-反向散度互补"方法家族里对比，强调该原则在 mid-training、RL、蒸馏等不同场景下反复出现。

### Table 2: 自回归视频扩散代码库基础设施对比

| Codebase | Bi. | Causal | TF | DF | SF | Replayed | FSDP2(Bi) | CP/SP(Bi) | SAC(Bi) | JVP(Bi) | FSDP2(Causal) | CP/SP(Causal) | SAC(Causal) | JVP(Causal) | KV Cache |
|----------|-----|--------|----|----|----|----------|-----------|-----------|---------|---------|----------------|-----------------|--------------|---------------|----------|
| Self-Forcing | ✗ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | △(v1) | ✗ | △(AC) | ✗ | ✓(post) |
| FastVideo | ✓ | ✓ | ✗ | ✓ | ✓ | ✗ | ✓ | ✓(F-U) | ✓ | ✗ | ✓ | ✗ | ✓ | ✗ | ✓(post) |
| FastGen | ✓ | ✓ | ✗ | ✓ | ✓ | ✗ | ✓ | ✗ | △(AC) | ✗ | ✓ | ✗ | △(AC) | ✗ | ✓(post) |
| (Causal-)rCM | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓(F-U) | ✓ | ✓ | ✓ | ✓(F-U) | ✓ | ✓ | ✓(pre/post) |

**说明**：✓ 支持、✗ 未发现、△ 部分/不明确/路径依赖。Causal-rCM 是唯一同时支持全部算法配方（TF/DF/SF/Replayed）和全部基础设施特性（双向与因果场景下的 FSDP2、CP/SP、SAC、JVP）的代码库。

### Table 3: Wan2.1 T2V 上的训练配置

| Configuration | Stage 1 (Wan2.1-1.3B TF/DF) | Stage 2 (Wan2.1-14B TF/DF) | Stage 3 (TF-dCM) | Stage 3 (TF-sCM) | SF-DMD |
|---|---|---|---|---|---|
| Global batch size | 256 | 64 | 32 | 32 | 64 |
| Context parallel size | 1 | 8 | 4 | 4 | 4 |
| Student optimizer | AdamW, lr=$10^{-5}$, $\beta$=(0.9,0.999), wd=0.01 | 同左 | AdamW, lr=$2\times10^{-6}$, $\beta$=(0,0.999), wd=0.01 | 同左 | 同左 |
| Fake-score optimizer | – | – | – | – | AdamW, lr=$4\times10^{-7}$, $\beta$=(0,0.999), wd=0.01 |
| CFG scale | – | – | 3.0 | 3.0 | 5.0 |
| Time sampling/weighting | TF: $p_G$=UniformShift(5), 共享 $t$, 高斯钟形权重；DF: 每 chunk 随机 $t$, 无权重 | 同左 | uniform RF grid, shift=3, steps=48, skip=1 | $p_G$=LogitNormal($\mu$=-0.8,$\sigma$=1.6) | $p_D$=UniformShift(5) |
| Specific hyperparameters | – | – | – | tangent warmup=1000 | max rollout steps=4, student update freq.=6 |
| Training iterations | 30k | 30k | 10k | 1k | varies |

**说明**：三阶段训练流水线的完整超参数配置，体现各阶段优化器、时间采样分布、训练迭代数的具体差异。

### Table 4: Wan2.1 T2V 上的主要流式视频生成结果

| Method | NFE | Total Score↑ | Quality Score↑ | Semantic Score↑ | Throughput (FPS)↑ | First Latency (s)↓ | Second Latency (s)↓ | SF-DMD iters |
|---|---|---|---|---|---|---|---|---|
| **Bidirectional** | | | | | | | | |
| Wan2.1-1.3B | 50×2 | 82.78 | 83.44 | 80.13 | 0.72 | – | – | – |
| Wan2.1-14B | 50×2 | 83.35 | 83.97 | 80.88 | 0.18 | – | – | – |
| **Frame-wise (c1-1)** | | | | | | | | |
| Causal Forcing (4-step) | 5 | 81.56 | 82.59 | 77.44 | 8.3 | 0.40 | 0.46 | – |
| Causal-rCM (4-step) | 5 | 84.29 | 85.27 | 80.36 | 8.3 | 0.40 | 0.46 | 1200 |
| Causal-rCM (2-step) | 3 | 84.63 | 85.46 | 81.31 | 12.2 | 0.40 | 0.31 | 3000 |
| Causal-rCM (2-step, noisy ctx) | 2 | 83.11 | 83.55 | 81.37 | 15.9 | 0.40 | 0.23 | 1500 |
| Causal-rCM (1-step) | 2 | 84.63 | 85.54 | 81.01 | 15.9 | 0.40 | 0.23 | 3000 |
| **Chunk-wise (c3-3)** | | | | | | | | |
| Self-Forcing (4-step) | 5 | 83.76 | 84.53 | 80.68 | 17.4 | 0.57 | 0.64 | – |
| LongLive (4-step) | 5 | 83.62 | 84.36 | 80.69 | 17.4 | 0.57 | 0.64 | – |
| Causal Forcing (4-step) | 5 | 83.96 | 84.94 | 80.04 | 17.4 | 0.57 | 0.64 | – |
| AnyFlow (4-step) | 5 | 84.31 | 85.15 | 80.94 | 17.4 | 0.57 | 0.64 | – |
| Causal-rCM (4-step) | 5 | 84.37 | 85.02 | 81.73 | 17.4 | 0.57 | 0.64 | 1250 |
| Causal-rCM (2-step) | 3 | 84.30 | 85.04 | 81.36 | 22.2 | 0.57 | 0.49 | 2500 |
| Causal-rCM (2-step, noisy ctx) | 2 | 84.24 | 84.96 | 81.36 | 25.6 | 0.57 | 0.41 | 1750 |
| Causal-rCM (1-step) | 2 | 84.01 | 84.71 | 81.22 | 25.6 | 0.57 | 0.41 | 3000 |

**说明**：Causal-rCM 在 frame-wise 和 chunk-wise 两种设定、1/2/4 步采样下全面超越 Self-Forcing、LongLive、Causal Forcing、AnyFlow 等基线，且吞吐量（FPS）随步数减少显著提升。值得注意 frame-wise 设定下 1-2 步模型反而优于 4 步（rollout 浅、exposure bias 小），chunk-wise 设定下 4 步模型因 chunk 内时序结构丰富而更优。

### Table 5: SF-DMD 初始化策略消融

**Frame-wise (c1-1)**

| Initialization | Total Score↑ | Quality Score↑ | Semantic Score↑ | SF-DMD iterations |
|---|---|---|---|---|
| DF | 83.11 | 83.85 | 80.16 | 2000 |
| TF | 82.62 | 83.62 | 78.61 | 1000 |
| DF-KD | 80.59 | 80.41 | 81.32 | 2000 |
| TF-KD | 83.49 | 84.50 | 79.43 | 1250 |
| TF-dCM | 84.29 | 85.27 | 80.36 | 1200 |
| TF-sCM | 83.84 | 84.67 | 80.55 | 1000 |

**Chunk-wise (c3-3)**

| Initialization | Total Score↑ | Quality Score↑ | Semantic Score↑ | SF-DMD iterations |
|---|---|---|---|---|
| DF | 84.80 | 85.58 | 81.65 | 1500 |
| TF | 84.95 | 85.82 | 81.47 | 1000 |
| DF-KD | 83.61 | 84.10 | 81.68 | 1500 |
| TF-KD | 83.79 | 84.41 | 81.30 | 1000 |
| TF-dCM | 84.33 | 85.22 | 80.75 | 3200 |
| TF-sCM | 84.37 | 85.02 | 81.73 | 1250 |

**说明**：六种初始化策略（DF、TF、DF-KD、TF-KD、TF-dCM、TF-sCM）的系统对比是本文的核心消融实验，结合 Figure 7/8 的训练曲线与可视化结果，证明 TF-CM（TF-dCM/TF-sCM）综合表现最稳健。

### Table 6: CM/CTM 蒸馏配方的高层视角

| Route | Setting | Ultimate pipeline | Related works |
|---|---|---|---|
| CM | Bidirectional | dCM → sCM → DMD/GAN (+CM/on-policy CM) | APT: dCM→GAN；[[rCM]]: sCM+DMD |
| CM | Causal | TF-dCM → TF-sCM → SF-DMD/SF-GAN (+TF-CM/SF-CM) | APT2: TF-dCM→SF-GAN；CF++: TF-dCM→SF-DMD；**Causal-rCM（本文）**: TF-dCM/TF-sCM→SF-DMD |
| CTM | Bidirectional | MeanFlow(FD) → MeanFlow(JVP) → DMD/GAN (+MeanFlow/on-policy MeanFlow) | Transition Matching: MeanFlow(FD)→DMD2-v；AnyFlow: MeanFlow(FD)→DMD+on-policy MeanFlow(FD) |
| CTM | Causal | TF-MeanFlow(FD) → TF-MeanFlow(JVP) → SF-DMD/SF-GAN | AnyFlow: TF-MeanFlow(FD)→SF-DMD+SF-MeanFlow(FD) |

**说明**：作者把现有蒸馏方法归纳为 CM 路线和 CTM（[[MeanFlow]]）路线、双向和因果两种场景的四象限，本文工作对应"CM + Causal"象限，并指出离散时间方法可作为连续时间 JVP 方法的预热阶段，是未来系统化整合的方向。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| rCM 合成 T2V 数据 | 由 Wan2.1-14B 双向教师生成（100 步 Euler，shift 3.0，CFG 5.0） | 纯合成数据，无真实视频 | 全流程训练（不依赖真实数据） |

### 实现细节

- **Backbone**：[[Wan2.2|Wan2.1]] T2V（1.3B 学生，14B 教师），480p 分辨率，$832\times480$ 空间分辨率，81 RGB 帧（对应 VAE 时序压缩后 21 个潜在帧）
- **因果 chunk 模式**：frame-wise（c1-1，逐潜在帧）与 chunk-wise（c3-3，每 chunk 3 个潜在帧）两种设定，TF/DF/TF-CM 掩码、SF-DMD rollout、KV 缓存推理、流式评估统一使用同一 chunk 模式
- **优化器**：AdamW，各阶段学习率从 $10^{-5}$（TF/DF 阶段）逐步降到 $2\times10^{-6}$（TF-CM/SF-DMD 阶段）
- **并行策略**：上下文并行（Context Parallel）规模 1~8，随模型规模和阶段调整
- **训练硬件**：未在正文明确列出具体 GPU 型号/数量（基础设施章节描述支持 FSDP2 + Ulysses CP/SP）

### 可视化结果

定性观察（对应 Table 5 + Figure 7/8）：DF/TF 直接初始化的 SF-DMD 模型 VBench 分数虽然有时更高，但水面、毛发、树叶等高频纹理出现过度平滑、过饱和，细节明显缺失；TF-CM（尤其 TF-dCM）初始化的模型纹理质量最稳健。在交互式世界模型应用（Figure 10）中，Cosmos 3 在相同初始场景下，依据左转/右转/直行的动作条件生成了符合预期的不同未来轨迹，验证了该配方迁移到更大规模 omnimodal 世界模型的可行性。

---

## 批判性思考

### 优点

1. **理论原则的跨场景迁移清晰**：把 [[rCM]] 在双向蒸馏中验证过的"前向散度保多样性 + 反向散度保质量"原则，干净地映射到因果自回归场景的 TF/SF 对应关系上，逻辑自洽且有 Table 1 的更大方法家族支撑。
2. **系统性消融研究**：六种初始化策略（DF/TF/DF-KD/TF-KD/TF-dCM/TF-sCM）在 frame-wise 和 chunk-wise 两种设定下都做了完整对比，同时结合定量指标（VBench）和定性可视化（Figure 8），结论可信度高。
3. **基础设施贡献扎实且可复用**：custom-mask FlashAttention-2 JVP 内核、FSDP2/CP 兼容方案是把"理论上可行"变成"大规模训练上可行"的关键工程贡献，Table 2 的代码库对比也表明工业界目前缺乏同时支持全部特性的开源方案。
4. **真实落地应用验证**：不仅停留在 Benchmark 数字，还把配方应用到 NVIDIA 自家的 Cosmos 3 omnimodal 世界模型上，构建出真正的动作条件交互式生成系统，说明方法的工程可扩展性。

### 局限性

1. **frame-wise 长 rollout 仍然脆弱**：作者在 Limitations 中坦诚 4 步 SF-DMD 模型在长时间训练后会出现摄像机方向性漂移（camera drift），且无法长时间训练，作者推测动作条件设定可能缓解此问题但未在 T2V 场景验证。
2. **初始化质量与最终性能不完全对齐**：TF-sCM 提供更强的初始化，但在 frame-wise 设定下 TF-dCM 在长 SF-DMD 精调中反而更稳定、峰值更高，说明"更好的起点"不等于"更好的终点"，这一发现削弱了 TF-sCM 相对 TF-dCM 的实际优势。
3. **联合训练（像 rCM 那样）在因果场景下不可行**：作者尝试类似 rCM 的联合优化但发现会降低 VBench 上限，转而采用串行三阶段流水线，这说明因果教师与双向教师之间的分布差异可能是更深层的未解决问题，而非简单的训练技巧问题。
4. **JVP 内核仍是 Triton 实现，性能尚未追平最新硬件**：TF-sCM 的单次迭代速度仅与使用标准 FlashAttention-2 的 TF-dCM 相当，落后于 FlashAttention-3/4，存在进一步加速空间。
5. **仅用合成数据训练**：虽然这是有意为之的设计选择（避免真实数据许可问题，且依赖 rCM 已有的合成数据管线），但合成数据本身受限于 Wan2.1-14B 教师模型的能力天花板，可能无法覆盖真实分布的长尾情况。

### 潜在改进方向

1. 探索动作条件信号作为显式运动先验来缓解 frame-wise T2V 设定下的摄像机漂移问题
2. 研究因果教师和双向教师分布差异的根本原因，可能能找到比"分阶段训练"更优雅的联合训练方案
3. 用更先进的 FlashAttention-3/4 或其他底层算子重写 JVP 内核以缩小与标准训练的速度差距
4. 按 Table 6 的路线图，系统性地把离散时间方法（dCM、MeanFlow-FD）作为连续时间 JVP 方法的预热阶段，进一步提高训练效率

### 可复现性评估

- [x] 代码开源（基于 [[rCM]] 仓库 `github.com/NVlabs/rcm` 扩展，论文为该仓库提供"Open Recipe"）
- [ ] 预训练模型权重未在论文中明确提供下载链接
- [x] 训练细节完整（Table 3 提供了完整的逐阶段超参数）
- [x] 数据集可获取（基于 rCM 已发布的合成 T2V 数据管线，使用公开的 Wan2.1 模型生成）

---

## 关联笔记

### 基于

- [[rCM]]: 本文直接扩展的双向扩散蒸馏框架，前向-反向散度互补思想的原始来源
- [[Self-Forcing]]: 因果训练范式之一，本文 SF-DMD 阶段的基础
- [[teacher forcing|Teacher-Forcing]]: 因果训练范式之一，本文 TF-CM 阶段的基础
- [[Diffusion Forcing]]: 另一种因果训练范式，作为初始化消融对比对象
- [[一致性蒸馏|Consistency Model]] / [[sCM]] / [[MeanFlow]]: 本文 TF-CM 蒸馏阶段使用的核心目标函数家族
- [[Distribution Matching Distillation|DMD]]: 本文 SF-DMD 阶段使用的核心蒸馏目标

### 对比

- [[Causal Forcing]] (Causal Forcing): 直接基线，用 ODE 初始化桥接架构差异后做蒸馏
- LongLive / AnyFlow: chunk-wise 流式视频生成基线，Table 4 中的对比对象
- [[Cosmos3|Cosmos 3]]: 本文方法的实际应用载体，验证了跨架构（MoT）迁移的可行性

### 方法相关

- [[JVP]]: 连续时间一致性目标（TF-sCM）依赖的核心微分算子
- [[FlashAttention]]: TF-sCM 自定义掩码 JVP 内核的基础算子
- [[KV 缓存]]: Self-Forcing rollout 和交互式推理的关键基础设施
- [[Stop-Gradient]]: SF-DMD 梯度截断、KV 缓存更新中的关键操作
- [[Rectified Flow]]: 本文采用的扩散噪声调度，决定速度目标和一致性函数参数化形式
- [[FSDP2]]: 大规模分布式训练的并行基础设施

### 硬件/数据相关

- [[Wan2.2|Wan2.1]]: 本文主要实验的 backbone 模型（1.3B 学生 / 14B 教师）
- [[VBench]]: 本文核心评测指标（VBench-T2V）

---

## 速查卡片

> [!summary] Causal-rCM
> - **核心**: 把 [[rCM]] 的"CM 前向散度 + DMD 反向散度"互补原则迁移到自回归视频扩散，用 Teacher-Forcing CM 初始化 Self-Forcing DMD
> - **方法**: 三阶段流水线（TF 转换 → TF-CM 蒸馏[支持首个连续时间 TF-sCM 实现] → SF-DMD 精调），配套 custom-mask FlashAttention-2 JVP 内核
> - **结果**: 1-2 步因果 Wan2.1-1.3B 达到 VBench-T2V 84.63，frame-wise/chunk-wise 双设定 SOTA；成功迁移到 Cosmos 3 实现交互式动作条件世界模型
> - **代码**: 基于 [github.com/NVlabs/rcm](https://github.com/NVlabs/rcm) 扩展

---

*笔记创建时间: 2026-06-26*
