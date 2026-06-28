---
title: "Kairos: A Native World Model Stack for Physical AI"
method_name: "Kairos"
authors: [Kairos Team]
year: 2026
venue: arXiv
tags: [world-model, world-action-model, linear-attention, physical-ai, video-generation, robot-manipulation, cross-embodiment]
zotero_collection: 3-Robotics/World Model
image_source: online
arxiv_html: https://arxiv.org/html/2606.16533v2
created: 2026-06-28
---

# 论文笔记：Kairos: A Native World Model Stack for Physical AI

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Kairos Team（SenseNova/商汤系） |
| 日期 | June 2026 |
| 项目主页 | [GitHub](https://github.com/kairos-agi/kairos-sensenova) / [HuggingFace](https://huggingface.co/kairos-agi) / [ModelScope](https://modelscope.cn/collections/kairos-team/kairos30) |
| 对比基线 | [[Cosmos3]]（Cosmos）、[[Fast-WAM]]、[[GigaWorld]]、π0.5、G0.5、LingBot-VA、MotuBrain、Being-H0.7 |
| 链接 | [arXiv](https://arxiv.org/abs/2606.16533) / [HTML](https://arxiv.org/html/2606.16533v2) / [Code](https://github.com/kairos-agi/kairos-sensenova) |

---

## 一句话总结

> Kairos 用 Cross-Embodiment 数据课程做原生预训练，用滑窗+扩张滑窗+[[GLA|门控线性注意力]] 的混合线性时序记忆统一理解-生成-预测三任务，并给出该混合记忆"长程误差严格可控"的理论证明，在 4B 参数规模下于 LIBERO-Plus、RoboTwin 2.0、WorldModelBench、DreamGen 等基准全面追平或超越十倍参数量的基线，同时推理延迟做到线性可扩展。

---

## 核心贡献

1. **Native Pre-training Paradigm（原生预训练范式）**：拒绝"通用视频生成器 + 下游微调"的解耦套路，提出 Cross-Embodiment Data Curriculum (CEDC)，把开放世界视频、人类行为数据、机器人交互数据组织成"观察 → 模仿 → 具身"三阶段渐进式课程，从训练起点就原生注入物理规律和行为语义。
2. **Native Unified Architecture + Hybrid Linear Temporal Attention（原生统一架构 + 混合线性时序注意力）**：用单一 [[Mixture-of-Transformers|MoT]] 骨架同时承载世界理解、世界生成、世界预测三个任务，时序建模按[[Sliding Window Attention|滑窗注意力]]（局部）、Dilated SWA（中程）、[[GLA|门控线性注意力]]（全局持久记忆）三层分工，整体保持线性复杂度。
3. **理论保证**：证明纯局部滑窗注意力在长程预测上存在不可消除的超额风险下界（Theorem 1），而该混合多尺度记忆的超额风险渐近上界严格可控（Theorem 2），为架构设计提供数学依据而非仅靠实验直觉。
4. **Deployment-Aware System Co-Design（部署感知系统协同设计）**：把硬件约束当作建模的一级原则而非后处理加速，通过时间步蒸馏 + 硬件感知量化，让 4B 模型在消费级 GPU 上做到比 14B/28B 级基线快 28×–85× 的推理延迟，为未来闭环自我进化提供运行基础。

---

## 问题背景

### 要解决的问题

世界模型正从"被动视觉生成器"转向"physical AI 的基础设施"，必须同时做到：原生地从异构经验（开放世界视频/人类行为/机器人交互）中获取世界知识；在长时间跨度上维持持久状态（物体永久性、延迟物理效应）；在真实部署约束下高效运行。论文把这归纳为四个相互耦合的挑战：(1) 异构经验数据碎片化、知识形式互不兼容；(2) 长时程世界状态维持困难；(3) 世界理解与具身控制之间存在鸿沟；(4) 部署与闭环运行的现实约束。

### 现有方法的局限

- 生成式像素级渲染路线（如 [[Cosmos3|Cosmos]]）专注视频生成质量，但通常靠下游微调对接具身控制，存在"学习世界"和"控制世界"的解耦。
- 预测式潜表征路线（V-JEPA 2、DINO-world）回避像素重建开销，但对长程一致性的架构保证缺乏理论刻画。
- 交互式环境建模路线（Genie 3、Dreamer 4）强调交互性和"worldness"，但很少同时兼顾跨具身泛化和边缘端部署效率。
- 多数长视频模型在短时程外观过渡上表现良好，但随时长增长会退化——论文指出这种退化是局部连续性假设下**结构性不可避免**的，而非简单的参数容量问题。

### 本文的动机

作者认为上述四个挑战不能分别用工程补丁解决,而要用一套"学习世界、维持世界、运行世界"三位一体的原生世界-动作模型栈来联合应对：用 CEDC 解决数据异构问题、用混合线性时序记忆解决长程状态维持问题（且给出理论保证）、用部署协同设计解决运行效率问题，从而把世界模型从"演示"真正变成"可部署的基础设施"。

---

## 方法详解

### 模型架构

Kairos 采用单一内生骨架统一 **World Understanding**、**World Generation**、**World Prediction** 三个模块，三者共享由[[GLA|混合线性时序记忆]]稳定的世界状态。

- **World Understanding**：以 [[Qwen|Qwen 系列]] [[VLM]] 作为基础理解层，把异构原始输入（物理描述、多模态传感流）转换为高层语义表征。
- **World Generation**：标准条件[[Diffusion Transformer|扩散]]范式，由三部分组成：(1) 高压缩比 video VAE 把原始视频映射到紧凑潜空间；(2) 多模态条件编码器产生语义嵌入；(3) 时序可扩展的 [[DiT]] 主干在潜空间做扩散建模。支持 T2V/I2V/TI2V 多模态条件，条件嵌入通过 [[Cross-Attention|交叉注意力]]注入主干。
- **World Prediction（World-Action Model）**：基于 [[Mixture-of-Transformers|MoT]] 架构，由 Video DiT（继承 World-Generation 预训练权重）和 Action DiT（约 1/5 规模，提升推理效率）组成。输入序列分三组 token：历史视频 token、未来视频 token、未来动作 token；历史 token 只能看历史，未来视频/动作 token 可看全部历史；未来视频 token 用稀疏时空注意力，未来动作 token 用全注意力以支持长程规划。推理时可关闭未来视频生成分支，仅生成动作 token，实现高效的"action-only"推理。

### 核心模块：Hybrid Linear Temporal Attention

为同时满足效率（线性复杂度）、长程建模、可扩展性三大需求，Kairos 设计了 LinearDiT 主干，把多种互补注意力机制交织在 $M$ 组混合 block 中：

#### 模块 1：长程记忆——[[GLA|Gated Linear Attention]]（GDN 实现）

用 GatedDeltaNet 风格的 [[GLA]] 作为全局信息传播的唯一长程通道，核心是 Delta 更新规则解决朴素线性注意力的"key 碰撞"问题：先按当前 key 检索旧值 $v_t^{old}$，再用门控强度 $\beta_t$ 插值得到新值 $v_t^{new}$，最后用"移除旧关联 + 写入新关联"的方式更新状态矩阵 $S_t$。在此基础上引入遗忘门 $\alpha_t$ 做全局衰减，得到 Gated Delta Update，使模型既能精确局部修正、又能自适应丢弃过时信息。GLA 是主干中唯一的全局注意力，其余注意力层全部限制在局部时序邻域内，形成"局部管运动、全局管一致性"的明确职责分离。

#### 模块 2：短/中程记忆——[[Sliding Window Attention|(Dilated) Sliding Window Attention]]

[[Sliding Window Attention|SWA]] 把注意力限制在大小为 $w$ 的固定时序窗口内建模局部运动；Dilated SWA (DSWA) 用相同窗口大小但引入时序扩张因子，覆盖约 1 秒级别的中程依赖，同时保持线性复杂度。两者均用 [[RoPE]] 编码局部时空相对位置；交替使用 $d=1$（SWA）与 $d \in \{6, 12\}$（DSWA）使主干逐级聚合不同时间尺度的信息。

### 模块可扩展性

混合注意力的解耦设计天然支持三类扩展：(1) 交互式世界建模——把动作条件 token 直接注入 GLA 门控/状态更新，可零延迟闭环控制；(2) 无限时长生成——SWA/DSWA 内存恒定、GLA 压缩历史，支持任意长视频的循环状态传递；(3) 跨模态融合——额外模态（音频、深度图）可作为新流接入混合 block。

---

## 关键公式

### 公式 1: [[GLA|GDN 特征提取与门控]]

$$
q_t = W_Q x_t,\quad k_t = W_K x_t,\quad v_t = W_V x_t,\quad \beta_t = \sigma(W_\beta x_t)
$$

**含义**：从输入 $x_t$ 投影出 query/key/value，并计算 sigmoid 门控强度 $\beta_t$ 控制新信息写入比例。

**符号说明**：
- $W_Q, W_K, W_V, W_\beta$: 可学习投影矩阵
- $\sigma(\cdot)$: sigmoid 函数

### 公式 2: 记忆检索与插值

$$
v_t^{old} = S_{t-1} k_t,\qquad v_t^{new} = \beta_t v_t + (1-\beta_t) v_t^{old}
$$

**含义**：用当前 key 从旧状态 $S_{t-1}$ 中检索出旧值，再与当前值按门控强度插值得到待写入的新值。

**符号说明**：
- $S_t \in \mathbb{R}^{d_v \times d_k}$: 存储 key-value 关联的可学习联想记忆矩阵

### 公式 3: [[GLA|Delta 状态更新]]

$$
S_t = \underbrace{S_{t-1} - v_t^{old} k_t^\top}_{\text{remove}} + \underbrace{v_t^{new} k_t^\top}_{\text{write}}
$$

**含义**：先移除旧的 key-value 关联，再写入新关联，等价于对在线回归损失 $\|v_t - Sk_t\|^2$ 做一步在线梯度下降（delta rule）。

### 公式 4: Gated Delta 遗忘门

$$
\alpha_t = \sigma(W_\alpha x_t)
$$

**含义**：计算全局衰减门 $\alpha_t \in (0,1)$，用于整体缩放历史状态的贡献。

### 公式 5-6: Gated Delta 更新规则

$$
S_t = \alpha_t S_{t-1} - v_t^{old} k_t^\top + v_t^{new} k_t^\top
$$

$$
S_t = \alpha_t S_{t-1} + \beta_t (v_t - v_t^{old}) k_t^\top
$$

**含义**：两个等价形式，把遗忘门 $\alpha_t$ 与 delta 校正结合，使模型同时具备精确局部关联修正能力和自适应长程遗忘能力。输出由 $o_t = S_t q_t$ 读出。

**符号说明**：
- $\alpha_t$: 遗忘门，全局缩放上一时刻状态
- $\beta_t(v_t - v_t^{old})$: delta 校正项

### 公式 7: [[Sliding Window Attention|滑窗注意力公式]]

$$
\mathrm{SWA}(Q,K,V)_i = \sum_{j \in [i-\frac{w}{2}, i+\frac{w}{2}]} \mathrm{Softmax}\left(\frac{Q_i K_j^\top}{\sqrt{d}}\right) V_j
$$

**含义**：查询 token $i$ 只与窗口 $[i-w/2, i+w/2]$ 内的 key/value 做标准 softmax 注意力，窗口大小 $w$ 按每帧 token 数 $L$ 的倍数设置。

**符号说明**：
- $w$: 窗口大小（= $L \times$ window\_size）
- $d$: 注意力头维度

### 公式 8: Dilated Sliding Window Attention (DSWA)

$$
\mathrm{DSWA}(Q,K,V) = \mathrm{SWA}(\mathrm{rearrange}(Q), \mathrm{rearrange}(K), \mathrm{rearrange}(V))
$$

**含义**：将输入从 $(B, F\!\cdot\!L, D)$ reshape 为 $(B\!\cdot\!d, \frac{F}{d}\!\cdot\!L, D)$ 后再做 SWA，从而用相同窗口大小覆盖更大的时序跨度（扩张因子 $d$），实现中程依赖建模且保持线性复杂度。

**符号说明**：
- $d$: 扩张因子，取值 $\{6, 12\}$（$d=1$ 时退化为普通 SWA）

### 公式 9-10: 局部窗口预测的超额风险恒等式（Theorem 1）

$$
R_w^\star - R_{full}^\star = \mathbb{E}\left[(m_t - m_t^{(w)})^2\right] = \mathbb{E}\left[\mathrm{Var}(m_t \mid W_t^{(w)})\right]
$$

$$
R_w^\star > R_{full}^\star \iff m_t \text{ 不是 } W_t^{(w)}\text{-可测的}
$$

**含义**：把限制在最近 $w$ 步窗口的最优预测风险与全历史最优风险之差，精确等价为给定窗口下真实条件期望的方差。超额风险严格大于零，当且仅当最优全历史预测量 $m_t$ 无法从窗口内信息恢复——这从数学上证明纯局部注意力对长程依赖建模存在结构性不足。

**符号说明**：
- $m_t = \mathbb{E}[Y \mid H_t]$: 给定全历史 $H_t$ 的最优预测
- $m_t^{(w)} = \mathbb{E}[Y \mid W_t^{(w)}]$: 给定最近 $w$ 步窗口的最优预测
- $R_w^\star, R_{full}^\star$: 窗口受限/全历史下的最优平方预测风险

### 公式 11: 超额风险下界（Corollary 1）

$$
R_w^\star - R_{full}^\star \geq P(E)\,\alpha(1-\alpha)(\mu_1 - \mu_2)^2
$$

**含义**：当一个影响未来目标 $Y$ 的关键历史事件 $E$（概率 $P(E)>0$）落在窗口之外，且对应两种可能的真实未来期望 $\mu_1,\mu_2$（条件概率 $\alpha, 1-\alpha$），超额风险存在与模型容量无关的信息论下界——扩大参数量或训练算力无法消除这一差距，唯有架构上显式保留超窗口信息才能解决。

**符号说明**：
- $\mu_1, \mu_2$: 两种隐藏历史状态对应的真实未来期望
- $\alpha$: 隐藏状态取值为状态 1 的条件概率

### 公式 12: 混合多尺度记忆渐近超额风险上界（Theorem 2）

$$
R_t(\hat\mu_t) - R_t^\star \leq \left(L\varepsilon + \frac{L_G \bar\xi}{1-\rho}\right)^2 \quad \text{as } t\to\infty
$$

**含义**：若学习到的混合预测器以最大误差 $\varepsilon$ 近似真实 Bayes 分解（局部 SWA、中程 DSWA、全局 GLA 三个组件），且全局记忆的 Gated Delta 更新是收缩映射（收缩系数 $\rho<1$），则长程超额风险渐近被严格限定在一个有限上界内，证明该混合架构对长程依赖建模是理论上"近似充分"的。

**符号说明**：
- $\varepsilon$: 局部/中程/全局组件的最大近似误差
- $\rho < 1$: Gated Delta 更新的收缩系数
- $\bar\xi$: 全局记忆更新的最大单步扰动误差
- $L, L_G$: 解码器的 Lipschitz 常数

### 公式 13-15: [[Flow Matching]] 训练目标

$$
z_\sigma = (1-\sigma) z_0 + \sigma \epsilon
$$

$$
u_\sigma = \frac{dz_\sigma}{d\sigma} = \epsilon - z_0
$$

$$
\mathcal{L}_{FM}(\theta) = \mathbb{E}_{z_0,\epsilon,\sigma,c}\left[\left\| V_\theta(z_\sigma, \sigma, c) - u_\sigma \right\|_2^2\right]
$$

**含义**：按 [[Rectified Flow|rectified-flow]] 线性插值参数化构造含噪潜变量 $z_\sigma$，真实速度沿路径恒定为 $\epsilon - z_0$；DiT 被训练为条件速度预测器 $V_\theta$，用 MSE 回归该恒定速度，统一支持 T2V/I2V/TI2V 等多种条件模式。

**符号说明**：
- $\sigma \in (0,1)$: 连续插值变量（噪声调度位置）
- $c$: 条件输入（文本/图像嵌入）
- $V_\theta$: DiT 参数化的条件速度预测器

### 公式 16-19: 形状感知指数时间步偏移

$$
\tilde\sigma_i = \frac{s\,\sigma_i^{(0)}}{1+(s-1)\sigma_i^{(0)}}\ ,\qquad \tilde\sigma_i = \mathrm{sigmoid}\!\left(\mathrm{logit}(\sigma_i^{(0)}) + \log s\right)
$$

$$
s = \exp(f(L))\sqrt{F},\qquad f(L) = mL+b,\qquad m = \frac{r_{max}-r_{min}}{L_{max}-L_{min}},\qquad b = r_{min} - mL_{min}
$$

**含义**：基础时间步调度 $\sigma_i^{(0)}$ 经指数偏移（等价于 logit 空间平移 $\log s$）重映射；偏移强度 $s$ 按潜空间形状（帧数 $F$、每帧 token 数 $L$）自适应设置，分辨率/时长越大，偏移越强，把调度步数重新分配到对预测误差更敏感的轨迹区域，解决多阶段训练（不同分辨率/帧数）共享同一固定调度器导致的鲁棒性下降问题。

**符号说明**：
- $s$: 偏移强度因子
- $L = H \times W$: 每帧潜空间 token 数；$F$: 编码后帧数
- $r_{min}, r_{max}, L_{min}, L_{max}$: 预定义映射范围超参数

### 公式 20: 联合 World-Action 训练损失

$$
\mathcal{L} = \mathcal{L}_{video} + \lambda \mathcal{L}_{action}
$$

**含义**：Video DiT 的视频生成目标与 Action DiT 的动作预测目标联合优化，使模型在学习环境动力学的同时获得可执行的控制策略，$\lambda$ 平衡两个目标的权重。

---

## 关键图表

### Figure 1: Motivation of Kairos

![Figure 1](https://arxiv.org/html/2606.16533v2/figures/motivation_v1.png)

**说明**：阐明 Kairos 不仅是生成模型，而是面向未来具身智能自我进化、原生设计的可部署基础设施，对比三类现有世界模型路线（像素生成、潜表征预测、交互式环境建模）。

### Figure 2: Framework of Kairos

![Figure 2](https://arxiv.org/html/2606.16533v2/figures/framework_new_v1.png)

**说明**：Kairos 整体框架图，展示"学习世界（CEDC）→ 维持世界（混合线性时序记忆）→ 运行世界（部署协同设计）"的三层闭环结构。

### Figure 3: 性能与效率对比

![Figure 3a](https://arxiv.org/html/2606.16533v2/x1.png)
![Figure 3b](https://arxiv.org/html/2606.16533v2/x4.png)
![Figure 3c](https://arxiv.org/html/2606.16533v2/x5.png)

**说明**：(a) World-Action Model 基准（RoboTwin 2.0 上 96.1 分超越 MotuBrain 96.0、SANTS 94.4；LIBERO-Plus 上 90.8 分超越 Being-H0.7 84.8）；(b) Embodied World Model 基准（WorldModelBench Robot 9.30 分、DreamGen 61.80 分均排名第一）；(c) 480p/720p 下逐 DiT 步推理时间，Kairos 严格线性扩展（$R^2=0.9997/0.9998$），720p 15s 视频仅需 8.95s 而 Cosmos2.5-14B 需 383.10s。

### Figure 4: Model Architecture of Kairos

![Figure 4](https://arxiv.org/html/2606.16533v2/figures/kairos_arch1.jpg)

**说明**：World Understanding（VLM）→ World Generation（DiT 扩散）→ World Prediction（MoT 双 DiT）三模块的完整架构图，展示条件注入路径和模块间的权重继承关系。

### Figure 5: DiT block 架构

![Figure 5](https://arxiv.org/html/2606.16533v2/x7.png)

**说明**：混合线性注意力 DiT block 的内部结构：Dilated SWA / GLA / SWA 三种时序注意力交织排布，配合 cross-attention 接入条件嵌入，构成 $M$ 组重复 block。

### Figure 6: GDN（门控线性注意力模块）架构

![Figure 6](https://arxiv.org/html/2606.16533v2/x8.png)

**说明**：GatedDeltaNet 核心计算图：Q/K/V 投影 → L2 归一化 → Gated Delta Rule 更新状态矩阵 $S_t$ → 输出读出，对应公式 1-6 的具体模块化实现。

### Figure 7: Cross-Embodiment Data Curriculum

![Figure 7](https://arxiv.org/html/2606.16533v2/x9.png)

**说明**：CEDC 三阶段金字塔——Physical Laws（物理规律，配合因果 Chain-of-Thought 文本推理示例如"苹果受重力与茎杆张力作用，成熟后茎杆断裂，自由落体加速下降"）→ Human Behaviors（人类行为意图）→ Robot Actions（机器人本体动作），底层到顶层数据量递减但动作接地程度递增。

### Figure 8: 形状感知指数时间步偏移曲线

**说明**：（HTML 未提供直接图片链接）展示不同训练阶段（分辨率、帧数组合）下偏移强度 $s$ 如何随潜空间 token 数和帧数变化，对应公式 16-19。分辨率越高、视频越长，有效偏移强度越大，调度步数向误差敏感区域偏移越明显。

### Figure 9: 跨具身生成样本

![Figure 9](https://arxiv.org/html/2606.16533v2/figures/kairos_2_compress.jpg)

**说明**：展示 Kairos 在单臂、双臂、灵巧手、人形等不同机器人本体上的跨具身泛化生成样本，体现"多本体多任务统一大脑"的能力。

### Table 1: 渐进式物理预训练阶段

| Task | Stage | Resolution | Max Frames |
|------|-------|-----------|-----------|
| T2I/N2I | Image Pretraining | 256P | 1 |
| T2V/TI2V/I2V/N2V | Pretraining | 256P | 81 |
| T2V/TI2V/I2V/N2V | Pretraining | 480P | 81 |
| T2V/TI2V/I2V/N2V | Pretraining | 720P | 81 |
| T2V/TI2V/I2V/N2V | CT | 720P | 241 |
| T2V/TI2V/I2V/N2V | Domain-specific SFT & Merging | 720P | 241 |
| T2V/TI2V/I2V/N2V | RL | 720P | 241 |

**表格说明**：训练从图像预训练逐步扩展到 720p、15 秒（241 帧）长视频，最后经过 domain-specific SFT + 模型融合（[[Model Soup]]、CART、TIES、DARE、WUDI-Merging）+ DPO 强化学习收尾。

### Table 4: 不同硬件平台延迟对比（Kairos-4B-robot 480P 5s 蒸馏模型）

| GPU | Resolution | Memory(GB) | 1GPU (s) | 4GPU (s) |
|-----|-----------|-----------|----------|----------|
| Nvidia A800 | 480P | 23.5 | 11.7 | 3.0 |
| Nvidia RTX5090 | 480P | 13.9 | 11.4 | 5.7 |

**表格说明**：480p 视频生成在专业卡（A800）和消费级卡（RTX5090）上均能跑出接近实时的延迟。

### Table 5: 不同模型延迟对比（TI2V 模式，720P，5s）

| Model | Memory (GB) | Complexity (PFlops) | 1GPU (s) | 4GPU (s) |
|-------|------------|---------------------|----------|----------|
| Lingbot-28B | 46.1 | 347.4 | 5525 | 1436 |
| Cosmos-Predict2.5-14B | 70.2 | 156.5 | 2526 | 687 |
| Wan2.2-5B | 23.4 | 16.6 | 201 | 85 |
| **Kairos-4B** | **23.5** | **2.3** | **43** | **9** |

**表格说明**：Kairos-4B 计算复杂度仅 2.3 PFlops（远低于其他模型），相对 Cosmos-Predict2.5-14B 提速 28×–85×，相对同参数量级的 Wan2.2-5B 也有 2.5×–3.7× 加速。

### Table 6: WorldModelBench Robot Set 评测

| Model | Param | Instruction Following | Physics Adherence (Overall) | Common Sense (Overall) | Total Score |
|-------|-------|------------------------|------------------------------|--------------------------|--------------|
| Lingbot* | 28B | 2.14 | 4.92 | 1.00/0.98 | 9.04 |
| Cosmos3-Nano* | 16B | 2.36 | 4.96 | 0.98/0.96 | 9.26 |
| Abot-Physworld* | 14B | 2.10 | 4.88 | 1.00/0.98 | 8.96 |
| Cosmos-Predict2.5* | 14B | 2.14 | 4.86 | 1.00/0.94 | 8.94 |
| Wan2.2* | 5B | 2.04 | 4.62 | 0.96/0.90 | 8.52 |
| Cosmos-Predict2.5* | 2B | 2.14 | 4.94 | 0.98/0.98 | 9.04 |
| GigaWorld-0* | 2B | 1.50 | 4.98 | 0.98/1.00 | 8.46 |
| **Kairos** | **4B** | **2.36** | **4.96** | **0.98/1.00** | **9.30** |

**表格说明**：4B 参数的 Kairos 取得总分第一（9.30），Instruction Following 与 16B 的 Cosmos3-Nano 并列最高（2.36），Physics Adherence 在牛顿力学/流体/重力上均拿满分。

### Table 7: DreamGen Bench 评测

| Method | Param | AVG_PA | AVG_IF | AVG_Score |
|--------|-------|--------|--------|-----------|
| Cosmos-Predict2.5* | 14B | 0.495 | 0.478 | 0.487 |
| Cosmos-Predict2.5* | 2B | 0.419 | 0.568 | 0.494 |
| GigaWorld-0 | 2B | 0.485 | 0.588 | 0.537 |
| Wan2.2* | 5B | 0.314 | 0.554 | 0.434 |
| Wan2.2 | 14B | 0.519 | 0.703 | 0.611 |
| Cosmos3-Nano* | 16B | 0.505 | 0.574 | 0.540 |
| ABot-PhysWorld* | 14B | 0.484 | 0.434 | 0.459 |
| Lingbot* | 28B | 0.466 | 0.569 | 0.518 |
| **Kairos** | **4B** | **0.538** | 0.698 | **0.618** |

**表格说明**：Kairos-4B 在 AVG_PA（物理符合度）和 AVG_Score（总分）均排名第一，AVG_IF 仅次于 14B 的 Wan2.2（0.703 vs 0.698），以远小参数量取得最佳综合表现。

### Table 8（节选）: PAI-Bench Robot Set 评测（部分大模型对比）

| Model | Param | Domain Score | Overall Score |
|-------|-------|---------------|----------------|
| Lingbot* | 28B | 82.98 | 79.97 |
| Cosmos3-Nano* | 16B | 88.04 | 82.62 |
| Abot-Physworld+DPO | 14B | 93.06 | 84.91 |
| Cosmos-Predict2.5* | 14B | 82.60 | 79.40 |
| Wan2.1 | 14B | – | – |

**表格说明**：Kairos-4B（表中未完整摘录的小模型组）在 <10B 模型组内 Domain Score 88.59、Overall Score 82.57 排名第一，逼近甚至超越多个 ≥10B 基线。

### Table 11: RoboTwin 2.0 基准结果

| Model | Clean | Randomized | Average |
|-------|-------|------------|---------|
| π0 | 65.9 | 58.4 | 62.2 |
| X-VLA | 72.9 | 72.8 | 72.9 |
| π0.5 | 82.7 | 76.8 | 79.8 |
| starVLA | 88.2 | 88.3 | 88.3 |
| ABot-M0 | 81.2 | 80.4 | 80.8 |
| LingBot-VLA | 86.5 | 85.3 | 85.9 |
| G0.5 | 93.7 | 92.8 | 93.2 |
| JEPA-VLA | 73.5 | – | – |
| GigaWorld-Policy | 86.0 | 85.0 | 85.5 |
| Motus | 88.7 | 87.0 | 87.8 |
| LingBot-VA | 92.9 | 91.6 | 92.2 |
| [[Fast-WAM]] | 91.9 | 91.8 | 91.8 |
| Being-H0.7 | 90.2 | 89.6 | 89.8 |
| AIM | 94.0 | 92.1 | 93.1 |
| SANTS | 94.6 | 94.2 | 94.4 |
| MotuBrain | 95.8 | 96.1 | 96.0 |
| **Kairos** | **96.9** | 95.2 | **96.1** |

**表格说明**：Kairos 在 50+ 任务的双臂操作基准上取得全部 WAM/VLA 基线中的最高平均成功率（96.1），验证联合建模世界动力学与动作演化对复杂双臂协调任务的提升。

### Table 12: LIBERO-Plus 基准结果

| Method | Camera | Robot | Language | Light | Background | Noise | Layout | Average |
|--------|--------|-------|----------|-------|-------------|-------|--------|---------|
| ACoT-VLA | 96.6 | 70.4 | 79.7 | 95.1 | 97.1 | 95.9 | 85.0 | 88.0 |
| π0 | 61.0 | 40.8 | 63.5 | 89.3 | 84.1 | 80.1 | 76.4 | 69.4 |
| π0.5 | 75.8 | 79.4 | 83.3 | 95.5 | 95.0 | 89.6 | 87.0 | 85.7 |
| ProGAL-VLA | 93.2 | 71.5 | 93.6 | 86.8 | 92.3 | 74.8 | 86.7 | 85.5 |
| Being-H0.7 | – | – | – | – | – | – | – | 84.8 |
| Kairos | 95.5 | 72.6 | 86.8 | 97.7 | 95.8 | 96.8 | 81.5 | 89.0 |
| **Kairos-joint** | **95.9** | **74.6** | **95.3** | 97.1 | **97.1** | 95.4 | 83.8 | **90.8** |

**表格说明**：Kairos-joint（推理时联合去噪视频与动作 token）相比 Kairos（关闭视频生成只输出动作）平均分由 89.0 提升到 90.8，验证显式"未来想象"对动作预测的额外增益。

### Table 13: 具身人类中心预训练消融

| Model | Avg. | Gain |
|-------|------|------|
| w/o human-centric data | 83.0 | – |
| w/ human-centric data | 89.0 | +6.0 |

**关键发现**：大规模人类行为数据可作为机器人轨迹数据的可扩展互补监督源，显著提升 LIBERO-Plus 表现。

### Table 14: 生成-预测联合训练消融

| Model | Avg. | Gain |
|-------|------|------|
| Action Prediction Only | 65.8 | – |
| Video Generation & Action Prediction | 89.0 | +23.2 |

**关键发现**：仅训练 Action DiT（去掉视频生成监督）导致性能大幅下降（-23.2），证明生成目标提供的世界建模监督是动作预测获得强条件信号的关键来源。

### Table 9: 人类中心数据扩展与 VLM 选择消融

| Human-Centric Scaling | Stronger VLM Encoder | Instruction Following ↑ | Total Score ↑ |
|------------------------|------------------------|---------------------------|------------------|
| – | – | 2.10 | 9.08 |
| ✓ | – | 2.33 | 9.25 |
| ✓ | ✓ | 2.36 | 9.30 |

**关键发现**：注入大规模人类中心数据使 Instruction Following 从 2.10 提升到 2.33；将 [[VLM]] 编码器由 Qwen2.5-VL-7B 升级为参数更少但多模态理解更强的 Qwen3.5-2B，进一步提升至 2.36/9.30，说明更强的语言-视觉理解对世界模型的指令对齐和生成质量都有直接增益。

---

## 实验

### 数据集

| 数据集/来源 | 规模 | 特点 | 用途 |
|--------------|------|------|------|
| 开源视频数据 | – | Koala-36M, Openhumanvid, VidGen, [[AgiBot World]]-Beta, [[DROID]] | Phase I-III 预训练 |
| 自建互联网数据 | 数千万级叶节点的层级标签体系 | 覆盖人类/机器人/通用场景/物理现象四大域 | Phase I 物理知识预训练 |
| 第一人称人类操作数据 | 100,000+ 小时 | egocentric 视角采集 | Phase II 人类行为预训练 |
| WorldModelBench / DreamGen Bench / [[PAI-Bench]] | – | 视觉质量、物理合理性、指令跟随评测 | World Generation 评测 |
| LIBERO-Plus / [[RoboTwin 2.0]] | 50+ 双臂任务（RoboTwin） | 场景泛化、视觉扰动鲁棒性 | World-Action Model 评测 |

### 实现细节

- **Backbone**: VLM 用 [[Qwen|Qwen2.5-VL / Qwen3.5]] 系列，DiT 用自研 LinearDiT（混合线性注意力）
- **优化器**: 全程 [[AdamW]]，各阶段学习率从图像预训练 5e-5 逐步衰减至 1e-6（DPO 阶段）
- **训练流程**: Stage I 物理预训练（图像→256p→480p→720p→241帧续训→domain SFT+模型融合→DPO）→ Stage II 具身人类中心预训练（人类中心→机器人中心→目标本体微调→DPO）→ Stage III 联合 World-Action 训练
- **数据处理**: PySceneDetect 镜头分割（>95% 精度、80% 召回），过滤损坏/重复/<5秒片段，最终得到数百万小时有效原始视频
- **推理优化**: 时间步蒸馏 + 权重量化（4-bit），在 Kairos-4B-robot 480P(5s) 蒸馏模型上 A800 单卡 11.7s/4卡 3.0s

### 可视化结果

Figure 9、16-24（论文附图）展示了 Kairos 在 WorldModelBench、DreamGen、PAI-Bench 等数据集上跨单臂/双臂/灵巧手/人形本体的生成样本，定性观察显示模型保持了较好的物体永久性、背景一致性和长时程（15 秒）因果连贯性。人类评测（Figure 19）显示 Kairos 在视频质量、物理合理性、任务完成率上均优于基线。

---

## 批判性思考

### 优点

1. **理论与架构设计的紧密耦合**：不止是经验性地堆叠 SWA/DSWA/GLA，而是给出 Theorem 1（必要性，纯局部注意力的超额风险下界）和 Theorem 2（充分性，混合记忆的渐近上界）两个定理支撑设计动机，理论严谨度明显高于一般的"工程组合式"论文。
2. **以小博大的效率-能力权衡**：4B 参数在多个基准上追平甚至超越 14B-28B 级竞品，且推理延迟做到严格线性扩展（$R^2 > 0.999$），这对实际部署（尤其消费级硬件）价值很高。
3. **数据课程设计有清晰的认知科学类比**：物理规律→人类行为→机器人动作的三阶段渐进顺序，与"先学物理常识、再学任务意图、最后学本体执行"的发展路径相呼应，且有 Table 13 的消融数据支撑（+6.0 平均分）。
4. **消融实验颇为系统**：覆盖人类中心数据规模、VLM 选择、生成-预测联合训练、联合去噪推理四个维度，且每个消融都对应明确的设计决策（如 Table 14 显示仅训练 Action DiT 会暴跌 23.2 分，强力证明了联合训练的必要性）。

### 局限性

1. **作为技术报告，部分关键细节描述较泛**：例如"数千万级叶节点标签体系""数百万小时视频"等数据规模描述缺乏更精确的可复现统计口径，专有数据的具体构成无法被外部完全复现。
2. **Theorem 2 的假设较强**：收缩系数 $\rho<1$ 和有界扰动 $\bar\xi$ 是理论保证成立的前提，但论文未给出经验验证这些假设在实际训练好的 GDN 模块中是否真正满足（如测量训练后模型的实际 $\rho$ 值）。
3. **基线比较中部分结果来自"团队复现"（标 * 号）**：如 Table 6、7 中多个基线标注"Models marked with * indicate results reproduced by our team"，复现误差可能引入对比偏差，缺乏第三方独立复现验证。
4. **联合去噪推理（Kairos-joint）与纯动作推理的取舍未充分讨论延迟代价**：Table 12 显示 Kairos-joint 比 Kairos 多 1.8 分，但论文未明确给出联合去噪相对纯动作推理增加的具体延迟开销，无法评估这一增益在实际闭环控制频率约束下是否值得。

### 潜在改进方向

1. 对 Theorem 2 中的收缩系数 $\rho$ 和扰动上界 $\bar\xi$ 做经验测量，验证理论假设与实际训练动力学的吻合程度。
2. 公开更详细的数据构成统计（如各类别视频的具体小时数分布），提升可复现性。
3. 探索把"自我进化"（Future Works 提到的 Recursive Imagination）落地为具体的在线闭环实验，而非仅作为展望。
4. 针对 Kairos-joint 做延迟/吞吐量的系统性测量，量化"显式未来想象"在实时控制场景下的真实代价收益比。

### 可复现性评估
- [x] 代码开源（[github.com/kairos-agi/kairos-sensenova](https://github.com/kairos-agi/kairos-sensenova)）
- [x] 预训练模型权重（HuggingFace / ModelScope 均已发布 Kairos-4B 系列）
- [x] 训练细节较完整（Table 1 给出分阶段分辨率/帧数/学习率配置）
- [ ] 专有数据集（互联网层级标签数据、egocentric 人类操作数据）不可获取，仅部分开源数据（Koala-36M、AgiBotWorld-Beta、DROID 等）可复现

---

## 关联笔记

### 基于
- [[GatedDeltaNet2|GatedDeltaNet]] / [[KDA]]: GLA 模块的底层 Delta Rule 与门控机制来源
- [[Flow Matching]] / [[Rectified Flow]]: World Generation 模块的训练目标范式
- [[DPO]]: 物理预训练与具身预训练阶段末尾的强化对齐方法
- [[Mixture-of-Transformers]]: World Prediction 模块（Video DiT + Action DiT）的架构基础

### 对比
- [[Cosmos3|Cosmos]]: 像素级生成式世界模型路线的代表，强调数字孪生与基础设施定位，是 Kairos 反复对标的主要基线
- [[Fast-WAM]]: World-Action Model 的高效推理路线（测试时是否需要未来想象），RoboTwin 2.0 上的直接对比基线
- [[WAM]]: World-Action Model 的通用范式定义，Kairos 的 World Prediction 模块属于 Joint WAM 一类
- [[Cross-Embodiment]]: CEDC 数据课程所依赖的跨本体泛化范式

### 方法相关
- [[GLA]]: 全局持久记忆的核心机制，混合线性注意力的关键组件
- [[Sliding Window Attention]]: 局部/中程时序建模的基础算子
- [[DiT]] / [[Diffusion Transformer]]: World Generation 与 World Prediction 的扩散骨架
- [[VLM]]: World Understanding 模块的基础（Qwen 系列）
- [[RoPE]]: SWA/DSWA 内部的相对位置编码方案
- [[Model Soup]]: 域专精 SFT 模型融合时使用的多种合并策略之一

### 硬件/数据相关
- [[RoboTwin 2.0]] / [[LIBERO-Plus]]: World-Action Model 能力的主要评测基准
- [[PAI-Bench]] / [[WorldModelBench]] / [[DreamGen]]: 通用世界模型生成质量与物理合理性评测基准
- [[DROID]] / [[AgiBot World]]: Phase III 机器人本体数据的开源来源

---

## 速查卡片

> [!summary] Kairos: A Native World Model Stack for Physical AI
> - **核心**: 用"学习世界（CEDC 数据课程）→ 维持世界（混合线性时序记忆）→ 运行世界（部署协同设计）"三位一体的原生世界模型栈替代"通用生成器+下游微调"的解耦套路
> - **方法**: SWA（局部）+ DSWA（中程）+ GLA（全局门控线性记忆）混合时序注意力统一理解-生成-预测；CEDC 三阶段（物理规律→人类行为→机器人动作）渐进式预训练；理论证明混合记忆的长程误差严格可控
> - **结果**: 4B 参数在 LIBERO-Plus（90.8）、RoboTwin 2.0（96.1）、WorldModelBench（9.30）、DreamGen（61.80）上追平/超越十倍参数量基线；推理延迟相对 14B 级模型快 28×–85×，且严格线性扩展
> - **代码**: [github.com/kairos-agi/kairos-sensenova](https://github.com/kairos-agi/kairos-sensenova)

---

*笔记创建时间: 2026-06-28*
