---
title: "HiLo-WAM: Decoupling Long-Horizon Task Evolution and Local Dynamics in World Action Models"
method_name: "HiLo-WAM"
authors: [Xingyu Wang, Yanjiang Guo, Zhijing Ding, Jiaxin Fang, Zeyu Lai, Jiayu Dong, Weikang Wan, Jianyu Chen, Yuxiang Gao, Mengdi Xu]
year: 2026
venue: arXiv
tags: [world-action-model, long-horizon-manipulation, bimanual-manipulation, flow-matching, video-diffusion, hierarchical-world-model, inference-acceleration]
zotero_collection: Robotics/World Model
image_source: online
arxiv_html: https://arxiv.org/html/2611.05023v1
created: 2026-11-05
---

# 论文笔记：HiLo-WAM: Decoupling Long-Horizon Task Evolution and Local Dynamics in World Action Models

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | 未在摘要中注明 |
| 日期 | November 2026 |
| 项目主页 | 暂无 |
| 对比基线 | [[Fast-WAM]], [[FAWAM]], [[LingBot-Video]] |
| 链接 | [arXiv](https://arxiv.org/abs/2611.05023) / Code 暂无 |

---

## 一句话总结

> HiLo-WAM 将机器人操作的视觉世界建模分解为稀疏长时程流（任务上下文）和密集局部流（机器人-物体交互），通过双流 DiT + 跨流注意力掩码生成动作，同时用渐进推理实现 2.93× 加速。

---

## 核心贡献

1. **双流视觉世界建模**: 将 [[世界动作模型（WAM）]] 的视觉预测解耦为稀疏长时程流（High-level, Hi）和密集局部流（Low-level, Lo），分别捕获任务进度和机器人-物体交互，消除两种不同时间尺度信号的相互干扰。
2. **结构化跨流注意力**: 设计非对称注意力掩码，使局部流和动作流能感知长时程流的近端关键帧，但对预测不确定性最高的远端未来帧设置隔离屏蔽，保证动作生成的稳定性。
3. **渐进推理（Progressive Inference）**: 三级加速机制（部分去噪 + 联合去噪 + 动态跳步），在不损失性能的前提下将推理延迟从 1114ms 降至 380ms（2.93× 加速）。

---

## 问题背景

### 要解决的问题

现有 [[世界动作模型（WAM）]] 主要聚焦于**短时程局部视觉预测**（dense short-horizon dynamics），缺乏对任务长时程演化的建模能力，导致在长流程操作任务（如组装积木、挂杯子等多步骤操作）上表现不稳定。

### 现有方法的局限

- **[[Fast-WAM]] / [[FAWAM]] 类短视程 WAM**：只预测未来几帧，缺少任务整体进度感知，容易在长任务中"迷失方向"（如反复错误地选择操作目标）。
- **朴素长视程 WAM 扩展**：将单一视觉轨迹的预测步长直接拉长，混淆了慢速任务级进展与快速局部交互，增加不确定性，计算量也大幅上升。
- **层次化策略（Hierarchical Policy）**：依赖语言计划或中间子目标分解，需要额外的高层规划器，系统更复杂，并非通过视觉世界建模本身解决问题。

### 本文的动机

不同时间尺度的视觉信号有本质区别：**任务进度信息稀疏且变化缓慢**（适合低帧率、长跨度采样），**物体交互信息密集且变化迅速**（适合高帧率、短跨度采样）。将二者解耦建模，让各流专注自己擅长的尺度，同时通过跨流注意力实现互补，可以在不引入多余规划器的情况下获得长时程理解能力。

---

## 方法详解

### 问题形式化

标准 WAM 建模：

$$
p(V_t, A_t \mid o_t, s_t, l)
$$

其中 $V_t$ 为预测的未来视觉帧，$A_t$ 为动作序列，$o_t$ 为当前观测，$s_t$ 为机器人状态，$l$ 为语言指令。

HiLo-WAM 将其扩展为以历史观测 $O_t = (o_{t-M}, \ldots, o_t)$ 为条件、同时预测两个时间尺度的未来视觉：

- **局部流（Lo）**：$V^L = \{o_{t+j}\}_{j=0}^{T}$，连续密集采样，捕获细粒度机器人-物体交互
- **长时程流（Hi）**：$V^H = \{o_{t+ik}\}_{i=-m}^{n}$，以步幅 $k$ 稀疏采样，涵盖更长时间跨度

### 模型架构

HiLo-WAM 采用 **[[Mixture-of-Transformers]]（双流 DiT）+ Action DiT** 架构：

- **视频骨干**: [[Video DiT]]，从 [[Wan 2.2-5B]] 初始化，沿用其 VAE 编码器
- **语言编码**: T5 编码器，提供语言条件
- **动作编码**: Action Encoder，生成状态 token $q_t$ 和动作 token $Z^A_t$
- **总参数**: Video DiT 30层（隐层维度 3072）+ Action DiT 30层（隐层维度 1024）
- **输出**: [[Action Chunking]] $a_{t:t+H}$，$H=32$，动作维度 14

**编码过程**（Eq. 2–3）：

$$
Z^H = E_v(V^H), \quad Z^L = E_v(V^L)
$$

两个视觉流用同一个 Video VAE 独立编码，然后与动作 token 合并：

$$
\text{tokens} = [Z^H,\; Z^L,\; q_t,\; Z^A]
$$

**长时程流 token 进一步分组**（Eq. 4）：

$$
Z^H = [Z^{H,\text{ctx}},\; Z^{H,\text{near}},\; Z^{H,\text{far}}]
$$

其中 ctx 为历史关键帧（$m=8$ 帧），near 为最近未来关键帧，far 为远端未来关键帧（不确定性最高）。

### 结构化跨流注意力掩码（Cross-Stream Attention Mask）

这是 HiLo-WAM 最核心的设计，各 token 的注意力权限如下：

| Token 类型 | 可关注的 token |
|------------|---------------|
| 长时程流（Hi）全部 token | 长时程流内部双向（ctx、near、far 互见） |
| 局部流（Lo）token | ctx、near（长时程近端）+ 自身局部流 |
| 动作 token $Z^A$ | 状态 token $q_t$ + 动作 token + Hi 和 Lo 双流 |
| 局部流和动作 token | **不可见** far-future token（隔离最高不确定性预测） |
| 视觉 token | **不关注**动作 token（单向信息流）|

这种非对称设计确保动作生成不依赖最不确定的远端预测，同时保持长时程任务上下文对局部决策的正向引导。

### 关键公式

#### 公式1: [[Flow-Matching|Flow Matching 插值]]

$$
y_\tau = (1 - \tau)\,y + \tau\,\varepsilon, \quad \varepsilon \sim \mathcal{N}(0, I),\quad \tau \in [0, 1]
$$

**含义**: 将干净数据 $y$ 与噪声 $\varepsilon$ 进行线性插值，构造扩散轨迹，$\tau$ 为时间步。

**符号说明**:
- $y$: 干净目标（视觉帧或动作）
- $\varepsilon$: 标准高斯噪声
- $\tau$: 流时间步（0=干净，1=噪声）

#### 公式2: [[Flow-Matching|Flow Matching 目标速度]]

$$
u = \varepsilon - y
$$

**含义**: 从噪声到干净数据的目标速度向量，网络需学习预测此速度场。

#### 公式3: [[Flow-Matching|Flow Matching 损失]]

$$
\mathcal{L}_\text{FM}(y) = \mathbb{E}_{\tau, \varepsilon}\left[\left\|f_\theta(y_\tau,\, \tau,\, C) - (\varepsilon - y)\right\|^2\right]
$$

**含义**: 网络 $f_\theta$ 在含噪数据 $y_\tau$ 和上下文 $C$（干净历史帧 + 注意力可见的 context）下预测目标速度，均方误差优化。

**符号说明**:
- $f_\theta$: DiT 网络
- $C$: 干净的条件上下文（历史帧 + 注意力可见的 context）
- $C$ 中包含可关注的 token，受跨流注意力掩码约束

#### 公式4: [[Flow-Matching|多流联合训练损失]]

$$
\mathcal{L} = \lambda_H \cdot \mathcal{L}_H + \lambda_L \cdot \mathcal{L}_L + \lambda_A \cdot \mathcal{L}_A
$$

**含义**: 对长时程流、局部流、动作三个模态分别计算 Flow Matching 损失后加权求和，各流的流时间步 $\tau$ 独立采样。

**符号说明**:
- $\lambda_H, \lambda_L, \lambda_A$: 各流损失权重
- $\mathcal{L}_H$: 长时程视觉流 FM 损失
- $\mathcal{L}_L$: 局部视觉流 FM 损失
- $\mathcal{L}_A$: 动作 FM 损失
- 各流的 $\tau$ 独立采样（允许不同去噪进度）

### 渐进推理（Progressive Inference）

三级推理加速机制（无需重新训练，仅修改推理过程）：

**1. 部分去噪（Partial Denoising）**

设推理步数约束：$n_H \leq n_L \leq n_A$（视觉流用更少步数，动作流用完整步数）。
视觉流提前完成去噪并缓存特征，动作流继续完整去噪，利用缓存的视觉特征。

**2. 联合去噪（Joint Denoising）**

所有流在同一时间步同时开始去噪（而非顺序），保证各流在相同噪声水平下特征对齐，使缓存特征更可靠。

**3. 动态跳步（Dynamic Skipping）**

借鉴 [[DreamZero]] 的思路：若连续两步预测的速度向量余弦相似度超过阈值 $\gamma$，则跳过下 $k$ 步的 DiT 计算，直接复用上一步速度。

---

## 关键图表

### Figure 1: 方法概览

![Figure 1 - HiLo-WAM Overview](https://arxiv.org/html/2611.05023v1/overview.png)

**说明**: 左侧为朴素 WAM（仅密集局部短视程预测），右侧为 HiLo-WAM（同时维护稀疏长时程流和密集局部流）。长时程流以步幅 $k$ 采样历史和未来关键帧，提供任务整体进度；局部流提供细粒度机器人-物体交互信息；两流共同引导动作生成。

### Figure 2: HiLo-WAM 架构

![Figure 2 - HiLo-WAM Architecture](https://arxiv.org/html/2611.05023v1/framework.png)

**说明**: 展示双流 DiT（Video DiT 共享权重处理 Hi 和 Lo 流）+ Action DiT 的整体结构，以及渐进推理的三步流程。T5 编码器提供语言条件；Video VAE 分别编码两个视觉流；层级联合注意力（layer-wise joint attention）连接 Video DiT 和 Action DiT。

### Figure 3: 跨流注意力掩码

![Figure 3 - Cross-Stream Attention Mask](https://arxiv.org/html/2611.05023v1/attention.png)

**说明**: 可视化各 token 组的注意力权限。核心设计：远端未来 Hi 流 token（$Z^{H,\text{far}}$）对 Lo 流和动作流不可见，防止高不确定性预测干扰控制；视觉 token 对动作 token 不可见，确保单向信息流。

### Figure 4: 仿真与真实场景定性对比

![Figure 4 - Qualitative Comparison](https://arxiv.org/html/2611.05023v1/performance.png)

**说明**: 对比 [[Fast-WAM]] 和 HiLo-WAM 在三个任务上的视觉预测和行为：
- **Put Bottles into Dustbin**：Fast-WAM 频繁切换目标瓶子，HiLo-WAM 一致地追踪同一目标。
- **Build House**：Fast-WAM 预测目标不稳定，HiLo-WAM 保持稳定的放置目标。
- **Mug Hanging**：HiLo-WAM 长时程流引导完成细粒度的杯钩对齐。

### Figure 5: 跨流注意力可视化

![Figure 5 - Cross-Stream Attention Visualization](https://arxiv.org/html/2611.05023v1/attention_vis.png)

**说明**: 单次推理中各流 token 间的注意力权重热图，展示长时程流将手臂位置和目标物体位置信息传递给局部流和动作流的机制。

### Figure 6: Place Can Basket 视频预测对比

![Figure 6 - Video Prediction Comparison](https://arxiv.org/html/2611.05023v1/video_predict.png)

**说明**: 在 Place Can Basket 任务上，对比仅使用局部流的预测（无长时程引导）vs HiLo-WAM 的双流预测 vs Ground Truth。HiLo-WAM 的局部流在长时程流引导下生成任务一致的抓取视频，仅局部流则抓取目标不稳定。

### Figure 7: 时间步幅对性能的影响

**（Figure 7 图片暂无在线 URL，详见原文）**

**说明**: RoboTwin 2.0 上不同时间步幅 $k$ 的平均成功率：
- $k=1$: 92.3%
- $k=4$: **93.7%**（最优）
- $k=8$: 92.8%

步幅 $k=4$ 获得最佳权衡：足够稀疏以覆盖长时程任务上下文，又不会让相邻关键帧冗余。

### Figure 8: 真实机器人任务流程

![Figure 8 - Real-World Task Progressions](https://arxiv.org/html/2611.05023v1/task.png)

**说明**: [[Fourier GR-3]] 人形机器人执行三个长流程双臂操作任务的关键帧：Build House（搭积木）、Disassemble House（拆积木）、Disassemble Heart（拆心形拼图）。每个任务包含 10 个里程碑式评分点。

### Figure 9: 真实世界评估结果

**（Figure 9 图片暂无在线 URL，详见原文）**

**说明**: 在 [[Fourier GR-3]] 上的定量评估（每任务 10 次试验）：
- **域内**：平均进度评分 6.9，成功率 33%，优于 π0.5（+1.0 分，+10%）和 Fast-WAM（+1.8 分，+13%）
- **域外**（变化光照、背景、杂物）：平均评分 6.0，成功率 30%，展示鲁棒性

### Table 1a: RoboTwin 2.0 平均成功率

| 方法 | Clean (%) | Rand. (%) | Avg. (%) |
|------|-----------|-----------|----------|
| π0 | — | — | — |
| π0.5 | — | — | — |
| X-VLA | — | — | — |
| StarVLA | — | — | — |
| Fast-WAM | 91.90 | 91.80 | 91.85 |
| LingBot-VA | 92.90 | 91.50 | 92.20 |
| AHA-WAM | 93.40 | 92.20 | 92.80 |
| **HiLo-WAM** | **93.87** | **93.52** | **93.70** |

**关键发现**: HiLo-WAM 超越最强 WAM 基线 AHA-WAM 0.9 个百分点（Avg），且在干净和随机场景下均最优。

### Table 1b: RoboDojo 长时程任务评估

| 方法 | 进度评分 | 成功率 (%) |
|------|----------|-----------|
| Fast-WAM | 9.14 | 5.17 |
| AHA-WAM | 8.61 | 2.67 |
| StarVLA | 14.15 | 6.50 |
| **HiLo-WAM** | **17.43** | **6.50** |

**关键发现**: 长时程任务上 HiLo-WAM 进度评分比 Fast-WAM 高 +8.29，比 AHA-WAM 高 +8.82，显示双流设计对长流程任务的显著增益。

### Table 2: 消融实验（RoboTwin 2.0）

| 配置 | Avg. 成功率 (%) | 说明 |
|------|----------------|------|
| 仅局部流（Lo only） | 90.58 | 无长时程引导 |
| + 长时程流（无历史 ctx 直接访问） | 93.22 | 有长时程但历史上下文不直达 |
| + 历史上下文直接访问（Full Model） | **93.70** | ctx 直接可见给局部和动作流 |
| 去掉局部流（Hi only） | 93.52 | 保留精细交互建模的必要性 |

**关键发现**: 长时程流贡献 +3.12%（93.70 vs 90.58），历史上下文直接访问额外贡献 +0.48%；局部流对精细动作不可或缺（93.52 vs 93.70）。

### Table 3: 渐进推理消融（效率评估）

| 配置 | $n_H / n_L$ | 延迟 (ms) | 加速比 | 成功率 (%) |
|------|-------------|-----------|--------|-----------|
| 基线（全步数）| 10/10 | 1114.46 | 1.00× | 93.54 |
| + 部分去噪 | 5/5 | 723.65 | 1.54× | 93.62 |
| + 联合去噪 | 5/5 | 619.85 | 1.80× | 93.70 |
| + 动态跳步 | 5/5 | 380.02 | **2.93×** | **93.70** |

**关键发现**: 三级加速叠加实现 2.93× 推理加速，且成功率不下降（甚至因联合去噪特征对齐而微升）。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| [[RoboTwin 2.0]] | 50 双臂任务 | 每任务 50 clean + 500 random demo，50 trials 评估 | 主要仿真 benchmark |
| [[RoboDojo]] | 42 任务（含 8 长流程任务）| 长流程任务 50 trials 评估，进度评分 | 长时程评估 |
| Fourier GR-3 真实世界 | 3 长流程双臂任务 | 50 小时遥操作数据，10 trials/任务 | 真实机器人验证 |

### 实现细节

- **视频骨干**: [[Video DiT]]，从 [[Wan 2.2-5B]] 初始化（30 层，隐层 3072）
- **Action DiT**: 30 层，隐层 1024
- **视频分辨率**: 240×320，视频 chunk $T=32$，动作 chunk $H=32$，动作/视频频率比 4
- **长时程流**: $m=8$ 历史帧，$n=8$ 未来帧，步幅 $k=4$
- **优化器**: AdamW，lr = 1e-4，weight decay = 0.01，cosine 学习率调度
- **推理**: 10 步动作去噪，视觉流 $n_H = n_L = 5$ 步（部分去噪），CFG scale = 1.0
- **动态跳步阈值**: 余弦相似度 $\gamma$（具体值见附录）

### 可视化结果

- 在 RoboTwin 2.0 的 Put Bottles into Dustbin 任务上，HiLo-WAM 长时程流始终追踪同一目标瓶子，而 Fast-WAM 在中途随机切换抓取目标，导致任务失败。
- 在 Mug Hanging 真实任务上，长时程流提前预见杯钩的接触位置，引导局部流完成亚厘米级精度的对齐动作。

---

## 批判性思考

### 优点

1. **设计直觉清晰**: 时间尺度解耦的思路有理论依据——任务级进度和物体级交互确实具有本质不同的时间尺度，分而治之合理。
2. **架构优雅**: 双流共享 Video DiT 权重，不额外增加大量参数，保持效率。
3. **推理加速实用**: 2.93× 加速且不降性能，工程价值高，渐进推理可应用于其他多模态扩散模型。
4. **全面验证**: 仿真 + 真实 + 鲁棒性测试，消融实验完整覆盖每个设计选择。

### 局限性

1. **依赖视频基础模型语义理解**: 性能上界受 [[Wan 2.2-5B]] 预训练能力约束，对该模型不擅长的场景（如非常规物体、极端遮挡）可能退化。
2. **长时程流需要准确的 $k$ 选择**: 步幅 $k$ 是手工设定的超参，不同任务的最优 $k$ 可能不同，自适应步幅选择是未来方向。
3. **不含显式子任务分解**: 对需要条件推理（conditional branching）的任务（如"如果 A 失败则尝试 B"）无能为力，作者也承认此局限并建议引入 VLM。
4. **真实数据规模**：50 小时遥操数据仅用于 3 个任务，数据效率评估不完整。

### 潜在改进方向

1. 引入 VLM 提供高层推理能力，解决需要显式子任务分解的场景。
2. 自适应时间步幅 $k$：让模型根据任务复杂度动态调整长时程流采样频率。
3. 将渐进推理推广到其他视觉-动作扩散模型（如 Diffusion Policy 类架构）。

### 可复现性评估

- [ ] 代码开源（暂无）
- [ ] 预训练模型（暂无）
- [x] 训练细节完整（附录 A 完整列出超参数）
- [ ] 数据集可获取（真实数据为内部采集数据）

---

## 关联笔记

### 基于

- [[Wan 2.2-5B]]: Video DiT 骨干网络，Video VAE 均从此初始化
- [[Flow-Matching]]: 训练目标，对三个模态分别优化 FM 损失
- [[Mixture-of-Transformers]]: 双流 DiT 的基础架构
- [[DreamZero]]: 动态跳步加速策略的来源

### 对比

- [[Fast-WAM]] / [[FAWAM]]: 短视程局部 WAM 基线，RoboTwin 91.85% vs HiLo-WAM 93.70%
- [[LingBot-Video]]: 语言引导视频 WAM 基线
- [[Action Chunking]]: 动作输出范式

### 方法相关

- [[世界动作模型（WAM）]]: 所属方法族
- [[Video DiT]]: 核心视觉骨干
- [[Dual-Stream Co-Training]]: 相关双流训练范式
- [[渐进式推理]]: 本文提出的推理加速概念

### 硬件/数据相关

- [[Fourier GR-3]]: 真实机器人验证平台
- [[RoboTwin 2.0]]: 主仿真 benchmark
- [[RoboDojo]]: 长时程评估 benchmark

---

## 速查卡片

> [!summary] HiLo-WAM
> - **核心**: 双流解耦 WAM，Hi 流（长时程稀疏）+ Lo 流（局部密集）
> - **方法**: 跨流注意力掩码 + 渐进推理（部分/联合去噪 + 动态跳步）
> - **结果**: RoboTwin 93.70%、RoboDojo 17.43 分、Fourier GR-3 成功率 33%、2.93× 推理加速
> - **代码**: 暂无

---

*笔记创建时间: 2026-11-05*
