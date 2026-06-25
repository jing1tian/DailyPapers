---
title: "SC3-Eval: Evaluating Robot Foundation Models via Self-Consistent Video Generation"
method_name: "SC3-Eval"
authors: [Wei-Cheng Tseng, Gashon Hussein, Yuzhu Dong, Allen Z. Ren, Lucy X. Shi, XuDong Wang, Sergey Levine, Zhaoshuo Li, Jinwei Gu, Florian Shkurti, Ming-Yu Liu, Quan Vuong]
year: 2026
venue: arXiv
tags: [world-model, policy-evaluation, video-generation, vla, flow-matching, simulation, robotics]
zotero_collection: 3-Robotics/World Model
image_source: online
arxiv_html: https://arxiv.org/html/2606.18610
created: 2026-06-25
---

# 论文笔记：SC3-Eval: Evaluating Robot Foundation Models via Self-Consistent Video Generation

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | University of Toronto, Vector Institute, NVIDIA, Physical Intelligence, Stanford University, UC Berkeley, Allen Institute for AI |
| 日期 | June 2026 (v1: 2026-06-17, v2: 2026-06-23) |
| 项目主页 | [weichengtseng.github.io/sc3-eval](https://weichengtseng.github.io/sc3-eval/) |
| 对比基线 | [[Ctrl-World]] / [[IRASim]] / [[Cosmos-Predict-2.5]] |
| 链接 | [arXiv](https://arxiv.org/abs/2606.18610) / [HTML](https://arxiv.org/html/2606.18610) |

---

## 一句话总结

> SC3-Eval 把 [[Cosmos3]] 视频基础模型改造成 [[VLA（视觉-语言-动作模型）|VLA 策略]] 的"数字孪生评测员"：通过正向-逆向动力学一致性、跨视角一致性、测试时一致性三重约束，让闭环视频 rollout 的成功率预测与真实机器人高度相关（Pearson 0.929）。

---

## 核心贡献

1. **正向-逆向动力学联合训练（Forward-Inverse Dynamics Consistency）**: 同一个 backbone 同时学习"动作→帧"（正向）和"帧→动作"（逆向）两个方向，逆向目标作为隐式正则化项，把自回归 rollout 锚定在物理可行的动作流形上，显著抑制误差累积漂移。
2. **跨视角一致性（Cross-View Consistency）**: 训练时随机遮蔽一个相机视角，让模型用其余视角去补全（inpaint）它，强制多视角观测在生成时保持互相一致，避免腕部相机等局部视角出现与第三人称视角矛盾的画面。
3. **测试时一致性 + 不确定性早停（Test-Time Consistency with Uncertainty-Driven Early Termination）**: 推理阶段复用逆向动力学模式，把"策略提议的动作"与"从生成帧反推出的动作"之间的偏差作为免费的不确定性信号，超过阈值就提前终止 rollout，防止把已经失真的视频继续滚下去污染评分。
4. **预测-执行视野解耦（Prediction-Execution Horizon Decoupling）**: 训练时使用比执行时更长的预测视野（$l'=24$ vs $l=16$），既保留预训练阶段对长视频的先验，又给每个 chunk 提供更丰富的监督信号。
5. **真实世界基准与系统性验证**: 构建 381 小时的真实「清理餐桌」（table bussing）操作数据集，覆盖 12 类物体，对 7 个 [[VLA（视觉-语言-动作模型）|VLA]] checkpoint（含分布外的反向任务）做离线/在线、同分布/分布外的全面对比，不仅报告聚合指标，还验证了失败模式的可复现性。

---

## 问题背景

### 要解决的问题

评测通用机器人操作策略（[[VLA（视觉-语言-动作模型）|VLA]]）在真实世界中代价高昂：每次评测都需要占用物理机器人时间、人工复位环境、人工监督打分，难以规模化扩展到大量 checkpoint 的迭代评测。[[Action-Conditioned World Model|动作条件视频世界模型]]提供了一种可扩展的替代方案——用生成的视频模拟策略 rollout，但这条路线面临三个核心挑战：

1. **自回归误差累积**：长视野 rollout 中，[[Action-Conditioned World Model|动作条件世界模型]]每一步的小误差会随时间不断复合放大（drift）。
2. **多视角不一致**：多相机观测必须保持互相一致，否则策略可能被视觉上不连贯的场景误导而做出错误判断。
3. **分布外泛化**：评测器需要泛化到训练分布之外的策略行为——被评测的策略本身可能就是产生 OOD 视频内容的来源。

### 现有方法的局限

- 物理仿真器（如 [[SIMPLER]]/[[SimplerEnv]]、RoboLab）依赖人工建模资产和物理参数，对真实场景的视觉/动力学细节覆盖有限，跨任务迁移成本高。
- Real-to-sim 方法（如 PolaRiS）仍需大量人工标定与资产重建。
- 纯前向视频世界模型（[[Ctrl-World]]、[[IRASim]]、[[Cosmos-Predict-2.5]] 等）只学习 $\mathcal{W}^{fd}(v_{t+1} \mid v_{1:t}, a_{1:t})$，缺乏内在机制约束生成轨迹停留在物理可行的动作流形上，长视野闭环 rollout 容易漂移；同时大多只建模单一视角，未显式处理多视角一致性。

### 本文的动机

观察到视频基础模型（如 [[Cosmos3]]）本身已经具备统一处理多种生成任务的能力，作者认为：如果把"动作生成"所需要的正向-逆向动力学结构，重新用于"策略评测"这个互补任务，那么逆向动力学目标天然提供了一个无需额外训练成本的自我一致性检验信号——既能在训练时充当正则化锚点抑制漂移，又能在推理时复用为不确定性度量，触发早停。

---

## 方法详解

### 问题形式化

设 $\Pi = \{\pi_1, \dots, \pi_n\}$ 为待评测的机器人操作策略集合。目标是构造一个世界模拟器 $\mathcal{W}$，使得策略 $\pi_i$ 在 $\mathcal{W}$ 内 rollout 得到的表现 $R_{\mathcal{W},i}$ 与该策略在真实世界中的表现 $R_i$ 高度相关——既要绝对数值上**校准**（用 [[Pearson Correlation|Pearson 相关系数]] $r(R, R_\mathcal{W})$ 衡量），也要在排序上**保序**（用 [[MMRV (Mean Maximum Rank Violation)|MMRV]] 衡量）。

### 模型架构

SC3-Eval 采用 **统一正向-逆向动力学视频模型** 架构：
- **Backbone**: 初始化自预训练的 [[Cosmos3|Cosmos3-Nano]] 权重，沿用其 [[Diffusion Transformer|DiT]] 主干与 [[Flow Matching|rectified-flow]] 生成范式
- **输入**: 多视角观测帧 $v_{1:t}$（含第三人称视角与腕部视角）+ 动作序列 $a_{1:t}$，动作以附加 token 的形式注入序列
- **核心模块**: 三种联合训练模式共享同一组参数，仅通过"哪些 token 是干净的、哪些是加噪的"来区分模式（见 Figure 2）
- **输出**: 正向模式输出未来视频帧；逆向模式输出动作估计 $\hat{a}_i$；跨视角模式输出被遮蔽视角的补全帧
- **训练数据**: 381 小时真实「清理餐桌」操作数据集

### 核心模块

#### 模块1: 正向-逆向动力学一致性（Forward-Inverse Dynamics Consistency）

**设计动机**: 传统视频世界模型只学习正向映射 $\mathcal{W}^{fd}(v_{t+1} \mid v_{1:t}, a_{1:t})$，没有任何机制约束生成的视频序列对应着物理上可执行的动作。SC3-Eval 联合训练正向与逆向两个方向，通过参数共享把生成过程"锚定"在物理可行的动作流形（physically plausible action manifold）上。

**具体实现**:
- 正向模式：给定历史帧与动作，预测下一段视频帧（标准的 [[Action-Conditioned World Model|动作条件世界模型]]目标）
- 逆向模式：给定相邻帧，反推产生该帧变化所需的动作（即 [[Inverse Dynamics Model|逆向动力学模型]]目标），用于把生成过程"拉回"物理可行域
- 两个模式共享同一 backbone，仅通过 mask 哪些 token 加噪来切换

#### 模块2: 跨视角一致性（Cross-View Consistency）

**设计动机**: 多相机系统中，如果某个视角（如腕部相机）的生成内容与第三人称视角矛盾（例如腕部相机显示物体已抓取，但第三人称视角显示物体仍在桌上），策略会被这种视觉不连贯误导。

**具体实现**:
- 训练时随机选择一个相机视角作为目标，用其余视角的信息去 inpaint（补全）该目标视角
- 每个视角因此被监督要与其余视角保持一致，而不是独立生成

#### 模块3: 联合训练的模式采样

三种目标按概率随机分配，每个训练样本只贡献单一模式的梯度：

| 模式 | 采样概率 |
|------|---------|
| 正向动力学（含多视角） | $p_{FD} = 0.8$ |
| 跨视角补全 | $p_{CVI} = 0.1$ |
| 逆向动力学 | $p_{ID} = 0.1$ |

#### 模块4: 推理时的测试时一致性（Test-Time Consistency）

**预测-执行视野解耦**: 策略 $\pi$ 从观测 $\mathcal{V}$ 提议长度为 $l'$ 的动作 chunk，正向模式 $\mathcal{W}^{fd}$ 渲染对应帧，但闭环 rollout 中只取前 $l$ 帧追加进历史（$l < l'$），随后策略重新规划。训练时使用更长的预测视野 $l' = 24$，执行时只取 $l=16$ 帧，原因有两点：
1. **预训练先验保留**：backbone 本身在更长视频片段上预训练，过短的训练片段会破坏这一先验
2. **单 chunk 监督信息更丰富**：过短的 chunk 暴露给模型的物体运动信息不足

**不确定性驱动的早停**: 推理阶段复用逆向动力学模式，把策略提议的动作与从生成帧反推出的动作之间的差异，作为该 chunk 的"自我一致性误差"，超过阈值 $\tau$ 时终止 rollout，避免继续在已经失真的视频上滚动评分。

---

## 关键公式

### 公式1: [[Pearson Correlation|策略表现校准]]

$$
r(R, R_{\mathcal{W}})
$$

**含义**: 衡量世界模拟器 $\mathcal{W}$ 中各策略的预测表现 $R_\mathcal{W}$ 与真实世界表现 $R$ 之间的线性相关程度，用于评估评测器的**绝对数值校准**能力。论文未给出展开形式，采用标准 [[Pearson Correlation|Pearson 相关系数]]定义。

**符号说明**:
- $R = (R_1, \dots, R_n)$: $n$ 个策略在真实世界中的表现（如成功率）
- $R_{\mathcal{W}} = (R_{\mathcal{W},1}, \dots, R_{\mathcal{W},n})$: 对应策略在世界模拟器 $\mathcal{W}$ 中 rollout 得到的预测表现

### 公式2: [[MMRV (Mean Maximum Rank Violation)|相对排序一致性]]

$$
\text{MMRV}(R, R_{\mathcal{W}})
$$

**含义**: 衡量评测器对策略两两排序的一致性，是 checkpoint 选择决策中比绝对数值更关键的属性（即使绝对值有偏差，只要相对排序正确，仍可用于挑选最佳 checkpoint）。该指标定义沿用先前工作（SIMPLER），论文未在正文中重新展开公式。

**符号说明**: 与公式1 共用 $R$、$R_{\mathcal{W}}$ 定义；数值越低代表排序违反越少，一致性越好。

### 公式3: 训练目标（[[Flow Matching|flow matching]] 目标）

模型在加噪 token 上使用 [[Flow Matching|flow matching]] 目标进行训练，继承 [[Cosmos3]] backbone 的 [[Rectified Flow|rectified-flow]] 形式化（论文未在正文重新展开具体损失表达式，三种模式仅通过加噪 token 的选择区分，均复用同一套 flow matching 目标）。

### 公式4: [[Inverse Dynamics Model|逐 chunk 一致性误差]]（测试时不确定性信号）

$$
U_{\text{chunk}}(t) = \frac{1}{l} \sum_{i=t}^{t+l-1} \left\| a_i - \hat{a}_i \right\|_2
$$

**含义**: 第 $t$ 个 rollout chunk 的自我一致性误差，是策略实际提议的动作与世界模型用逆向动力学模式从生成帧反推出的动作之间的平均 $\ell_2$ 距离。该值越大说明生成的视频越"不像"策略实际想做的动作所对应的结果，意味着 rollout 可能已经发生漂移。

**符号说明**:
- $l$: 执行视野长度（实验中 $l=16$）
- $a_i$: 策略在时刻 $i$ 实际提议的动作
- $\hat{a}_i$: 世界模型用逆向动力学模式从生成的第 $i$ 帧反推出的动作估计
- $t$: 当前 rollout chunk 的起始时刻

### 公式5: 早停判定规则

$$
\text{Terminate at } t \iff U_{\text{chunk}}(t) > \tau
$$

**含义**: 当某个 chunk 的一致性误差超过阈值 $\tau$ 时，立即终止该条 rollout，不再继续生成后续帧，从而避免已经失真的视频继续污染该策略的评分。

**符号说明**:
- $\tau$: 早停阈值，在留出子集上调优，实验中取 $\tau = 0.02$（对应约 4.88% 的 rollout 被提前终止，见 Table 5）

---

## 关键图表

### Figure 1: SC3-Eval 三大一致性轴概览 / The three consistency axes of SC3-Eval

![Figure 1](https://arxiv.org/html/2606.18610v2/fig/teaser_v6.png)

**说明**: 三个子图分别展示：(a) 正向-逆向动力学一致性——对比真实轨迹与"仅正向训练"基线的漂移情况；(b) 多视角一致性——从第三人称视角预测腕部视角的好/坏案例对比；(c) 测试时一致性——通过动作分歧实现早停的示意。

### Figure 2: 自一致性训练流程 / Self-consistency training for SC3-Eval

![Figure 2](https://arxiv.org/html/2606.18610v2/fig/training_mode_v4.png)

**说明**: 展示三种联合训练模式（正向动力学、跨视角补全、逆向动力学）共享同一 backbone，仅通过哪些 token 是干净/加噪来区分模式，对应正文模块1-3 的具体实现。

### Figure 3: 真实世界实验配置 / Real-world experiment setup

![Figure 3](https://arxiv.org/html/2606.18610v2/fig/exp_setup_v3.png)

**说明**: (a) 工作空间布局，橙色箭头表示「清理餐桌」任务流程，绿色箭头表示反向变体（OOD 任务）；(b) 三个同步相机视角；(c) 三种成功判定标准下的示例轨迹。

### Figure 4: 预测表现与真实表现的相关性 / Correlation between predicted and real-world policy performance

![Figure 4](https://arxiv.org/html/2606.18610v2/fig/corr_v3.png)

**说明**: (a) 离线评测（开环）与 (b) 在线评测（闭环）下，预测成功率与真实成功率的散点相关图，分别展示同分布（in-distribution）与分布外（out-of-distribution）情形，对应 Table 1 的数值结果。

### Figure 5: 在线生成的定性 rollout / Qualitative rollouts for online generation

![Figure 5](https://arxiv.org/html/2606.18610v2/fig/qual_v2.png)

**说明**: 展示初始观测、真实世界 rollout、以及 SC3-Eval 在线生成 rollout 三者的并排对比，体现生成轨迹对真实失败模式的复现能力。

### Figure 6: 在线评测下各类别结果复现率 / Per-category outcome reproduction rate under online evaluation

![Figure 6](https://arxiv.org/html/2606.18610v2/x1.png)

**说明**: 按"语言跟随失败""物体抓取失败""物体放置失败""无失败"四个结果类别，统计各 baseline 与 SC3-Eval 复现真实失败模式的比率，对应 Table 6。

### Figure 7: 逆向动力学锚定的定性效果 / Qualitative effect of inverse dynamics grounding on offline rollouts

![Figure 7](https://arxiv.org/html/2606.18610v2/fig/qual_id_ablation.png)

**说明**: 对比"有/无逆向动力学一致性"训练得到的离线 rollout 视觉效果，展示逆向动力学如何抑制自回归漂移、保持画面物理合理性。

### Figure 8: 跨视角补全对腕部视角重入场景的定性效果 / Qualitative effect of cross-view inpainting on wrist-view scene re-entry

![Figure 8](https://arxiv.org/html/2606.18610v2/fig/qual_cv_ablation.png)

**说明**: 展示腕部相机视角因抓取/放置动作"离开再重新进入"场景时，有无跨视角一致性训练对画面连贯性的影响。

### Figure 9: 不确定性驱动早停的示例 rollout / Uncertainty-driven early termination on example rollout

![Figure 9](https://arxiv.org/html/2606.18610v2/fig/uncertainty_examples.png)

**说明**: 展示某条 rollout 中逐 chunk 一致性误差 $U_{\text{chunk}}(t)$ 随时间变化的曲线，以及超过阈值 $\tau$ 触发早停的具体时刻。

### Table 1: 策略评测主结果（Pearson r ↑ / MMRV ↓）

| Method | InD Offline r↑ | InD Offline MMRV↓ | InD Online r↑ | InD Online MMRV↓ | OOD Offline r↑ | OOD Offline MMRV↓ | OOD Online r↑ | OOD Online MMRV↓ |
|--------|------|------|------|------|------|------|------|------|
| Ctrl-World | 0.878 | 0.185 | 0.871 | 0.191 | 0.821 | 0.191 | 0.832 | 0.179 |
| IRASim | 0.773 | 0.176 | 0.730 | 0.188 | 0.700 | 0.188 | 0.663 | 0.364 |
| Cosmos-Predict 2.5 | 0.807 | 0.148 | 0.897 | 0.090 | 0.808 | 0.145 | 0.871 | 0.195 |
| **SC3-Eval (Ours)** | **0.959** | **0.018** | **0.984** | **0.022** | **0.962** | **0.022** | 0.870 | 0.171 |

**说明**: InD = in-distribution（同分布，正向「清理餐桌」任务），OOD = out-of-distribution（分布外，反向「清理餐桌」任务）。SC3-Eval 在七项指标中六项最优，在线 OOD 的 Pearson r 与最佳 baseline（Cosmos-Predict 2.5 的 0.871）基本持平，整体 closed-loop Pearson 0.929、MMRV 0.119（论文摘要中的汇总数字）。

### Table 2: 消融实验（在线闭环）

| Variant | Pearson r ↑ | MMRV ↓ |
|---------|-----------|--------|
| **Full model** | **0.929** | **0.119** |
| w/o 逆向动力学 (inverse dynamics) | 0.842 | 0.175 |
| w/o 跨视角补全 (cross-view inpainting) | 0.802 | 0.199 |
| w/o 早停 (early termination) | 0.871 | 0.151 |
| w/o 视野解耦 (horizon decoupling) | 0.807 | 0.177 |

**说明**: 去掉跨视角补全对相关性影响最大（0.929 → 0.802），其次是视野解耦（→0.807）和逆向动力学（→0.842），说明三种一致性机制对最终评测精度都有不可替代的贡献。

### Table 3: 七个策略 checkpoint 的真实世界成功率

| ID | 任务族 | 语言跟随 (Language Following) | 物体抓取 (Object Lifting) | 物体放置 (Object Placing) |
|----|--------|-------|--------|--------|
| 1 | 清理餐桌（同分布） | 0.946 | 0.610 | 0.126 |
| 2 | 清理餐桌（同分布） | 0.876 | 0.688 | 0.306 |
| 3 | 清理餐桌（同分布） | 0.959 | 0.720 | 0.350 |
| 4 | 清理餐桌（同分布） | 0.446 | 0.154 | 0.008 |
| 5 | 反向清理餐桌（分布外） | 0.745 | 0.444 | 0.286 |
| 6 | 反向清理餐桌（分布外） | 0.675 | 0.105 | 0.057 |
| 7 | 反向清理餐桌（分布外） | 0.926 | 0.785 | 0.449 |

**说明**: 7 个 checkpoint 成功率跨度很大（语言跟随 0.446–0.959），为评测器提供了足够的区分度去检验相关性与排序一致性。

### Table 4: 离线 rollout 的 PSNR

| Method | In-distribution | Out-of-distribution |
|--------|-----------------|----------------------|
| Ctrl-World | 14.80 | 14.61 |
| IRASim | 13.99 | 13.12 |
| Cosmos-Predict 2.5 | 14.83 | 14.81 |
| w/o 逆向动力学 | 14.97 | 14.07 |
| w/o 跨视角补全 | 15.11 | 15.01 |
| w/o 视野解耦 | 14.87 | 14.76 |
| **SC3-Eval (Ours)** | **15.44** | **15.12** |

**说明**: 像素级 [[PSNR]] 与最终评测相关性指标并非严格对应（如 w/o 跨视角补全 的 PSNR 反而略高于完整模型），说明仅用图像重建质量无法完全反映评测器的可靠性，这正是论文引入 Pearson/MMRV 而非单纯 PSNR 作为核心指标的原因。

### Table 5: 早停阈值消融

| 阈值 τ | 终止率 | InD r↑ | InD MMRV↓ | OOD r↑ | OOD MMRV↓ |
|-------------|------------------|--------|-----------|--------|------------|
| ∞（不早停） | 0% | 0.956 | 0.093 | 0.853 | 0.197 |
| 0.05 | 2.21% | 0.974 | 0.027 | 0.881 | 0.170 |
| **0.02** | **4.88%** | **0.984** | **0.022** | 0.870 | 0.171 |
| 0.01 | 10.22% | 0.941 | 0.101 | **0.902** | **0.159** |

**说明**: $\tau=0.02$ 在 InD 上综合最优，被选为最终配置；$\tau=0.01$ 在 OOD 上指标更好但终止率显著更高（10.22% 的 rollout 被提前截断），存在过早放弃有效轨迹的风险，体现阈值选择上的精度-覆盖率权衡。

### Table 6: 各失败类别结果复现率（离线 / 在线）

| Method | Language（语言跟随失败） | Lifting（抓取失败） | Placing（放置失败） | No Failure（无失败） | Average ↑ |
|--------|----------|---------|---------|-----------|-----------|
| Ctrl-World | 0.25 / 0.47 | 0.90 / 0.71 | 0.81 / 0.25 | 0.25 / 0.35 | 0.55 / 0.45 |
| IRASim | 0.03 / 0.56 | 0.92 / 0.64 | 0.80 / 0.11 | 0.05 / 0.12 | 0.45 / 0.36 |
| Cosmos-Predict 2.5 | 0.91 / 0.57 | 0.77 / 0.53 | 0.15 / 0.22 | 0.02 / 0.20 | 0.46 / 0.38 |
| **SC3-Eval (Ours)** | 0.91 / 0.61 | 0.72 / 0.80 | 0.47 / 0.31 | **0.38 / 0.53** | **0.62 / 0.56** |

**说明**: SC3-Eval 在离线和在线两种设定下平均复现率均最高（0.62 / 0.56），尤其在"无失败"类别上大幅领先（0.38/0.53 vs 第二名 0.25/0.35），表明其不仅能预测聚合成功率，还能真实复现策略具体的失败/成功模式，这是单纯看汇总指标无法体现的能力。

---

## 实验结果

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| 真实「清理餐桌」操作数据集 | 381 小时 | 12 类物体，多视角相机，正向+反向任务变体 | 训练 + 同分布/分布外评测 |

### 实现细节

- **Backbone**: [[Cosmos3|Cosmos3-Nano]] 预训练权重初始化
- **优化器**: AdamW，学习率 $1\times10^{-4}$，weight decay 0.05
- **Batch Size**: 有效 batch size 512 条轨迹/step
- **训练步数**: 共 24,000 步
- **硬件**: 32 块 NVIDIA GB200 GPU（8 节点 × 4 GPU），约 2.2 个 wall-clock 天
- **数据增强**: 伪动作增强（pseudo-action augmentation，$p=0.5$），10Hz / 20Hz 多帧率训练
- **关键超参**: 执行视野 $l=16$ 帧，预测视野 $l'=24$ 帧，早停阈值 $\tau=0.02$
- **推理速度**: 单块 GB200 GPU 上闭环在线评测约 2.3 秒/chunk，比基于物理的仿真器慢数个量级

### 可视化结果

定性 rollout（Figure 5、7、8）显示：去掉逆向动力学一致性后，离线 rollout 在长视野下会出现明显的视觉漂移（物体形变、消失）；去掉跨视角一致性后，腕部相机在"离开-重新进入"场景时容易与第三人称视角矛盾。Figure 9 的不确定性曲线显示一致性误差能在视觉失真发生前的早期阶段被检测出来，为早停提供有效的提前预警信号。

---

## 批判性思考

### 优点
1. **零额外训练成本的不确定性信号**：逆向动力学模式本身是训练目标的一部分，推理时复用作为不确定性度量，无需额外训练专门的不确定性估计模块，设计上非常优雅。
2. **不止于聚合指标**：通过 Table 6 的失败模式复现率实验，证明了该评测器不仅"打分准"，还能"为什么打这个分"地复现具体失败原因，这对策略调试比单一数字更有价值。
3. **系统性的真实世界验证规模**：381 小时数据、7 个真实策略 checkpoint、覆盖同分布与分布外（反向任务）双重维度，是同类工作中较为扎实的真实世界实验设计。

### 局限性
1. **推理速度瓶颈**：单 GB200 上 2.3 秒/chunk，比物理仿真器慢"数个量级"，难以用于大规模、高频率的策略迭代评测（如 RL 训练中的在线评测）。
2. **短视野验证**：训练和验证都局限于约 20 秒的短时操作任务，论文自己承认在更长视野下可能出现"局部校正信号不足以抑制长期漂移"和"超出时序连贯性先验范围的视觉退化"两种新失败模式，尚未验证。
3. **MMRV/Pearson 的公式细节未展开**：论文正文未重新给出这两个核心指标的数学定义（依赖读者查阅 SIMPLER 等先前工作），对完全独立复现略有不便。
4. **OOD 在线指标提升有限**：Table 1 显示在线 OOD 场景下 SC3-Eval（0.870）相对最佳 baseline Cosmos-Predict 2.5（0.871）并无优势，说明分布外泛化仍是该方法的相对薄弱环节。

### 潜在改进方向
1. 探索蒸馏或缓存机制降低推理延迟，使其能用于在线 RL 微调场景下的快速评测反馈。
2. 针对长视野任务，探索全局一致性约束（而非仅逐 chunk 局部约束）以缓解长期漂移。
3. 进一步研究早停阈值 $\tau$ 的自适应选取（如按任务难度或策略置信度动态调整），而非固定全局阈值。

### 可复现性评估
- [ ] 代码开源（论文/项目主页未明确提及代码仓库链接）
- [ ] 预训练模型（未提及权重发布）
- [x] 训练细节完整（优化器、学习率、batch size、步数、硬件均有详细说明）
- [ ] 数据集可获取（381 小时数据集为自采集，未提及公开计划）

---

## 关联笔记

### 基于
- [[Cosmos3]]: SC3-Eval 的 backbone 初始化权重来源，沿用其 [[Diffusion Transformer|DiT]] 架构与 [[Flow Matching|rectified-flow]] 训练范式
- [[Inverse Dynamics Model]]: 正向-逆向联合训练中的逆向分支直接对应该概念
- [[Action-Conditioned World Model]]: 正向分支的基础范式

### 对比
- [[Ctrl-World]]: 主要对比 baseline 之一，传统单向视频世界模型
- [[IRASim]]: 主要对比 baseline 之一
- [[Cosmos-Predict-2.5]]: 主要对比 baseline，在 OOD 在线场景下与 SC3-Eval 表现最接近
- [[SIMPLER]]: 物理仿真器路线的代表，MMRV 指标的来源工作

### 方法相关
- [[Flow Matching]]: 训练目标基础
- [[Rectified Flow]]: Cosmos3 backbone 沿用的生成范式
- [[Pearson Correlation]]: 核心评测指标之一
- [[MMRV (Mean Maximum Rank Violation)]]: 核心评测指标之一

### 硬件/数据相关
- [[VLA（视觉-语言-动作模型）]]: 被评测的策略类型
- 381 小时真实「清理餐桌」数据集（自采集，未公开）

---

## 速查卡片

> [!summary] SC3-Eval: Evaluating Robot Foundation Models via Self-Consistent Video Generation
> - **核心**: 用三重自一致性（正向-逆向动力学、跨视角、测试时不确定性早停）把视频基础模型改造成可信赖的机器人策略评测器
> - **方法**: 在 [[Cosmos3]] backbone 上联合训练正向动力学（$p=0.8$）、跨视角补全（$p=0.1$）、逆向动力学（$p=0.1$）三种模式，推理时复用逆向模式做早停不确定性信号
> - **结果**: 闭环 Pearson r = 0.929，MMRV = 0.119，超越 [[Ctrl-World]] / [[IRASim]] / [[Cosmos-Predict-2.5]] 三个强基线，并泛化到训练分布外的反向任务
> - **代码**: 未公开（截至笔记创建时）

---

*笔记创建时间: 2026-06-25*
