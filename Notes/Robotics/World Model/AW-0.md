---
title: "Agentic World Action Models"
method_name: "AW-0"
authors: [Zhuoyang Liu, Dantong Niu, Botong Zhang, Boning Shao, Jing Wang, Ruijie Zheng, Yuvan Sharma, Runyu Lu, Jiaming Liu, Renrui Zhang, Roei Herzig, Yutong Bai, Ken Goldberg, Danfei Xu, Yuke Zhu, Guanzhi Wang, "Linxi 'Jim' Fan", Trevor Darrell]
year: 2026
venue: arXiv
tags: [world-action-model, agentic-robotics, code-as-policies, mixture-of-transformers, skill-learning, bimanual-manipulation, flow-matching]
zotero_collection: Robotics/World Model
image_source: online
arxiv_html: https://arxiv.org/html/2610.30136v1
created: 2026-10-23
---

# 论文笔记：Agentic World Action Models

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | UC Berkeley, NVIDIA, University of Pennsylvania |
| 日期 | October 2026 |
| 项目主页 | [agentic-wam.github.io](https://agentic-wam.github.io) |
| 对比基线 | [[π0.5]]、[[DreamZero]]、[[EgoScale]]、[[WLA]] |
| 链接 | [arXiv](https://arxiv.org/abs/2610.30136) |

---

## 一句话总结

> AW-0 将代码生成嵌入世界动作模型，通过三流（代码、视频、动作）联合去噪，实现规划、预测、执行的统一，并支持技能库复用的智能体机器人执行。

---

## 核心贡献

1. **三流混合专家架构（Mixture-of-Transformers）**: 将 [[WAM|世界动作模型]] 的视频-动作双流扩展为代码-视频-动作三流，用 [[Mixture-of-Transformers]] 联合去噪；代码使用[[吸收掩码扩散|吸收状态掩码扩散]]，视频和动作使用 [[Flow-Matching|Flow Matching]]，通过块状因果注意力（C→V→A 顺序）实现跨专家信息流
2. **自动程序标注与技能库构建管线**: 用编码智能体对机器人和人类 ego 视频自动标注参数化技能程序，并通过 LLM 协调器将重复子程序蒸馏为可复用的[[技能库|技能库]]，无需人工标注
3. **混合执行策略**: 测试时每步重规划，由有效性和进度检查决定切换到代码执行（[[混合执行]]）还是直接动作块执行，结合两者优势

---

## 问题背景

### 要解决的问题

现有[[WAM|世界动作模型]]能联合预测未来视觉状态和机器人动作，但缺乏对技能组合复用的显式支持；而智能体机器人系统（代码规划+技能调用）虽然具备组合性，但缺乏物理动力学建模。

### 现有方法的局限

- [[WAM]] 类方法（如 WLA、DreamZero）仅做视频-动作联合预测，没有语义级别的技能规划，语义推理和语言跟随能力有限
- 智能体机器人方法（[[Code as Policies]]、[[Agentic Robot]] 等）把代码规划置于策略之外，无法建模物理动力学，执行受限于预定义基元
- 两类方法割裂，无法在同一统一框架内做到"规划即预测即执行"

### 本文的动机

将代码作为世界动作模型内部的**规划接口**：代码既结构化多步预测与控制，又作为可复用程序供智能体执行，实现"agentic code enables connection between planning, prediction and execution within WAMs"。

---

## 方法详解

### 模型架构

AW-0 采用 [[Mixture-of-Transformers]] 架构，包含三个专家：

- **输入**: 观测 $\mathcal{O}_t$（RGB 帧 + 手部图像）、语言指令 $L$、机器人状态 $S_t$（关节角度/力矩/工具状态）、记忆 $M_t$（前序代码片段的压缩槽）
- **代码专家**: 冻结 Qwen3-VL ViT + SDAR-1.7B 初始化的掩码扩散语言骨干；[[吸收掩码扩散|吸收掩码扩散]] 生成参数化技能程序 $C_0$
- **视频专家**: [[Diffusion Transformer (DiT)|DiT]] 初始化自 Cosmos-Predict2.5-2B；[[Flow-Matching]] 生成未来视频帧潜码 $Z_0$
- **动作专家**: 独立 DiT 从头预训练；[[Flow-Matching]] 生成动作块 $A_0$（64 维统一状态-动作空间，水平 32 步）
- **注意力结构**: 专家内双向注意力；跨专家块状因果注意力，顺序 C→V→A
- **总参数**: ~4B（三专家合计）

### 核心模块

#### 模块 1: 块状因果跨专家注意力（Block Causal Cross-Expert Attention）

**设计动机**: 利用 [[Block Causal Attention]] 建立代码→视频→动作的单向信息流，代码提供语义先验指导视频和动作生成。

**具体实现**:
- 代码 token 可与自身双向注意力，但视频/动作专家不能反向看代码
- 视频专家可看代码的输出，但不能看动作专家
- 动作专家可看代码和视频的输出

#### 模块 2: 噪声重采样（Noise Resampling）

**设计动机**: 视频专家预训练时用偏向低噪声的 Beta 分布，直接联合训练会导致噪声分布不匹配，需要重采样以平衡预训练和联合训练目标。

**具体实现**:
- 视频噪声级别从预训练分布 $p_\text{pre}$ 和均匀分布 $p_\text{ref}$ 的混合中采样（$\alpha = 0.5$）
- 用重要性权重 $w_v(u) = p_\text{pre}(u)/q(u)$ 恢复预训练分布的视频损失
- 用 $w_g(u) = p_\text{ref}(u)/q(u)$ 驱动动作和代码损失的均匀覆盖

#### 模块 3: 混合执行（Hybrid Execution）

**设计动机**: 代码生成的参数化技能在精细操作中有时不如直接动作块精确，需要动态切换。

**具体实现**:
- 每步重规划（0.23s/次，RTX 5090）
- 解码当前代码切片，通过有效性检查（语法正确 + API 匹配）和进度检查（前序代码片段完成度）
- 通过检查则执行代码（调用[[技能库|技能库]]中的技能），否则退回直接动作块

#### 模块 4: 自动标注管线

**设计动机**: 大规模程序化标注人工不可行，需要自动化。

**具体实现**:
1. **子任务程序生成**: Gemini 3.1 Pro 对每个轨迹写参数化程序（仅声明式 API）
2. **技能库蒸馏**: Claude Opus 4.8 担任协调器，识别重复子程序，蒸馏为可复用技能库（YAM：23 个技能，人类 ego：27 个技能）
3. **缓存标注**: 冻结技能库后，通过接触模式和运动模式匹配为剩余轨迹复用已有程序

---

## 关键公式

### 公式 1: [[WAM|联合条件分布]]

$$
p_\theta(C_0, Z_0, A_0 \mid \mathcal{C}_t)
$$

**含义**: AW-0 联合建模代码 $C_0$、视频潜码 $Z_0$、动作块 $A_0$ 在时刻 $t$ 的条件分布。

**符号说明**:
- $C_0$: 参数化技能程序的代码 token 序列
- $Z_0$: 视频帧潜码（来自预训练 VAE）
- $A_0$: 动作块（机器人关节空间，水平 32 步）
- $\mathcal{C}_t = \{\mathcal{O}_t, L, S_t, M_t\}$: 当前条件上下文（观测、语言、状态、记忆）

### 公式 2: [[Mixture-of-Transformers|三专家联合预测]]

$$
(\ell^c_\theta, v^v_\theta, v^a_\theta) = F_\theta(\tilde{C}_r, Z_{\sigma_v}, A_{\sigma_a}, \sigma_v, \sigma_a; \mathcal{C}_t)
$$

**含义**: 三专家模型 $F_\theta$ 同时输出代码的 logit 预测 $\ell^c_\theta$、视频速度场 $v^v_\theta$、动作速度场 $v^a_\theta$。

**符号说明**:
- $\tilde{C}_r$: 吸收掩码比例为 $r$ 的带噪代码序列
- $Z_{\sigma_v}$: 噪声级别为 $\sigma_v$ 的视频潜码
- $A_{\sigma_a}$: 噪声级别为 $\sigma_a$ 的动作块
- $\sigma_v, \sigma_a$: 视频和动作的连续噪声级别

### 公式 3: [[Flow-Matching|流匹配扰动]]

$$
X^m_{\sigma_m} = (1 - \sigma_m) X^m_0 + \sigma_m \varepsilon_m, \quad \varepsilon_m \sim \mathcal{N}(0, I)
$$

**含义**: 对视频/动作进行线性插值加噪，$\sigma_m = 0$ 为干净样本，$\sigma_m = 1$ 为纯高斯噪声。

**符号说明**:
- $X^m_0$: 干净的视频潜码或动作块（$m \in \{v, a\}$）
- $\sigma_m$: 噪声级别，由噪声偏移函数从均匀随机变量映射而来

### 公式 4: [[Flow-Matching|噪声级别偏移（Noise Shift）]]

$$
\sigma_m = \varphi_{\gamma_m}(u_m) = \frac{\gamma_m u_m}{1 + (\gamma_m - 1) u_m}
$$

**含义**: 将均匀采样的 $u_m \in [0,1]$ 非线性映射到实际噪声级别 $\sigma_m$，$\gamma_m = 5$ 时偏向高噪声区域。

**符号说明**:
- $u_m \sim \mathcal{U}(0,1)$: 均匀随机变量
- $\gamma_m = 5$: 噪声偏移系数（视频和动作相同）

### 公式 5: [[吸收掩码扩散|代码吸收掩码扩散]]

$$
\tilde{c}_i = [\text{MASK}] \text{ if } b_i = 1 \text{ else } c_i, \quad b_i \sim \text{Bernoulli}(r), \quad r \sim \mathcal{U}(0,1)
$$

**含义**: 对代码 token 序列以随机比例 $r$ 随机掩码，模型需要预测被掩码的 token（吸收状态 Masked Diffusion）。

**符号说明**:
- $c_i$: 第 $i$ 个代码 token
- $r$: 掩码比例，从均匀分布随机采样
- $b_i$: 第 $i$ 个位置是否被掩码的伯努利指示变量

### 公式 6: [[噪声重采样|视频噪声重采样混合分布]]

$$
u_v \sim q(u) = \alpha \, p_\text{pre}(u) + (1 - \alpha) \, p_\text{ref}(u), \quad p_\text{ref} = \mathcal{U}(0,1), \quad \alpha = 0.5
$$

**含义**: 以 50/50 混合从预训练分布（偏低噪）和均匀分布中采样视频噪声级别，确保联合训练覆盖全噪声范围。

**符号说明**:
- $p_\text{pre}(u)$: 视频预训练时的噪声级别分布（$u_v = s \cdot b$，$b \sim \text{Beta}(7,1)$，$s=0.999$）
- $p_\text{ref}(u) = \mathcal{U}(0,1)$: 均匀参考分布，确保高噪声区域也有训练信号
- $\alpha = 0.5$: 混合权重

### 公式 7: 重要性权重

$$
w_v(u) = \frac{p_\text{pre}(u)}{q(u)}, \quad w_g(u) = \frac{p_\text{ref}(u)}{q(u)}
$$

**含义**: $w_v$ 用于视频损失，恢复预训练分布；$w_g$ 用于动作和代码损失，恢复均匀覆盖。

**符号说明**:
- $w_v$: 视频分支的重要性权重
- $w_g$: 动作和代码分支的重要性权重（"g" 来自 guidance）

### 公式 8: [[WAM|训练目标]]

$$
\mathcal{L} = \mathbb{E}_{u_v \sim q} \left[ \lambda_v \bar{w}_v \ell_v + \bar{w}_g \left( \lambda_a \ell_a + \lambda_c \ell_c \right) \right]
$$

**含义**: 三流的加权联合损失，视频用预训练分布权重，动作和代码用均匀覆盖权重。

**符号说明**:
- $\lambda_v, \lambda_a, \lambda_c$: 视频、动作、代码的损失权重
- $\bar{w}_v, \bar{w}_g$: 归一化后的重要性权重
- $\ell_v$: 视频速度场 Flow Matching 损失（目标：$\varepsilon_v - X^v_0$）
- $\ell_a$: 动作速度场 Flow Matching 损失
- $\ell_c$: 代码的交叉熵掩码扩散损失

### 公式 9: 统一状态-动作空间维度

$$
2 \times (9 + 22 + 1) = 64
$$

**含义**: 双臂机器人（YAM）状态-动作空间维度：2臂 × (9 关节角度 + 22 关节力矩 + 1 工具状态) = 64 维。

---

## 关键图表

### Figure 1: AW-0 系统概览

![Figure 1](https://arxiv.org/html/2610.30136v1/teaser_v6_1003.png)

**说明**: AW-0 的整体概览。上方展示三流生成框架（代码、未来视频、动作），下方展示从预训练数据到自动标注管线到真实机器人部署的完整流程，包含技能库和[[混合执行]]示例。

### Figure 2: Mixture-of-Transformers 模型架构

![Figure 2](https://arxiv.org/html/2610.30136v1/method_v6_0926.png)

**说明**: [[Mixture-of-Transformers]] 三专家架构。左侧展示代码、视频、动作专家的并行去噪过程，中间展示块状因果跨专家注意力（C→V→A），右侧展示各专家的初始化模型（Qwen3-VL、Cosmos-Predict2.5、scratch DiT）和记忆压缩槽 $M_t$。

### Figure 3: 自动程序标注管线

![Figure 3](https://arxiv.org/html/2610.30136v1/code_label_pipeline_v3_1003.png)

**说明**: 自动标注流程三阶段：(1) Gemini 3.1 Pro 对每条轨迹生成参数化子任务程序；(2) Claude Opus 4.8 作为协调器，发现重复模式并蒸馏为[[技能库|可复用技能库]]；(3) 冻结技能库后，通过接触/运动模式匹配为剩余轨迹缓存复用程序。

### Figure 4: RoboTwin 2.0 平均成功率对比

*（HTML 内嵌图，无独立 URL）*

**说明**: 展示 AW-0 在 RoboTwin 2.0 Clean/Random 两种设置下与 π0.5、DreamZero、EgoScale、WLA 的平均成功率对比。AW-0 在 Clean 设置下以 80.8% 大幅领先；AW-0 (hybrid) 在 Random 设置下以 29.2% 最高。

### Figure 5: 代码专家与注意力顺序消融

*（HTML 内嵌图，无独立 URL）*

**说明**: 三种注意力流向（C→V→A、V→C→A、仅 V 专家）下的平均 rubric score。C→V→A 最优（71.1），证明代码作为语义先验对视频和动作预测有正向引导作用。

### Figure 6: 推理效率与性能权衡

*（HTML 内嵌图，无独立 URL）*

**说明**: 平均 rubric score（柱）与 RTX 5090 上的推理延迟（红线）的对比。[[混合执行]] 策略下联合去噪（0.23s）比自回归生成（0.55s）快 2.4×，同时性能更高。

### Figure 7: 代码内容消融

*（HTML 内嵌图，无独立 URL）*

**说明**: 不同代码标注内容的 rubric score：完整程序（71.1）> 无注释代码 > 无技能调用代码 > 语言子目标，证明技能调用是关键贡献。

### Figure 8–17: 技能关键帧（YAM 机器人数据集）

- **Figure 8** — Skill I `fold_edge_over`（任务：用丝带包裹卷毯）  
  ![Figure 8 start](https://arxiv.org/html/2610.30136v1/figures/skills/fold_edge_over/f000354_start.jpg)

- **Figure 9** — Skill II `alternating_smoothing_pats`（任务：卷毛巾）  
  ![Figure 9 start](https://arxiv.org/html/2610.30136v1/figures/skills/alternating_smoothing_pats/f000130_start.jpg)

- **Figure 10** — Skill III `flip_object_over`（任务：折叠餐巾并放置餐具）  
  ![Figure 10 start](https://arxiv.org/html/2610.30136v1/figures/skills/flip_object_over/f004748_start.jpg)

- **Figure 11** — Skill IV `adjust_in_place`（任务：将书签插入书中）  
  ![Figure 11 start](https://arxiv.org/html/2610.30136v1/figures/skills/adjust_in_place/f000936_start.jpg)

- **Figure 12** — Skill I（Ego）`open_closure`（任务：组装相册）  
  ![Figure 12 start](https://arxiv.org/html/2610.30136v1/figures/skills/open_closure/f000000_start.jpg)

- **Figure 13** — Skill II（Ego）`pour_into_target`（任务：将米饭盛入碗中）  
  ![Figure 13 start](https://arxiv.org/html/2610.30136v1/figures/skills/pour_into_target/f000645_start.jpg)

- **Figure 14** — Skill III（Ego）`stir_contents`（任务：用搅拌棒混合溶液）  
  ![Figure 14 start](https://arxiv.org/html/2610.30136v1/figures/skills/stir_contents/f000758_start.jpg)

- **Figure 15** — Skill IV `wipe_surface_back_and_forth`（任务：擦拭桌面）  
  ![Figure 15 start](https://arxiv.org/html/2610.30136v1/figures/skills/wipe_surface_back_and_forth/f001028_start.jpg)

- **Figure 16** — Skill V `fold_over_and_crease`（任务：折叠衣物并压痕）  
  ![Figure 16 start](https://arxiv.org/html/2610.30136v1/figures/skills/fold_over_and_crease/f000529_start.jpg)

- **Figure 17** — Skill VI `wrap_around_target`（任务：缠绕目标物体）  
  ![Figure 17 start](https://arxiv.org/html/2610.30136v1/figures/skills/wrap_around_target/f000412_start.jpg)

### Figure 18: YAM 技能库嵌入空间可视化

![Figure 18](https://arxiv.org/html/2610.30136v1/skill-viz-yam.png)

**说明**: YAM 数据集 23 个技能在嵌入空间中的聚类分布，展示技能库的语义组织结构和技能间的相似性。

### Figure 19: 人类 Ego 技能库嵌入空间可视化

![Figure 19](https://arxiv.org/html/2610.30136v1/skill-viz-human.png)

**说明**: 人类 ego 数据集 27 个技能在嵌入空间中的聚类，跨日常任务类型分布，验证了[[技能库]]的泛化性。

### Figure 20: 真实机器人实验平台

| 平台 | 图片 |
|------|------|
| (a) YAM 双臂站台 | ![YAM setup](https://arxiv.org/html/2610.30136v1/hardware_setup_yam.png) |
| (b) Tianji-Sharpa 灵巧手站台 | ![Tianji setup](https://arxiv.org/html/2610.30136v1/hardware_setup_tianji_sharpa.png) |

**说明**: 两个真实世界实验平台。YAM 用于双臂操作 6 个任务，Tianji-Sharpa 用于灵巧手操作 2 个任务（排列水果/代金牌、解魔方）。

### Figure 21–28: 真实任务关键帧与代码标注

| 任务 | 图片 |
|------|------|
| Arrange Fruits | ![](https://arxiv.org/html/2610.30136v1/arrange_fruits.png) |
| Pack Backpack | ![](https://arxiv.org/html/2610.30136v1/pack_backpack.png) |
| Replace Lightbulb | ![](https://arxiv.org/html/2610.30136v1/replace_lightbulb.png) |
| Fold and Stack Towels | ![](https://arxiv.org/html/2610.30136v1/fold_and_stack_towels.png) |
| Arrange Fruits with Tongs | ![](https://arxiv.org/html/2610.30136v1/arrange_fruits_sharpa.png) |
| Deal Poker | ![](https://arxiv.org/html/2610.30136v1/deal_poker.png) |
| Flip Scoreboard | ![](https://arxiv.org/html/2610.30136v1/flip_scoreboard.png) |
| Solve Rubik's Cube | ![](https://arxiv.org/html/2610.30136v1/solve_rubiks_cube.png) |

**说明**: 每组展示任务关键帧序列和对应的结构化代码标注（技能调用 + 参数），展示了[[技能库]]中技能在不同任务上的复用。

### Figure 29–30: 预训练代码生成可视化

| 来源 | 图片 |
|------|------|
| YAM 验证样本 | ![](https://arxiv.org/html/2610.30136v1/pretrained_code_generation_yam.png) |
| Ego 人类验证（Squeeze toothpaste）| ![](https://arxiv.org/html/2610.30136v1/pretrained_code_generation_human.png) |

**说明**: 预训练后的代码生成与参考程序的对比，展示模型能正确调用技能库中的技能并给出合理参数。

### Figure 31–32: 在线代码生成（部署时）

| 任务 | 图片 |
|------|------|
| Pack Backpack（YAM）| ![](https://arxiv.org/html/2610.30136v1/posttrained_code_generation_yam.png) |
| Solve Rubik's Cube（Sharpa）| ![](https://arxiv.org/html/2610.30136v1/posttrained_code_generation_sharpa.png) |

**说明**: 真实部署时的在线代码生成示例，展示了模型在不同任务中动态生成并执行技能调用序列的能力。

### Figure 33–34: 未来帧预测质量对比

| 任务 | 图片 |
|------|------|
| Arrange Fruits | ![](https://arxiv.org/html/2610.30136v1/video_quality_comparison.png) |
| Deal Poker | ![](https://arxiv.org/html/2610.30136v1/video_quality_comparison_deal_poker.png) |

**说明**: AW-0 与两个 baseline（WLA、DreamZero）的未来帧预测质量对比，展示 AW-0 生成的视频帧更清晰准确，物体状态预测更精确。

### Figure 35: 真实机器人混合执行（Fold and Stack Towels）

![Figure 35](https://arxiv.org/html/2610.30136v1/real_hybrid_progress.png)

**说明**: 展示[[混合执行]]在 Fold and Stack Towels 任务中的切换逻辑，代码块（技能调用）和直接动作块交替使用，提升了复杂任务的成功率（90.6% vs 84.4%）。

### Figure 36: 失败案例分析

![Figure 36](https://arxiv.org/html/2610.30136v1/failure_analysis.png)

**说明**: 代表性失败案例分类，包括：视觉分布偏移导致的预测误差、程序参数错误（语法有效但执行错误）、以及接触密集场景中的动作精度问题。

### Figure 37: RoboTwin 2.0 混合执行可视化

| 任务 | 图片 |
|------|------|
| blocks_ranking_rgb | ![](https://arxiv.org/html/2610.30136v1/blocks_ranking_rgb_progress.png) |
| pick_dual_bottles | ![](https://arxiv.org/html/2610.30136v1/pick_dual_bottles_progress.png) |

**说明**: 仿真环境中[[混合执行]]的代码/动作步骤切换示例，随机视觉化环境下代码驱动占比更高（35.0% successful vs 5.6% overall）。

### Figure 38: 声明式 API（Grippers 和 Dexterous Hands）

*（图片路径未收录于可访问内容，仅限附录）*

**说明**: AW-0 代码标注所依赖的机器人操作 API 声明（平行夹爪和灵巧手两种形式），是技能库参数化的基础接口定义。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| YAM 双臂机器人数据 | ~2,000 小时 | 双臂操作任务，含精细接触 | 预训练（代码+视频+动作）|
| 人类 Ego 视频数据 | ~1,000 小时 / 248,642 集 | 第一人称视角，日常操作 | 预训练（跨形态泛化）|
| RoboTwin 2.0 | 50 个任务，Clean+Random | 仿真 benchmark，随机视觉化 | 评估 |
| 8 个真实任务 | 100 个示范/任务 | YAM + Tianji-Sharpa 灵巧手 | 下游微调 + 评估 |

### 实现细节

- **Backbone**: 代码专家：Qwen3-VL ViT（冻结）+ SDAR-1.7B；视频专家：Cosmos-Predict2.5-2B；动作专家：DiT（从头训练）
- **架构配置**: 28 层，2048 维注意力，64 维状态-动作空间，动作水平 32 步
- **下游微调**: 每任务 100 示范，20 epoch
- **推理硬件**: RTX 5090（0.23s/重规划，联合去噪）
- **预训练硬件**: H100 GPU

### Table 1: RoboTwin 2.0 成功率对比（每任务 50 次 rollout）

| 方法 | Clean (%) | Random (%) | 平均 (%) |
|------|-----------|------------|---------|
| [[π0.5]] | 54.9 | 27.7 | 41.3 |
| EgoScale | 26.2 | 2.9 | 14.5 |
| DreamZero | 33.6 | 17.6 | 25.6 |
| WLA | 64.8 | 3.6 | 34.2 |
| **AW-0** | **80.8** | 20.6 | **50.7** |
| AW-0 (hybrid) | 63.1 | **29.2** | 46.2 |

**说明**: AW-0 在 Clean 设置下最优（+16.0% vs WLA），但随机设置下混合执行更有优势。视觉随机化使代码路径更有价值（语义鲁棒性），但视频预测精度受分布偏移影响导致 Clean 设置下 hybrid 低于纯动作模式。

### Table 2: 真实机器人任务平均 Rubric Score（%）

| 方法 | YAM 6 任务 | Tianji-Sharpa 2 任务 | 平均 |
|------|-----------|----------------------|------|
| [[π0.5]] | — | — | 43.6 |
| EgoScale | — | — | 35.9 |
| DreamZero | — | — | 46.0 |
| WLA | — | — | 39.9 |
| **AW-0** | — | — | **71.1** |

**说明**: AW-0 平均超过最强 baseline DreamZero +25.1%，在需要长时序推理和语义规划的任务（Pack Backpack、Solve Rubik's Cube）上优势最为显著。

### Table 3: 消融实验——注意力顺序（Rubric Score）

| 注意力顺序 | 平均 Rubric Score (%) |
|-----------|----------------------|
| 仅视频专家（V-only） | — |
| V→C→A | — |
| **C→V→A（AW-0）** | **71.1** |

**说明**: C→V→A 顺序最优，代码作为语义先验引导视频和动作生成是关键设计。

### Table 4: 消融实验——代码条件化方式

| 方法 | 平均 Rubric Score (%) |
|------|----------------------|
| Wan2.2 VAE 特征 | 55.6 |
| **专用 ViT（AW-0）** | **71.1** |

**说明**: 专用 Qwen3-VL ViT 作为视觉编码器比 Wan2.2 VAE 特征提升 +15.5%，说明高质量视觉理解对代码生成至关重要。

### Table 5: 消融实验——生成范式

| 生成范式 | Rubric Score (%) | 推理延迟 (s) |
|---------|-----------------|-------------|
| 自回归生成 | 54.5 | 0.55 |
| **联合去噪（AW-0）** | **71.1** | **0.23** |

**说明**: 联合去噪既快（2.4× 加速）又好（+16.6%），自回归生成存在误差累积问题。

### Table 6: 消融实验——代码标注内容

| 代码内容 | Rubric Score (%) |
|---------|-----------------|
| 语言子目标 | — |
| 无技能调用代码 | — |
| 无注释代码 | — |
| **完整带技能调用代码** | **71.1** |

**说明**: 技能调用是关键，仅有自然语言子目标不能替代结构化程序；注释也对语义理解有正向贡献。

### Table 7（附录 Table 10）: 预训练代码专家验证指标

| 指标 | 值 |
|------|-----|
| Teacher-forced Loss | 0.535 |
| Teacher-forced Accuracy | 0.897 |
| Grounding Gap | 1.774 |
| Parse Rate | 0.875 |
| Entity F1 | 0.534 |
| Primitive F1 | 0.365 |
| Skill-call Rate | 0.292 |

**说明**: 预训练代码专家能以 89.7% 的 teacher-forced 精度和 87.5% 的解析率生成有效程序，但 skill-call rate 仅 29.2%，说明微调仍有提升空间。

### Table 8（附录 Table 12）: 混合执行步骤分布（RoboTwin 2.0）

| 场景 | 代码步比例 (%) | 直接动作步比例 (%) |
|------|--------------|-----------------|
| Clean, 全部 | 8.5 | 91.5 |
| Clean, 成功 | 14.5 | 85.5 |
| Clean, 失败 | 6.3 | 93.7 |
| Random, 全部 | 5.6 | 94.4 |
| Random, 成功 | **35.0** | 65.0 |
| Random, 失败 | 3.7 | 96.3 |

**说明**: 成功的 rollout 使用更多代码步（尤其随机化场景 35%），验证了代码路径在语义鲁棒性上的价值。失败案例主要依赖直接动作，说明代码路径的有效执行是成功的关键。

### 可视化结果

- 代码生成质量：预训练后已能生成语义一致的技能调用序列，微调后进一步精准
- 未来帧预测：AW-0 比 WLA/DreamZero 预测更清晰，尤其在接触密集场景
- 混合执行：在 Fold and Stack Towels 等复杂任务上，代码-动作切换显著提升完成度

---

## 批判性思考

### 优点

1. **统一框架**: 将规划（代码）、感知预测（视频）、执行（动作）统一在单一模型中，避免系统性误差累积
2. **技能可复用性**: 自动构建的技能库可跨任务复用，为持续学习和智能体扩展提供基础
3. **强大的实验验证**: 在仿真（50 任务）和真实硬件（YAM + 灵巧手）上均显著超过多个强 baseline
4. **自动标注可扩展**: 无需人工标注，数据管线可大规模扩展

### 局限性

1. **代码参数正确性**: 语法合法的代码不能保证执行正确，程序参数错误（如错误坐标）是主要失败原因之一
2. **视觉分布偏移**: 混合执行在随机视觉化下优势明显，但 Clean 设置下反而低于纯动作模式，说明视频预测对 OOD 仍然敏感
3. **在线学习缺失**: 目前不支持从交互中在线更新技能库或模型，限制了持续自主学习能力
4. **硬件局限**: 真实世界混合执行仅在 YAM 平台验证，灵巧手任务未做混合执行评估

### 潜在改进方向

1. **迭代程序修订**: 基于执行反馈自动修正代码参数
2. **技能库在线扩充**: 从新任务的成功轨迹中自动发现并添加新技能
3. **自主持续学习**: 机器人与环境交互积累经验，迭代更新 WAM 和技能库

### 可复现性评估

- [ ] 代码开源（项目主页尚无代码链接）
- [ ] 预训练模型（未公开）
- [x] 训练细节完整（附录详细描述）
- [ ] 数据集可获取（YAM 数据集未公开）

---

## 关联笔记

### 基于

- [[WAM|世界动作模型（WAM）]]: AW-0 在标准 WAM（视频-动作联合生成）基础上扩展了代码流
- [[Flow-Matching]]: 视频和动作生成的核心训练范式
- [[Mixture-of-Transformers]]: 多专家架构设计基础
- [[Code as Policies]]: 代码规划部分的核心前置工作

### 对比

- [[π0.5]]: 最强 VLA baseline，仅 action 生成，无世界模型
- DreamZero: 世界动作模型 baseline，无代码流
- WLA: 世界动作模型 baseline，有子任务指令但无结构化代码
- EgoScale: 以人类 ego 数据训练的策略

### 方法相关

- [[吸收掩码扩散]]: 代码生成的核心扩散方式
- [[技能库]]: 可复用参数化技能组织形式
- [[混合执行]]: 代码路径与动作块的动态切换机制
- [[Diffusion Transformer (DiT)|DiT]]: 视频和动作专家的基础架构
- [[Block Causal Attention]]: 跨专家信息流的注意力机制
- [[Agentic Robot]]: 智能体机器人领域背景

### 硬件/数据相关

- YAM（双臂机器人平台）
- Tianji-Sharpa（灵巧手机器人平台）

---

## 速查卡片

> [!summary] Agentic World Action Models (AW-0)
> - **核心**: 将代码生成嵌入 WAM，通过三流（代码+视频+动作）联合去噪实现规划-预测-执行统一
> - **方法**: Mixture-of-Transformers + 吸收掩码扩散代码生成 + 自动技能库标注管线 + 混合执行
> - **结果**: RoboTwin 2.0 Clean 80.8%（+16% vs WLA），真实机器人 71.1%（+25.1% vs DreamZero）
> - **代码**: [agentic-wam.github.io](https://agentic-wam.github.io)

---

*笔记创建时间: 2026-10-23*
