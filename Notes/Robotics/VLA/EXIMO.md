---
title: "EXIMO: VLM Guided Exploration of VLA Policies"
method_name: "EXIMO"
authors: [Bhavya Sukhija, Oliver Groth, Mohit Shridhar, Tim Hertweck, Michael Bloesch, Markus Wulfmeier, Abbas Abdolmaleki, Martin Riedmiller]
year: 2026
venue: arXiv
tags: [vla, reinforcement-learning, imitation-learning, vlm-planning, robot-manipulation, exploration, knowledge-distillation]
zotero_collection: Robotics/VLA
image_source: online
arxiv_html: https://arxiv.org/html/2608.19891v1
created: 2026-08-22
---

# 论文笔记：EXIMO: VLM Guided Exploration of VLA Policies

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Google DeepMind |
| 日期 | August 2026 |
| 项目主页 | 未公开 |
| 对比基线 | [[GROD]] |
| 链接 | [arXiv](https://arxiv.org/abs/2608.19891) / Code (未公开) |

---

## 一句话总结

> EXIMO 通过让 [[VLM]] 在探索阶段分解复杂任务来编排 [[GROD|VLA]] 采集成功轨迹，再将这些轨迹蒸馏回 VLA 并结合残差 RL 精炼，实现在 22 个操作任务上远超纯 BC 或直接 RL 微调的成功率。

---

## 核心贡献

1. **VLM 引导的探索（Explore）**: 利用 [[VLM]]（Gemini）作为高层规划器将复杂目标分解为 [[GROD|VLA]] 可执行的原子子任务，在不采集额外遥操作数据的情况下生成高质量演示轨迹
2. **知识蒸馏式模仿学习（Imitate）**: 将 VLM 编排数据蒸馏回 [[GROD|VLA]]，训练时用 VLM 中间目标监督、推理时仅用高层目标，消除 VLM 推理延迟并简化后续 RL
3. **残差 RL 精炼（Optimize）**: 基于蒸馏后 VLA 的较高基准成功率，训练一个轻量 [[Residual Policy|残差策略]] 做残差修正，用 [[MPO]] 算法实现高效在线 RL

---

## 问题背景

### 要解决的问题

预训练 [[VLA（视觉-语言-动作模型）|VLA]] 在训练分布内的原子技能上表现出色，但难以泛化到训练分布外的复杂组合目标（如多步骤序列、需要语义推理的指令）。适配这类任务有两条常见路径：收集更多遥操作数据（代价高昂）或直接做 RL 微调（对长序列任务极低效）。

### 现有方法的局限

- **[[Behavior Cloning|纯行为克隆 (BC)]]**: 收集遥操作数据成本高，且难以覆盖长序列组合任务
- **直接 RL 微调 VLA**: 大型 [[Diffusion Policy|扩散头]] VLA 难以做 on-policy 梯度更新；稀疏奖励下长序列任务探索困难，样本效率极低
- **运行时 VLM 编排（无蒸馏）**: 推理延迟高，每步都需要 VLM 查询；且评估时若移除 VLM 则性能大幅下降（训练/评估分布不匹配）

### 本文的动机

将 [[VLM]] 的语义世界知识用于**数据采集阶段**而非推理阶段：让 VLM 在探索时提供中间子目标，为 [[GROD|VLA]] "搭桥"，收集到更多成功样本后再蒸馏回 VLA。这样既规避了遥操作需求，又为后续 RL 提供了一个非平凡的初始化策略，从根本上缓解了 RL 的稀疏奖励探索难题。

---

## 方法详解

### 模型架构

EXIMO 是一个**三阶段算法**，而非单一网络架构。核心组件包括：

- **基础 VLA**: [[GROD]]（3B 参数，[[PaliGemma]] backbone + [[Diffusion Policy|扩散策略头]]，在 Aloha 遥操作数据上训练）
- **编排器**: [[VLM]]（Gemini），仅在 Explore 阶段使用
- **残差策略 $\pi^{ref}$**: 轻量网络，输出残差修正 $\Delta a$
- **RL 算法**: [[MPO]]（Maximum A Posteriori Policy Optimisation），off-policy

三阶段流程：

1. **Explore** → VLM 编排 + VLA 执行 → 收集成功轨迹到数据缓冲区 $\mathcal{D}$
2. **Imitate** → 在 $\mathcal{D}$ 上对 [[GROD]] SFT，输入高层目标 $g$（非中间子目标）
3. **Optimize** → 残差 [[MPO]] 在线 RL，在蒸馏后 VLA 基础上做精炼

### 核心模块

#### 模块 1: VLM 编排（Explore Phase）

**设计动机**: 利用 [[VLM]] 的语义知识将超出 VLA 分布的复杂目标分解为 VLA 可执行的原子子目标，无需采集任何额外遥操作数据。

**具体实现**:
- 每个时间步，Gemini 接收多视角图像（timestep 0, 50, 100, 150, 200）和高层目标 $g$
- VLM 在 `<think>` 块中推理当前机器人状态，在 `<answer>` 块中输出下一个原子指令（如 "pick up the blue plate with your left hand"）
- VLA 以该中间目标执行动作，VLM 以闭环方式持续更新指令
- 仅保留被 ground-truth 成功检测器确认的轨迹存入缓冲区 $\mathcal{D}$

**指令约束**（Figures 8-9）:
- 仅允许 "pick [object]" 和 "put [object] in [location]" 两类指令
- 需精确描述颜色、类型，不能使用模糊表述，一次只给一个动作

#### 模块 2: 知识蒸馏（Imitate Phase）

**设计动机**: 将 VLM 的组合推理能力内化到 VLA 参数中，消除部署时对 VLM 的依赖和推理延迟。

**具体实现**:
- 在 $(s, g_{original}, a)$ 三元组上对 [[GROD]] 继续 [[Behavior Cloning|行为克隆]] 训练
- **关键设计**: 训练输入是高层目标 $g$（而非 VLM 的中间子目标 $g_t$），迫使 VLA 自行内化分解逻辑
- 使用与 GROD 原始训练一致的 BC 目标（预测 [[Action Chunking|动作块]] $a_{0:K-1}$）

#### 模块 3: 残差 RL 精炼（Optimize Phase）

**设计动机**: [[GROD]] 的 [[Diffusion Policy|扩散策略头]] 难以直接做 RL 梯度更新；蒸馏后 VLA 成功率非平凡，为轻量 [[Residual Policy|残差策略]] 的 RL 训练提供了可行基础。

**具体实现**:
- [[Residual Policy|残差策略]] $\pi^{ref}$ 以扩展状态 $x = (s, a^{VLA})$ 为输入，输出残差修正 $\Delta a$
- 最终执行动作为 $a = a^{VLA} + \Delta a$
- 奖励为稀疏成功信号 $r_t = \mathbf{1}_{s_t \in \text{Success}(g)}$
- 使用 [[MPO]] 算法进行 off-policy 在线训练

---

## 关键公式

### 公式 1: [[VLA（视觉-语言-动作模型）|VLA 策略]]

$$
a \sim \pi^{VLA}(\cdot \mid s, g)
$$

**含义**: VLA 策略以当前状态 $s$ 和语言目标 $g$ 为条件采样动作 $a$。

**符号说明**:
- $s \in \mathcal{S}$: 环境观测状态（多视角图像）
- $g \in \mathcal{G} \subseteq \mathcal{T}^L$: 语言目标（token 序列）
- $a \in \mathcal{A}$: 输出动作（[[Action Chunking|动作块]]）

### 公式 2: [[VLM]] 编排目标生成

$$
g_t \sim \pi^{VLM}(\cdot \mid s_{\leq t},\, g)
$$

**含义**: VLM 编排器在每个时间步 $t$，根据历史状态序列 $s_{\leq t}$ 和高层目标 $g$ 生成中间子目标 $g_t$。

**符号说明**:
- $g_t$: 当前时间步的中间子目标（自然语言短语，如 "pick up blue plate"）
- $s_{\leq t}$: 历史状态序列（多时间步图像）
- $g$: 原始高层任务目标

### 公式 3: [[Residual Policy|残差策略]] 动作合成

$$
a = a^{VLA} + \Delta a
$$

**含义**: 最终执行动作由 VLA 的基准动作加上残差策略的修正量组成。

**符号说明**:
- $a^{VLA}$: [[GROD|VLA]] 蒸馏策略输出的基准动作
- $\Delta a$: 残差策略 $\pi^{ref}$ 输出的修正量
- $a$: 实际执行的最终动作

### 公式 4: [[Residual Policy|残差 MDP]] 状态定义

$$
x = (s,\, a^{VLA})
$$

**含义**: 残差策略的状态空间将环境状态与 VLA 输出的动作拼接，使残差策略能感知 VLA 当前意图。

### 公式 5: [[MPO|RL 优化目标]]

$$
\max_{\pi^{ref} \in \Pi}\; \mathbb{E}_{\pi^{ref},\, s_0 \sim \rho}\!\left[\sum_{t=0}^{\infty} \gamma^t \cdot \mathbf{1}_{s_t \in \text{Success}(g)}\right] = J(\pi^{ref},\, \rho)
$$

**含义**: 在残差策略类 $\Pi$ 中找到使累积折扣稀疏奖励最大的策略，奖励仅在状态进入成功集合时为 1。

**符号说明**:
- $\pi^{ref}$: 待优化的残差策略
- $\rho$: 初始状态分布
- $\gamma$: 折扣因子
- $\mathbf{1}_{s_t \in \text{Success}(g)}$: 稀疏成功指示函数
- $\text{Success}(g) \subset \mathcal{S}$: 任务 $g$ 对应的成功状态集合

---

## 关键图表

### Figure 1: VLM 编排交互示例

![Figure 1](https://arxiv.org/html/2608.19891v1/x1.png)

**说明**: Explore 阶段 VLM 的实际交互流程。VLM 接收序列图像和任务描述（"put the plate, bowl on the rack"），在 `<think>` 块中推理当前状态，在 `<answer>` 块中生成原子指令（如 "pick up the blue plate with your left hand"），VLA 执行后 VLM 继续更新指令。

### Figure 2: Explore 阶段三方法对比

![Figure 2](https://arxiv.org/html/2608.19891v1/x2.png)

**说明**: 22 个任务上的成功率（上）、成功耗时（中）、Episode 长度（下）对比。三条件：无编排的 base VLA、运行时 VLM 编排（VLA + VLM）、VLM 编排数据蒸馏后的 VLA（EXIMO SFT）。蒸馏后 VLA 在成功率和效率上均优于运行时 VLM 编排，验证了知识蒸馏的有效性。

### Figure 3: 在线 RL 阶段样本效率对比

![Figure 3](https://arxiv.org/html/2608.19891v1/x3.png)

**说明**: 20 个任务上 5 个随机种子的 RL 训练曲线（成功率与成功耗时）。GROD + SFT 初始值更高，收敛到更好的最终性能，而 base GROD 即使获得更多 environment steps 也无法追上，印证了 SFT 预热对 RL 可行性的关键作用。

### Figure 4: RL 精炼后最终性能

![Figure 4](https://arxiv.org/html/2608.19891v1/x4.png)

**说明**: 柱状图对比 SFT alone、SFT + RL（EXIMO 完整）和 base GROD + RL 的最终成功率与成功耗时。EXIMO（SFT + RL）在所有任务上一致领先，组合两个阶段比任意单一阶段都强。

### Figure 5: 自由指令 vs 限制指令对比

![Figure 5](https://arxiv.org/html/2608.19891v1/x5.png)

**说明**: 5 个任务上，VLM 生成自由格式指令与限制为 pick-and-place 格式指令的性能对比。两者性能相近，说明 [[GROD]] 对两种格式都有较好的鲁棒性，验证了指令格式约束不是关键瓶颈。

### Figure 6: VLM 直接蒸馏到残差策略的消融

![Figure 6](https://arxiv.org/html/2608.19891v1/x6.png)

**说明**: 尝试将 VLM 知识直接蒸馏到 [[Residual Policy|残差策略]]（离线 + 在线 RL）而非先蒸馏到 VLA。左图：离线 RL 随数据量提升有改善；右图：在线 RL 阶段性能下滑，归因于分布偏移——残差策略训练分布（VLM 编排）与评估分布（无 VLM）不匹配。

### Figure 7: RL 训练期间使用 VLM 编排的消融

![Figure 7](https://arxiv.org/html/2608.19891v1/x7.png)

**说明**: 在 RL 训练中以不同概率 $p$ 使用 VLM 编排（$p=0$ 为纯 RL，$p=1$ 为全程编排）。上行：评估性能（无 VLM）随 $p$ 增大而下降；下行：数据采集性能随 $p$ 增大而提升。结论：训练时使用 VLM 编排会造成训练/评估分布不匹配，纯 RL（$p=0$）在评估时最优。

### Figure 8-9: VLM 编排 Prompt 模板

![Figure 8](https://arxiv.org/html/2608.19891v1/x8.png)
![Figure 9](https://arxiv.org/html/2608.19891v1/x9.png)

**说明**: 完整的 VLM Prompt 结构，包括系统角色定义（"expert robot programmer"）、任务描述插槽、多视角图像观测、逐步推理要求（`<thinking>` 块）、以及详细的指令生成规范（颜色+类型描述、禁止模糊词、每次仅一个动作、禁止连续动作等）。

### Table 1: 22 个操作任务套件

| ID | 任务名 | 自然语言目标 | 类型 |
|----|--------|-------------|------|
| T2 | BowlGlassOnRack | put the bowl and glass on the rack | 技能链接 |
| T3 | BananaInBowl-Reasoning0 | put the item that a monkey can eat into the bowl | 语义推理 |
| T4 | MugOnPlate | put the mug on the plate | 技能链接 |
| T5 | MugOnPlate-Reasoning0 | put the object you pour coffee in on the plate | 语义推理 |
| T6 | MugOnPlate-Reasoning1 | put the object with a handle on top of the flat object | 属性推理 |
| T7 | PenInContainer | put the pen into the white container | 技能链接 |
| T8 | PenInContainer-Reasoning0 | put the object you use to write into the white container | 语义推理 |
| T9 | PenInContainer-Reasoning1 | put the thinnest object into the white container | 属性推理 |
| T10 | CanOpenerInCaddy-Left-Reasoning0 | place the can opener in the left compartment of the caddy | 空间理解 |
| T11 | CanOpenerInCaddy-Right-Reasoning0 | place the can opener in the right compartment of the caddy | 空间理解 |
| T12 | MagnifierCanOpenerInCaddy | put the magnifier and can opener in the caddy | 多物体链 |
| T13 | MagnifierInCaddy-Left-Reasoning0 | place the magnifier in the left compartment of the caddy | 空间理解 |
| T14 | MagnifierInCaddy-Right-Reasoning0 | place the magnifier in the right compartment of the caddy | 空间理解 |
| T15 | ScissorsInCaddy-Left-Reasoning0 | place the scissors in the left compartment of the caddy | 空间理解 |
| T16 | ScissorsInCaddy-Right-Reasoning0 | place the scissors in the right compartment of the caddy | 空间理解 |
| T17 | ScissorsMagnifierInCaddy | put the scissors and magnifier in the caddy | 多物体链 |
| T18 | ScissorsScrewdriverInCaddy | put the scissors and screwdriver in the caddy | 多物体链 |
| T19 | ScrewdriverInCaddy-Left-Reasoning0 | place the screwdriver in the left compartment of the caddy | 空间理解 |
| T20 | ScrewdriverInCaddy-Right-Reasoning0 | place the screwdriver in the right compartment of the caddy | 空间理解 |
| T21 | ScrewdriverMagnifierInCaddy | put the screwdriver and magnifier in the caddy | 多物体链 |
| T22 | PlateBowlOnRack | put the plate and bowl on the rack | 技能链接 |
| T23 | PlateGlassOnRack | put the plate and glass on the rack | 技能链接 |

**说明**: 任务覆盖四类挑战：(i) 需要技能链接的餐具放置；(ii) 需要语义推理的目标识别（"monkey can eat"）；(iii) 需要空间理解的分隔间定位（left/right compartment）；(iv) 多物体顺序放置。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| GROD 预训练数据 | 大规模 Aloha 遥操作 | 仿真 Aloha 双臂原子技能 | 预训练 VLA |
| EXIMO 探索数据 | VLM 编排采集（过滤后成功轨迹） | 无人工遥操作，自动过滤 | SFT 微调 |

### 实现细节

- **Base VLA**: [[GROD]]（Gemini Robotics On-Device），3B 参数，[[PaliGemma]] VLM backbone + [[Diffusion Policy|扩散策略头]]
- **编排 VLM**: Gemini（最新版本）
- **机器人平台**: ALOHA 双臂操作机器人（仿真）
- **RL 算法**: [[MPO]]（Maximum A Posteriori Policy Optimisation），off-policy
- **奖励函数**: 稀疏成功信号（ground-truth 成功检测器）
- **评估规模**: 每个 method-task 组合 1000 个 episode
- **RL 实验**: 5 个随机种子，报告均值 ± 2标准误差

### 实验结论

**Explore 阶段**（Figure 2）:
- VLM 编排将 base VLA 成功率大幅提升，尤其对长序列任务（PlateBowlOnRack）和推理任务
- 蒸馏后 VLA 进一步超越运行时 VLM 编排，同时消除 VLM 推理延迟

**RL 阶段**（Figures 3-4）:
- GROD + SFT 比 base GROD 收敛更快、最终性能更高
- 完整 EXIMO（SFT + RL）在所有任务上一致优于 SFT alone 和 base GROD + RL

**消融实验**（Figures 5-7）:
- 指令格式（自由 vs 限制）对 [[GROD]] 影响不大（Figure 5）
- 直接将 VLM 蒸馏到残差策略在在线 RL 阶段失败，确认需要先蒸馏到 VLA（Figure 6）
- RL 训练期间加入 VLM 编排会造成评估分布不匹配，$p=0$ 时评估性能最优（Figure 7）

---

## 批判性思考

### 优点

1. **实用性强**: 三阶段设计直接且可解释，每个阶段对应清晰的工程决策，无需复杂的端到端联合训练
2. **零遥操作**: 不依赖人工遥操作采集新任务数据，仅利用现有 VLA 和 VLM 即可适配新任务
3. **充分消融**: 三组消融实验（Figures 5-7）逐一验证了关键设计决策，增强说服力
4. **任务多样性**: 22 个任务覆盖技能链接、语义推理、空间理解三大类，评估较为全面

### 局限性

1. **仿真局限**: 所有实验在仿真环境下进行，实物机器人泛化性未验证
2. **依赖 GT 成功检测器**: 探索阶段的成功过滤依赖环境提供的 ground-truth 检测，现实中难以获得
3. **依赖 VLA 已掌握原子技能**: 若目标任务涉及 VLA 从未见过的操作动作，VLM 编排也无法奏效
4. **VLM 查询成本**: 探索阶段 Gemini 查询成本较高（每 50 步一次），规模化时可能成为瓶颈

### 潜在改进方向

1. **VLM 替代成功检测器**: 用 VLM 本身评估任务完成度，实现全自动学习闭环
2. **VLM 驱动环境重置**: 用 VLM 编排 "undo" 动作序列来重置环境，减少人工干预
3. **On-policy 蒸馏**: 探索类自蒸馏框架延伸，进一步提升 Imitate 阶段的数据效率

### 可复现性评估

- [ ] 代码开源（未发布）
- [ ] 预训练模型（GROD 未公开）
- [ ] 训练细节完整（较充分，消融细节详细）
- [ ] 数据集可获取（仿真环境未公开）

---

## 关联笔记

### 基于

- [[GROD]]: 基础 VLA 模型（Gemini Robotics On-Device，3B）
- [[PaliGemma]]: GROD 的 VLM backbone
- [[Diffusion Policy]]: GROD 的动作预测头架构

### 对比

- [[GROD]]: 无 VLM 编排的基线 VLA
- [[Residual Policy]]: RL 精炼中采用的策略范式
- [[MPO]]: 残差 RL 使用的具体算法

### 方法相关

- [[VLM]]: 编排器核心组件（Gemini）
- [[Action Chunking]]: VLA 输出动作块格式
- [[Behavior Cloning]]: Imitate 阶段的训练目标
- [[MPO]]: Optimize 阶段使用的 RL 算法

### 硬件/数据相关

- [[ALOHA]]: 双臂操作机器人平台（仿真）

---

## 速查卡片

> [!summary] EXIMO: VLM Guided Exploration of VLA Policies
> - **核心**: VLM 编排探索 → SFT 蒸馏 → 残差 RL 精炼
> - **方法**: Gemini 分解任务 + GROD 执行 + MPO 在线优化
> - **结果**: 22 任务上成功率和样本效率大幅超越纯 BC 或直接 RL 微调
> - **代码**: 未公开

---

*笔记创建时间: 2026-08-22*
