---
title: "pico-WAM: Evaluating Counterfactual Prediction and Planning Error in a Compact World Action Model"
method_name: "pico-WAM"
authors: [Keivalya Pandya]
year: 2026
venue: arXiv
tags: [world-action-model, model-predictive-control, counterfactual-learning, imitation-learning, push-t, manipulation]
zotero_collection: Robotics/World Model
image_source: local
arxiv_html: https://arxiv.org/html/2611.01039v1
created: 2026-11-04
---

# 论文笔记：pico-WAM: Evaluating Counterfactual Prediction and Planning Error in a Compact World Action Model

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Khoury College of Computer Sciences, Northeastern University |
| 日期 | October 2026 |
| 项目主页 | [GitHub](https://github.com/keivalya/pico-wam) |
| 对比基线 | [[Behavior Cloning]], [[Action Diffusion\|Diffusion Policy]] |
| 链接 | [arXiv](https://arxiv.org/abs/2611.01039) / [Code](https://github.com/keivalya/pico-wam) |

---

## 一句话总结

> pico-WAM 是一个紧凑的[[世界动作模型（WAM）|World Action Model]]，通过反事实分支训练和受控评分干预，揭示了轨迹打分（而非提案生成）才是当前WAM规划的主要瓶颈。

---

## 核心贡献

1. **诊断框架**: 提出将规划失败拆分为"提案生成"和"轨迹打分"两个独立瓶颈，并设计交叉因子实验（LL/LE/EL/EE）加以分离
2. **[[Counterfactual Branch Training|反事实分支训练]]**: 从每第四个状态出发、加扰动生成的反事实分支数据，使短时序代价排名的rank相关从0.49提升至0.80，并将 restart-over-warm 搜索的覆盖率优势扩大4.8个百分点
3. **打分为主要干预目标**: 用模拟器真实代价替换学习代价（保留学习提案），覆盖率提升12.2个百分点；仅替换提案生成无可检测增益，从而定位打分为有效改进目标

---

## 问题背景

### 要解决的问题

当 [[世界动作模型（WAM）|World Action Model]] 在控制任务上失败时，失败来自哪里？是**提案生成**（policy提出的候选动作序列质量差），还是**轨迹打分**（模型对候选序列的预测代价排序不准确）？

### 现有方法的局限

- [[PETS]] / [[MBRL]] 通过不确定性感知搜索和受限模型展开来处理模型误差，但仍无法分离两个失败源
- 世界模型仅在专家演示动作上训练，推理时必须评估模型从未见过的反事实候选动作（off-manifold），导致外推误差
- [[MBPO]] 等方法向模型回放中加入模拟器分支，但机器人操控场景中没有现成的可微分物理引擎

### 本文的动机

用一个**可溯源**的紧凑模型（单共享表示同时支持policy、动力学、状态读出），在严格匹配训练条件下，分别用真实模拟器代价/状态替换学习部分，以手术式干预定位瓶颈。

---

## 方法详解

### 模型架构

pico-WAM 采用**共享隐变量**架构：

- **输入**: 64×64 图像 + 固定语言指令 + 8-D 物理状态（agent位置/速度 + 方块位置/朝向 $\cos\theta, \sin\theta$）
- **视觉编码**: 3层 CNN + 16个 [[Spatial Softmax]] 关键点
- **语言编码**: Embedding + [[GRU]]
- **状态编码**: 8-D 状态 MLP
- **融合**: Fusion MLP + LayerNorm → 128-D 共享隐变量 $z_t$
- **Policy 头**: $3 \times 512$ MLP → 预测 8 个2-D [[Action Chunking|动作块]]
- **状态读出 R**: 线性映射 $128 \to 8$（标准化物理状态）
- **动力学模型**: 残差 MLP（见 Eq. 1）

图像解码器**禁用**，物理状态在推理时显式提供，使模型聚焦于policy–dynamics交互，而非视觉感知或语言泛化。

### 核心模块

#### 模块1：共享隐变量表示

**设计动机**: 单一 $z_t$ 同时为 policy、动力学、状态读出服务，使每步决策完全可溯源（观测 → 编码 → 提案 → 想象后果 → 打分 → 执行）。

**具体实现**:
- 图像、语言、物理状态三路并行编码后通过 Fusion MLP 融合为 $z_t \in \mathbb{R}^{128}$
- 预测梯度流向动力学和读出头；未来隐变量目标施加 **stop-gradient**，作为隐一致性损失的目标
- 推理时：policy 以 imagined latents 展开 12 步形成初始提案

#### 模块2：反事实分支训练（Counterfactual Branches）

**设计动机**: 演示数据仅覆盖专家动作；规划器必须对未见过的候选动作打分。

**具体实现**:
- 从每第四个录制状态出发，对8个演示动作加 Gaussian 扰动（SD=0.3，截断至 $[-1,1]^2$）
- 在真实模拟器中执行扰动动作，获得真实状态转移作为预测监督目标
- 生成 6,013 个分支、48,104 个转移
- 分支**只提供预测目标**，不提供专家动作标签（不影响 BC 损失）

#### 模块3：规划循环（Planning Loop）

基于 [[MPC]] 的单步滚动规划：

- Policy 展开12步生成初始提案
- 256个候选：1个 policy 提案 + 1个搜索中心 + 254个噪声扰动（SD=0.1，截断至 $[-1,1]^2$）
- 两种搜索初始化：
  - **Restart**：每步以当前 policy 提案为搜索中心（重新启动）
  - **Warm**：将上一步最优计划前移一步（热启动）
- 只执行代价最低计划的第一个动作，然后重新规划

---

## 关键公式

### 公式1：[[世界动作模型（WAM）|残差动力学模型]]（Eq. 1）

$$
\hat{z}_{t+1} = f_\theta(z_t, a_t) = \text{LN}\!\left(z_t + \text{MLP}[z_t, a_t]\right)
$$

**含义**: 以残差方式在当前隐变量上施加动作条件增量，LayerNorm 稳定训练

**符号说明**:
- $z_t \in \mathbb{R}^{128}$：当前时刻的共享隐变量
- $a_t \in \mathbb{R}^2$：当前动作（2-D 相对位移）
- $\text{MLP}[\cdot]$：两个 256 单元隐层的全连接网络
- $\text{LN}(\cdot)$：Layer Normalization

### 公式2：[[Behavior Cloning|训练总损失]]（Eq. 2）

$$
\mathcal{L} = \mathcal{L}_\text{BC} + 0.5\!\left(\mathcal{L}_{s,0} + \mathcal{L}_{s,1:H}\right) + \mathcal{L}_z
$$

**含义**: 四项损失联合优化，使共享表示同时支持行为克隆、当前/未来状态预测和隐一致性

**符号说明**:
- $\mathcal{L}_\text{BC}$：行为克隆损失，对8个动作块回归专家标签（均方误差）
- $\mathcal{L}_{s,0}$：当前时刻状态重建损失
- $\mathcal{L}_{s,1:H}$：未来状态预测损失（水平 $H=5$）
- $\mathcal{L}_z$：隐一致性损失（以 stop-gradient 的真实未来编码为目标）

### 公式3：[[MPC|规划代价函数]]（Eq. 3）

$$
J(\hat{s}_{1:L}) = \frac{1}{L}\sum_{k=1}^{L}\!\left(w_k\, d_T(\hat{s}_k,\, g) + 0.05\, d_A(\hat{s}_k)\right)
$$

**含义**: 对预测状态序列的几何代价求时序加权均值；末尾步权重更高，以鼓励将方块推向目标

**符号说明**:
- $L=12$：规划水平
- $w_k$：时间步权重，从 0.5 线性增至 1.5
- $d_T(\hat{s}_k, g)$：预测方块角点到目标角点的平均像素距离
- $d_A(\hat{s}_k)$：agent 到方块的距离减去15px半径（$\max(\|p_A - c_T\|_2 - 15, 0)$）
- $g$：目标位置

### 公式4：[[Counterfactual Branch Training|主要对比指标]]（Eq. 4）

$$
\Delta = \left(C_\text{restart}^b - C_\text{warm}^b\right) - \left(C_\text{restart}^d - C_\text{warm}^d\right)
$$

**含义**: 分支训练是否扩大了 restart 相对于 warm 搜索的覆盖率优势（双重差分估计量）

**符号说明**:
- $C$：覆盖率（mean maximum coverage, %）
- 上标 $b$：有反事实分支的训练条件
- 上标 $d$：仅用演示数据的训练条件

---

## 关键图表

### Figure 1: Comparing Candidate Choices / 候选选择与后果对比

![[picoWAM_fig1_candidates.png]]

**说明**: 从相同初始状态出发，演示训练模型（选择遗憾2.82）与分支训练模型（选择遗憾1.37）各从256个候选中选取12步计划。线框显示预测终点与模拟器真实终点。两个模型选出的计划均未移动方块，说明低代价可能仅反映agent靠近方块而非完成任务。

### Figure 2: pico-WAM Architecture / 模型架构

![[picoWAM_fig2_architecture.png]]

**说明**: 三个面板展示 pico-WAM 完整流程。(a) 编码路径：图像经 [[Spatial Softmax]] CNN、语言经 [[GRU]]、物理状态经MLP，三路融合为共享 $z_t$；(b) 动力学：动作条件残差MLP推进隐变量，读出头预测物理状态；(c) 规划：policy给出提案，动力学想象后果，几何代价打分，执行最优计划首步后重新编码。

### Figure 3: Branches Help Restart Search / 分支训练放大Restart优势

![[picoWAM_fig3_branches.png]]

**说明**: (a) 15个训练种子的覆盖率曲线（从warm到restart的连线斜率）：分支训练（橙）在restart搜索下明显更高；(b) 配对覆盖率差分：数据×搜索交互（主效应）集中在正值区间，均值+4.8个百分点，15个种子中12个为正。

### Figure 4: Prediction, Ranking, and Control Diagnostics / 预测-排名-控制的不同误差模式

![[picoWAM_fig4_diagnostics.png]]

**说明**: (a) 随机动作下的方块 nMSE：演示模型超过零动作基准（即预测"不动"还更准），分支训练降低约三倍；(b) [[Selection Regret|选择遗憾]]：分支训练从7.56降至2.77，提升所有15个种子；(c) 预测进度 vs. 真实进度：分支训练的 restart 预测10.9px进展但实际仅1.2px，系统性乐观。

### Figure 5: Factorial Scoring Study / 因子交叉打分研究

![[picoWAM_fig5_factorial.png]]

**说明**: 2×2 因子实验（提案 L/E × 打分 L/E）。(a) 覆盖率均值矩阵：LL→LE 提升12.2pp（主效应），LL→EL无可检测增益，表明**打分而非提案生成是有效干预目标**；(b) 配对差值：改变打分（蓝）效果显著，改变提案（橙）置信区间跨零。

### Figure 6: Imitation Baselines / 与模仿学习基线的比较

![[picoWAM_fig6_imitation.png]]

**说明**: pico-WAM restart 搜索 vs. BC、Tuned BC、[[Action Diffusion|Diffusion Policy]]（5个种子）。左图覆盖率差无显著优势；右图成功率：vs. BC +9.5pp，vs. Diffusion Policy **−16.7pp**，说明更好的排名不能直接转化为更好的闭环控制。

### Figure 7: Door Opening Generalization / 门开启任务泛化

![[picoWAM_fig7_door.png]]

**说明**: [[Meta-World]] door-open-v3 上的独立研究（使用纯状态空间动力学模型，非 pico-WAM 架构）。(a) 混合数据（demo+noisy expert）将选择遗憾从 $7.15\times10^{-4}$ 降至 $4.23\times10^{-4}$，所有10个种子一致；(b) 规划成功率 mixed 88.2% > demo-only 78.6%，但两者均低于反应式BC基线98.6%。

### Figure 8: Recorded Demonstrations Below 95% Threshold / 演示覆盖率直方图

![[picoWAM_fig8_demos.png]]

**说明**: 206个 LeRobot [[PushT]] 演示的最高录制覆盖率直方图（均值84.8%），无一超过95%成功阈值。说明控制成功需要超越录制演示的表现，所有成功案例均为对演示的真实超越。

### Table 1: Matched Study — Coverage and Success（15 seeds, 200 starts）

| Training | Control | Coverage, % | Success, % |
|----------|---------|-------------|------------|
| BC | Reactive | 78.3 ± 3.3 | 23.3 |
| Current, demos | Reactive | 72.9 ± 4.2 | 22.2 |
| Current, + branches | Reactive | 71.1 ± 3.4 | 21.6 |
| WAM, demos | Reactive | 72.1 ± 2.6 | 22.6 |
| WAM, demos | Warm | 21.3 ± 2.9 | 0.0 |
| WAM, demos | Restart | 43.8 ± 4.1 | 1.8 |
| WAM, + branches | Reactive | 71.4 ± 2.9 | 20.4 |
| WAM, + branches | Warm | 51.1 ± 4.4 | 5.6 |
| **WAM, + branches** | **Restart** | **78.4 ± 4.1** | **32.8** |

**关键发现**: 无分支时，两种搜索均劣于同检查点的反应式 policy；加入分支后 restart 达到与BC相当的覆盖率并显著提升成功率。

### Table 2: Prediction and Ranking Diagnostics（means over 15 seeds）

| 指标 | Demos only | + Branches | Difference |
|------|-----------|------------|------------|
| Block RMSE, random, 10 steps (test demos, px) | 14.8 | 7.1 | −7.8 [−9.3, −6.2] |
| Block RMSE, random, 10 steps (fresh episodes, px) | 17.3 | 10.6 | −6.7 [−8.5, −4.9] |
| Rank correlation with true cost | 0.49 | 0.80 | +0.32 [0.30, 0.34] |
| Overlap of top 32 | 0.40 | 0.65 | +0.26 [0.24, 0.27] |
| Selection regret | 7.56 | 2.77 | −4.79 [−5.11, −4.47] |
| Selection optimism | 5.93 | 2.27 | −3.66 [−4.72, −2.60] |

**关键发现**: 演示模型在随机动作下误差超过零动作基准（预测能力倒退），分支训练修复此问题并大幅改善排名；但计划仍系统性高估实际进展。

### Table 3: Architecture Ablations（difference from full model, 5 seeds）

| Variant | Reactive | Restart |
|---------|----------|---------|
| Auxiliary gradients blocked | +5.1 [−5.1, 15.3] | **−44.3** [−50.2, −38.4] |
| Image only | −31.5 | −41.6 |
| State only | +0.7 | −3.3 |
| No instruction | +1.5 | −4.1 |
| No future-state loss | +0.4 | −6.4 |
| No latent loss | +0.5 | −5.2 |
| **Full model** | **71.6%** | **81.7%** |

**关键发现**: 阻断预测梯度进入编码器对规划的损害（−44.3pp）远超反应式 policy；去除物理状态输入同样大幅损害双模式性能；其余辅助损失无可检测效果。

### Table 4: Imitation Baselines and Dynamics Study

| Study, seeds | Reference | Coverage (pp) | Success (pp) |
|--------------|-----------|---------------|--------------|
| Matched, 15 | BC | 0.1 [−3.1, 3.4] | **9.5** [3.4, 15.6] |
| Horizon-matched, 15 | BC | 0.2 [−4.0, 4.4] | **10.0** [4.7, 15.3] |
| Original gym-pusht, 15 | BC | −1.4 [−4.6, 1.7] | 8.1 [1.7, 14.5] |
| Tuned BC, 15 | tuned BC | 2.7 [−0.6, 6.0] | −0.1 [−7.3, 7.1] |
| **Diffusion Policy, 5** | DP | −1.0 [−7.6, 5.5] | **−16.7** [−27.9, −5.5] |
| State dynamics, 15 | latent | −1.9 [−4.7, 1.0] | 12.0 [5.7, 18.3] |
| **Oracle dynamics, 15** | latent | **13.1** [10.6, 15.5] | **45.3** [38.5, 52.0] |

**关键发现**: restart 搜索覆盖率未超过 Diffusion Policy 且成功率落后16.7pp；但Oracle动力学（真实模拟器）使覆盖率+13.1、成功率+45.3，说明动力学质量是关键上限。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| [[PushT]] (LeRobot) | 206 demos, 25,444 transitions @ 10Hz | 无一演示超过95%成功阈值 | 训练/评估 |
| [[Meta-World]] door-open-v3 | 80 scripted + 80 noisy-expert episodes | 39-D 状态, 4-D 动作 | 门开启迁移研究 |

### 实现细节

- **编码器**: 3-层 CNN (16 [[Spatial Softmax]] 关键点) + [[GRU]] + 状态MLP → 128-D 共享隐变量
- **Policy**: 3×512 MLP，预测8个2-D 动作
- **动力学**: 2×256 残差 MLP + LayerNorm
- **状态读出**: 线性 128→8
- **优化器**: Adam, 30,000 updates, batch size 64
- **训练种子**: 15个（共享初始参数、动作 minibatch、初始起点）
- **硬件**: V100-SXM2, ~21.3 GPU-hours（主研究）
- **分支生成**: 6,013个分支，48,104个转移（每第四帧 × SD=0.3扰动）
- **仿真器**: [[MuJoCo]] (Push-T), Meta-World v3.0.0 (门开启)
- **规划**: 256候选 × 12步展开，13ms/步（含诊断回放）

---

## 批判性思考

### 优点

1. **方法论严格**: 严格匹配训练条件（相同初始参数、动作批次、更新次数），手术式因子实验，避免混淆
2. **诊断工具集完整**: 分离预测误差（nMSE）、候选排名（rank相关、overlap、regret）和控制覆盖率，提供可复现的诊断协议
3. **诚实呈现局限**: 明确指出pico-WAM并非新型架构，不声称硬件迁移或一般性优越；Oracle实验清晰展示动力学质量上限

### 局限性

1. **任务和规模受限**: 仅在[[PushT]]（2-D桌面推块）上进行主研究；物理状态显式提供，回避了视觉感知挑战
2. **计划仍系统性乐观**: 分支训练 restart 预测10.9px进展实际仅1.2px，根本原因未解决
3. **控制性能未超越Diffusion Policy**: 覆盖率持平但成功率低16.7pp，说明更好的排名≠更好的控制

### 潜在改进方向

1. 探索更有效的分支生成策略（如主动学习选择扰动方向，而非随机Gaussian噪声）
2. 解决计划乐观性：改进短时序代价与真实进展的相关性，或引入不确定性估计
3. 扩展到视觉输入（移除显式状态供给），验证方法在更真实设置下的可迁移性

### 可复现性评估

- [x] 代码开源（GitHub: keivalya/pico-wam）
- [x] 训练细节完整（30K steps, batch=64, 分支生成协议均公开）
- [x] 数据集可获取（LeRobot Push-T 公开）
- [ ] 预训练模型（需运行 reproduce.py 重新训练）

---

## 关联笔记

### 基于

- [[PETS]]: 不确定性感知的模型预测控制，处理模型误差的早期基准
- [[MBPO]]: 向模型回放中加入模拟器分支，pico-WAM 的反事实分支有类似动机
- [[Behavior Cloning]]: pico-WAM 包含BC损失，同时作为主要对比基线
- [[世界动作模型（WAM）|World Action Model]]: pico-WAM 是WAM的诊断实例化

### 对比

- [[Action Diffusion|Diffusion Policy]]: 成功率高出16.7pp，说明更好的规划排名不等同于更好的控制
- [[MBRL]]: pico-WAM 属于基于模型的RL范畴但侧重诊断而非性能SOTA

### 方法相关

- [[Counterfactual Branch Training]]: 核心训练技术，扩充off-manifold预测监督
- [[Selection Regret]]: 核心评估指标，衡量计划选择质量
- [[MPC]]: 规划循环基础框架（单步滚动重规划）
- [[Action Chunking]]: policy预测8步动作块
- [[GRU]]: 语言编码器
- [[Spatial Softmax]]: 视觉特征提取

### 硬件/数据相关

- [[PushT]]: 主实验benchmark（LeRobot版本）
- [[Meta-World]]: 门开启迁移研究
- [[MuJoCo]]: 物理模拟引擎（提供真实分支状态和Oracle打分）

---

## 速查卡片

> [!summary] pico-WAM
> - **核心**: 诊断WAM规划失败：打分（而非提案生成）是关键瓶颈
> - **方法**: 共享隐变量 WAM + 反事实分支训练 + 因子交叉评分干预
> - **结果**: 分支训练 → rank相关 0.49→0.80，restart coverage +4.8pp；真实打分 → coverage +12.2pp；仍落后 Diffusion Policy 16.7pp（成功率）
> - **代码**: [github.com/keivalya/pico-wam](https://github.com/keivalya/pico-wam)

---

*笔记创建时间: 2026-11-04*
