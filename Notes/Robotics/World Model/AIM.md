---
title: "Training Controllable World Models via Action Influence Maximization"
method_name: "AIM"
authors: [Samuel M. Bateman, Tenny Yin, Chongyi Zheng, Lei Huang, Brian Wang, Benjamin Eysenbach, Jaime Fernández Fisac, Dhruv Shah]
year: 2026
venue: CoRL 2026
tags: [world-model, controllability, active-data-collection, reinforcement-learning, manipulation, video-prediction, mutual-information]
zotero_collection: Robotics/World Model
image_source: online
arxiv_html: https://arxiv.org/html/2610.27387v1
created: 2026-10-22
---

# 论文笔记：Training Controllable World Models via Action Influence Maximization

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Princeton University, Columbia University |
| 日期 | October 2026 |
| 项目主页 | — |
| 对比基线 | [[DIAYN]], [[METRA]], [[IQL]], [[SAC]] |
| 链接 | [arXiv](https://arxiv.org/abs/2610.27387) / [HTML](https://arxiv.org/html/2610.27387v1) |

---

## 一句话总结

> 专家演示数据的"动作影响力"不足导致 [[Action-Conditioned World Model|动作条件世界模型]] 无法学会听从动作指令；AIM 通过最大化数据集条件互信息来主动收集高影响力数据，以 500 条主动轨迹匹配 300k 条专家轨迹的可控性，实现 85× 数据效率提升。

---

## 核心贡献

1. **动作影响力诊断**: 定义"解释动作信息量"（Explained Action Information）作为[[可控性|世界模型可控性]]的量化指标，并证明其上界是训练数据集的条件[[互信息]]  $I(A; S' \mid S)$
2. **可控性上界定理**: 证明数据集 MI 严格限制了世界模型可控性的天花板——数据不够多样，模型再大也无法学会服从动作
3. **AIM 算法**: 设计结合动作信息奖励（$r_{\text{MI}}$）与状态覆盖奖励（$r_{\text{cov}}$）的主动数据采集算法，以批量在线 RL 循环实现持续优化
4. **实证验证**: 在 Four Rooms 离散导航和 Push-T 像素级操作任务中验证，AIM 是唯一能在像素级操作中持续提升可控性的方法

---

## 问题背景

### 要解决的问题

[[Action-Conditioned World Model|动作条件世界模型]]（如视频预测模型）在训练时接收动作作为输入，理论上应能预测不同动作的不同未来。然而，用成功专家演示训练的模型往往**忽略输入的动作**，直接预测"专家会做什么"，导致可控性极差。

### 现有方法的局限

- **数据规模**方案：堆砌更多专家数据（如 300k 轨迹），但这只在数据量极大时才有边际改善，且代价极高
- **探索/内在动机**方案：[[DIAYN]]、[[METRA]]、[[Curiosity-driven Exploration|ICM]]、DADS 等方法设计用于状态空间探索，但针对的不是世界模型可控性
- **随机策略**方案：均匀随机采集的数据有更好的动作多样性，但不在目标评估分布内

### 本文的动机

关键洞察：专家数据的问题不在量少，而在于**所有轨迹的动作结果太过相似**——专家无论采取哪个动作，结果都差不多（都是成功）。这导致数据集的条件[[互信息]] $I(A; S' \mid S)$ 极低，而这个量是世界模型可控性的**理论上界**。因此，解决方案不是更多数据，而是**不同类型的数据**。

---

## 方法详解

### 总体框架

**AIM（Action Influence Maximization）** 是一个批量在线 RL 框架，通过最大化数据集的条件[[互信息]]来主动收集训练数据：

- **输入**: 预训练的专家演示数据 $\mathcal{D}_0$（~3k 轨迹）+ 评估策略分布 $\rho_{\text{eval}}$
- **核心循环**: 每轮用学习的内在奖励运行 [[SAC]]/[[IQL]]+[[AWR]] 策略，收集 100 条新轨迹，更新密度模型
- **输出**: 增强数据集 $\mathcal{D}_k$，在此上训练的[[Action-Conditioned World Model|动作条件世界模型]] 具有更高可控性
- **世界模型**: 基于 [[Diffusion Forcing|Latent Diffusion Forcing]] 架构

### 核心模块

#### 模块 1：解释动作信息量（Explained Action Information）

**设计动机**: 量化世界模型在给定状态下，是否真的利用了动作输入来区分未来状态。

**具体实现**:
- 用条件动作信息衡量：模型的动作条件预测 $\hat{p}_\theta(\cdot | s, a)$ 是否显著优于边际预测 $\hat{p}_{\theta, \pi}(\cdot | s)$
- 通过 KL 散度之差量化这个"利用程度"
- 在评估分布 $\rho_{\text{eval}}$ 下取期望

#### 模块 2：AIM 奖励

**设计动机**: 主动采集那些让动作与转移高度相关（高 $r_{\text{MI}}$）且覆盖评估状态分布（高 $r_{\text{cov}}$）的数据。

**具体实现**:
- 动作信息项 $r_{\text{MI}}$：用[[Flow-Matching|流匹配]]密度估计器拟合逆动力学模型 $q_k(a|s,s')$ 和行为策略 $\beta_k(a|s)$
- 覆盖项 $r_{\text{cov}}$：用密度估计器拟合评估状态分布 $\rho_{\text{eval}}$ 和当前数据集状态分布 $d_k(s)$
- z-score 归一化两项再加和，稳定训练

#### 模块 3：批量在线 RL 实现

**设计动机**: 奖励模型在每轮数据后需要重新拟合，因此用批量在线框架（而非持续在线）

**具体实现**:
- 每轮固定奖励模型，运行 2×10⁵ 梯度步
- 用 [[AWR]]（Advantage-Weighted Regression）做 KL 约束策略更新，防止探索偏离行为数据太远（Figure 11 的信任域限制）
- 可选实用版本：加入新奇度奖励 $r^{\text{nov}}$ 进一步促进覆盖

#### 模块 4：世界模型架构（Latent Diffusion Forcing）

**设计动机**: 需要一个能做视频预测的基础架构，评估可控性

**具体实现**:
- 基于 [[Diffusion Forcing]] 的潜空间扩散模型
- 接收动作序列和历史帧作为条件，预测未来帧
- 所有对比方法共享相同架构，仅训练数据不同

---

## 关键公式

### 公式 1：[[互信息|解释动作信息量]]

$$
\hat{I}(\rho_{\text{eval}}, \pi_{\text{eval}}) := \mathbb{E}_{d_{\text{eval}}(s,a)}\Big[D_{\text{KL}}\big(p(\cdot|s,a) \| \hat{p}_{\theta, \pi_{\text{eval}}}(\cdot|s)\big) - D_{\text{KL}}\big(p(\cdot|s,a) \| \hat{p}_\theta(\cdot|s,a)\big)\Big]
$$

**含义**: 衡量世界模型在评估分布下"从动作中获得的额外预测精度"，即模型到底有多听从动作输入

**符号说明**:
- $\rho_{\text{eval}}$: 评估状态分布（如 BC 策略的访问分布）
- $\pi_{\text{eval}}$: 评估动作策略
- $p(\cdot|s,a)$: 真实环境动力学
- $\hat{p}_{\theta,\pi}(\cdot|s)$: 世界模型的边际预测（无动作条件）
- $\hat{p}_\theta(\cdot|s,a)$: 世界模型的动作条件预测

### 公式 2：[[互信息|Theorem 3.1 — 数据集 MI 上界]]

$$
\hat{I}(\rho, \pi) \leq I(\rho, \pi) := \mathbb{E}_{\rho(s)\pi(a|s)}\Big[D_{\text{KL}}\big(p(\cdot|s,a) \| p_\pi(\cdot|s)\big)\Big] = I_{\rho \pi p}(A; S' | S)
$$

**含义**: 世界模型的可控性（解释动作信息量）上界是数据集的条件互信息 $I(A;S'|S)$——数据天花板决定了模型能学到的极限

**符号说明**:
- $I_{\rho \pi p}(A; S' | S)$: 数据集轨迹上的条件互信息
- $p_\pi(\cdot|s)$: 在策略 $\pi$ 下的边际状态转移分布

### 公式 3：[[互信息|MI 上下文 Bandit 目标]]

$$
\max_\pi \; I(\rho_{\text{eval}}, \pi) = \mathbb{E}_{s \sim \rho_{\text{eval}}}\Big[\mathbb{E}_{a \sim \pi(\cdot|s)}\Big[D_{\text{KL}}\big(p(\cdot|s,a) \| p_\pi(\cdot|s)\big)\Big]\Big]
$$

**含义**: 最大化可控性上界等价于：对评估分布中的每个状态，选择使动作条件转移尽可能偏离边际转移的动作

**符号说明**:
- $\pi$: 数据采集策略（可优化的）
- $\rho_{\text{eval}}$: 固定的评估状态分布

### 公式 4：[[Action Influence|AIM 奖励]]

$$
r_{\text{AIM}}^k(s,a,s') := \underbrace{\log q_k(a|s,s') - \log \beta_k(a|s)}_{r_{\text{MI}}} + \underbrace{\log \rho_{\text{eval}}(s) - \log d_k(s)}_{r_{\text{cov}}^k}
$$

**含义**: AIM 奖励由两项构成：动作信息项鼓励采集"从结果能推断出动作"的数据，覆盖项鼓励访问评估分布中但当前数据集未覆盖的状态

**符号说明**:
- $q_k(a|s,s')$: 逆动力学模型（从转移推断动作），用[[Flow-Matching|流匹配]]拟合
- $\beta_k(a|s)$: 当前数据集中的行为策略密度
- $\rho_{\text{eval}}(s)$: 评估状态分布密度（如 BC 策略访问密度）
- $d_k(s)$: 当前数据集 $\mathcal{D}_k$ 的状态边际分布

### 公式 5：[[Action Influence|最优数据采集策略]]

$$
\pi_k^\star(a|s) \propto \beta_k(a|s) \cdot \exp\!\Big(\mathbb{E}_{s' \sim p(\cdot|s,a)}[r_{\text{AIM}}^k(s,a,s')]\Big)
$$

**含义**: 最优采集策略是行为策略 $\beta_k$ 的软调整版本，通过指数加权 AIM 奖励——由 [[AWR]] 近似实现

**符号说明**:
- $\pi_k^\star$: 第 $k$ 轮的最优采集策略
- $\beta_k$: 当前行为策略（KL 约束基准）

### 公式 6：[[Action Influence|严格改进保证]]（Theorem 3.2）

$$
I(\rho, \pi_k^\star) > I(\rho, \beta_k) \quad \text{如果 } \beta_k \neq \pi_k^\star \text{ 在评估概率非零的状态上}
$$

**含义**: 只要最优采集策略与当前行为策略不同，数据集 MI 就严格增大，即可控性上界严格提升

### 公式 7：[[Curiosity-driven Exploration|新奇度奖励]]（实用版补充项）

$$
r^{\text{nov}}_{t,e} = \min_{j \in \mathcal{T}_{i,t}} \| \mathbf{x}^T_{i,t} - \mathbf{x}^T_j \|_2
$$

**含义**: 当前状态到最近已访问状态的最小距离，鼓励探索未曾到达的区域

**符号说明**:
- $\mathbf{x}^T_{i,t}$: 第 $i$ 轮第 $t$ 步的状态特征
- $\mathcal{T}_{i,t}$: 历史访问状态集合

### 公式 8：[[Action Influence|实用版 AIM 奖励]]

$$
\hat{r}_{\text{AIM}}(s,a,s') = \text{z\_score}\big(\log \hat{q}_{\beta_k}(a|s,s') - \log \hat{\beta}_k(a|s)\big) - \text{z\_score}\big(\log \hat{d}_k(s)\big)
$$

**含义**: 两项分别 z-score 归一化后相减，稳定两项的量级差异；可选加入新奇度项作为覆盖的额外保障

---

## 关键图表

### Figure 1：动机——专家数据导致可控性偏差

![Figure 1: Expert vs AIM data motivation](https://arxiv.org/html/2610.27387v1/20261013-corl-motivation.png)

**说明**: 左侧展示用接近专家的数据（低动作影响力）训练的世界模型：无论输入什么动作，预测结果几乎相同，表现出"专家成功偏置"。右侧展示 AIM 数据训练后的模型，能够区分不同动作的结果。

### Figure 2：AIM 总览

![Figure 2: AIM overview teaser](https://arxiv.org/html/2610.27387v1/20261012-corl2026-teaser.png)

**说明**: AIM 的整体流程：从专家数据出发，用 AIM 奖励训练采集策略，将新采集的高影响力数据加入训练集，迭代提升世界模型可控性。

### Figure 3：动作信息量与仿真保真度的相关性（Four Rooms）

*（此图 URL 在 arXiv HTML 中未提取到，详见 [arXiv HTML 页面](https://arxiv.org/html/2610.27387v1)）*

**说明**: 解释动作信息量与仿真保真度（Sim Fidelity）在 Four Rooms 任务中高度相关，Pearson 相关系数 $r = 0.979$。验证了可控性指标作为仿真质量代理的可行性。

### Figure 4：基线方法的状态-动作覆盖模式对比

![Figure 4: Baseline state-action coverage](https://arxiv.org/html/2610.27387v1/baseline_modes_composite.png)

**说明**: 近专家策略（PPO）将覆盖集中在成功执行附近（高度聚集），随机策略提供更宽泛但无结构的覆盖，AIM 在评估分布附近实现有目标的探索。

### Figure 5：Four Rooms 与 Push-T 上通过主动数据采集改善可控性

*（此图 URL 在 arXiv HTML 中未提取到，详见 [arXiv HTML 页面](https://arxiv.org/html/2610.27387v1)）*

**说明**: 主要结果图。AIM 是唯一在 Push-T 像素级操作任务中持续提升可控性的方法，其他探索基线（DIAYN、METRA、ICM、DADS）在部分指标上甚至不如不加任何探索的基线。

### Figure 6：Four Rooms 与 Push-T 上的数据规模趋势

*（此图 URL 在 arXiv HTML 中未提取到，详见 [arXiv HTML 页面](https://arxiv.org/html/2610.27387v1)）*

**说明**: 展示可控性随数据规模（轨迹数）的变化趋势，说明专家数据规模从 3k 扩展到 300k 时可控性提升缓慢，而 500 条 AIM 数据就能跨越这一差距。

### Figure 7：Latent Diffusion Forcing 世界模型架构

![Figure 7: World model architecture](https://arxiv.org/html/2610.27387v1/corl2026_model_arch.png)

**说明**: 所有实验使用的 [[Diffusion Forcing|Latent Diffusion Forcing]] 世界模型架构。输入历史帧（通过 VAE 编码为潜变量）和动作序列，通过扩散过程预测未来潜帧，解码为未来视频帧。

### Figure 8：低数据量导致动作不敏感（Four Rooms 消融）

![Figure 8a: Low data action insensitivity](https://arxiv.org/html/2610.27387v1/figures/sim_realism_dozen/sim_realism_dozen_pppo_0_25pct.png)

**说明**: 用 0.25% 数据（约 150 条轨迹）训练的模型完全忽略动作输入，预测行为与训练策略的"最可能动作"相同。用 100% 数据（300k 轨迹）训练的模型接近完全动作敏感。这表明今天的大型世界模型可能也处于这个"低数据坍缩"区间。

### Figure 9：AIM vs 随机采集的状态访问密度比（Four Rooms）

![Figure 9: State visitation iter 1](https://arxiv.org/html/2610.27387v1/ratio_aim_vs_random_100_fixed_start.png)

**说明**: 对数比热图显示 AIM 采集的数据相对于均匀随机数据在状态空间的分布差异。AIM 访问的状态明显更远离起始状态，且每一轮迭代覆盖几乎不重叠的状态集合，说明 AIM 在持续扩展有效覆盖范围。

### Figure 11：Push-T 任务中 AIM 奖励的热图

![Figure 11: AIM reward heatmap](https://arxiv.org/html/2610.27387v1/grid_seed47_64x96_pair_composite_viridis.png)

**说明**: 显示学习到的 AIM 奖励在末端执行器位置网格上的分布。奖励在常规工作空间外的区域较高，说明未约束优化会将探索推向无意义区域——这正是用 [[AWR]] 做 KL 约束更新的理由。

### Figure 12-13：Push-T 拓展结果（BC 评估 / Push Away 评估）

*（这些图 URL 在 arXiv HTML 中未提取到，详见 [arXiv HTML 页面](https://arxiv.org/html/2610.27387v1)）*

**说明**: Figure 12 展示在 BC 轨迹评估下 AIM 显著改善末端执行器可控性和全图像像素召回率。Figure 13 展示在离策略的 "Push Away" 分布上的类似但更弱的结果（离策略任务更难泛化）。

### Table 1：Push-T 主要结果

| 实验 | BC Joint Recall (H=33) | BC Preference ELO |
|------|------------------------|-------------------|
| PPO-300K（300k 专家） | .224 ± .012 | 843 ± 155 |
| PPO-3K（3k 专家基线） | .203 ± .011 | 732 ± 155 |
| **PPO-3K + AIM500** | **.344 ± .018** | **1190 ± 167** |
| Recovery-3K | .266 ± .014 | 1059 ± 158 |
| **Recovery-3K + AIM500** | **.361 ± .019** | **1412 ± 183** |
| AIM w/o re-estimation | .379 ± .019 | 1388 ± 174 |

**关键发现**: 仅 500 条 AIM 轨迹加入 3k 专家数据，联合召回率从 .203 提升到 .344（提升 69%），超过 300k 专家数据的 .224，实现约 85× 数据效率增益。

### Table 2：Four Rooms 专家策略统计

| 专家策略 | 训练步数 | 动作熵 | Eval Return |
|----------|----------|--------|-------------|
| POMDP PPO（局部视野） | 300M | 0.054 | 0.657 |
| Early Stop PPO（低质量） | 50M | 0.252 | 0.587 |
| State Oracle PPO（完整状态） | 300M | 0.022 | 0.839 |

**关键发现**: 动作熵越低，专家策略越确定，对应的动作影响力越低，训练出的世界模型可控性越差。

### Table 6：探索基线超参数

| 方法 | 内在奖励 | 关键超参 | 学习率 |
|------|----------|----------|--------|
| [[DIAYN]] | 技能可辨别性 | 50 个离散技能 | 3×10⁻⁴ |
| DADS | 技能-动力学 MI | $z \in [-1,1]^8$，4 个混合 | 3×10⁻⁴ |
| [[Curiosity-driven Exploration\|ICM]] | 前向模型误差 | η=50, β=0.3 | 3×10⁻⁴ |
| [[METRA]] | 时序度量技能 | 4 个连续技能 | 1×10⁻⁴ |

---

## 实验

### 环境设置

| 任务 | 类型 | 评估指标 | 数据来源 |
|------|------|----------|----------|
| Four Rooms | 离散导航（部分可观测） | 解释动作信息量、仿真保真度（Pixel RMSE） | 均匀随机、POMDP PPO、Early Stop PPO、Oracle PPO |
| Push-T | 像素级操作（连续动作） | 末端执行器 masked MAE Recall（10% 阈值）、ELO | BC 扩散策略、Push Away PPO、BC Failure |

### 实现细节

- **世界模型**: Latent Diffusion Forcing，所有方法共享架构
- **批量在线 RL 协议**: 预训练于 ~3,005 条专家轨迹（约 1.1×10⁵ 转移），5 轮 × 100 条新轨迹，每轮 2×10⁵ 梯度步
- **基础 RL**: [[IQL]] + [[AWR]]（AWR 做 KL 约束更新），[[SAC]] 骨干用于对比基线
- **密度估计**: [[Flow-Matching|流匹配]] 密度估计器，z-score 归一化奖励
- **评估**: 每个模型 3 个推理种子 × 100 rollout，bootstrap 重采样（n=10,000）
- **人类评测**: 1,545 次 A/B 比较（20 名用户），Bradley-Terry ELO 评分

### 关键实验结果

**Four Rooms**:
- 解释动作信息量与仿真保真度高度线性相关（r = 0.979）
- 用 0.25% 数据训练时模型完全坍缩（忽略动作）；100% 数据时接近完全可控
- AIM 迭代采集到距起始状态更远的状态，每轮覆盖几乎不重叠区域

**Push-T**:
- 所有对比探索方法（DIAYN、DADS、ICM、METRA）在大多数指标上无法超过不做探索的 PPO-3K 基线
- AIM 是**唯一**持续提升可控性的方法
- PPO-3K + AIM500：BC 联合召回率 .344（vs PPO-300K 的 .224），ELO 1190（vs 843）
- **消融**：去掉奖励重估影响不大；去掉 MI 项会损害 BC 评估上的表现；单纯新奇度（无 MI）在 off-policy Push Away 评估上表现更好

---

## 批判性思考

### 优点
1. **理论基础扎实**: 通过互信息界证明了数据质量（而非数据量）是世界模型可控性的瓶颈，有严格的定理保证
2. **实际效果惊人**: 85× 数据效率增益，500 条主动轨迹打败 300k 专家轨迹，结论明确有力
3. **系统方法**：分离数据采集问题与模型架构问题，为世界模型社区提供新的优化维度

### 局限性
1. **计算代价高**: 每轮都需要重新拟合多个流匹配密度模型，计算开销显著
2. **仅在仿真验证**: 所有实验在仿真环境中进行，真实场景中安全运行主动探索策略面临挑战
3. **RL 近似无保证**: 批量在线 RL 循环只是理论最优策略的近似，Theorem 3.2 的严格改进保证在实践中不成立
4. **Tee 位置覆盖难**: 即使 300k 专家数据也无法解决 Push-T 中 tee 位置的可控性，根因在于预训练数据的状态覆盖不足

### 潜在改进方向
1. 用更轻量的密度估计器降低每轮的奖励模型训练代价
2. 将 AIM 扩展到真实机器人上，结合安全约束的探索
3. 理论分析批量在线 RL 近似与真实最优策略的差距

### 可复现性评估
- [ ] 代码开源（论文未提及）
- [ ] 预训练模型（未提及）
- [x] 训练细节完整（附录 10 有详细超参数）
- [x] 数据集可获取（Push-T 为公开任务，Four Rooms 为自建）

---

## 关联笔记

### 基于
- [[Diffusion Forcing]]: 世界模型底层架构
- [[IQL]]: 数据采集 RL 实现的值函数组件
- [[AWR]]: KL 约束策略更新实现

### 对比
- [[DIAYN]]: 探索基线，目标为技能多样性而非可控性
- [[METRA]]: 探索基线，时序度量空间技能学习
- [[Action-Conditioned World Model]]: 本文优化的目标架构类型

### 方法相关
- [[互信息]]: 核心理论工具，数据集条件 MI 决定可控性上界
- [[Flow-Matching]]: 密度估计器的底层方法
- [[Action Influence]]: 本文提出的核心概念

### 硬件/数据相关
- [[SAC]]: 对比探索基线使用的 RL 骨干

---

## 速查卡片

> [!summary] Training Controllable World Models via Action Influence Maximization (AIM)
> - **核心**: 专家数据的低动作影响力限制了世界模型可控性；AIM 通过最大化数据集条件 MI 主动采集高影响力数据
> - **方法**: AIM 奖励 = 动作信息项（$\log q_k(a|s,s') - \log \beta_k(a|s)$）+ 覆盖项；批量在线 RL + AWR 实现
> - **结果**: 500 条 AIM 轨迹 + 3k 演示 = 300k 专家轨迹可控性，85× 数据效率
> - **代码**: 未公开

---

*笔记创建时间: 2026-10-22*
