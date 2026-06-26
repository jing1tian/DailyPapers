---
title: "FORCE: Efficient VLA Reinforcement Fine-Tuning via Value-Calibrated Warm-up and Self-Distillation"
method_name: "FORCE"
authors: [Shuyi Zhang, Yunfan Lou, Hongyang Cheng, Yichen Guo, Chuyao Fu, Yaoxu Lyu, Xiaojie Zhang, Haoran Li, Pengwei Wang, Zhongyuan Wang, Shanghang Zhang]
year: 2026
venue: arXiv
tags: [vla-finetuning, reinforcement-learning, offline-to-online-rl, sample-efficient-rl, value-calibration, self-distillation, robot-manipulation]
zotero_collection: Robotics/VLA
image_source: online
arxiv_html: https://arxiv.org/html/2606.26006
created: 2026-06-26
---

# 论文笔记：FORCE: Efficient VLA Reinforcement Fine-Tuning via Value-Calibrated Warm-up and Self-Distillation

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | 论文未在摘要/HTML 正文中明确列出机构信息 |
| 日期 | June 2026 |
| 项目主页 | 未提供（论文中未出现 project page / GitHub 链接） |
| 对比基线 | [[Cal-QL]], [[CQL]], [[ConRFT]], [[PA-RL]], [[Octo]], [[π₀]], [[π₀.₅]] |
| 链接 | [arXiv](https://arxiv.org/abs/2606.26006) / [HTML](https://arxiv.org/html/2606.26006) |

---

## 一句话总结

> FORCE 用三阶段流程（离线 [[Cal-QL]] 预训练 → 分布式校准 Warm-up → 在线 [[VGPD|价值引导策略自蒸馏]]）解决 VLA 强化学习微调中的"灾难性初始遗忘"和低质量探索问题，在仿真和真实机器人任务上实现近乎完美的成功率且无需人类干预。

---

## 核心贡献

1. **三阶段无干预 offline-to-online RL 微调框架**：从离线预训练到在线微调全程不需要人类介入纠正，解决了 [[VLA]] RL 微调的样本效率瓶颈。
2. **分布式校准 Warm-up（Distributional Warm-up）**：在切换到在线训练之前，先用 on-policy rollout 数据让 Q 函数的支撑集（support）与策略访问分布对齐，消除"初始遗忘"现象。
3. **价值引导策略自蒸馏（[[VGPD|Value-Guided Policy Distillation]]）**：把策略改进问题转化为带 KL 约束的能量加权蒸馏目标，并引入动态优势过滤器，过滤低价值的探索动作，保证策略改进的单调性和稳定性。
4. **大幅的实证提升**：仿真中比此前最优 RL 方法（[[ConRFT]]）平均成功率提升 79% 的绝对值/相对优势体现，真实机器人 6 任务平均成功率从 BC 的 45% 提升到 98.3%，训练步数减少约 32.5%。

---

## 问题背景

### 要解决的问题

[[VLA]] 模型通过 [[模仿学习|纯模仿学习]]（behavior cloning, BC）训练，其性能被示范数据质量直接限制，存在所谓的"**imitation ceiling**"（模仿天花板）：策略永远不会比示范数据更好。[[强化学习|强化学习微调]]（RL fine-tuning）理论上可以突破这一上限，但在真实物理交互场景中存在严重的**样本效率**问题，限制了其实际可用性。

### 现有方法的局限

论文指出两个根本性困难：

1. **"灾难性初始遗忘"（Catastrophic Initial Unlearning）**：当策略从离线预训练（通常使用保守的 Q 值估计，如 [[CQL]]）切换到在线微调时，会出现广为人知的性能崩溃现象。原文描述："This 'initial unlearning' is caused by a Q-value scale mismatch, where the highly underestimated offline Q-function is 'deceived' by new online data, leading to a catastrophic adjustment period"（高度低估的离线 Q 函数在遇到新的在线数据时被"欺骗"，导致灾难性的调整期）。本质上是离线数据分布与策略实际访问分布之间的协变量偏移（covariate shift）在 Q 值尺度上的体现。
2. **低质量探索数据导致的低效策略更新**：在线探索初期产生大量低价值、高方差的动作样本，如果不加区分地用于更新策略，会拖慢甚至误导策略改进方向。

已有的离线-在线方法（如 [[Cal-QL]]、[[CQL]]、[[PA-RL]]）多聚焦于解决纯 Q 值层面的保守性问题，但没有同时处理"分布对齐"和"探索数据质量过滤"两个维度；[[ConRFT]] 等方法虽结合了 BC 与 Q 学习，但仍需要较多在线样本才能收敛，且未对探索噪声做显式过滤。

### 本文的动机

作者认为，离线到在线过渡失败的根源是 **Q 函数支撑集与策略访问分布不匹配**，因此在真正进入在线 RL 之前，应先有一个"热身阶段"用 on-policy rollout 让 Q 函数适应新分布；同时，在在线阶段策略更新时，应当用 Q 值对探索样本做"价值引导的自蒸馏"过滤，只学习高价值、低风险的动作，从而保证策略改进的单调性，而不依赖人类不断纠错。

---

## 方法详解

### 模型架构

FORCE 是一个 **[[Actor-Critic|Actor-Critic]] 式的三阶段训练流程**，可应用于不同的 VLA backbone（论文中分别在 [[Octo]] 和 [[π₀]] 上验证）：

- **输入**：观测 $o_t$（视觉 + 状态）+ 语言指令 $l$
- **Critic 网络**：2 层 MLP（隐藏层 256，[[Tanh]] 激活），双 Critic 集成（dual-critic ensemble），输出 $Q_\theta(s,a)$
- **Policy 网络**：2 层 MLP（隐藏层 256，Tanh 激活）的 [[Diffusion Policy|扩散式 / 一致性策略]]，基于 [[Octo]] 或 [[π₀]] 的视觉编码器（[[ResNet-18]]）提取特征
- **核心模块**：[[Cal-QL]] 离线预训练 → 分布式校准 Warm-up → [[VGPD]] 在线微调
- **输出**：动作 $a$（或动作块）

### 核心模块

#### 阶段一：离线 RL 预训练（[[Cal-QL]] Warm-up）

**设计动机**：在完全在线训练之前，先用离线示范数据 $\mathcal{D}_E$（expert buffer）训练出一个具备保守性、不会过度高估 OOD 动作价值的初始 Q 函数和策略。

**具体实现**：
- Critic 沿用 [[Cal-QL]] 的损失，在标准 TD 误差基础上加入校准约束项，防止 Q 值在 offline 数据分布外被过度高估
- Policy 损失同时包含 [[行为克隆|行为克隆]]（BC）项和 Q 值引导项，用权重 $\eta, \lambda$ 平衡模仿与价值最大化

#### 阶段二：分布式校准 Warm-up（Distributional Warm-up）

**设计动机**：直接从阶段一切换到纯在线 RL 会导致"灾难性初始遗忘"，因为 Q 函数的支撑集（support）仍局限在离线策略 $\pi_\beta$ 的访问范围内，没有覆盖当前策略 $\pi_\phi$ 实际会访问的状态-动作对。

**具体实现**：
- 让当前策略在环境中执行 on-policy rollout，收集数据 $\mathcal{D}_{warm}$，与离线数据合并得到 $\mathcal{D}_{mix} = \mathcal{D}_{offline} \cup \mathcal{D}_{warm}$，从而把 Q 函数的支撑集从 $\text{supp}(\pi_\beta)$ 扩展为 $\text{supp}(\pi_\beta) \cup \text{supp}(\pi_\phi)$
- BC 损失变为**成功率加权**（success-weighted）形式：只对标记为成功（$y=1$）的轨迹施加模仿约束，并除以成功率归一化因子 $\rho$，避免对失败轨迹做无意义的模仿
- 这一阶段本质上是让 Critic 提前"适应"在线分布，为正式进入在线 RL 做好缓冲，避免分布突变带来的价值崩塌

#### 阶段三：在线微调与 [[VGPD|价值引导策略自蒸馏]]（VGPD）

**设计动机**：进入真正的在线交互阶段后，探索数据质量参差不齐，如果不分辨地用全部探索样本更新策略，会引入大量噪声梯度。VGPD 把策略更新问题构造为一个带 KL 正则的能量加权蒸馏问题，只从高价值候选动作中学习。

**具体实现**：
- 维护两个缓冲区：**Expert Buffer** $\mathcal{D}_E$（离线数据 + 成功的在线轨迹）和 **Policy Buffer** $\mathcal{D}_\pi$（全部在线 rollout，包含失败样本）
- Critic 在 $\mathcal{D}_E \cup \mathcal{D}_\pi$ 上用标准 TD 损失继续更新
- Policy 改进被建模为带 [[KL 散度]] 约束的优化问题：在不过度偏离旧策略 $\pi_{old}$ 的前提下最大化 Q 值期望，其解析解是能量加权形式 $\pi^*(a|s) \propto \pi_{old}(a|s)\exp(Q_\theta(s,a)/\tau)$
- 实际优化时把上述能量加权分布投影回参数化策略 $\pi_\phi$，等价于一个加权对数似然最大化（**KL 投影**）
- **动态优势过滤器**：对策略当前采样出的 $K$ 个候选动作 $\{\hat{a}_k\}$，计算其 Q 值均值 $q_{mean}(s)$ 作为动态基线；若 $q_{mean}(s)$ 超过 Policy Buffer 中已执行动作 $a_{buf}$ 的 Q 值，则认为探索候选优于历史经验，启用蒸馏目标；否则退回模仿 $a_{buf}$
- 在蒸馏目标内部，进一步过滤掉 Q 值低于 $q_{mean}(s)$ 的候选动作，只对高于均值的候选做指数加权归一化，构造目标分布 $\mu_{VGPD}$
- 效果上，该机制天然形成一种课程学习：训练初期策略候选质量参差，过滤器更多回退到模仿历史经验；训练后期策略候选质量提升，蒸馏机制逐渐主导，实现自适应的"先模仿、后探索"过渡（对应 Figure 6 的可视化）

---

## 关键公式

### 公式 1：[[Cal-QL|Stage 1 Critic 校准损失]]

$$
\mathcal{L}_Q^{S1}(\theta) = \mathbb{E}_{(s,a,s')\sim\mathcal{D}_E}\Big[\big(Q_\theta(s,a) - \mathcal{B}^\pi \bar{Q}_{\bar\theta}(s,a)\big)^2\Big] + \alpha\Big(\mathbb{E}_{s\sim\mathcal{D}_E,\,a\sim\pi(\cdot|s)}\big[\max(Q_\theta(s,a), Q_\mu(s,a))\big] - \mathbb{E}_{s,a\sim\mathcal{D}_E}\big[Q_\theta(s,a)\big]\Big)
$$

**含义**：标准 TD 误差项 + [[Cal-QL]] 校准正则项。校准项约束策略采样动作的 Q 值不低于参考值 $Q_\mu$（避免过度保守），同时不让其超过离线数据本身的 Q 值估计太多，防止 OOD 高估。

**符号说明**：
- $\theta, \bar\theta$：在线 Critic 与目标 Critic 的参数
- $\mathcal{B}^\pi$：Bellman backup 算子，$\mathcal{B}^\pi \bar{Q}_{\bar\theta}(s,a) = r(s,a) + \gamma\,\mathbb{E}_{a'\sim\pi(\cdot|s')}[\bar{Q}_{\bar\theta}(s',a')]$
- $\mathcal{D}_E$：Expert Buffer（离线示范数据）
- $Q_\mu$：参考价值函数（用于校准下界，来自行为策略的蒙特卡洛回报估计）
- $\alpha$：校准约束权重

### 公式 2-4：[[行为克隆|Stage 1 Policy 损失]]

$$
\mathcal{L}_\pi^{S1}(\phi) = \eta\,\mathcal{L}_\pi^{BC,S1}(\phi) + \lambda\,\mathcal{L}_\pi^{Q,S1}(\phi)
$$

$$
\mathcal{L}_\pi^{BC,S1}(\phi) = \mathbb{E}_{(s,a)\sim\mathcal{D}_E}\big[-\log \pi_\phi(a|s)\big]
$$

$$
\mathcal{L}_\pi^{Q,S1}(\phi) = -\mathbb{E}_{s\sim\mathcal{D}_E,\,a\sim\pi_\phi(\cdot|s)}\big[Q_\theta(s,a)\big]
$$

**含义**：策略损失是 [[行为克隆]]（模仿离线数据）和 Q 值最大化（价值引导）的加权组合，离线阶段以模仿为主导。

**符号说明**：
- $\eta, \lambda$：BC 项与 Q 引导项的权重系数
- $\pi_\phi$：参数为 $\phi$ 的策略网络

### 公式 5-7：[[VGPD|Stage 2 分布式 Warm-up 损失]]

$$
\mathcal{L}_\pi^{S2}(\phi) = \eta\,\mathcal{L}_\pi^{BC,S2}(\phi) + \lambda\,\mathcal{L}_\pi^{Q,S2}(\phi)
$$

$$
\mathcal{L}_\pi^{BC,S2}(\phi) = \frac{1}{\rho}\,\mathbb{E}_{(s,a,y)\sim\mathcal{D}_{mix}}\Big[\mathbb{1}\{y=1\}\cdot\big(-\log \pi_\phi(a|s)\big)\Big]
$$

$$
\mathcal{L}_\pi^{Q,S2}(\phi) = -\mathbb{E}_{s\sim\mathcal{D}_{mix},\,a\sim\pi_\phi(\cdot|s)}\big[Q_\theta(s,a)\big]
$$

**含义**：在混合数据集 $\mathcal{D}_{mix}$（离线数据 + on-policy rollout）上继续训练，但 BC 项只对成功轨迹（$y=1$）生效，用 $1/\rho$ 归一化抵消成功样本占比偏低带来的梯度尺度问题。

**符号说明**：
- $\mathcal{D}_{mix} = \mathcal{D}_{offline} \cup \mathcal{D}_{warm}$：离线数据与 on-policy warm-up rollout 的合并数据集
- $y$：轨迹结果标签（1 表示成功，0 表示失败）
- $\rho$：成功率归一化因子（mini-batch 中成功样本占比）
- $\mathbb{1}\{\cdot\}$：指示函数

### 公式 8-9：[[VGPD|Stage 3 在线损失]]

$$
\mathcal{L}_Q^{S3}(\theta) = \mathbb{E}_{(s,a,s')\sim(\mathcal{D}_E\cup\mathcal{D}_\pi)}\Big[\big(Q_\theta(s,a) - \mathcal{B}^\pi \bar{Q}_{\bar\theta}(s,a)\big)^2\Big]
$$

$$
\mathcal{L}_\pi^{S3}(\phi) = \lambda\,\mathcal{L}_\pi^{Q,S3}(\phi) + \eta\,\mathcal{L}_\pi^{Distill,S3}(\phi)
$$

**含义**：Critic 在 Expert Buffer 与 Policy Buffer 的并集上继续做标准 TD 更新；Policy 损失从"BC + Q 引导"切换为"Q 引导 + 价值蒸馏"，正式从模仿过渡到自蒸馏驱动的策略改进。

**符号说明**：
- $\mathcal{D}_\pi$：Policy Buffer（全部在线 rollout，含失败样本）
- $\mathcal{L}_\pi^{Distill,S3}$：[[VGPD|价值引导自蒸馏]]损失（见公式 11）

### 公式 10：[[VGPD|正则化策略改进目标]]

$$
\mathcal{J}(\pi) = \mathbb{E}_{s\sim\mathcal{D}}\Big[\mathbb{E}_{a\sim\pi(\cdot|s)}[Q_\theta(s,a)] - \tau\, D_{KL}\big(\pi(\cdot|s)\,\|\,\pi_{old}(\cdot|s)\big)\Big]
$$

**含义**：策略改进被建模为在不偏离旧策略太远（KL 约束）的前提下最大化期望 Q 值，这是标准的信赖域式策略优化目标（类比 [[近端策略优化|PPO]] / MPO 类方法）。

**符号说明**：
- $\pi_{old}$：上一轮迭代的策略（蒸馏的教师分布）
- $\tau$：温度系数，控制 KL 约束的强度（$\tau$ 越大越保守）
- $D_{KL}$：[[KL 散度]]

### 公式 11：[[VGPD|能量加权 KL 投影]]

该目标的解析最优解为能量加权形式 $\pi^*(a|s) \propto \pi_{old}(a|s)\exp(Q_\theta(s,a)/\tau)$，将其投影回参数化策略 $\pi_\phi$ 等价于求解：

$$
\max_\phi\ \mathbb{E}_{s\sim\mathcal{D},\,a\sim\pi_{old}}\Big[\exp\big(Q_\theta(s,a)/\tau\big)\,\log \pi_\phi(a|s)\Big]
$$

**含义**：用旧策略采样的动作作为支撑，按其 Q 值的指数加权重要性进行加权对数似然训练，Q 值越高的动作权重越大，实现"自蒸馏"式的策略改进。

**符号说明**：
- $\exp(Q_\theta(s,a)/\tau)$：能量加权因子（softmax 形式的重要性权重，温度 $\tau$ 控制尖锐程度）

### 公式 13：[[VGPD|动态价值基线]]

$$
q_{mean}(s) = \frac{1}{K}\sum_{k=1}^{K} Q_\theta(s, \hat{a}_k), \qquad \{\hat{a}_k\}_{k=1}^{K} \sim \pi_\phi(\cdot|s)
$$

**含义**：从当前策略采样 $K$ 个候选动作，取其 Q 值均值作为动态基线，用以判断当前探索候选是否整体优于历史经验动作。

**符号说明**：
- $K$：每个状态采样的候选动作数量
- $\hat{a}_k$：第 $k$ 个候选动作

### 公式 14：[[VGPD|目标分布构造（动态优势过滤）]]

$$
\mu_{VGPD}(\cdot|s) = \big(1-\zeta(s)\big)\,\delta_{a_{buf}}(\cdot) + \zeta(s)\sum_{k=1}^{K} \tilde{w}_k(s)\,\delta_{\hat{a}_k}(\cdot)
$$

其中过滤指示函数：

$$
\zeta(s) = \mathbb{1}\big\{q_{mean}(s) > Q_\theta(s, a_{buf})\big\}
$$

**含义**：若候选动作的平均价值超过 Policy Buffer 中已执行动作 $a_{buf}$ 的价值（$\zeta(s)=1$），则蒸馏目标切换为对高价值候选的加权混合分布；否则（$\zeta(s)=0$）退回模仿 $a_{buf}$，避免向更差的探索方向更新。

**符号说明**：
- $\delta_{(\cdot)}$：Dirac 分布（退化到单点的概率质量）
- $a_{buf}$：从 Policy Buffer 采样的已执行动作
- $\zeta(s)$：动态优势过滤指示函数

### 公式 15：[[VGPD|指数加权归一化权重]]

$$
\tilde{w}_k(s) = \frac{\mathbb{1}\{Q_\theta(s,\hat{a}_k)\geq q_{mean}(s)\}\,\exp\big(Q_\theta(s,\hat{a}_k)/\tau\big)}{\sum_{j=1}^{K}\mathbb{1}\{Q_\theta(s,\hat{a}_j)\geq q_{mean}(s)\}\,\exp\big(Q_\theta(s,\hat{a}_j)/\tau\big)}
$$

**含义**：在 $K$ 个候选动作中，先用指示函数过滤掉低于均值基线的候选（二次过滤），再对剩余候选做能量加权 softmax 归一化，得到最终的蒸馏目标权重。

**符号说明**：
- $\mathbb{1}\{Q_\theta(s,\hat{a}_k)\geq q_{mean}(s)\}$：二值过滤项，只保留高于均值基线的候选
- $\tilde{w}_k(s)$：第 $k$ 个候选的归一化蒸馏权重，满足 $\sum_k \tilde{w}_k(s)=1$（在通过过滤的候选范围内）

---

## 关键图表

### Figure 1: FORCE 框架总览（Overview of the FORCE Framework）

![Figure 1](https://arxiv.org/html/2606.26006v1/x1.png)

**说明**：展示三阶段流程——Stage 1 离线 [[Cal-QL]] 预训练（用 Expert Buffer）→ Stage 2 分布式校准 Warm-up（on-policy rollout 扩展 Q 函数支撑集）→ Stage 3 在线微调（[[VGPD]] 价值引导自蒸馏）。每一阶段逐步校准价值估计并稳定策略改进。

### Figure 2: VGPD 在线阶段模块图（VGPD Module in the Online Phase）

![Figure 2](https://arxiv.org/html/2606.26006v1/x2.png)

**说明**：VGPD 作为正则化策略改进机制，维护 Expert Buffer 和 Policy Buffer 两个缓冲区；图中展示动态价值基线 $q_{mean}(s)$ 如何与 Policy Buffer 中动作的 Q 值比较，决定是走蒸馏分支还是回退模仿分支。

### Figure 3: 真实世界实验任务（Real-world Experiment Tasks）

![Figure 3](https://arxiv.org/html/2606.26006v1/x3.png)

**说明**：使用单臂 [[Franka Emika Panda]] 机器人，配备两个 [[Intel RealSense|RealSense D435]] 相机（手腕视角 + 侧视角），展示 6 个真实任务的实验台设置。

### Figure 4: 学习曲线（Learning Curves）

![Figure 4](https://arxiv.org/html/2606.26006v1/x4.png)

**说明**：FORCE 与各 baseline 在 [[ManiSkill3]] 六个任务上的训练曲线对比（3 个随机种子），FORCE 收敛更快且最终成功率更高。

### Figure 5: 消融实验（Ablation Study）

![Figure 5](https://arxiv.org/html/2606.26006v1/x5.png)

**说明**：在四个任务上对比完整 FORCE、去掉分布式 Warm-up 的变体、去掉 [[VGPD]] 的变体的训练曲线，验证两个核心模块各自的贡献。

### Figure 6: VGPD 的自适应特性（Adaptive Nature of VGPD / Adaptive Distillation Dynamics）

![Figure 6](https://arxiv.org/html/2606.26006v1/x6.png)

**说明**：可视化训练过程中蒸馏目标来源于"策略自身候选"还是"Policy Buffer 历史动作"的比例随训练步数的变化，体现自然涌现的课程学习现象：早期偏向模仿历史经验，后期逐渐转向自蒸馏驱动的探索。

### Figure 7: 仿真实验任务总览（Simulation Experiments on Six ManiSkill Manipulation Tasks）

![Figure 7](https://arxiv.org/html/2606.26006v1/x7.png)

**说明**：[[ManiSkill3]] 中六个操作任务（PlaceSphere、StackCube、PickCube、PushCube、PullCube、PullCubeTool）的视觉效果总览。

### Figure 8: 真实世界成功轨迹相机视角（Camera View of Successful Trajectories）

![Figure 8](https://arxiv.org/html/2606.26006v1/x8.png)

**说明**：展示真实机器人实验任务中成功执行轨迹的相机画面示例。

### Table 1: ManiSkill 任务成功率对比

| Method | StackCube | PullCube | PushCube | PullCubeTool | PlaceSphere | PickCube | Avg(%) |
|--------|-----------|----------|----------|--------------|--------------|----------|--------|
| Octo (BC) | 0 | 0 | 13 | 0 | 0 | 8.5 | 3.58 |
| π₀ (BC) | 60 | 87.5 | 70 | 7.5 | 27.5 | 2.5 | 42.5 |
| π₀.₅ (BC) | 50 | 87.5 | 100 | 7.5 | 15 | 5 | 44.17 |
| Cal-QL | 65.2 | 84.9 | 96.7 | 0 | 0 | 24.2 | 45.2 |
| CQL | 0 | 0 | 92.2 | 0 | 0 | 2.1 | 15.7 |
| PA-RL | 73.9 | 81.1 | 93.7 | 0 | 0 | 52.4 | 50.2 |
| ConRFT (no HIL) | 82.1 | 91.7 | 95.9 | 0 | 69.8 | 87.2 | 71.1 |
| **FORCE (Octo)** | **94.2** | **99** | **100** | **19** | **85.1** | **96.7** | **82.3** |
| **FORCE (π₀)** | **93.2** | **100** | **100** | **36.7** | **97.5** | **94.1** | **86.9** |

**关键发现**：FORCE 在几乎所有任务上大幅领先各类 baseline，尤其是在 [[CQL]]、[[Cal-QL]] 等纯 offline RL 方法完全失败的 PullCubeTool、PlaceSphere 任务上仍能取得有效成功率，体现分布式 Warm-up 和 VGPD 对探索效率的提升。

### Table 2: 真实世界任务结果

| Task | BC Success% | FORCE Success%（中间→最终） | BC 平均步数 | FORCE 平均步数（中间→最终） |
|------|-------------|------------------------------|-------------|------------------------------|
| Pick Cup | 35 | 45 → 90 | 134.5 | 125.2 → 36.4 |
| Open Drawer | 35 | 55 → 100 | 57.4 | 63.3 → 38.2 |
| Insert USB | 75 | 60 → 100 | 44.5 | 42.7 → 16.0 |
| Pick Corn | 40 | 45 → 100 | 142.3 | 153.3 → 52.4 |
| Stack Cube | 40 | 35 → 100 | 155.2 | 170.4 → 46.2 |
| Clean Board | 45 | 50 → 100 | 142.6 | 136.5 → 44.3 |
| **平均** | **45.0** | **48.3 → 98.3** | **112.8** | **115.2 → 38.9** |

**关键发现**：FORCE 在线微调后真实机器人 6 项任务平均成功率从 BC 基线的 45.0% 提升到 98.3%，同时完成任务所需步数从 112.8 步降到 38.9 步（约 3-4 倍执行效率提升），如 Stack Cube 从 170.4 步降到 46.2 步。

### Table 3: 达到 80% 成功率所需训练步数（样本效率消融）

| Task | w/o Pre-Cal | w/o Pre-Cal & VGPD | FORCE |
|------|-------------|---------------------|-------|
| StackCube | 28k | 28k | 18k |
| PickCube | 14k | 20k | 12k |
| PushCube | 8k | 10k | 4k |
| PlaceSphere | 30k | 30k | 20k |

**关键发现**：去掉分布式预校准（Pre-Cal）和 VGPD 后，达到 80% 成功率所需的训练步数显著增加，验证两个模块均能独立提升样本效率，完整 FORCE 平均减少约 32.5% 的训练步数。

### Table 4: 真实世界任务专属超参数

| Parameter | Pick Cup | Open Drawer | Insert USB | Pick Corn | Stack Cube | Clean Board |
|-----------|----------|-------------|------------|-----------|------------|--------------|
| Max episode length | 500 | 300 | 300 | 300 | 1000 | 1000 |
| Offline steps | 20k | 24k | 20k | 14k | 26k | 20k |
| Transition steps | 24k | 26k | 24k | 16k | 30k | 24k |
| Online steps | 32k | 30k | 28k | 20k | 36k | 32k |
| Negative reward | -0.05 | -0.01 | -0.01 | -0.05 | -0.01 | -0.01 |
| Online training time | 40 min | 20 min | 20 min | 20 min | 30 min | 40 min |

**关键发现**：所有真实任务的在线训练时间均在 20-40 分钟内完成，体现 FORCE 在真实物理交互场景下的高样本效率与可部署性。

---

## 实验

### 数据集 / 任务环境

| 环境 | 规模/任务 | 特点 | 用途 |
|------|-----------|------|------|
| [[ManiSkill3]] | 6 个操作任务（StackCube、PullCube、PushCube、PullCubeTool、PlaceSphere、PickCube） | GPU 并行化仿真，覆盖堆叠、推、拉、工具使用、精密放置等技能 | 仿真训练与评测 |
| 真实世界任务 | 6 个任务（Pick Cup、Open Drawer、Insert USB、Pick Corn、Stack Cube、Clean Board） | 接触丰富、精密插拔等多样化操作 | 真实机器人验证 |

### 实现细节

- **VLA Backbone**: [[Octo]] 和 [[π₀]]（两套独立验证）
- **Critic / Policy 网络**: 2 层 MLP，隐藏层 256，[[Tanh]] 激活
- **视觉编码器**: 预训练 [[ResNet-18]]，处理 640×480 RGB 图像
- **真实机器人平台**: [[Franka Emika Panda]] 单臂，Robotiq / 软体夹爪可选
- **相机配置**: 双 [[Intel RealSense|RealSense D435]]（腕部视角 + 侧视角）
- **状态空间**: 7 维（6 维末端执行器 delta pose + 1 维夹爪状态）
- **控制频率**: 10 Hz
- **奖励设计**: 任务相关的负奖励项（如 -0.01 ~ -0.05），鼓励快速完成任务

### 可视化结果

Figure 8 展示了真实世界多任务的成功轨迹相机画面；从执行步数对比（Table 2）可见 FORCE 微调后的策略在动作流畅度和效率上显著优于 BC 基线，体现 RL 微调不仅提升了成功率，也优化了执行路径的效率。

---

## 批判性思考

### 优点

1. **完全无人类干预**：与 [[HIL-SERL]]、[[EXPO-FT]] 等依赖人类在环纠错的方法不同，FORCE 全程自动化，更适合大规模无人值守部署。
2. **同时解决两个独立问题**：分布式 Warm-up 解决"价值分布错配"，VGPD 解决"探索数据质量"，两者在消融实验（Table 3、Figure 5）中均显示独立的正向贡献。
3. **跨 backbone 泛化性**：在 [[Octo]] 和 [[π₀]] 两种不同架构的 VLA 上都验证了有效性，说明方法本身与具体 backbone 解耦。
4. **真实机器人验证充分**：6 个真实任务、不止报告成功率还报告执行步数，能更全面体现策略质量的提升（不仅是"能不能做到"，还有"做得多熟练"）。
5. **VGPD 的理论动机清晰**：将策略改进构造为带 KL 约束的能量加权投影问题，是 [[近端策略优化|信赖域策略优化]]思想在 offline-to-online 场景下的合理延伸，且动态优势过滤器提供了直觉清楚的安全机制（不优于历史经验就不学）。

### 局限性

1. **计算开销**：VGPD 在每个状态需要采样 $K$ 个候选动作并逐一过 Critic 评估，这在大模型 VLA 推理本就昂贵的前提下进一步增加了在线训练的计算负担（论文在 Conclusion 中也明确承认这一局限）。
2. **依赖任务特定 Critic**：当前框架为每个任务单独训练 Critic，尚不具备跨任务泛化的通用价值函数或奖励模型。
3. **PullCubeTool 任务成功率仍然偏低**（FORCE 最高仅 36.7%），说明对高精度工具使用类任务，方法仍有较大提升空间。
4. **缺乏开源代码/项目主页**：截至本笔记撰写时未发现公开代码仓库或项目主页，可复现性存疑。
5. **机构信息缺失**：论文摘要/HTML 版本未明确列出作者机构，无法评估实验资源规模的可信度背景。

### 潜在改进方向

1. **动作候选缓存机制**：论文 Conclusion 中提到的未来方向，通过缓存历史候选动作的 Q 值评估结果，减少 VGPD 在线采样 $K$ 个候选的重复计算开销。
2. **通用奖励模型/价值函数**：训练跨任务共享的 Critic 或奖励模型，降低每个新任务都要从零训练价值函数的成本。
3. **结合人类先验进一步加速冷启动**：可探索把 FORCE 的自动化机制与少量人类干预结合，专门处理像 PullCubeTool 这类高精度任务的冷启动困难。

### 可复现性评估

- [ ] 代码开源（未发现）
- [ ] 预训练模型（未提及）
- [x] 训练细节完整（附录提供任务专属超参数表，见 Table 4）
- [x] 数据集/仿真环境可获取（[[ManiSkill3]] 公开仿真器）

---

## 关联笔记

### 基于

- [[Cal-QL]]: Stage 1 离线预训练直接采用 Cal-QL 的校准 Critic 损失
- [[CQL]]: Cal-QL 的基础保守 Q 学习思想来源
- [[VLA]]: FORCE 微调的对象是预训练好的 VLA 策略

### 对比

- [[ConRFT]]: 当前最强 RL 微调 baseline（无人类干预版本），FORCE 在仿真平均成功率上超越约 10 个百分点以上
- [[PA-RL]]: 另一离线-在线过渡方法，作为 baseline 对比
- [[Octo]]: 作为 VLA backbone 之一，同时也是纯 BC baseline
- [[π₀]] / [[π₀.₅]]: 作为 VLA backbone 之一以及纯 BC baseline
- [[HIL-SERL]]: 依赖人类在环干预的 RL 微调方法，FORCE 旨在去除这种依赖
- [[EXPO-FT]]: 另一条 VLA RL 微调路线，同样关注样本效率但保留人类干预机制

### 方法相关

- [[强化学习]]: 整体优化框架
- [[Actor-Critic]]: FORCE 的网络结构范式（Critic 评估价值，Policy 做决策）
- [[VGPD]]: 本文提出的核心在线微调机制（价值引导策略自蒸馏）
- [[KL 散度]]: VGPD 正则化策略改进目标的核心约束项
- [[行为克隆]]: Stage 1/2 中 BC 损失的理论基础
- [[模仿学习]]: imitation ceiling 问题的根源讨论对象
- [[近端策略优化]]: VGPD 信赖域式策略改进思想的相关参照
- [[经验回放]]: Expert Buffer / Policy Buffer 的实现基础

### 硬件/数据相关

- [[Franka Emika Panda]]: 真实机器人实验平台
- [[ManiSkill3]]: 仿真实验环境
- [[ResNet-18]]: 视觉编码器骨干

---

## 速查卡片

> [!summary] FORCE (2026)
> - **核心**: 三阶段 offline-to-online RL 微调框架，解决 VLA RL 微调的"灾难性初始遗忘"和低质量探索两大瓶颈
> - **方法**: Cal-QL 离线预训练 → 分布式校准 Warm-up（扩展 Q 函数支撑集） → VGPD 在线价值引导自蒸馏（KL 约束能量加权 + 动态优势过滤）
> - **结果**: 仿真平均成功率 82.3%-86.9%（vs. ConRFT 71.1%），真实机器人 6 任务平均成功率 45%→98.3%，训练步数减少 32.5%
> - **代码**: 未公开

---

*笔记创建时间: 2026-06-26*
