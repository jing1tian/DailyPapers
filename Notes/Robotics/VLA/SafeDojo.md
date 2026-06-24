---
title: "SafeDojo: Safe Reinforcement Learning for VLA via Interactive World Model"
method_name: "SafeDojo"
authors: [Kai Tang, Peidong Jia, Zhong Chu, Jixian Wu, Rui Ma, Jiajun Cao, Fangyuan Zhao, Sixiang Chen, Yichen Guo, Xiaowei Chi, Chun-Kai Fan, Kevin Zhang, Jinchang Xu, Fubing Yang, Weishi Mi, Xiaozhu Ju, Jian Tang, Shanghang Zhang]
year: 2026
venue: arXiv
tags: [safe-reinforcement-learning, vision-language-action, world-model, constrained-mdp, grpo, collision-avoidance]
zotero_collection: Robotics/VLA
image_source: online
arxiv_html: https://arxiv.org/html/2606.20698
created: 2026-06-24
---

# 论文笔记：SafeDojo: Safe Reinforcement Learning for VLA via Interactive World Model

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Peking University (State Key Laboratory of Multimedia Information Processing), Beijing Innovation Center of Humanoid Robotics, Nanyang Technological University, HKUST, UESTC |
| 日期 | June 2026 |
| 项目主页 | 未公开（论文未提供项目主页或代码链接） |
| 对比基线 | [[SafeVLA]] |
| 链接 | [arXiv](https://arxiv.org/abs/2606.20698) / Code: 未开源 |

---

## 一句话总结

> 在交互式视频世界模型的想象空间中分别预测任务进度与安全代价，用 Lagrangian 约束化 GRPO 训练出兼顾任务成功率与避碰安全的 VLA 策略。

---

## 核心贡献

1. **世界模型驱动的奖励-代价解耦评估**: 构建基于动作条件视频世界模型的交互式 rollout 机制，搭配双分支评估器，从想象出的未来帧/隐变量中分别估计任务奖励与安全代价。
2. **Lagrangian 安全约束 GRPO**: 提出在 [[GRPO]] 框架内对 reward advantage 和 cost advantage 分别归一化、再用自适应 Lagrange 乘子组合的约束化策略优化目标，避免固定惩罚系数的脆弱性。
3. **显著的安全-性能联合提升**: 在 [[SafeLIBERO]] 上同时取得最佳任务成功率、最佳安全成功率与最佳执行效率，Level I 安全成功率比最强基线 [[SafeVLA]] 提升 8.25 个百分点；真实 Franka 机械臂部署的 5 个任务上同样取得最佳平均成功率和安全成功率。

---

## 问题背景

### 要解决的问题

让 [[Vision-Language-Action Model|VLA]] 策略在开放世界、障碍物丰富的环境中学会**避碰安全的操作行为**，同时不牺牲任务完成率。这要求策略在执行前就能预判潜在碰撞，而不是事后补救——因为现实世界中一次碰撞就可能导致任务失败、硬件损坏或环境破坏。

### 现有方法的局限

- **推理时控制**（如 [[CBF]] 安全层）：提供形式化安全保证，但需要精确的几何建模和相机标定，难以扩展到处理高维视觉观测、面对开放世界未知障碍配置的 VLA。
- **Model-free 约束 RL**（如 PPO-Lagrangian、[[SafeVLA]]）：通过惩罚项强制安全，但需要真实环境中的探索试错，无法在违规发生前进行预判（non-anticipatory）。
- **既有 model-based 安全 RL**：多面向低维状态空间，依赖仿真器提供的手工设计标量代价函数，既不能直接迁移到 VLA 的高维视觉观测，也无法学习视觉安全信号。

### 本文的动机

任务进度和避碰是**两个可能冲突的目标**：一条轨迹可能在推进任务的同时发生碰撞，也可能完全无碰撞却没完成任务。SafeDojo 的核心思路是：既然真实世界探索代价高且危险，那就在交互式世界模型的"想象空间"里完成策略改进的探索过程，并把任务奖励和安全代价作为两路独立信号分别估计，再用显式约束优化将二者协调起来。

---

## 方法详解

### 模型架构

SafeDojo 建立在 [[Constrained MDP|CMDP]] 形式化之上，将标准 VLA 的 [[Markov Decision Process|MDP]] $\mathcal{M}=(\mathcal{S},\mathcal{A},P,r,\rho_0,\gamma)$ 扩展为带安全代价的 $\mathcal{M}^c=(\mathcal{S},\mathcal{A},P,r,c,\rho_0,\gamma)$，要求策略 $\pi_\theta$ 在满足安全预算 $J^c(\pi_\theta)\le d$ 的前提下最大化任务回报 $J^r(\pi_\theta)$。

整体流程分三步：
- **输入**: 初始视觉观测 $o_0$ + 语言指令 $l$，当前 VLA 策略 [[OpenVLA-OFT]] 采样一组候选动作轨迹 $\{\tau_i\}_{i=0}^{G-1}$
- **Rollout 引擎**: 基于 [[Wan 2.2]] 改造的交互式视频世界模型 $p_\phi$，在想象空间中逐 chunk 滚动生成未来帧与隐变量
- **核心模块**: 双分支评估器——[[ResNet|ResNet]] 微调的任务进度分类器 + 轻量级安全代价头，分别输出解耦的任务奖励与安全代价
- **优化目标**: 基于 [[GRPO]] 的 Lagrangian 安全约束化扩展，用自适应乘子 $\eta$ 平衡奖励与代价的 advantage
- **输出**: 经安全约束优化后的 VLA 策略 $\pi_\theta^*$，每个 [[Action Chunking|动作块]] 包含 $H=8$ 个控制步，每步为 7-DoF 指令（6-DoF delta 末端位姿 + 夹爪动作），遵循 [[OpenVLA-OFT]] 的动作表示

### 核心模块

#### 模块1: Chunk-wise 交互式世界模型 Rollout

**设计动机**: 避免在真实物理环境中采集策略 rollout 带来的风险与成本，转而在 [[World Model|世界模型]]的想象空间中做 model-in-the-loop 探索。

**具体实现**:
- 在 [[Wan 2.2]] 基础上加入 cross-attention 动作条件机制，构建动作条件视频预测器 $p_\phi$
- 每个候选轨迹包含 $K$ 个动作块，世界模型递归地以 $(\hat o_{i,k}, \hat z_{i,k}, a_{i,k}, l)$ 为条件预测下一时刻的 $(\hat o_{i,k+1}, \hat z_{i,k+1})$
- 针对 chunk-wise 自回归预测中训练-测试分布不匹配（exposure error）问题，训练时引入**静态视频增强**：随机冻结条件帧，让模型适应不完美的中间预测结果

#### 模块2: 解耦的任务奖励与安全代价估计

**设计动机**: 任务进度和碰撞风险并不总是对齐的——把两者揉成一个标量奖励会丢失结构信息，不利于后续的约束化优化。

**具体实现**:
- 任务奖励分支 $f_{\text{task}}$：用按帧成功标签微调的 ResNet 成功分类器，从想象帧 $\hat o_{i,k(h)}$ 和指令 $l$ 中估计逐步任务进度 $r_{i,h}$
- 安全代价分支 $f_{\text{safe}}$：紧凑卷积隐变量编码器 + 动作条件 MLP 头，从隐变量 $\hat z_{i,k(h)}$ 和动作块 $a_{i,h}$ 中预测逐步障碍接触概率 $c_{i,h}$
- 聚合时，任务奖励取全程平均；安全代价取 Top-$M$（$M=16$）高风险步的平均值，突出最危险时刻而非被低风险步稀释

#### 模块3: Lagrangian 安全约束 GRPO

**设计动机**: 简单的 reward shaping（$\tilde r_i = r_i - \lambda c_i$）把安全当作固定惩罚项，对人工设定的权重 $\lambda$ 高度敏感，且不能显式保证安全预算被满足。

**具体实现**:
- 借鉴 Constrained GRPO 思路，将 reward 和 cost 分别做组内归一化得到 $\hat A_i^r$、$\hat A_i^c$，保留各自的相对结构
- 用非负 Lagrange 乘子 $\eta$ 将两路 advantage 线性组合为 $\hat A_i^{\text{safe}}$
- $\eta$ 根据 batch 级安全约束违反程度自适应更新：违反时增大 $\eta$ 加重惩罚，满足时减小 $\eta$ 释放任务优化空间

---

## 关键公式

<!-- 公式标题使用 [[概念|名称]] 格式链接到概念库 -->

### 公式1: [[Constrained MDP|约束 MDP 形式化]]

$$
\mathcal{M}^{c}=(\mathcal{S},\mathcal{A},P,r,c,\rho_{0},\gamma), \qquad c:\mathcal{S}\times\mathcal{A}\rightarrow[0,1]
$$

$$
J^{r}(\pi_{\theta})=\mathbb{E}_{\pi_{\theta}}\Big[\sum_{t}\gamma^{t}r_{t}\Big] \quad \text{s.t.} \quad J^{c}(\pi_{\theta})=\mathbb{E}_{\pi_{\theta}}\Big[\sum_{t}\gamma^{t}c_{t}\Big]\leq d
$$

**含义**: 把标准 VLA 的 MDP 扩展为带安全约束的 CMDP，策略需要在累计安全代价不超过预算 $d$ 的前提下最大化累计任务回报。

**符号说明**:
- $r:\mathcal{S}\times\mathcal{A}\rightarrow\{0,1\}$: 任务奖励指示函数
- $c:\mathcal{S}\times\mathcal{A}\rightarrow[0,1]$: 安全代价函数（碰撞风险）
- $d$: 允许的安全预算
- $\gamma$: 折扣因子

### 公式2: [[GRPO|GRPO 原始目标]]

$$
\mathcal{L}_{\mathrm{GRPO}}=-\mathbb{E}_{i,t}\left[\min\left(\rho_{i,t}(\theta)\hat{A}_{i},\ \mathrm{clip}\left(\rho_{i,t}(\theta),1-\epsilon,1+\epsilon\right)\hat{A}_{i}\right)\right]
$$

**含义**: SafeDojo 的基础优化算法，无需价值网络，通过组内相对奖励归一化得到 advantage 并做 PPO 式 clip 更新。

**符号说明**:
- $\hat{A}_{i}=(R_{i}-\mu_{G}(R))/(\sigma_{G}(R)+\epsilon)$: 组内归一化优势
- $\rho_{i,t}(\theta)=\pi_{\theta}(a_{i,t}\mid s_{i,t})/\pi_{\theta}^{\mathrm{old}}(a_{i,t}\mid s_{i,t})$: 新旧策略概率比
- $G$: 每组采样的候选轨迹数（$G=16$）
- $\epsilon$: clip 范围超参数

### 公式3: 世界模型逐 Chunk 滚动（基于 [[World Model|交互式世界模型]]）

$$
(\hat{o}_{i,k+1},\hat{z}_{i,k+1})=p_{\phi}(\hat{o}_{i,k},\hat{z}_{i,k},a_{i,k},l),\quad k=0,\ldots,K-1
$$

**含义**: 交互式世界模型以当前想象观测、隐变量、动作块和语言指令为条件，递归生成下一个动作块结束时的观测和隐变量。

**符号说明**:
- $p_\phi$: 基于 [[Wan 2.2]] 构建、带 cross-attention 动作条件的视频世界模型
- $\hat o_{i,k}, \hat z_{i,k}$: 第 $i$ 条轨迹第 $k$ 个动作块结束时的想象观测和隐变量
- $K$: 每条轨迹的动作块数

### 公式4: 逐步奖励-代价估计（解耦任务奖励与安全代价）

$$
r_{i,h}=f_{\mathrm{task}}(\hat{o}_{i,k(h)},l),\qquad c_{i,h}=f_{\mathrm{safe}}(\hat{z}_{i,k(h)},a_{i,h})
$$

**含义**: 在每个控制步 $h$（而非仅在轨迹末尾）分别估计任务进度和安全代价，提供稠密反馈信号。

**符号说明**:
- $h$: 控制步索引；$k(h)=\lfloor h/H \rfloor$ 将控制步映射回所属动作块
- $f_{\mathrm{task}}$: 微调的 ResNet 成功分类器
- $f_{\mathrm{safe}}$: 卷积隐变量编码器 + 动作条件 MLP 安全头

### 公式5: 轨迹级奖励-代价聚合

$$
r_{i}=\frac{1}{KH}\sum_{h=1}^{KH}r_{i,h},\qquad c_{i}=\frac{1}{M}\sum_{c\in\mathrm{TopM}(\{c_{i,h}\}_{h=1}^{KH})}c
$$

**含义**: 任务奖励取全程逐步均值；安全代价只取最危险的 $M$ 个时间步均值，避免被大量安全步稀释风险信号。

**符号说明**:
- $K, H$: 动作块数与每块控制步数
- $M$: 参与安全聚合的高风险步数量，实验中取 $M=16$
- $\mathrm{TopM}(\cdot)$: 取集合中数值最大的 $M$ 个元素

### 公式6: Lagrangian 安全约束优势组合（基于 [[GRPO]] 的扩展）

$$
\hat{A}_{i}^{r}=\frac{r_{i}-\mu_{G}(r)}{\sigma_{G}(r)+\epsilon},\qquad \hat{A}_{i}^{c}=\frac{c_{i}-\mu_{G}(c)}{\sigma_{G}(c)+\epsilon}
$$

$$
\hat{A}_{i}^{\mathrm{safe}}=\hat{A}_{i}^{r}-\eta\hat{A}_{i}^{c}
$$

**含义**: 任务奖励和安全代价分别做组内归一化（保留各自相对结构），再用非负 Lagrange 乘子 $\eta$ 组合为统一的安全 advantage，区别于直接归一化预先加权的标量奖励。

**符号说明**:
- $\mu_G(\cdot), \sigma_G(\cdot)$: 同组 $G$ 条候选轨迹上的均值与标准差
- $\eta \ge 0$: 自适应 Lagrange 乘子，控制安全代价的相对权重

### 公式7: Lagrange 乘子自适应更新

$$
\eta\leftarrow\left[\eta+\alpha_{\eta}\left(\frac{1}{G}\sum_{i=1}^{G}c_{i}-d\right)\right]_{+}
$$

**含义**: 根据当前 batch 的平均安全代价是否超出预算 $d$ 来动态调整 $\eta$：超出预算时增大乘子加重惩罚，未超出则减小乘子释放任务优化空间。

**符号说明**:
- $\alpha_\eta$: 乘子学习率，实验中取 $0.05$
- $d$: 安全预算，实验中取 $0.2$
- $[\cdot]_+$: 投影到非负区间的算子

### 公式8: 最终 Safe-GRPO 损失

$$
\mathcal{L}_{\mathrm{Safe\text{-}GRPO}}=-\mathbb{E}_{i,t}\Big[\min\big(\rho_{i,t}(\theta)\hat{A}_{i}^{\mathrm{safe}},\ \mathrm{clip}(\rho_{i,t}(\theta),1-\epsilon,1+\epsilon)\hat{A}_{i}^{\mathrm{safe}}\big)\Big]+\beta D_{\mathrm{KL}}\big(\pi_{\theta}\|\pi_{\theta}^{\mathrm{ref}}\big)
$$

**含义**: 用安全约束后的优势 $\hat A_i^{\mathrm{safe}}$ 替换原始 GRPO 中的 $\hat A_i$，并保留可选的 KL 正则项约束到参考策略，构成 SafeDojo 的最终训练目标。实验中设 $\beta=0$，只依赖 GRPO 的 clip 机制做信任域约束。

**符号说明**:
- $\pi_\theta^{\mathrm{ref}}$: 参考策略（通常是 SFT 初始策略）
- $\beta$: KL 正则强度系数

---

## 关键图表

<!-- 图片：外链优先，找不到再本地下载 -->

### Figure 1: Overview of SafeDojo / 整体概览

![Figure 1](https://arxiv.org/html/2606.20698v1/x1.png)

**说明**: SafeDojo 用世界模型驱动的奖励-代价评估和安全 GRPO 增强 VLA 策略，在仿真与真实场景中同时提升安全成功率和执行效率。

### Figure 2: Detailed SafeDojo Pipeline / 详细流水线

![Figure 2](https://arxiv.org/html/2606.20698v1/x2.png)

**说明**: SafeDojo 完全在交互式视频世界模型内部优化 VLA 策略——将候选动作轨迹滚动为想象未来动态，解耦任务奖励与安全代价，再通过 Lagrangian 约束化 [[GRPO]] 联合优化，在不进行潜在危险真实 rollout 的前提下提升任务成功率并降低安全风险。

### Figure 3: Real-World Experiment Visualization / 真实世界实验可视化

![Figure 3](https://arxiv.org/html/2606.20698v1/figs/visual.png)

**说明**: SafeDojo 在真实任务中安全完成任务，而基线方法或任务失败、或违反安全约束、或只能依靠不安全接触才能成功。

### Figure 4: Ablation Studies / 消融实验

![Figure 4](https://arxiv.org/html/2606.20698v1/x3.png)

**说明**: 在 SafeLIBERO Level I Spatial Task 0 上的三组消融：(a) 逐一移除各组件；(b) 比较三种奖励-代价估计的输入来源配置；(c) 初始安全权重 $\eta$ 的敏感性分析。

### Figure 5: Representative SafeDojo Real-World Demos / 真实世界演示快照

![Figure 5](https://arxiv.org/html/2606.20698v1/figs/real_world_appendix_demos.png)

**说明**: 5 个真实世界任务上 SafeDojo 执行过程的代表性快照（附录图）。

### Table 1: SafeLIBERO Level I 量化结果

| Type | Method | TSR Spa. | TSR Goal | TSR Obj. | TSR Long | TSR Avg. | SSR Spa. | SSR Goal | SSR Obj. | SSR Long | SSR Avg. | ETS Spa. | ETS Goal | ETS Obj. | ETS Long | ETS Avg. |
|------|--------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|
| SFT | OpenVLA-OFT | 29.50 | 83.50 | 50.50 | 45.00 | 52.12 | 22.50 | 73.50 | 26.00 | **34.00** | 39.00 | 263.49 | 170.35 | 252.46 | 465.44 | 287.94 |
| CBF | AEGIS | 27.50 | 57.00 | 32.00 | 13.00 | 32.38 | 26.50 | 56.50 | 20.00 | 11.00 | 28.50 | 269.10 | 214.18 | 272.82 | 486.60 | 310.67 |
| Model-free RL | PPO | 35.00 | 84.00 | 48.50 | **49.00** | 54.12 | 28.00 | 73.00 | 27.00 | 31.50 | 39.88 | 253.67 | 170.84 | 256.92 | **461.24** | 285.67 |
| Model-free RL | SafeVLA | 41.00 | 83.50 | 54.50 | 38.00 | 54.25 | 34.50 | **79.50** | 35.00 | 31.00 | 45.00 | 236.70 | **166.97** | 249.98 | 525.10 | 294.69 |
| Model-based RL | WMPO | 54.00 | **85.50** | 63.50 | 39.50 | 60.62 | 41.00 | 71.50 | 37.50 | 29.50 | 44.88 | 217.74 | 167.18 | 238.82 | 472.32 | 274.02 |
| Model-based RL | WoVR | 53.50 | 77.00 | 29.00 | 43.00 | 50.62 | 40.00 | 65.50 | 14.00 | 31.00 | 37.62 | 216.98 | 183.23 | 273.17 | 478.56 | 287.99 |
| **Ours** | **SafeDojo** | **59.50** | 80.00 | **73.50** | 45.00 | **64.50** | **51.50** | 78.50 | **49.00** | 34.00 | **53.25** | **209.25** | 171.12 | **227.13** | 470.15 | **269.41** |

**说明**: TSR = Task Success Rate，SSR = Safe Success Rate，ETS = Execution Time Steps（越低越好）。SafeDojo 在平均 TSR、SSR 和 ETS 三项指标上全部最优，其中平均 SSR 比最强基线 SafeVLA 高 8.25 个百分点。

### Table 2: SafeLIBERO Level II 量化结果（泛化评估）

| Type | Method | TSR Spa. | TSR Goal | TSR Obj. | TSR Long | TSR Avg. | SSR Spa. | SSR Goal | SSR Obj. | SSR Long | SSR Avg. | ETS Spa. | ETS Goal | ETS Obj. | ETS Long | ETS Avg. |
|------|--------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|
| SFT | OpenVLA-OFT | 57.00 | 62.00 | 63.00 | 26.50 | 52.12 | 44.00 | 48.00 | 57.50 | 24.00 | 43.38 | 216.62 | 201.25 | 223.33 | 505.69 | 286.72 |
| CBF | AEGIS | 36.00 | 47.50 | 37.00 | 12.00 | 33.13 | 36.00 | 37.50 | 36.50 | 12.00 | 30.50 | 248.23 | 224.29 | 255.54 | **481.69** | 302.44 |
| Model-free RL | PPO | 57.50 | 64.50 | 62.00 | 24.50 | 52.12 | 47.00 | 48.50 | 57.50 | 20.00 | 43.25 | 217.10 | 195.35 | 222.30 | 507.28 | 285.51 |
| Model-free RL | SafeVLA | 68.50 | **67.50** | **67.00** | 23.00 | **61.00** | 55.50 | **62.50** | 19.00 | 49.50 | 49.50 | 194.68 | **189.60** | **221.31** | 553.87 | 289.87 |
| Model-based RL | WMPO | 72.00 | 64.50 | 59.50 | 27.00 | 55.75 | 58.50 | 48.00 | 54.00 | 24.00 | 46.12 | 188.63 | 192.27 | 228.23 | 499.56 | 277.17 |
| Model-based RL | WoVR | 66.00 | 62.50 | 51.00 | 21.00 | 50.12 | 60.00 | 51.00 | 45.50 | 19.50 | 44.00 | 201.03 | 192.72 | 233.31 | 516.00 | 285.76 |
| **Ours** | **SafeDojo** | **74.50** | 64.00 | 60.00 | **29.50** | 57.00 | **60.50** | 56.00 | 55.50 | **26.50** | **49.62** | **187.50** | 190.23 | 225.99 | 504.29 | **277.00** |

**说明**: 所有方法仅用 Level I 演示训练，直接在 Level II（障碍更远但沿运动路径）上评估泛化性。SafeDojo 取得最佳 TSR、最佳 SSR 和最低 ETS，说明想象 rollout 学到的安全信号能迁移到未见过的障碍配置。

### Table 3: 安全 RL 范式对比（附录 Table 3）

| Family | Representative methods | Anticipatory | Learned safety |
|--------|------------------------|--------------|-----------------|
| Lagrangian / CMDP | CPO, PPO-Lag, FOCOPS, SafeVLA | ✗ | ✗ |
| Control-theoretic | CBF-QP, AEGIS, Safe-Night VLA | ✓ | ✗ |
| Model-based (state-based) | SMBPO, SafeDreamer, FOSP | ✓ | ✗ |
| Gaussian process | Safe-GP, SAMBA | ✗ | ✗ |
| **Model-based (video)** | **SafeDojo (ours)** | **✓** | **✓** |

**说明**: SafeDojo 是表中唯一同时满足"预判性"（事前避免违规）和"安全信号可学习"（而非手工设计）两个轴的方法，作者认为这正是开放世界 VLA 部署所需要的组合。

### 消融实验关键发现（Figure 4 对应文字结果）

| 配置 | TSR (%) | SSR (%) | ETS | 说明 |
|------|---------|---------|-----|------|
| Full Model (η 固定 0.3) | 60.0 | 48.0 | 203.0 | 完整方法 |
| w/o 安全头 | — | 38.0 | — | 安全代价估计对安全成功率至关重要 |
| w/o 任务进度奖励分支 | 3.0 | 3.0 | 298.3 | 任务奖励反馈是目标导向改进的必要条件 |
| GRPO → 无约束优化 | 40.0 | 34.0 | — | 安全约束化更新的重要性得到验证 |
| w/o 自适应 Lagrange 乘子 | 68.0 | 46.0 | — | 自适应平衡奖励/代价对提升安全性而不牺牲任务完成必要 |
| 仅用 future 预测作为输入 | 52.0 | 36.0 | 217.3 | 仅靠未来想象状态不足以稳定评估任务进度与安全 |
| 当前观测+动作+future 预测 | 58.0 | 50.0 | 216.1 | 累积预测误差可能引入噪声评估信号 |
| 当前观测+动作（默认配置） | 60.0 | 48.0 | 203.0 | 在三种输入配置中取得最佳综合表现 |
| η=0.1 | 46.0 | 38.0 | 222.0 | 安全权重不足，未能充分抑制风险动作 |
| η=0.2（默认） | 74.0 | 58.0 | 186.3 | 任务完成、安全执行与效率的最佳平衡点 |
| η=0.6 | 36.0 | 26.0 | 244.4 | 安全权重过大导致策略过度保守，损害效率 |

**关键发现**: 安全头和任务奖励分支都是不可或缺的（去掉任一个都会显著掉点，去掉任务奖励分支后 TSR/SSR 几乎归零）；约束化 GRPO 优于无约束优化；当前观测+动作的输入配置优于引入未来预测（误差累积问题）；初始安全权重 $\eta=0.2$ 是效果最佳的折中点。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| [[SafeLIBERO]] | 4 个任务套件（Spatial/Goal/Object/Long）× 4 个任务 × 2 个障碍难度等级 | 基于 [[LIBERO]] / [[Robosuite]] 构建的安全关键基准，单臂 [[Franka Emika Panda]] 7-DoF 操作 | 训练（仅 Level I 演示）+ 测试（Level I & II） |
| 真实世界 Franka 任务 | 5 个任务（4 个单臂对应 SafeLIBERO 套件 + 1 个双臂协同任务） | 物理机器人部署验证 | 真实世界测试 |

### 实现细节

- **Backbone**: [[OpenVLA-OFT]]，离散动作 token 输出，所有 baseline 共享同一 SFT checkpoint
- **世界模型**: 基于 [[Wan 2.2]] 微调，1.5K SafeLIBERO 轨迹，5000 步优化，学习率 $1\times10^{-5}$
- **任务奖励分类器/安全代价头**: 各训练 20 epoch，学习率分别为 $1\times10^{-4}$ 和 $3\times10^{-4}$
- **策略优化**: 安全约束 GRPO，组大小 $G=16$，学习率 $2\times10^{-5}$，安全预算 $d=0.2$，乘子学习率 $\alpha_\eta=0.05$
- **评测指标**: TSR（任务成功率）、SSR（安全成功率，要求无碰撞完成任务，比 TSR 更严格）、CAR（无碰撞完成的episode占比）、CSC（平均接触步数）、ETS（平均执行步数，衡量效率）

### 可视化结果

真实世界演示（Figure 3、Figure 5）显示 SafeDojo 生成的轨迹既直接又具有避碰意识，避免不必要绕路；基线方法则表现出与障碍物碰撞、安全过滤后过度保守、或无法从不安全中间状态恢复等问题。

---

## 批判性思考

### 优点
1. 把"任务进度"和"安全代价"解耦为两路独立信号，而不是塞进一个手工权重的标量奖励，使约束优化更稳健、对超参数（如安全权重）的敏感度更低
2. 完全在世界模型的想象空间中完成策略改进的探索过程，理论上避免了真实环境中危险碰撞带来的硬件损坏风险，这对依赖真实试错的传统安全 RL 是质的提升
3. 安全成功率在 Level I → Level II 的跨域泛化实验中仅下降 3.63 个百分点且仍保持最优，说明学到的视觉安全信号具备一定的迁移能力，而非过拟合到训练时的障碍布局

### 局限性
1. 安全头只用二元碰撞标签训练，无法刻画更细粒度的安全需求（如接触力大小、与障碍物的距离余量），论文自己在 Appendix B 中也承认了这一点
2. 世界模型的预测误差会随 rollout horizon 增长累积，对长程任务的任务进度/安全评估可靠性是隐患；虽然用了静态视频增强缓解 exposure error，但没有给出长时序下误差增长的量化分析
3. SafeDojo 本质上仍是训练时机制，推理阶段没有硬安全保证（不像 CBF 那样有形式化证明），论文也明确指出这是未来需要与推理时安全层（如 CBF）结合的方向
4. 评测场景局限于桌面操作和静态障碍物，未覆盖动态障碍或人机共存场景，真实开放世界部署的安全性仍待验证
5. 论文未开源代码、未提供项目主页，复现难度较高（尤其是 1.5K 轨迹微调 Wan 2.2 这类重型世界模型的工程细节）

### 潜在改进方向
1. 将安全代价从二元碰撞标签扩展为连续风险估计（如预测接触力或安全余量），结合本文已有的逐步密集信号设计
2. 探索误差感知的世界模型训练或自适应 rollout 长度，缓解长程想象中的误差累积问题
3. 把 SafeDojo 学到的"软"安全信号与 [[CBF]] 等推理时"硬"安全层结合，形成训练时+推理时的双重防护

### 可复现性评估
- [ ] 代码开源（论文未提供 GitHub 链接）
- [ ] 预训练模型（未提供）
- [x] 训练细节完整（学习率、批大小、优化步数等超参数在正文和附录中给出）
- [x] 数据集可获取（[[SafeLIBERO]] / [[LIBERO]] 公开可用，但本文新增的安全标注、真实机器人数据未说明是否开放）

---

## 关联笔记

### 基于
- [[GRPO]]: SafeDojo 的策略优化算法基础，在此之上引入安全约束
- [[OpenVLA-OFT]]: 所有方法（包括 SafeDojo 和全部 baseline）共享的 VLA 策略骨干
- [[Wan 2.2]]: SafeDojo 交互式世界模型的基础视频生成模型

### 对比
- [[SafeVLA]]: model-free 安全约束 RL baseline，训练时安全感知优化但不具预判性；SafeDojo 是其 model-based、世界模型想象版的后续工作
- [[CBF]]: 推理时安全干预 baseline（AEGIS 方法），具有形式化保证但不可学习、依赖几何建模

### 方法相关
- [[Constrained MDP|CMDP]]: SafeDojo 的安全 RL 问题形式化基础
- [[World Model|世界模型]]: 提供想象空间中的安全策略改进能力
- [[Action Chunking]]: 动作输出的基本单位

### 硬件/数据相关
- [[Franka Emika Panda]]: 真实世界部署所用机械臂
- [[SafeLIBERO]] / [[LIBERO]] / [[Robosuite]]: 仿真训练与评测环境

---

## 速查卡片

> [!summary] SafeDojo: Safe Reinforcement Learning for VLA via Interactive World Model
> - **核心**: 在交互式视频世界模型的想象空间里解耦预测任务奖励与安全代价，用 Lagrangian 约束化 GRPO 训练安全 VLA 策略
> - **方法**: Wan2.2 改造的动作条件世界模型 + ResNet 成功分类器/安全代价头双分支评估 + 自适应 Lagrange 乘子的约束 GRPO
> - **结果**: SafeLIBERO Level I 安全成功率 53.25%，比最强基线 SafeVLA 高 8.25 个百分点；真实 Franka 五任务平均 TSR 76.00% / SSR 70.00%，均为最佳
> - **代码**: 未开源

---

*笔记创建时间: 2026-06-24*
