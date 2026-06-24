---
type: concept
aliases: [CMDP, 约束马尔可夫决策过程]
---

# Constrained MDP

## 定义
在标准 [[Markov Decision Process|MDP]] 基础上额外引入安全代价信号的形式化框架，记为 $\mathcal{M}^c=(\mathcal{S},\mathcal{A},P,r,c,\rho_0,\gamma)$，要求策略在累计安全代价不超过预算 $d$ 的前提下最大化任务回报，是安全强化学习问题的标准数学描述。

## 数学形式
$$
\mathcal{M}^{c}=(\mathcal{S},\mathcal{A},P,r,c,\rho_{0},\gamma), \qquad c:\mathcal{S}\times\mathcal{A}\rightarrow[0,1]
$$
$$
J^{r}(\pi_{\theta})=\mathbb{E}_{\pi_{\theta}}\Big[\sum_{t}\gamma^{t}r_{t}\Big] \quad \text{s.t.} \quad J^{c}(\pi_{\theta})=\mathbb{E}_{\pi_{\theta}}\Big[\sum_{t}\gamma^{t}c_{t}\Big]\leq d
$$

## 核心要点
1. 把任务奖励 $r$ 和安全代价 $c$ 作为两路独立信号显式分开，而非合并为单一标量奖励
2. 常见求解方式包括 Lagrangian 松弛（如 PPO-Lagrangian、CPO）将约束优化转化为对偶问题，用自适应乘子 $\eta$ 或 $\lambda$ 平衡两个目标
3. 是 model-free 安全 RL（如 [[SafeVLA]]）和 model-based 安全 RL（如 [[SafeDojo]]）共同采用的问题形式化基础，区别在于代价信号是来自真实环境探索还是世界模型想象

## 代表工作
- [[SafeDojo]]: 将标准 VLA 的 MDP 扩展为 CMDP，在交互式世界模型的想象空间中估计奖励与代价，用 Lagrangian 约束化 GRPO 求解
- [[SafeVLA]]: model-free 约束 RL 框架下的代表性安全 VLA 微调方法

## 相关概念
- [[Markov Decision Process]]
- [[GRPO]]
- [[SafeVLA]]
- [[CBF]]
