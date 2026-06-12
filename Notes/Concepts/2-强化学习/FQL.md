---
type: concept
aliases: [Flow Q-Learning, 流Q学习]
---

# FQL

## 定义
Flow Q-Learning，将 flow matching 生成模型与 Q-value 约束结合的 offline RL 方法，用于从混合质量离线数据中提取最优策略。

## 数学形式
$$\pi^* = \arg\max_\pi \mathbb{E}_{a \sim \pi(\cdot|s)} [Q(s, a)] - \alpha \cdot D_{KL}(\pi \| \pi_\beta)$$

其中通过 flow 生成过程参数化策略，$\pi_\beta$ 是行为策略。

## 核心要点
1. 用 flow matching 参数化策略分布，比 DDPM-based 方法推理更快
2. Q 值作为隐式约束，避免 out-of-distribution 动作外推
3. ForesightFlow 在此基础上引入势能引导，进一步改进 ODE 轨迹
4. 对比 [[IDQL]]、[[AWR]] 等方法，在接触丰富的操作任务上有优势

## 代表工作
- [[ForesightFlow]]: 扩展 FQL 引入势能引导
- Zhendong Wang et al. (2023): Flow Q-Learning

## 相关概念
- [[CFM]]
- [[IDQL]]
- [[AWR]]
- [[offline RL]]
