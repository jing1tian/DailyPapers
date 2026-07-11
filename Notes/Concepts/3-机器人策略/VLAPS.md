---
type: concept
aliases: [VLA Predictive Search, VLA-guided planning with tree search]
---

# VLAPS

## 定义

VLAPS（VLA + tree search planning）是将预训练 [[VLA（视觉-语言-动作模型）]] 策略的输出分布作为先验，结合 [[MCTS]] 在仿真环境中进行树搜索的机器人规划方法，由 Neary et al. 2025 提出。

## 数学形式

VLAPS 在 MCTS 节点选择中用 VLA 先验分布与 UCB 探索项的乘积评分候选动作块：

$$
U_{\text{VLAPS}}(v, a^i) = \psi_{\Phi_v}(a^i \mid I_t, L_\mathcal{T}) \times \frac{\sqrt{N(v, a^i)}}{1 + N(v, a^i)}
$$

其中 $\psi_{\Phi_v}$ 为节点 $v$ 处 VLA 对候选动作块的先验分布，$N(v, a^i)$ 为候选被选次数。

## 核心要点

1. 使用预训练 VLA（如 [[Octo]]）生成动作 chunk 候选，作为 MCTS 的先验策略
2. 在仿真器中执行候选动作块并推进树搜索
3. **缺点**：依赖策略自身的分布先验，缺乏显式的值函数来纠正策略偏差
4. V-VLAPS 在此基础上引入学习到的值头，实现值引导的树搜索

## 代表工作

- [[VLAPS]]: Neary et al. 2025，提出方法
- [[V-VLAPS]]: 在 VLAPS 基础上增加值函数头，实现值引导规划
