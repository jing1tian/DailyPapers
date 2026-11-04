---
type: concept
aliases: [选择遗憾, 计划选择遗憾, Plan Selection Regret]
---

# Selection Regret

## 定义

在一组候选轨迹中，模型实际选出的计划的真实代价，减去该候选池中代价最优计划的真实代价，用于衡量轨迹打分/排名机制的选择质量。

## 数学形式

$$
\text{SelectionRegret} = J_\text{true}(\hat{\tau}^*_\text{model}) - \min_{\tau \in \mathcal{P}} J_\text{true}(\tau)
$$

其中：
- $\hat{\tau}^*_\text{model}$：模型根据预测代价选出的最优计划
- $\mathcal{P}$：候选轨迹池
- $J_\text{true}(\cdot)$：模拟器/真实环境中的代价（非预测代价）

## 核心要点

1. **测量排名质量而非执行质量**: 选择遗憾评估的是在固定候选池中选出好计划的能力，不涉及执行后的闭环反馈
2. **Selection Optimism（选择乐观性）**: 相关指标 $J_\text{true}(\hat{\tau}^*) - J_\text{predicted}(\hat{\tau}^*)$，衡量模型对选出计划的代价预测有多乐观（系统性低估真实代价）
3. **与控制覆盖率的关系**: 选择遗憾降低（排名改善）不必然导致控制性能提升——更好的选择需配合有效的闭环执行

## 代表工作

- [[pico-WAM]]: 在1,911个共享状态上评估256候选；反事实分支训练将选择遗憾从7.56降至2.77（−4.79），但计划仍系统性高估实际进展（优化后 predicted 10.9px vs. realized 1.2px）

## 相关概念

- [[Counterfactual Branch Training]]
- [[MPC]]
- [[世界动作模型（WAM）|World Action Model]]
