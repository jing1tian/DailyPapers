---
type: concept
aliases: [残差策略, 残差修正器, 残差控制, Residual Corrector]
---

# Residual Policy

## 定义
在基础策略输出动作的基础上，叠加一个轻量级网络预测的残差修正量，以实现更精细的在线控制或异常恢复。

## 数学形式

$$
\mathbf{a}^{cor} = \mathbf{a}^{base} + \kappa \cdot \Delta\mathbf{a}^{res}
$$

其中 $\kappa \in \{0, 1\}$ 为二值门控，$\Delta\mathbf{a}^{res}$ 为残差修正量。

## 核心要点

1. **解耦设计**：残差网络独立训练，不影响基础策略，可叠加在任意已有策略上
2. **门控机制**：通常配合阈值化门控，仅在检测到异常时触发修正，避免干扰正常执行
3. **训练数据**：常从人类干预 / 专家纠正 episode 中学习，捕捉"什么情况下需要修正"
4. **执行频率**：可以高于基础策略（如基础 5 Hz，残差 10 Hz），实现更快的反应

## 代表工作

- [[FAWAM]]: 力引导残差修正器，以力跟踪误差为触发信号，三层 MLP 实现 10 Hz 闭环修正

## 相关概念

- [[Wrench Tracking Error]]
- [[Closed-Loop Control]]
- [[Action Chunking]]
- [[ForceVLA]]
