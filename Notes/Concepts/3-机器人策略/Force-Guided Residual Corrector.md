---
type: concept
aliases: [力引导残差修正器, FGRC]
---

# Force-Guided Residual Corrector

## 定义
FAWAM 的在线执行模块：以力跟踪误差（实测力 vs 预测力）为触发信号，通过三层 MLP 预测残差动作修正量，以 10 Hz 频率实现闭环力反馈控制。

## 数学形式

$$
\Delta\mathbf{a}^{res},\; g^{res} = \text{Res}_\phi(\mathbf{e}^w,\; \mathbf{c}_t,\; \mathbf{a}^{base},\; \mathbf{w}^{pred},\; \mathbf{q}_t)
$$

$$
\mathbf{a}^{cor} = \mathbf{a}^{base} + \kappa \cdot \Delta\mathbf{a}^{res}, \quad \kappa = \mathbb{1}[g^{res} > \tau]
$$

## 核心要点

1. **触发条件**：力跟踪误差 $\mathbf{e}^w$ 显著时（门控 $g^{res} > \tau = 0.8$），激活残差修正
2. **训练来源**：从 FACTR 力反馈遥操采集的 20 条人类干预 episode 中学习
3. **平衡采样**：50% 干预 + 50% 正常样本，保留基础策略能力
4. **轻量级实现**：三层 MLP，可以 10 Hz 实时运行，延迟极低

## 代表工作

- [[FAWAM]]: Force-Guided Residual Corrector 是 FAWAM 的在线闭环控制核心

## 相关概念

- [[Wrench Tracking Error]]
- [[Residual Policy]]
- [[Closed-Loop Control]]
- [[FACTR]]
- [[Force-Envisioned Action Model]]
