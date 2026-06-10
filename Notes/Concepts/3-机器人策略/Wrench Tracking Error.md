---
type: concept
aliases: [力跟踪误差, 扳手跟踪误差, Force Tracking Error]
---

# Wrench Tracking Error

## 定义
机器人执行过程中，实时测量的末端力/力矩值与策略预测的期望力/力矩值之间的偏差，用于检测接触异常并触发闭环修正。

## 数学形式

$$
\mathbf{e}_{t-H_e+1:t}^w = \mathbf{w}_{t-H_e+1:t}^{meas} - \mathbf{w}_{t-H_e+1:t}^{pred}
$$

其中 $H_e$ 为误差历史窗口长度，$\mathbf{w}^{meas}$ 为实测力，$\mathbf{w}^{pred}$ 为策略预测力。

## 核心要点

1. 正常执行时误差应接近零；出现接触偏差（如末端位置漂移、障碍阻挡）时误差会显著增大
2. 作为残差修正器的关键输入，提供"什么时候需要修正"的信号
3. 需要策略在动作预测的同时联合预测力轨迹，否则无参考值可比较

## 代表工作

- [[FAWAM]]: 将力跟踪误差序列输入三层 MLP 残差修正器，检测异常并触发实时修正

## 相关概念

- [[Force Feature Encoding]]
- [[Residual Policy]]
- [[Closed-Loop Control]]
- [[FACTR]]
