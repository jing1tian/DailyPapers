---
type: concept
aliases: [时间集成, 时序集成, Temporal Ensemble]
---

# Temporal Ensembling

## 定义

在机器人策略推理时，对历史多步预测结果做加权平均以获得当前执行动作的技术，权重随预测滞后程度指数衰减。

## 数学形式

$$
\hat{a}_{t} = \frac{\sum_{i=0}^{B-1} w_{i} \, a_{t}^{[t-i]}}{\sum_{i=0}^{B-1} w_{i}}, \quad w_{i} = e^{\lambda i}
$$

其中 $a_t^{[t-i]}$ 是在 $t-i$ 时刻对当前步 $t$ 的预测，$\lambda < 0$ 时越近的预测权重越大。

## 核心要点

1. **平衡稳定性与响应速度**: $\lambda$ 越负则越依赖最新预测（响应快但可能抖动），越接近 0 则越平均（稳定但滞后）
2. **缓解 Action Chunking 抖动**: 动作块之间的拼接往往产生突变，时间集成平滑过渡
3. **计算开销极低**: 仅需维护一个大小为 $B$（动作步长）的滑动缓冲区

## 代表工作

- [[ACT]]: 首次提出并系统研究 Temporal Ensembling 在 Transformer 策略中的应用
- [[FurnitureVLA]]: 在双臂家具装配中验证最优 $\lambda = -0.1$

## 相关概念

- [[Action Chunking]]: 时间集成的应用前提，输出动作块
- [[Diffusion Policy]]: 常与时间集成结合使用
