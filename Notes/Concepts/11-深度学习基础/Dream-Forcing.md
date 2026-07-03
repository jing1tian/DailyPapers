---
type: concept
aliases: [Dream Forcing]
---

# Dream-Forcing

## 定义

Dream Forcing 是一种训练技术，强制模型在训练时用自身预测的 rollout（"梦境"）而非真实观测作为输入，使训练分布与推理时的 autoregressive 模式对齐，减少 train-test distribution shift。

## 核心要点

1. 训练时用模型自身预测替代真实 ground truth 输入
2. 减轻 exposure bias（训练时见真实值，推理时用预测值的偏差）
3. 与 Teacher Forcing 相对：Teacher Forcing 用真实值，Dream Forcing 用预测值
4. 在 [[ABot-M0.5]] 的 WAM 框架中用于对齐 world model 的训练和推理

## 数学形式

训练时：$\hat{x}_{t+1} = f(x_t)$，下一步用 $\hat{x}_{t+1}$ 而非真实 $x_{t+1}$ 作为输入

## 相关概念

- [[WAM]]
- [[RSSM]]
- [[DreamerV3]]
