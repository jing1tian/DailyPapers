---
type: concept
aliases: [去同步化训练, Async Training, Desync Training, 异步训练对齐]
---

# Desynchronized Training（去同步化训练）

## 定义

去同步化训练是一种针对异步多分支策略的训练技巧：在训练时随机将慢速（高延迟）分支的输入向前偏移若干帧，使训练数据分布与部署时该分支输出陈旧的情况相匹配，从而弥合训练-部署差距。

## 数学形式

$$
\big(o^{\text{vlm}}, s^{\text{vlm}}\big) = \big(o_{t-\Delta}, s_{t-\Delta}\big), \quad \Delta \sim \mathcal{U}\{0, 1, \ldots, \Delta_{\max}\}
$$

$$
\Delta_{\max} = \left\lceil \frac{\Delta^{\max}_{\text{obs}}}{\tau_{\text{data}}} \right\rceil
$$

## 核心要点

1. **问题**: 同步训练时两分支输入同帧；部署时 VLM 分支输出必然陈旧，导致训练/部署分布不匹配
2. **解决方案**: 训练时随机将 VLM 分支输入偏移 $\Delta$ 帧，$\Delta$ 从 $[0, \Delta_{\max}]$ 均匀采样
3. **$\Delta_{\max}$ 设置**: 由实际测量的最大部署延迟 $\Delta^{\max}_{\text{obs}}$ 决定，硬件相关
4. **无需修改模型结构**: 只需在数据采样时引入时序偏移，不需要额外网络输入

## 效果（来自 DASH 实验）

| 配方 | 4B 仿真成功率 |
|------|------------|
| 异步推理（无去同步化训练） | 62.3% |
| 异步推理（有去同步化训练，即 DASH） | 69.5% |

## 代表工作

- [[DASH]]: 提出 Desynchronized Training，用于训练 VLM 冻结骨干 + DiT 动作头的异步策略

## 相关概念

- [[Asynchronous Inference]]
- [[Decoupled Policy]]
- [[Cross-Timescale Fusion]]
