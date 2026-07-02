---
type: concept
aliases: [树状前缀分支, Prefix Tree, Tree-Structured Branching]
---

# Tree-Structured Prefix Branching

## 定义

Tree-Structured Prefix Branching 是 [[Shared-Prefix GRPO]] 的层级扩展：在 GRPO rollout 组构建时引入多个中间分叉点，形成树状 rollout 结构。只有最早的完全共享根段被掩码，后续部分共享段在其所属 cluster 分叉后保持可训练，从而减少过度掩码，让更多前缀段的动作块贡献到策略梯度。

## 核心要点

1. **动机**: [[Shared-Prefix GRPO]] 的平坦变体掩码整个前缀，丢弃接近阶段的可训练信号；树状分支减少过度掩码，保留部分共享前缀的学习信号
2. **分叉点位置**: $b_\ell = \lfloor P / 2^{k-\ell} \rfloor$（$\ell = 1, \ldots, k$），组大小 $G = 2^k$，在最大分叉深度 $P$ 内均匀排布
3. **掩码策略**: 只有最早的完全共享根段被掩码；后续每层 cluster 分叉后，该 cluster 内的段变为可训练（因为不同 cluster 可获得不同奖励和学习信号）
4. **batch size 补偿**: Prefix Tree 的有效比较组减少，需相应增大 batch size 以保持同等数量的有效学习组

## 稳定性权衡

- **优点**: 更强的峰值成功率，更快的策略损失下降（TurnOnSinkFaucet 消融实验）
- **缺点**: 更多前缀段参与策略更新，对分叉点配置、前缀长度和 rollout 随机性更敏感；训练稳定性不如平坦 [[Shared-Prefix GRPO]]

## 代表工作

- [[Z-1]]: 在 [[GRPO]] 后训练 flow-based VLA 模型时引入此方法

## 相关概念

- [[GRPO]]
- [[Shared-Prefix GRPO]]
- [[Action Chunking]]
