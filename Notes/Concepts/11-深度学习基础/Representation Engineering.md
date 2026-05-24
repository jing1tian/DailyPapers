---
type: concept
aliases: [表征工程, RepE]
---

# Representation Engineering

## 定义

通过直接操控神经网络的内部表征（激活、子空间）来控制模型行为的技术框架，无需修改模型权重。

## 核心要点

1. 将模型行为控制问题转化为激活空间中的几何问题
2. 主要手段：激活加法（[[CAA]]）、子空间投影（[[Conceptor]]）、稀疏特征分离（SAE）
3. 与 fine-tuning 互补：RepE 快速灵活，微调则更持久
4. Zou et al. (2023) 将其系统化，证明线性表征假设在 LLM 中的有效性

## 代表工作

- [[COAST]]: 将表征工程扩展到机器人 VLA 领域，通过 Conceptor 引导操作策略

## 相关概念

- [[Activation Steering]]
- [[CAA]]
- [[Conceptor]]
- [[Residual Stream]]
