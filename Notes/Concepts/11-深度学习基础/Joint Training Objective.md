---
type: concept
aliases: [联合训练目标, 多任务损失, Multi-task Loss, Joint Loss]
---

# Joint Training Objective

## 定义
联合训练目标是将多个子任务的损失函数加权求和，在一次训练中同时优化多个目标，使模型的各个组件协同学习。

## 数学形式

PrismWAM 的联合目标：

$$
\mathcal{L} = \mathcal{L}_R + \mathcal{L}_D + \mathcal{L}_A
$$

其中每项为 [[Flow Matching]] 的噪声加权 MSE：

$$
\mathcal{L}_m = \mathbb{E}_{\sigma_m}\left[w(\sigma_m) \cdot \left\| \hat{v}_m - v_m \right\|^2 \right]
$$

## 核心要点

1. **多模态协同**: 各模态损失同时反向传播，使共享参数（如 $B_V$）同时服务多个目标
2. **权重平衡**: 各损失项权重的平衡至关重要，可通过等权重或任务特定权重调节
3. **梯度干扰问题**: 不同子任务的梯度可能相互冲突，需要架构设计（如 [[Chain Attention]]）配合
4. **稀疏 + 密集混合**: PrismWAM 中 RGB 稀疏目标监督与深度密集序列监督结合

## 代表工作
- [[PrismWAM]]: RGB 目标预测 + 深度序列预测 + 动作生成三路联合训练

## 相关概念
- [[Flow Matching]]
- [[Noise Weighting]]
- [[Mixture-of-Transformers]]
