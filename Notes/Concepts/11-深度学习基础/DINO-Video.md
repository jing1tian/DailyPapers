---
type: concept
aliases: [DINO-Video, DinoVideo]
---

# DINO-Video

## 定义

面向机器人场景的时序视频表示模型，基于 [[DINOv3]] 扩展块因果注意力（Block-wise Causal Attention）和 3D 旋转位置编码（[[Rotary Position Encoding|3D RoPE]]），在 500 万视频片段上预训练，具备显式因果时序感知能力。

## 核心要点

1. **骨干网络**: 在 [[DINOv3]]（ViT-L，303M 参数）基础上，将标准注意力替换为块因果注意力，使当前帧 token 只能注意到历史帧，捕获因果时序依赖。
2. **3D RoPE**: 在空间（patch 位置）和时间（帧序号）两个维度施加旋转位置编码，解耦空间和时序感知。
3. **训练数据**: 500 万视频片段（含机器人操作和人类活动），无需额外标注。
4. **LARYBench 表现**: Composite Robot 71.97（超越 V-JEPA 2 的 70.43），RoboCOIN 误差 0.20（最优），在等参数量下全面超越 [[DINOv3]] 和 V-JEPA 2。

## 数学形式

用于 VLA 时序感知蒸馏时的视频蒸馏损失：

$$
\mathcal{L}_{\text{video}} = \mathbb{E}\Bigl[\| \operatorname{Proj}_{\text{video}}(\mathbf{Q}_t) - \mathbf{Z}_t \|_F^2 + \| \operatorname{Proj}_{\text{video}}(\mathbf{Q}_{t+T}) - \mathbf{Z}_{t+T} \|_F^2\Bigr]
$$

其中 $\mathbf{Z}_t, \mathbf{Z}_{t+T}$ 是 DINO-Video 在因果前向传播下对当前帧和未来帧输出的 patch 级特征。

## 代表工作

- [[LingBot-VLA-2.0]]: 将 DINO-Video 作为时序教师模型，通过 Frobenius 范数损失对 VLA 动作查询进行时序感知蒸馏。

## 相关概念

- [[DINOv3]]：骨干网络
- [[Rotary Position Encoding]]：3D RoPE 的基础
- [[Knowledge Distillation]]：DINO-Video 在 VLA 中的应用方式
- [[VLA]]：蒸馏目标模型
