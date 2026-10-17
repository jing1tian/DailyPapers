---
type: concept
aliases: [VJEPA-2, Video Joint Embedding Predictive Architecture 2]
---

# VJEPA-2

## 定义
Meta 提出的视频 JEPA 第二代，学习视频的 joint embedding 预测表示，不依赖像素重建，擅长学习物理动态和动作结果。

## 数学形式
$$\hat{z}_{target} = g_\theta(z_{context}, \text{mask})$$

预测器 $g_\theta$ 在 latent space 预测被 mask 掉的 target patch 的表示，避免像素空间的 trivial 解。

## 核心要点
1. 纯 latent space 预测，不重建像素，表示更抽象
2. 视频帧间的 temporal masking 迫使模型学习动态
3. 适合作为 action-conditioned world model 的骨干

## 代表工作
- [[PAVXploreRL]]: 以 VJEPA-2 为骨干构建 PAV 三目标 world model
- [[VeriVLA]]: 冻结 VJEPA-2 作为潜在世界模型验证器的视觉编码骨干，配合动作条件预测器进行 Delta Progress 估计

## 相关概念
- [[JEPA]]
- [[LeWM]]
