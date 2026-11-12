---
type: concept
aliases: [ICLR, 交互中心潜推理, Interaction-Centric Latent Expert]
---

# Interaction-Centric Latent Reasoning

## 定义
在 [[世界动作模型（WAM）]] 的世界专家与动作专家之间插入轻量潜空间专家，以末端执行器轨迹曲线 + 目标物体 bounding box 作为离线视觉监督目标，使中间表征聚焦于局部机器人–物体交互区域而非全局场景语义。

## 数学形式

$$
\mathcal{L}_\text{lat} = 1 - \frac{1}{M} \sum_{i=1}^{M} \cos(\hat{z}_i,\, z_i^{\text{GT}})
$$

其中 $z_i^\text{GT}$ 由冻结 [[SigLIP]] 对叠加末端轨迹曲线 + 目标框图像进行 patch 聚合得到，$\hat{z}_i$ 为 Latent Expert 的预测输出。

## 核心要点

1. 监督信号可从标注数据自动生成：末端执行器轨迹由运动学正解得到，2D 投影需相机外参；目标框由 Grounding DINO 在抓取/释放事件时刻自动选取。
2. Latent Expert 由 [[Janus]] 初始化，使用可学习 query token（M=3，每段 1 个）条件化于世界专家输出。
3. 冻结 [[SigLIP]] encoder，训练 MLP 聚合头提取交互注释图像特征，避免破坏视觉先验。
4. 通过 blockwise 因果注意力掩码，潜专家可访问世界专家的 Key-Value 但不回传梯度。

## 代表工作

- [[LaST0.5]]: 提出 ICLR，在 [[RLBench]] 达 88.0% mean S.R.，[[LIBERO]] 达 99.5%

## 相关概念

- [[世界动作模型（WAM）]]
- [[Mixture-of-Transformers]]
- [[可供性（Affordance）]]
- [[Future Latent Alignment]]
- [[Action Expert]]
- [[SigLIP]]
