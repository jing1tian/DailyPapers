---
type: concept
aliases: [Tactile JEPA, 触觉 JEPA, 触觉潜在预测架构]
---

# Tactile-JEPA

## 定义

一种专为稀疏触觉信号设计的自监督预训练目标，在潜在空间中用部分可见触觉片段预测被遮盖 taxel 的完整表征，以避免直接重建背景零值带来的优化退化。

## 数学形式

$$
\mathcal{L} = \frac{1}{|M|D} \sum_{i \in M} \|\hat{z}_i - \mathrm{sg}(\bar{z}_i)\|_1 + \frac{\lambda}{|C|D} \sum_{i \in C} \|\hat{z}_i - \mathrm{sg}(\bar{z}_i)\|_1
$$

其中 $M$ 为被遮盖位置集，$C$ 为可见位置集，$D$ 为特征维度，$\lambda=0.5$，$\mathrm{sg}$ 为停止梯度。

## 核心要点

1. **潜在预测而非信号重建**：触觉信号高度稀疏（大多数 taxel 大多数时刻无信号），VAE / MAE 对零背景的重建会随数据增大而退化；潜在空间目标始终由有意义的接触特征驱动
2. **全片段管状掩码**：同一 taxel 在整段片段所有帧均被遮盖，迫使模型从空间邻域和时序演变中推断缺失接触
3. **掩码热身（Mask Warm-up）**：从 40% 线性增长至 80%，共 300 步；稀疏信号中高掩码率过早使用会让所有接触证据被遮盖，导致梯度爆炸
4. **多层次目标**：从目标编码器 4 个深度各取特征，分别 LayerNorm 后拼接，多尺度监督提升表征质量
5. **目标编码器 EMA**：目标编码器为上下文编码器的指数移动平均（梯度停止），防止特征塌缩

## 代表工作

- [[LingBot-Touch]]: 首次将 JEPA 目标适配到异构触觉传感器，提出 Tactile-JEPA 名称

## 相关概念

- [[Joint Embedding Predictive Architecture]]: 原始 I-JEPA、V-JEPA 框架
- [[V-JEPA 2.1]]: 多层次监督目标的来源
- [[Masked Autoencoding]]: 对比方法（信号重建），触觉场景下性能更差
- [[Haptic Transformer]]: Tactile-JEPA 训练的共享 Transformer 骨干
- [[指数移动平均]]: 目标编码器更新机制
