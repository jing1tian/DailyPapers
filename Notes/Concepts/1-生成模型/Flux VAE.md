---
type: concept
aliases: [Flux VAE, FluxVAE]
---

# Flux VAE

## 定义
Flux VAE 是 Flux 生成模型中使用的变分自编码器，提供高质量的连续视觉潜码表示，相比 VQ-VAE 具有更强的表达能力。

## 核心要点
1. 输出连续潜码（区别于 VQ-VAE 的离散码本），保留更细粒度的视觉信息
2. 编码器部分可冻结用于视觉特征提取，无需重新训练
3. 在 AffordanceVLA 中替代原有 VQ-VAE，用于 Which2Act 的目标物体潜码编码

## 数学形式
$$
z_q = \text{Encoder}_{\text{FluxVAE}}(I_{crop})
$$

其中 $I_{crop}$ 为根据目标边界框裁剪的图像区域。

## 代表工作
- [[AffordanceVLA]]: 使用冻结 Flux VAE 编码器实现 Which2Act 物体定位

## 相关概念
- [[VAE]]
- [[VQ-VAE]]
- [[可供性预测]]
