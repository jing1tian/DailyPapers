---
type: concept
aliases: [Visual Encoder, 视觉编码器, Image Encoder, Vision Backbone]
---

# Visual Encoder（视觉编码器）

## 定义

视觉编码器是将原始图像映射到紧凑潜变量表示的神经网络模块，通常为预训练的大型视觉基础模型（如 DINO-V2、SigLIP、ViT），被机器人策略和世界模型广泛复用。

## 数学形式

$$
Z = E(x), \quad x \in \mathbb{R}^{H \times W \times 3},\; Z \in \mathbb{R}^{N_Z}
$$

## 核心要点

1. **共享复用**：多个 WAM 和 VLA 策略共享同一个公开预训练视觉编码器（如 DINO-V2、SigLIP）
2. **攻击暴露面**：共享编码器是可迁移[[对抗补丁攻击|对抗攻击]]的核心利用点——优化一次补丁即可迁移至所有使用同一编码器的策略
3. **[[Attention Mechanism|注意力]]瓶颈**：编码器中的注意力层使局部扰动能扩散至全局空间位置
4. **冻结使用**：在 WAM 策略中视觉编码器通常冻结，其权重公开可得

## 代表工作

- [[TAPDreamer]]: 仅利用公开视觉编码器（无目标策略访问）优化对抗补丁，将攻击成功率最大化

## 相关概念

- [[WAM]]
- [[Attention Mechanism]]
- [[VAE]]
- [[对抗补丁攻击]]
- [[TAPDreamer]]
