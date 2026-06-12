---
type: concept
aliases: [Wan2.2, Wan2.2 TI2V, Wan 2.2]
---

# Wan2.2-TI2V-5B

## 定义

阿里巴巴推出的 5B 参数文本+图像引导视频生成（Text-Image-to-Video）扩散 Transformer 模型，采用 VAE（空间步长 $s=16$，时间步长 4，$C=48$ 通道）和 30 块 Transformer（hidden 3072，FFN 14336，24 头）。

## 数学形式

$$
\text{Transformer hidden dim} = 3072, \quad \text{FFN dim} = 14336, \quad \text{Heads} = 24, \quad \text{Blocks} = 30
$$

$$
\text{VAE: } W' = W/16, \; H' = H/16, \; T' = T/4, \; C=48
$$

## 核心要点

1. **大规模预训练**: 5B 参数，具备强大的视频生成先验
2. **VACE 块**: 专用条件注入块，可接入 ControlNet 风格侧分支
3. **潜空间特性**: $C=48$ 通道与 Mirage 潜空间记忆的特征维度直接对应
4. **Flow Matching 训练**: 使用 flow matching 目标函数

## 代表工作

- [[Mirage]]: 以 Wan2.2-TI2V-5B 为骨干，通过 ControlNet 侧分支注入潜空间记忆

## 相关概念

- [[VAE]]
- [[ControlNet]]
- [[LoRA]]
