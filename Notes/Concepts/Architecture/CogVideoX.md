---
type: concept
aliases: [CogVideoX, Cog Video X]
---

# CogVideoX

## 定义

CogVideoX 是由智谱 AI 开发的大规模文本到视频扩散 Transformer 模型，基于专家 Transformer（Expert Transformer）架构，在约 3500 万视频片段上预训练，具备高质量视频生成能力。

## 核心要点

1. 采用 3D VAE 对时序视频进行统一压缩，降低计算成本
2. 使用专家 Transformer（Expert Transformer）分离图像和文本的注意力机制
3. 在大规模视频数据上预训练，具备广泛的视觉先验
4. 在机器人操作领域常被用作世界模型的视频生成骨干（微调后）

## 数学形式

流匹配训练目标：

$$
\mathcal{L} = \mathbb{E}_{z_0, z_1, t}\left[\|v_\theta(z_t, t, c) - v^*(z_t, t)\|_2^2\right]
$$

## 代表工作

- [[GEM-4D]]: 以 CogVideoX 为视频骨干，通过几何蒸馏增强帧间对应一致性
- [[TesserACT]]: 在 CogVideoX 基础上联合生成 RGB、深度和法线

## 相关概念

- [[Diffusion Transformer]]
- [[Flow Matching]]
- [[视频生成世界模型]]
