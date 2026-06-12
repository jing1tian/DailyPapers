---
type: concept
aliases: [Latent Consistency Model, 潜空间一致性模型]
---

# LCM

## 定义
Latent Consistency Model：通过一致性蒸馏（Consistency Distillation）把多步扩散模型压缩为 1-4 步推理，在潜空间（Latent Diffusion）上操作，大幅加速生成速度。

## 数学形式
$$\mathbf{x}_\theta(\mathbf{z}_t, t) \approx \mathbf{x}_\theta(\mathbf{z}_{t'}, t')$$
一致性约束：同一轨迹上任意两点 $(t, t')$ 的一致性函数输出相同。

## 核心要点
1. 基于 [[一致性蒸馏]]，教师模型是预训练的 DDPM/[[流匹配]] 模型
2. 1-4 NFE（Number of Function Evaluations）可完成生成，比原始扩散模型快 10-50×
3. 适合图像/视频生成加速，但在多模态（视频+动作）联合生成中需要 modality-aware 调整（见 [[Flash-WAM]]）
4. 对比 [[一致性蒸馏]]：LCM 专门针对潜空间扩散（Latent Diffusion），而非像素空间

## 代表工作
- Luo et al. (2023): LCM 原始论文
- [[Flash-WAM]]: 把 LCM 蒸馏扩展到 WAM 的视频+动作联合生成

## 相关概念
- [[一致性蒸馏]]
- [[扩散模型]]
- [[视频扩散模型]]
- [[Flash-WAM]]
