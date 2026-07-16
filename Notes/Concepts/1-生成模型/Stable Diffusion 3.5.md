---
type: concept
aliases: [SD3.5, Stable Diffusion 3.5, SD 3.5]
---

# Stable Diffusion 3.5

## 定义

Stability AI 推出的第三代扩散模型，采用 Multimodal Diffusion Transformer（MMDiT）架构，在文本-图像对齐和生成质量上显著优于前代，是 Orca 视觉 Readout 的解码器骨干。

## 核心要点

1. 基于 Flow Matching 训练，生成质量和采样速度均优于 DDPM 类方法
2. MMDiT 使文本和图像 token 在 Transformer 中联合建模，对齐更精准
3. 在 Orca 中通过 MLP Adapter（556.9M 参数）+ LoRA 接入，用于将世界隐状态解码为预测图像
4. 目标分辨率 768×768，聚焦物理合理性而非写实度

## 代表工作

- [[Orca]]: 以 SD3.5 作为视觉 Readout 解码器

## 相关概念

- [[Flow Matching]]
- [[DiT]]
- [[LoRA]]
- [[Sora]]
