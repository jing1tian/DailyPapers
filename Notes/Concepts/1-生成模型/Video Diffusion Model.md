---
type: concept
aliases: [视频扩散模型, Video DiT, Video Diffusion Transformer]
---

# Video Diffusion Model

## 定义

将扩散模型扩展到视频域的生成模型，通过逐步去噪从高斯噪声生成时序连贯的视频序列，通常基于 [[Diffusion Transformer]]（DiT）架构。

## 核心要点

1. **时序建模**: 在空间注意力基础上增加时序注意力或 3D 注意力，捕获帧间一致性
2. **潜在压缩**: 通过 VAE 将视频压缩到潜在空间（时间+空间），降低计算量
3. **条件生成**: 支持文本、图像、动作序列等多种条件，用于机器人预测、视频编辑等
4. **表征局限**: 优化像素重建的隐状态编码密集外观细节，不适合直接用于动作控制（见 [[AGRA]]）

## 代表工作

- [[Cosmos-Predict-2.5]]: 2B 参数视频 DiT，在 WAM 中作为视频骨干
- [[AGRA]]: 揭示视频扩散模型表征的 action-grounding gap，提出 DINOv2 对齐方案
- [[World Action Model]]: 将视频扩散模型与动作解码器耦合的框架

## 相关概念

- [[Cosmos-Predict]]
- [[World Action Model]]
- [[Noise Conditioning]]
- [[Representation Alignment]]
