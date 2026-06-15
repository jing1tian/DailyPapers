---
type: concept
aliases: [Cosmos-Predict 2.5, Cosmos Predict 2.5B]
---

# Cosmos-Predict-2.5

## 定义

NVIDIA Cosmos 系列的视频预测模型，基于视频 [[Diffusion Transformer]] 架构，在 World Action Model 场景中作为视频骨干使用。

## 核心要点

1. **规模**: 2B 参数，28 层 Video DiT
2. **输入分辨率**: 192×336（机器人操作场景）
3. **时序压缩**: 时序压缩比 4，5 个潜在视频帧
4. **设计目标**: 优化像素级重建质量，生成时序连贯的高质量视频预测

## 与 AGRA 的关系

Cosmos-Predict-2.5 的隐状态因优化像素重建而编码密集外观细节，不适合直接用于动作控制。[[AGRA]] 通过将第 8 层隐状态对齐至 [[DINOv2]] 语义特征来弥补这一缺陷。

## 代表工作

- [[AGRA]]: 以 Cosmos-Predict-2.5-2B 为视频骨干，添加 AGRA 对齐目标
- [[Cosmos3]]: Cosmos 系列相关工作

## 相关概念

- [[Cosmos-Predict]]
- [[Video Diffusion Model]]
- [[World Action Model]]
