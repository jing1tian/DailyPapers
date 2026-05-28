---
type: concept
aliases: [EfficientVLA, Efficient Vision-Language-Action]
---

# EfficientVLA

## 定义

EfficientVLA 是一个专注于 VLA 推理效率优化的方法，通过减少 visual token 数量、优化推理路径等方式降低 VLA 实时部署的计算成本。

## 核心要点

1. **推理效率优先**: 目标是在保持任务成功率的同时显著降低延迟
2. **Token 压缩**: 核心思路是减少处理的 visual token 数量
3. **VLA 部署瓶颈**: VLA 推理速度是实时机器人控制的关键瓶颈（控制频率要求通常 10-25 Hz）

## 相关概念

- [[FastV]]
- [[SparseVLM]]
- [[FitPrune]]
- [[DivPrune]]
- [[OpenVLA]]
