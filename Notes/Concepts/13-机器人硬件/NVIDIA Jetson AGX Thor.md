---
type: concept
aliases: [Jetson AGX Thor, Jetson Thor, AGX Thor]
---

# NVIDIA Jetson AGX Thor

## 定义

NVIDIA Jetson AGX Thor 是 NVIDIA 面向人形机器人和高性能边缘 AI 设计的片上系统（SoC）模组，集成高性能 GPU（用于神经网络推理）与 CPU，主打高算力低功耗的边缘部署场景。

## 核心要点

1. **目标场景**：人形机器人机载计算，无需云端推理
2. **典型用途**：VLA 模型端侧推理（如 π0.5 动作生成 + 语言生成）
3. **算力约束**：相比数据中心 GPU（RTX 4090），内存和算力更有限，对推理优化要求更高
4. **OxyGen 测试**：[[OxyGen]] 在 Jetson AGX Thor 上实现 π0.5 端到端部署，将总推理延迟从 1030 ms 降至 393 ms

## 代表工作

- [[OxyGen]]: 在 Jetson AGX Thor 上部署 MoT VLA，验证统一 KV 管理在边缘设备的有效性

## 相关概念

- [[Multi-Task Parallelism]]
- [[KV Cache]]
- [[π0.5]]
