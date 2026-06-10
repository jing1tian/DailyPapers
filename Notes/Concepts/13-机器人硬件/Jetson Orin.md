---
type: concept
aliases: [Jetson AGX Orin, Orin Nano, NVIDIA Jetson Orin, Jetson]
---

# Jetson Orin

## 定义

NVIDIA Jetson Orin 系列是专为边缘 AI 应用设计的嵌入式计算模块，结合 ARM CPU 与 Ampere GPU，常用作机器人板载计算平台。

## 核心要点

1. **型号层级**:
   - **AGX Orin**: 高性能型，32GB / 64GB 内存，最高 275 TOPS
   - **Orin NX**: 中端型，16GB 内存
   - **Orin Nano**: 入门型，8GB 内存，约 40 TOPS
2. **机器人部署关键参数**:
   - Orin Nano 8GB 是 VLA 部署的"最低配置基准"
   - GR00T-N1.6/N1.7（6GB+ 权重）无法在 Orin Nano 上运行
   - SmolVLA / BitVLA / Evo-1 可在 Orin Nano 运行（需 2–2.5 GiB）
3. **延迟基准**（来自 [[vla.cpp]]）:
   - SmolVLA: AGX Orin 65ms / Orin Nano 142ms
   - π₀: AGX Orin 28ms / Orin Nano 39ms（需 simulator offload）

## 代表工作

- [[vla.cpp]]: 系统性测试 RTX 3060 / AGX Orin / Orin Nano 三个硬件层级的 VLA 推理延迟与内存占用

## 相关概念

- [[vla.cpp]]: 面向 Jetson Orin 的 VLA 推理运行时
- [[ALOHA]]: 常与 Jetson 配合使用的机器人平台
