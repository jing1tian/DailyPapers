---
type: concept
aliases: [EmbodiedCpp, Embodied Inference Runtime]
---

# Embodied.cpp

## 定义
一个可移植的 C++ 推理运行时框架，支持 VLA 和 WAM 类 embodied AI 模型在异构机器人硬件（多 NPU/GPU/CPU 混合边缘设备）上的闭环部署。

## 核心要点
1. **统一 API**：同时支持 VLA 和 WAM 两种模型家族，提供统一接口
2. **Multi-Rate 执行**：支持闭环控制中不同频率的感知-动作流，满足 embodied 部署的 latency-first 要求
3. **异构硬件支持**：针对工控机、ARM 芯片、多 NPU 板卡等非标准硬件
4. 支持 OpenVLA、LaWAM、LingBot-VA 等多个模型

## 核心问题解决
现有推理框架（TensorRT、vLLM 等）针对 request-response 服务设计，不满足 embodied 部署的闭环控制要求（multi-rate、latency-first、real-time feedback）。

## 代表工作
- Embodied.cpp: A Portable Inference Runtime of Embodied AI Models on Heterogeneous Robots (2026)

## 相关概念
- [[VLA]]
- [[WAM]]
- [[LaWAM]]
