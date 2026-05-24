---
type: concept
aliases: [GR00T N1, NVIDIA GR00T, Isaac GR00T]
---

# GR00T

## 定义
NVIDIA 发布的通用人形机器人基础模型，采用 Mixture of Transformers（MoT）架构，分别处理语言/视觉（language/vision expert）和动作（action expert）。

## 核心要点
1. MoT 架构：language expert + action expert 分离，支持多任务并行推理
2. 训练数据覆盖多种机器人形态和操作任务
3. 支持 KV cache 共享优化（见 [[OxyGen]]）
4. 与 [[π₀]] 同属 MoT-VLA 范式

## 代表工作
- [[OxyGen]]: 针对 GR00T 的 KV cache 统一管理优化

## 相关概念
- [[VLA]]
- [[MoE]]
- [[Action Expert]]
