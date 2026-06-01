---
type: concept
aliases: [BitCPM 4, BitCPM-4]
---

# BitCPM4

## 定义
MiniCPM4 的 INT4 量化版本，使用量化感知训练（QAT）在保持较高精度的同时将模型权重压缩到 4-bit，是 MiniCPM 系列在端侧部署的极限压缩方案。

## 核心要点
1. 基于 QAT 而非 PTQ，量化精度损失更小
2. 目标是在移动端/边缘设备上运行 LLM，INT4 量化可降低约 4× 内存占用
3. 需要与 MiniCPM4 的 ArkInfer 推理引擎配合使用

## 代表工作
- [[MiniCPM4]]: BitCPM4 是其 INT4 量化版本

## 相关概念
- [[QAT]]
- [[PTQ]]
- [[GPTQ]]
