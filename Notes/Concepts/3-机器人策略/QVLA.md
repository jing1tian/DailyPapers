---
type: concept
aliases: [Quantized VLA]
---

# QVLA

## 定义
Quantized VLA：对 VLA 模型权重进行量化压缩，使其能在边缘设备上高效推理的方法类别。

## 核心要点
1. 对 VLA 的 LLM backbone 进行 PTQ 或 QAT 量化
2. 目标是降低内存占用和推理延迟
3. 挑战：动作 token 对精度敏感，激进量化会导致操控失败

## 代表工作
- [[ActQuant]]: sub-4-bit 的 action-guided 量化，使用 HSIC 保护关键权重

## 相关概念
- [[ActQuant]]
- [[OpenVLA]]
