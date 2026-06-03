---
type: concept
aliases: [FAST]
---

# FAST

## 定义
Frequency-based Action Sequence Tokenization，将机器人连续动作序列转为离散 token 的编码方法，用于 VLA 的动作解码头。

## 核心要点
1. 用 DCT（离散余弦变换）在频域压缩动作序列
2. 相比逐步预测，减少 token 数量，提升推理速度
3. 被 Discrete Diffusion VLA 等工作采用作对比

## 代表工作
- [[DD-VLA]]: 对比 FAST tokenization 方案

## 相关概念
- [[VLA（视觉-语言-动作模型）]]
- [[扩散策略]]
