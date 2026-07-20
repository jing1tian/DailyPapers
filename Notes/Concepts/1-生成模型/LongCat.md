---
type: concept
aliases: [LongCat-Video]
---

# LongCat

## 定义
一种支持长视频生成的 autoregressive 视频扩散模型，采用 causal 架构实现低延迟、任意长度视频输出。

## 核心要点
1. Autoregressive（AR）帧预测，逐段生成保证因果一致性
2. 支持无限长视频输出而无需重新训练
3. 常作为 few-step AR 视频生成领域的 baseline

## 代表工作
- [[OPSD-V]]: 在此基础上做 on-policy self-distillation 改进

## 相关概念
- [[DiT]]
- [[Self-Forcing]]
- [[CausVid]]
