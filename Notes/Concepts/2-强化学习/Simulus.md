---
type: concept
aliases: [Simulus world model, VQ-VAE world model for Atari]
---

# Simulus

## 定义

Simulus 是一种基于 VQ-VAE tokenizer 的 Transformer 世界模型，针对 Atari 100K 基准设计，是该基准上目前最强的 token-based 世界模型之一。

## 核心要点

1. **VQ-VAE Tokenizer**: 使用学习到的 VQ-VAE 将 Atari 帧编码为离散 token，而非简单的 patch lookup
2. **Atari 100K 性能**: IQM = 0.990，Optimality Gap = 0.412
3. **与 ITC 的关系**: ITC 以 Simulus 为基础，在其 VQ-VAE tokenizer 之上插入 OT 求解层，进一步提升到 IQM = 1.092

## 代表工作

- [[ITC]]: 在 Simulus 基础上进一步提升 Atari 100K IQM（1.092 vs. 0.990）

## 相关概念

- [[World Model]]
- [[ITC]]
- [[Transformer]]
