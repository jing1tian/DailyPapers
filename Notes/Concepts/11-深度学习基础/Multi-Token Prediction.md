---
type: concept
aliases: [MTP, Multi-Token Prediction, 多 Token 预测]
---

# Multi-Token Prediction

## 定义

Multi-Token Prediction（MTP）是一种语言模型训练目标增强方法：在标准下一 token 预测的基础上，增加辅助头同时预测多个未来 token，提供更密集的训练信号，改善样本效率和模型对长程依赖的学习能力。

## 数学形式

$$
\mathcal{L}_{\text{MTP}} = \mathcal{L}_{\text{next}} + \sum_{k=2}^{K} w_k \cdot \mathcal{L}_k^{\text{aux}}
$$

其中 $\mathcal{L}_k^{\text{aux}}$ 为预测未来第 $k$ 个 token 的辅助损失。

## 核心要点

1. **密集监督**: 每个位置同时提供 $K$ 个预测信号，相比标准 next-token 预测信息利用率更高
2. **长程依赖**: 强制模型在当前表示中编码更远未来的信息
3. **推理加速**: 多头预测可用于投机解码（speculative decoding），加速推理
4. **轻量辅助头**: 通常采用共享骨干 + 独立预测头，额外训练开销可控

## 代表工作

- Meta（2024）: "Better & Faster Large Language Models via Multi-Token Prediction"
- [[NextForcing]]: 将 MTP 思想迁移到连续视频潜变量，提出 [[Multi-Chunk Prediction]]

## 相关概念

- [[Multi-Chunk Prediction]]: MTP 在连续视频域的对应扩展
- [[Teacher Forcing]]: MTP 所改进的标准训练范式
- [[Transformer]]: 主流 MTP 实现所基于的架构
