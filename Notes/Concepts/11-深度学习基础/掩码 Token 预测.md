---
type: concept
aliases: [Masked Token Prediction, MTP, Masked Prediction, 掩码预测]
---

# 掩码 Token 预测（Masked Token Prediction）

## 定义
一种自监督/生成学习方法：随机掩盖序列中部分 token，训练模型根据上下文预测被掩盖的 token。BERT 中首次大规模应用，后扩展到视觉和运动生成领域。

## 数学形式

$$
\mathcal{L} = -\sum_{t \in \mathcal{M}} \log p(z_t \mid z_{\setminus \mathcal{M}})
$$

其中 $\mathcal{M}$ 为被掩盖的位置集合，$z_{\setminus \mathcal{M}}$ 为未掩盖的 token。

## 核心要点

1. **迭代细化**: 生成时先预测所有掩码位置，再根据置信度迭代更新低置信度 token
2. **并行生成**: 与自回归逐步生成不同，掩码预测可并行解码，速度更快
3. **SONIC 应用**: 运动学规划器以起止帧为锚点，迭代填充中间运动 token
4. **置信度筛选**: $\text{Prob}(z_t) = \sigma(h)$ 估计 token 置信度，低置信度 token 被重新掩盖

## 代表工作

- [[SONIC]]: 运动学规划器核心机制，生成 0.8–2.4s 可变长度运动片段

## 相关概念

- [[有限标量量化]]: 提供被预测的离散 token 表示
- [[临界阻尼弹簧模型]]: 与掩码 token 预测配合生成根轨迹
