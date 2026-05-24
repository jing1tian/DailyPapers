---
type: concept
aliases: [Orthogonal Fine-Tuning, 正交微调]
---

# OFT

## 定义
正交微调方法，用正交矩阵乘法替代 LoRA 的低秩分解，对预训练权重施加正交变换，保留权重的超球面结构（hyperspherical energy），防止过拟合和语义漂移。

## 数学形式
$$W' = R \cdot W_0, \quad R^\top R = I, \quad R \in \mathbb{R}^{d \times d}$$

实际中对权重矩阵分块做正交变换，降低计算量。

## 核心要点
1. 正交变换保持向量间的角度关系（超球面结构）不变
2. 不改变权重的 Frobenius 范数，对预训练语义保护更好
3. 参数量与 LoRA 相近，但通过正交约束提升泛化
4. BehaviorVLA 在跨环境 VLA 训练中使用 OFT 做参数高效微调

## 代表工作
- [[OFT]]：Qiu et al. 2023，ICLR 2024
- [[BehaviorVLA]]：跨分布 VLA 行为表征中使用 OFT

## 相关概念
- [[LoRA]]
- [[DoRA]]
- [[正交约束]]
