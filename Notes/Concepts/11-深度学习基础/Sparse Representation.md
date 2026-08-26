---
type: concept
aliases: [sparse coding, 稀疏表示, Lp norm representation]
---

# Sparse Representation

## 定义
用少量非零元素的向量表示输入特征，通过 $L_p$（$p < 2$）范数约束来诱导稀疏性，兼顾信息压缩和可解释性。

## 数学形式
$$\min_z \|x - Dz\|_2^2 + \lambda \|z\|_p, \quad p < 2$$

对于 $p=1$（LASSO）稀疏解有闭式条件；$p < 1$ 更稀疏但优化非凸。

## 核心要点
1. 与稠密表示（isotropic Gaussian）相比，稀疏激活更具可解释性
2. 可通过 ReLU 或 RepReLU 等激活函数隐式实现
3. 能防止 JEPA 类架构的表示坍塌，同时保持紧凑表示

## 代表工作
- [[LpWM]]: 在 world model latent 空间引入稀疏表示

## 相关概念
- [[JEPA]]
- [[Joint Embedding Predictive Architecture]]
- [[LeWM]]
