---
type: concept
aliases: [JEPA, I-JEPA, V-JEPA, joint embedding predictive]
---

# Joint Embedding Predictive Architecture

## 定义
LeCun 提出的自监督学习框架：在隐空间中预测未来/目标表示，而非在像素空间重建，从而学到紧凑的抽象表示。

## 数学形式
$$\mathcal{L}_\text{JEPA} = \|f_\theta(z_x) - sg(f_\phi(z_y))\|_2^2$$

其中 $z_x$ 是上下文编码，$z_y$ 是目标编码，$sg$ 表示停止梯度，防止坍塌。

## 核心要点
1. 预测 latent 表示而非像素，避免预测不相关细节（纹理、光照）
2. 用各向同性 Gaussian 约束防止表示坍塌（稠密版）
3. LpWM 用稀疏 $L_p$ 约束替代 Gaussian 约束

## 代表工作
- [[LpWM]]: 稀疏 JEPA 世界模型
- [[JEPA-WAM]]: JEPA 架构用于 World Action Model
- [[LeWM]]: Le World Model，JEPA 在 RL 中的应用

## 相关概念
- [[Sparse Representation]]
- [[World Model]]
- [[LeWM]]
