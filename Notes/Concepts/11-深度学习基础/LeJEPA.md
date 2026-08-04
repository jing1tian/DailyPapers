---
type: concept
aliases: [Learnable JEPA, Latent Energy JEPA]
---

# LeJEPA

## 定义
Latent World Model 的 JEPA 变体，用可学习的正则化目标替代 JEPA 中的固定 stability heuristic，使 latent 空间分布更接近目标 Gaussian。

## 核心要点
1. 继承 [[JEPA]] 的 latent 预测范式（在表示空间预测未来，而非像素空间）
2. 用 Epps-Pulley（EP）目标正则化 latent 分布，但 EP 在尾部梯度消失（参见 [[QQWorld]] 的改进）
3. 主要用于 physical parameter identifiability 研究（[[PokeWorld]] 实验平台上评测）
4. 相比 DreamerV3 等生成式 WM，LeJEPA 不预测像素，计算更高效

## 代表工作
- [[LeWM]]: LeWorldModel，LeJEPA 的具体实现
- [[QQWorld]]: 改进 LeJEPA 的 EP 正则化目标
- [[WCM]]: 基于 LeJEPA 构建轻量 VLA-RL critic，联合预测未来潜在状态与价值估计

## 相关概念
- [[JEPA]]
- [[LeWM]]
- [[QQWorld]]
