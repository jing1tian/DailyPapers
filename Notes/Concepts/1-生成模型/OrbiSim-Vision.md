---
type: concept
aliases: [状态引导扩散视觉, State-Guided Diffusion Vision]
---

# OrbiSim-Vision

## 定义

OrbiSim 的视觉渲染模块，以预测的物理状态为条件，通过潜扩散模型生成高保真视觉观测，实现动力学与视觉渲染的解耦。

## 核心要点

1. 空间条件图（Spatial Condition Map）：基于预测物理状态生成几何对齐的视觉线索
2. 对象 token 通过交叉注意力调制 U-Net 内部特征
3. VAE 编解码在潜空间操作，降低计算开销
4. 视觉上下文随机稀疏化，提升自回归鲁棒性
5. 与 OrbiSim-Dynamics 解耦，梯度主要流经动力学模块

## 代表工作

- [[OrbiSim]]: 提出 OrbiSim-Vision 的论文

## 相关概念

- [[潜扩散模型]]
- [[VAE]]
- [[U-Net]]
- [[交叉注意力]]
- [[OrbiSim-Dynamics]]
