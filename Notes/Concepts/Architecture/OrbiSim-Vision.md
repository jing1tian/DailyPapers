---
type: concept
aliases: [OrbiSim-Vision, OrbiSim Vision, 状态引导扩散视觉]
---

# OrbiSim-Vision

## 定义
OrbiSim 框架的视觉渲染模块，以 OrbiSim-Dynamics 预测的物理状态为条件，通过潜扩散模型生成高保真视觉观测帧，实现动力学与渲染的解耦。

## 数学形式

$$
\hat{o}_t \sim p_\phi^{vis}(o_t \mid o_{t-k:t-1}, \hat{x}_t, \bar{x})
$$

$$
\mathcal{L}_{vis} = \mathbb{E}_{\sigma, \varepsilon}\left[w(\sigma) \| D_\phi(\cdot)_t - y_{t,(0)} \|_2^2\right]
$$

## 核心要点
1. **空间条件图**：基于预测物理状态 $\hat{x}_t$ 生成几何对齐的视觉线索。
2. **对象 token 注入**：通过交叉注意力将实体 token 调制 U-Net 内部特征。
3. **视觉上下文增强**：训练时随机稀疏化历史帧，提升自回归鲁棒性。
4. **解耦设计**：梯度主要流经 OrbiSim-Dynamics 而非视觉渲染器，避免梯度退化。
5. 使用 VAE 在潜空间操作，降低计算开销。

## 代表工作
- [[OrbiSim]]: 完整系统描述

## 相关概念
- [[潜扩散模型]]
- [[U-Net]]
- [[交叉注意力]]
- [[VAE]]
- [[OrbiSim-Dynamics]]
