---
type: concept
aliases: [Covariant Hamiltonian Optimization for Motion Planning]
---

# CHOMP

## 定义
CHOMP（Ratliff et al. 2009）：通过协变梯度下降在轨迹空间中优化平滑性和障碍物代价，是经典的基于梯度的运动规划方法。

## 数学形式
$$U[\xi] = \int_0^1 \left[ \frac{1}{2} \| \dot{\xi}(t) \|^2 + c(\xi(t)) \right] dt$$

其中 $c(\xi)$ 是障碍物代价场，通过 Fréchet 导数在轨迹的黎曼流形上做梯度下降。

## 核心要点
1. 联合优化平滑性（动能代价）和障碍物代价，避免局部最优的贪心避障
2. 用预条件化的功能梯度替代欧氏梯度，保持轨迹时间一致性
3. 局限：对初始轨迹敏感，在狭窄通道中容易陷入局部最优

## 相关概念
- [[AIT*]]
- [[NeurRAFT]]
- [[Safety-aware MPPI]]
