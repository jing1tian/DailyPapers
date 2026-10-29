---
type: concept
aliases: [VLSA, VLA Safety Layer, Vision-Language-Action Safety]
---

# VLSA

## 定义

一种基于控制障碍函数（CBF）的即插即用 VLA 安全滤波器：在 VLA 策略输出的标称控制指令基础上，通过求解二次规划（QP）实时修正末端执行器速度或关节速度，保证对已知几何障碍的碰撞安全性。

## 数学形式

$$
\min_{\mathbf{u},\, \{\mathbf{u}_{z,ij}\}} \frac{w_m}{2}\|\mathbf{u} - \mathbf{u}_{\mathrm{nom}}\|_2^2 + \frac{w_z}{2}\sum_{i,j}\|\mathbf{u}_{z,ij} - \mathbf{u}_{z,ij}^{\mathrm{nom}}\|_2^2
$$

$$
\text{s.t.} \quad \mathbf{a}_{v,ij}^\top \mathbf{v}_i + \mathbf{a}_{\omega,ij}^\top \boldsymbol{\omega}_i + \mathbf{a}_{z,ij}^\top \mathbf{u}_{z,ij} + \alpha h_{ij} \ge 0
$$

其中 $\mathbf{u}_{\mathrm{nom}}$ 为 VLA 标称指令，$h_{ij}$ 为椭球障碍物的 CBF 值。

## 核心要点

1. **两个变体**: VLSA-EE（末端执行器速度控制）和 VLSA-ARM（关节速度控制）
2. **实时 QP**: 每步在线求解二次规划，计算开销低
3. **几何建模依赖**: 需要障碍物的精确椭球几何表示
4. **动态场景弱**: 基于瞬时速度约束，对动态障碍的适应性有限

## 代表工作

- Hu et al., 2025: VLSA 原始论文

## 相关概念

- [[CBF]]: 控制障碍函数，VLSA 的理论基础
- [[VLA-Stem]]: 基于 RL 的替代安全机制，在动态障碍场景下显著优于 VLSA
- [[VLA（视觉-语言-动作模型）]]: VLSA 保护的上层策略
