---
type: concept
aliases: [解耦策略, 因子化策略, Decoupled World Model Policy]
---

# Decoupled Policy

## 定义
解耦策略是一种将机器人操作策略分解为**多个独立子问题**的方法，通过引入中间变量（如世界状态轨迹）将动作生成与状态预测解耦，各子问题分别建模后组合。

## 数学形式

PrismWAM 的三阶段解耦策略：

$$
\pi_\theta(a_{1:H} \mid r_{\leq 0}, d_0, c) = \iint p_\theta(a_{1:H} \mid z_D, c) \cdot p_\theta(z_D \mid d_0, z_R, c) \cdot p_\theta(z_R \mid r_{\leq 0}, c) \, dz_R \, dz_D
$$

其中 $z_R$ 为语义规划潜变量（来自 RGB），$z_D$ 为物理展开潜变量（来自深度）。

## 核心要点

1. **中间变量解耦**: 通过引入世界状态轨迹 $x_{1:F}$（或其潜变量 $z_R, z_D$）将端到端策略分解
2. **物理与语义分离**: 语义规划（"做什么"）和物理展开（"状态如何演变"）分别用不同模态建模
3. **可解释性增强**: 各子问题有明确的物理含义，便于分析和调试
4. **灵活组合**: 各子模块可独立改进或替换

## 代表工作
- [[PrismWAM]]: 提出基于 RGB（语义）+ 深度（物理）的三阶段解耦策略

## 相关概念
- [[Factorized Policy]]
- [[World Action Model]]
- [[Decoupled Representation]]
- [[Chain Attention]]
