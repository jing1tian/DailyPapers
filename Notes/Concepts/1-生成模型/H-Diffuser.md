---
type: concept
aliases: [Hierarchical Diffuser, H-Diffuser Planner]
---

# H-Diffuser

## 定义

在 Diffuser 基础上引入层次化结构的扩散规划器，高层扩散模型生成粗粒度子目标序列，低层扩散模型在相邻子目标间规划细粒度轨迹，是 ID-WM 最直接的对比基线。

## 核心要点

1. **层次化扩散**: 高层和低层均使用扩散模型，分别建模子目标分布和局部轨迹分布
2. **相比 Diffuser 更快**: 层次化分解使每级扩散规模更小，但总推理时间仍约 48 分钟
3. **无显式逆动力学**: 不像 ID-WM 那样将低层世界模型用作逆动力学求解器
4. **Maze 成功率 0.92**: 在 Medium Maze 上略低于 ID-WM（0.99）

## 代表工作

- [[IDWM]]: H-Diffuser 作为层次化扩散规划基线，推理约 48 分钟，成功率 0.92（Maze）

## 相关概念

- [[Diffuser]]
- [[扩散模型]]
- [[层次化规划]]
- [[IDWM]]
