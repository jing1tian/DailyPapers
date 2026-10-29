---
type: concept
aliases: [B样条, B-spline Curve, 三次B样条]
---

# B-spline（B 样条）

## 定义

B-spline（Basis spline）是一种以**控制点**定义的分段多项式曲线，通过少量控制点 $C \in \mathbb{R}^{K \times D}$ 和固定基矩阵 $B$ 紧凑表示高维序列数据，具有局部支撑和 $C^{p-1}$ 连续性（$p$ 为阶数）。

## 数学形式

对于轨迹 $\Delta P \in \mathbb{R}^{T \times D}$，用 B-spline 逼近：

$$
\widetilde{\Delta P} \approx B C, \quad C \in \mathbb{R}^{K \times D}
$$

- $B \in \mathbb{R}^{T \times K}$: 固定 B-spline 基矩阵（由节点向量决定）
- $C$: $K$ 个控制点（$K \ll T$，实现压缩）
- $D$: 维度（如 3 维轨迹）

## 核心要点

1. **压缩效率**: $K \ll T$ 时显著减少参数量，如 $T=32$ 步压缩为 $K=10$ 个控制点
2. **平滑约束**: B-spline 天然强制曲线平滑连续，隐式正则化轨迹预测
3. **固定基矩阵**: 推理时 $B$ 固定，预测控制点 $C$ 后矩阵乘法即可解码完整轨迹
4. **机器人轨迹**: 常用于压缩机器人关节轨迹或视觉点追踪序列

## 代表工作

- [[TraceWAM]]: 用 $K=10$ 控制点压缩 $T=32$ 步三维点运动轨迹，ADE 比直接预测降低 23.5%

## 相关概念

- [[点追踪]]: B-spline 常用于压缩点追踪序列
- [[TraceWAM]]: 将 B-spline 用于运动轨迹压缩的代表工作
