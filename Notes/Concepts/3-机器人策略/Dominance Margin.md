---
type: concept
aliases: [显性边距, 视角显性度, View Dominance]
---

# Dominance Margin

## 定义

衡量 VLA 模型在某一时间步对全局相机和局部相机注意力权重之间差异绝对值的指标，用于量化视角依赖的不均衡程度。

## 数学形式

$$
D_t = |a_{\text{global},t} - a_{\text{local},t}|
$$

其中 $a_{\text{global},t}$ 和 $a_{\text{local},t}$ 分别为第 $t$ 步模型对全局和局部视图 token 的平均注意力权重。

## 核心要点

1. $D_t$ 越大，说明模型在该步对某一视角的依赖越强（视角不均衡）
2. 失败轨迹相比成功轨迹平均 Dominance Margin 更大，印证了"[[View-Dependency Inertia|视角依赖惯性]]"的存在
3. 配合局部依赖变化度 $V = \text{Var}(\Delta L_t)$ 共同评估模型的视角切换能力

## 代表工作

- [[FocusVLA]]: 定义 Dominance Margin 作为分析静态融合 VLA 局限性的核心度量

## 相关概念

- [[View-Dependency Inertia]]
- [[Dynamic View Routing]]
- [[VLA]]
