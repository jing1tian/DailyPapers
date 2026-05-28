---
type: concept
aliases: [SFA, Slow Feature Analysis, 慢特征分析]
---

# SFA

## 定义

SFA（Slow Feature Analysis，慢特征分析）是一种无监督学习方法，从时序数据中提取随时间变化最慢的特征，基于"语义信息变化比感知信息慢"的假设。

## 数学形式

$$\min_{g} \langle \dot{y}_j^2 \rangle_t \quad \text{s.t. } \langle y_j \rangle_t = 0, \quad \langle y_j^2 \rangle_t = 1, \quad \langle y_j y_i \rangle_t = 0 \ (j > i)$$

最小化输出 $y_j = g_j(x_t)$ 的时间导数方差，约束为零均值、单位方差、去相关。

## 核心要点

1. **时序慢性假设**: 语义相关的表示（如物体身份）比感知输入（像素）变化慢
2. **无监督**: 只需时序数据，无需标签
3. **与 ICA 互补**: ICA 基于统计独立性，SFA 基于时序慢性；两者都能提取有意义因子
4. **JEPA 理论**: LeJEPA 分析中用 SFA 框架理解慢变特征学习

## 代表工作

- Wiskott & Sejnowski 2002: SFA 原始论文

## 相关概念

- [[ICA]]
- [[JEPA]]
- [[V-JEPA]]
