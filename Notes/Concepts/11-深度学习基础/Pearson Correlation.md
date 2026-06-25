---
type: concept
aliases: [Pearson Correlation, Pearson 相关系数, Pearson's r, 皮尔逊相关系数]
---

# Pearson Correlation（Pearson 相关系数）

## 定义

衡量两组数值变量之间**线性相关程度**的统计量，取值范围 $[-1, 1]$，常用于评测"预测值与真实值是否在数值上线性一致"，是回归/校准类任务最常用的校准（calibration）指标之一。

## 数学形式

$$
r(X, Y) = \frac{\sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^{n} (x_i - \bar{x})^2} \sqrt{\sum_{i=1}^{n} (y_i - \bar{y})^2}}
$$

## 符号说明

- $x_i, y_i$: 第 $i$ 个样本在两组变量下的取值
- $\bar{x}, \bar{y}$: 两组变量各自的均值
- $r=1$: 完全正线性相关；$r=-1$: 完全负线性相关；$r=0$: 无线性相关

## 核心要点

1. **仅衡量线性关系**：$r$ 高不代表预测值与真实值数值相等，只代表两者按比例同向变化；常与 [[MMRV (Mean Maximum Rank Violation)|MMRV]] 等排序一致性指标搭配使用，分别衡量"绝对校准"与"相对排序"两个互补维度
2. **对异常值敏感**：少数极端样本会显著拉低或拉高 $r$ 值
3. **在策略评测中的应用**：常用于衡量"模拟器/世界模型中预测的策略成功率"与"真实世界实测成功率"之间的整体线性一致性

## 代表工作

- [[SC3-Eval]]: 用 $r(R, R_{\mathcal{W}})$ 衡量世界模型评测器预测表现 $R_\mathcal{W}$ 与真实世界表现 $R$ 的线性校准程度，闭环评测下达到 $r=0.929$

## 相关概念

- [[MMRV (Mean Maximum Rank Violation)]]
- [[SIMPLER]]
