---
type: concept
aliases: [PCA, 主成分分析]
---

# Principal Component Analysis（主成分分析）

## 定义

通过特征分解协方差矩阵，将数据投影到方差最大的正交方向（主成分）上，实现降维和子空间提取。

## 数学形式

$$
R = \frac{1}{N}X^\top X, \quad R = U\Lambda U^\top
$$

前 $k$ 个主成分投影：$Z = XU_k$，其中 $U_k$ 为前 $k$ 列。

## 核心要点

1. [[Conceptor]] 是 PCA 的软化版本：PCA 硬截断前 $k$ 个方向，Conceptor 通过孔径参数软加权
2. 激活子空间分析中常用 PCA 可视化高维激活的聚类结构
3. COAST 用 PCA 可视化跨任务成功/失败激活的联合分布

## 代表工作

- [[COAST]]: 用联合 PCA 可视化证明跨任务失败几何共享性

## 相关概念

- [[Conceptor]]
- [[Aperture Parameter]]
- [[Subspace Similarity]]
