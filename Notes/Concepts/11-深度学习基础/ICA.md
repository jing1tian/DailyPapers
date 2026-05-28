---
type: concept
aliases: [ICA, Independent Component Analysis, 独立分量分析]
---

# ICA

## 定义

ICA（Independent Component Analysis，独立分量分析）是一种无监督表示学习方法，将观测数据分解为统计独立的源信号，是识别数据真实生成因子的经典框架。

## 数学形式

假设观测 $x = As$，其中 $s$ 是独立源信号（$p(s) = \prod_i p_i(s_i)$），$A$ 是混合矩阵。ICA 估计：

$$\hat{s} = Wx, \quad W \approx A^{-1}$$

通过最大化非高斯性（如 FastICA 用峭度或负熵）求解 $W$。

## 核心要点

1. **统计独立性**: ICA 假设源信号互相统计独立（比非相关更强的假设）
2. **非高斯性**: ICA 利用源信号的非高斯性作为分解准则（高斯信号的混合仍是高斯的，无法区分）
3. **可识别性**: ICA 在非高斯且独立假设下可识别真实因子（除排列和缩放歧义外）
4. **JEPA 连接**: LeJEPA 理论分析中用 ICA 框架证明表示的可识别性条件

## 代表工作

- Hyvärinen & Oja 2000: FastICA 算法
- [[LeJEPA]]：用 ICA 框架分析 JEPA 世界模型的表示可识别性

## 相关概念

- [[SFA]]
- [[JEPA]]
- [[Representation Collapse]]
