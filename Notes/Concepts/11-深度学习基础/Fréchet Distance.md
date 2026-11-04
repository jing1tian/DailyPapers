---
type: concept
aliases: [Fréchet Distance, Fréchet距离, 弗雷歇距离, Frechet Distance]
---

# Fréchet Distance

## 定义

Fréchet 距离（Fréchet Distance，$d_F$）是衡量两条曲线形状相似度的度量，直觉上等价于"人牵狗散步"：人沿一条路径走、狗沿另一条路径走，两者都只能向前不能后退，所需牵绳的最小长度即为 Fréchet 距离。

## 数学形式

H-JEPA 中用 Fréchet 路径保真度衡量规划质量（Eq. 7）：

$$
\text{fidelity} = \frac{d_F(\mathbf{0},\,\tilde{p}) - d_F(p,\,\tilde{p})}{d_F(\mathbf{0},\,\tilde{p})}
$$

其中：
- $p$：规划路径（xyz 累计位置序列）
- $\tilde{p}$：专家参考路径
- $\mathbf{0}$：零点路径（作为基线，衡量"完全不动"的 Fréchet 距离）

取值范围：趋向 1 表示规划路径高度忠实于专家轨迹。

## 核心要点

1. **路径形状感知**: 不同于点对点位移，Fréchet 距离对路径的形状和走向敏感
2. **有序性约束**: 参数化方向必须单调递增（人和狗都不能回头），区别于 Hausdorff 距离
3. **离散近似**: 实际计算用离散 Fréchet 距离（动态规划），复杂度 $O(mn)$
4. **局限性**: 高保真度路径不一定操作成功（路径对但抓取失败仍可高分）

## 代表工作

- [[H-JEPA]]: 在 DROID 离线规划评估中使用 Fréchet 路径保真度作为规划质量指标

## 相关概念

- [[Hierarchical Planning]]: Fréchet 保真度应用于层次规划质量评估
- [[DROID]]: H-JEPA 使用该指标评估 DROID 数据集上的规划性能
