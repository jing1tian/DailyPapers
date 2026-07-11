---
type: concept
aliases: [Distance-based DP-Means, DP均值聚类, 非参数聚类, Dirichlet Process Means]
---

# DP-Means（DP 均值聚类）

## 定义

DP-Means 是一种非参数聚类算法，基于 Dirichlet Process 混合模型的确定性近似（硬分配版本）。与 K-Means 不同，DP-Means **自动确定聚类数量**：当样本与所有现有类中心的距离超过阈值 $\lambda$ 时，自动创建新类别。

## 数学形式

$$
z_i = \begin{cases}
\arg\min_k \|x_i - \mu_k\|^2 & \text{if } \min_k \|x_i - \mu_k\|^2 \leq \lambda \\
K + 1 \text{ (新类)} & \text{otherwise}
\end{cases}
$$

其中 $\lambda$ 为超参数，控制类别粒度；$\mu_k$ 为第 $k$ 个类的中心。

## 核心要点

1. **自适应类别数**：无需预设 $K$，超参数仅一个（距离阈值 $\lambda$）
2. **在线可用**：支持流式数据，新样本若距离所有现有中心 >$\lambda$ 则自动开辟新类
3. **WP-WM 中的用途**：消解离散符号碰撞——冻结投影可能将多个物体类型映射到同一符号 ID，DP-Means 对同符号的不同嵌入聚类以区分它们
4. **必要性**：无此层时 36 物体场景准确率从 97.2% 跌至 22.2%（-75 pp）

## 代表工作

- Kulis & Jordan (2012): *Revisiting k-means: New algorithms via Bayesian nonparametrics*（原始 DP-Means 论文）
- [[WP-WM]]: 用 DP-Means 解决[[Discrete Bottleneck|冻结投影]]的符号碰撞问题

## 相关概念

- [[Discrete Bottleneck]]
- [[Symbol Grounding]]
- [[Blackboard Architecture]]
