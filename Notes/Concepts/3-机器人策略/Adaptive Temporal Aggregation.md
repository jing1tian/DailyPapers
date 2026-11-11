---
type: concept
aliases: [自适应时序聚合, Adaptive Chunk Aggregation, 自适应动作块对齐, 时序自适应聚合]
---

# Adaptive Temporal Aggregation（自适应时序聚合）

## 定义

自适应时序聚合是一种用于重叠 [[Action Chunking|动作块]] 对齐的方法：通过计算每个新到动作块相对于执行周期的分数偏移，并累积分数来决定整数跳步数和线性插值权重，从而在动态延迟下实现无漂移的亚步级时序对齐。

## 数学形式

**分数偏移:**

$$
\rho_k = \frac{T_k}{T_{\text{exec}}}
$$

**整数跳步 $n_k$ 和插值权重 $f_k$:**

$$
n_k = \Big\lfloor \sum_{j \le k} \rho_j \Big\rfloor - \Big\lfloor \sum_{j < k} \rho_j \Big\rfloor, \quad f_k = \Big\{ \sum_{j \le k} \rho_j \Big\}
$$

**线性插值执行动作（每维度独立）:**

$$
\tilde{a} = (1 - f_k)\, a_i + f_k\, a_{i+1}
$$

## 核心要点

1. **问题**: 固定偏移聚合器不能处理动作块到达间隔的随机波动，长时间运行会产生累积漂移
2. **分数偏移**: 将实际到达间隔 $T_k$ 相对执行周期 $T_{\text{exec}}$ 归一化，得到 $\rho_k \approx 1$
3. **累积分数**: 通过对 $\rho_k$ 求前缀和并取整/取小数，分别得到整数跳步和插值系数，消除漂移
4. **线性插值**: 在相邻动作步之间插值，生成平滑的执行动作，避免控制跳变

## 效果（来自 DASH 真实机器人实验）

| 方法 | 成功率 |
|------|-------|
| 固定偏移聚合器 | 66.7% |
| 自适应时序聚合（DASH） | 95.8% |

## 代表工作

- [[DASH]]: 提出 Adaptive Temporal Aggregation，替代固定偏移聚合器，解决真实机器人部署中的时序漂移问题

## 相关概念

- [[Action Chunking]]
- [[Asynchronous Inference]]
- [[Decoupled Policy]]
