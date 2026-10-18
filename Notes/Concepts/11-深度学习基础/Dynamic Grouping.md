---
type: concept
aliases: [动态分组, Budget-Constrained Grouping, 预算约束动态分组]
---

# Dynamic Grouping（动态分组）

## 定义

在多玩家/多智能体 Transformer 中，基于玩家间相机轨迹实时计算相关性矩阵，并在计算预算约束下自动决定哪些玩家进入同一通信组的动态分组算法。

## 数学形式

**玩家对相关性**（取场景相关性与互视相关性的最大值）：

$$
R^{(t)}_{ij} = \max\!\left(R^{(t),\mathrm{scene}}_{ij},\ R^{(t),\mathrm{mutual}}_{ij}\right)
$$

**互视相关性**：

$$
r^{\mathrm{mutual}}_{ij}(\tau) = \sqrt{Q_{i \to j}(\tau) \cdot Q_{j \to i}(\tau)}
$$

**组容量**（由预算反推）：

$$
K = \min\!\left(N,\ \left\lfloor \tfrac{1}{2} + \sqrt{\tfrac{1}{4} + B_{\mathrm{pair}}} \right\rfloor\right)
$$

## 核心要点

1. **相机轨迹驱动**: 仅使用相机位置和朝向（无语义特征），轻量且可解释，每48帧更新一次
2. **预算约束**: 要求总 FLOP 不超过中心化方法的代价 $\mathcal{C}_\mathrm{central}$，组容量 $K$ 由剩余预算解析推导
3. **锚节点机制**: 允许玩家至多属于两个组，跨组共享玩家作为锚节点传递信息，提升覆盖率
4. **贪心优化**: 以最大化高相关性玩家对的覆盖数为目标，贪心选择分组方案

## 代表工作

- [[Multics]]: 提出预算约束动态分组，相比固定分组、随机分组和基于组方法在 GeoKPM 上分别提升 +3.46%、+7.41%、+5.91%

## 相关概念

- [[Group Token]]
- [[Budget-Constrained Grouping]]
- [[自注意力机制]]
