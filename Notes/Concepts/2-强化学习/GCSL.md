---
type: concept
aliases: [Goal-Conditioned Supervised Learning, 目标条件监督学习]
---

# GCSL

## 定义
将目标条件强化学习转化为监督学习问题：将轨迹中的未来状态作为当前步的"标签目标"，从而无需 reward 函数直接进行目标条件策略训练。

## 数学形式
$$\pi^* = \arg\max_\pi \mathbb{E}_{(s,g,a) \sim \mathcal{D}} \left[ \log \pi(a | s, g) \right]$$

其中数据集 $\mathcal{D}$ 由轨迹重标记（hindsight relabeling）构建，$g$ 来自同一轨迹的未来状态。

## 核心要点
1. 不需要 reward 函数，只需轨迹数据
2. 利用 hindsight 重标记：将轨迹中任意后续状态作为"达成目标"
3. 迭代训练：策略生成新轨迹 → 重标记 → 再训练
4. 对高维目标空间泛化能力有限

## 代表工作
- [[INTACT]]: 将 GCSL 作为对比 baseline，用于验证意图坐标的优势

## 相关概念
- [[JEPA]]
- [[PRISM]]
