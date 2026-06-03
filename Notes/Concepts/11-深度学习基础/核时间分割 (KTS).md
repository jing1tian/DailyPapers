---
type: concept
aliases: [Kernel Temporal Segmentation, KTS, 核时序分割]
---

# 核时间分割 (KTS)

## 定义

KTS（Kernel Temporal Segmentation）是一种基于核方法的时序片段划分算法，通过最小化段内核矩阵方差找到将时间序列分割为 $K$ 段的最优边界点，常用于视频摘要和关键帧提取。

## 数学形式

$$
\min_{\mathcal{K}} \sum_{i=0}^{K} \left[ \sum_{t \in S_i} G_{tt} - \frac{1}{|S_i|} \sum_{t, t' \in S_i} G_{tt'} \right] + \lambda |\mathcal{K}|
$$

通过[[动态规划]]在 $O(T^2 K)$ 时间内精确求解。

## 核心要点

1. **核相似度驱动**: 基于任意核函数（常用 RBF 核），无需手工设计特征变化检测器
2. **全局最优**: DP 保证给定关键帧数量 $K$ 下的全局最优分割
3. **灵活扩展**: 可将多模态核矩阵融合后输入，自然支持多源信号

## 代表工作

- [[SKIP]]: 将多模态 RBF 核矩阵融合后输入 KTS，实现机器人感知关键帧选择

## 相关概念

- [[RBF 核]]
- [[动态规划]]
- [[多模态特征融合]]
