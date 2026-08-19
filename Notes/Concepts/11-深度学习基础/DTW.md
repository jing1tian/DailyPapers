---
type: concept
aliases: [Dynamic Time Warping, 动态时间规整]
---

# DTW (Dynamic Time Warping)

## 定义
一种度量两个时间序列相似度的算法，通过动态规划允许序列在时间轴上弹性对齐，处理时间偏移和速度变化。

## 数学形式
$$\text{DTW}(X, Y) = \min_{\text{path}} \sum_{(i,j)\in\text{path}} d(x_i, y_j)$$

其中 path 为满足单调性和连续性约束的最优对齐路径。

## 核心要点
1. 时间弹性对齐：比欧氏距离更适合对比速度不同的动作序列
2. 复杂度 $O(nm)$，但有快速近似算法
3. 在 [[DriveCache]] 中用于度量驾驶动作序列相似度，决定 attention 缓存复用策略

## 相关概念
- [[DriveCache]]
- [[TeaCache]]
