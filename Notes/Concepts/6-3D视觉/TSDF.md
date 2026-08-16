---
type: concept
aliases: [Truncated Signed Distance Function, 截断符号距离函数, 截断有符号距离函数]
---

# TSDF

## 定义

截断符号距离函数（Truncated Signed Distance Function）是一种用于三维重建的体素级隐式曲面表示，将空间离散化为体素网格，每个体素存储其到最近表面的有符号距离值（负值在表面内，正值在表面外），并通过"截断"将距离限制在一定范围内以节省存储。

## 数学形式

**TSDF 体素值:**

$$
\Psi(v) = \min\left(1, \frac{d(v)}{\tau}\right) \cdot \operatorname{sign}(d(v))
$$

**TSDF 加权平均融合（用于多帧整合）:**

$$
\Psi_t(v) = \frac{W_{t-1}(v)\Psi_{t-1}(v) + w_t \psi_t(v)}{W_{t-1}(v) + w_t}, \quad W_t(v) = W_{t-1}(v) + w_t
$$

## 核心要点

1. **有符号距离**: 体素值正负号区分曲面内外，零值面（等值面）即为重建表面
2. **截断范围 $\tau$**: 仅保留距表面 $\tau$ 以内的体素值，超出范围置为 $\pm 1$，大幅减少需要更新的体素数
3. **加权融合**: 多帧深度图通过加权平均整合，观测置信度较高的帧获得更大权重
4. **Marching Cubes**: 通过在 TSDF 体积上运行等值面提取算法可得到三角网格
5. **在 VLA 中的应用**: AtlasVLA 将 TSDF 加权融合思想迁移到潜特征空间的体素聚合，用观测置信度作为特征融合权重

## 代表工作

- KinectFusion (2011): 首次将 TSDF 用于实时 RGB-D 三维重建
- [[AtlasVLA]]: 借鉴 TSDF 加权融合思想，实现 4D 潜特征的持久空间记忆

## 相关概念

- [[深度引导反投影]]: 从深度图构建 TSDF 所需的几何操作
- [[体素哈希]]: TSDF 的高效空间存储索引结构
- [[Persistent World State Memory]]: AtlasVLA 中受 TSDF 启发的潜特征空间记忆机制
