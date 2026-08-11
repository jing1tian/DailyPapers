---
type: concept
aliases: [持久世界状态记忆, 4D Spatial Memory, 世界状态记忆, World State Memory]
---

# Persistent World State Memory

## 定义

一种机器人操作中的持久化空间记忆机制，通过将腕部相机的 2D 观测序列提升到 4D 潜空间（3D 空间 + 时间），并用体素哈希策略维护全局空间地图，使机器人能"记住"已扫视过但当前不在视野内的空间区域，从根本上解决腕部相机的部分可观测性问题。

## 数学形式

**空间反投影（2D → 3D）:**

$$
m_t, P_t = \text{Back-Projection}(X_w^t, D^w_t, \mathbf{T}^{in}, \mathbf{T}^{ex}_t)
$$

**时空嵌入融合:**

$$
\widehat{m}_t = m_t + \mathcal{E}_{spatial}(P_t) + \mathcal{E}_{temporal}(t)
$$

**体素加权聚合更新（受 TSDF 启发）:**

$$
\mathcal{M}_t(v) = \frac{\mathcal{W}_{t-1}(v)\mathcal{M}_{t-1}(v) + w_t m_t(v)}{\mathcal{W}_{t-1}(v) + w_t}, \quad \mathcal{W}_t(v) = \lambda\mathcal{W}_{t-1}(v) + w_t(v)
$$

## 核心要点

1. 将 2D 视觉 token 通过深度估计和相机内外参反投影到 3D 潜空间，赋予特征精确几何坐标
2. 空间位置编码提供几何感知，时序位置编码防止空间混叠（不同时刻同一位置的特征可区分）
3. 体素哈希结构实现 $O(1)$ 空间访问，加权聚合保留多次观测的最优特征
4. "永久初始化"规则：第一帧状态作为全局锚点，即使机器人移动也能保持参考系一致
5. 指数衰减权重 $\lambda$ 控制历史特征的置信度，记忆容量上限为 2048 token

## 代表工作

- [[AtlasVLA]]: 提出该机制，将其与自我工作状态记忆结合，实现腕部相机 VLA 超越多视角基线

## 相关概念

- [[深度引导反投影]]: 2D → 3D 提升的核心操作
- [[TSDF]]: 体素加权聚合的灵感来源
- [[体素哈希]]: 空间索引数据结构
- [[Ego-Working State Memory]]: 配套的任务进度记忆模块
- [[MemoryVLA]]: 先前的时序缓存记忆方法（缺乏显式空间建模）
