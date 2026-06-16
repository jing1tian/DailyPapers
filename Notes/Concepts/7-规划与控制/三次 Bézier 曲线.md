---
type: concept
aliases: [Cubic Bezier, Cubic Bézier, Bézier 平滑, C1 连续拼接]
---

# 三次 Bézier 曲线（Cubic Bézier Curve）

## 定义

由四个控制点参数化的三次多项式曲线，在机器人运动规划中用于在动作块边界实现 C¹ 连续（位置与切线方向均连续）的轨迹拼接。

## 数学形式

$$\mathbf{B}(t) = (1-t)^3\mathbf{P}_0 + 3(1-t)^2 t\mathbf{P}_1 + 3(1-t)t^2\mathbf{P}_2 + t^3\mathbf{P}_3, \quad t\in[0,1]$$

C¹ 端点约束：
$$\mathbf{B}(0) = \mathbf{h}_0,\quad \mathbf{B}(1) = \mathbf{f}_c,\quad \dot{\mathbf{B}}(0) \parallel \hat{d}_{hist},\quad \dot{\mathbf{B}}(1) \parallel \hat{d}_{fut}$$

控制点构造（$\lambda = \sigma\|\mathbf{P}_3 - \mathbf{P}_0\|$）：
$$\mathbf{P}_1 = \mathbf{P}_0 + \lambda\hat{d}_{hist},\quad \mathbf{P}_2 = \mathbf{P}_3 - \lambda\hat{d}_{fut}$$

## 核心要点

1. 四个控制点唯一确定一段三次曲线，计算高效
2. C¹ 约束通过将 $\mathbf{P}_1, \mathbf{P}_2$ 沿对应切线方向布置实现
3. 切线方向由历史执行轨迹（$\hat{d}_{hist}$）和预测未来块（$\hat{d}_{fut}$）估计
4. 在 VLA 动作块拼接中消除块边界 C⁰ 不连续，显著减少真实机器人抖动

## 代表工作

- [[HyVLA-0.5]]: 提出异步 Bézier 平滑用于动作块边界 C¹ 连续拼接

## 相关概念

- [[Action Chunking]]
- [[Delta-Chunk 动作表示]]
