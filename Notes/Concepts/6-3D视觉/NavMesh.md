---
type: concept
aliases: [Navigation Mesh, 导航网格]
---

# NavMesh

## 定义

NavMesh（Navigation Mesh）是一种用于路径规划的多边形网格数据结构，将可行走区域表示为凸多边形集合，供智能体进行无碰撞路径搜索。

## 核心要点

1. **可行走区域编码**: 将 3D 场景的地面/平台等可通行区域抽象为 2D 多边形网格
2. **碰撞检测**: 智能体在 NavMesh 内移动时自动避开障碍物和边界
3. **路径搜索**: 结合 A* 等算法在多边形图上高效搜索最优路径
4. **精化策略**: 可通过射线投射（Ray-casting）、边界腐蚀（Boundary Erosion）、桥接合成（Bridge Synthesis）等方法改善导航网格质量

## 数学形式

NavMesh 将场景建模为图 $G = (V, E)$，其中节点 $V$ 为多边形中心，边 $E$ 连接相邻可通行多边形；路径搜索为：

$$
\text{path}^* = \arg\min_{path} \sum_{e \in path} w(e)
$$

## 代表工作

- [[HYWorld2]]: 在 WorldNav 中使用 NavMesh 精化，结合射线投射和边界腐蚀确保 3D 场景轨迹规划的无碰撞导航

## 相关概念

- [[3D Gaussian Splatting]]
- [[World Model]]
