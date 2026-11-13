---
type: concept
aliases: [连通域, 连通分量, Connected Components Analysis]
---

# Connected Component

## 定义

图像或点云处理中，将相互连通的像素/体素归为同一组的算法，常用于目标检测的候选区域提取。

## 核心要点

1. **基于深度图**：RoboScript 中对工作空间高度图做连通域分析，每个分量提升为 3D 边界体
2. **几何优先感知**：先用深度/几何信息生成候选，再用 VLM/Grounding 解析语义
3. **标准算法**：BFS/DFS 或 Union-Find 实现

## 代表工作

- [[RoboScript]]: 感知流水线的第一步（深度图 → 高度图 → 连通域 → 3D 边界体）

## 相关概念

- [[Grounding Model]]
- [[VLM]]
