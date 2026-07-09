---
type: concept
aliases: [时序几何, 动态几何, 时变几何关系]
---

# Temporal Geometry

## 定义

时序几何（Temporal Geometry）描述场景中物体或特征点之间的空间关系如何随时间演变，特别是在机器人操作场景中对"接近 → 接触 → 搬运 → 释放"等交互阶段的几何变化进行建模和监督。

## 数学形式

跨关键帧的关系变化量（归一化）：

$$\Delta(R^k, R^{k+}) = \frac{R^{k+} - R^k}{|R^{k+}| + |R^k| + \epsilon}$$

时序几何蒸馏损失：

$$\mathcal{L}_{\text{tem}} = \frac{\sum_{(k,k^+) \in \mathcal{A}_{\mathcal{K}}, i,j} w_{ij}^{k,k^+} D_{ij}^{k,k^+}}{\sum w}$$

其中 $\mathcal{A}_{\mathcal{K}}$ 为相邻关键帧对集合，$D_{ij}^{k,k^+}$ 为 token 对在帧间的变化预测误差。

## 核心要点

1. **捕获交互阶段**: 接近（距离减小）、接触（急剧变化）、搬运（维持接触）、释放（距离增大）各阶段的几何变化具有独特模式
2. **补充帧内几何**: 单帧内的几何关系无法捕获运动轨迹信息，时序几何提供动态上下文
3. **动作耦合**: 结合动作感知权重，聚焦于与机器人动作变化耦合最强的视觉区域

## 代表工作

- [[MECo-WAM]]: 通过时序几何蒸馏损失 $\mathcal{L}_{\text{tem}}^{\text{act}}$ 监督跨关键帧的几何关系演变
- [[WAM4D]]: 在推理时显式预测 4D 时序几何

## 相关概念

- [[Relational Distillation]]
- [[VGGT]]
- [[4D Reconstruction]]
- [[World Action Model]]
