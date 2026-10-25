---
type: concept
aliases: [动态视角路由, View Routing, Dynamic View Selection, Action-Side Routing]
---

# Dynamic View Routing

## 定义

在多视角机器人操作策略中，根据当前任务阶段的预测意图，动态调整全局相机（第三人称）和局部相机（腕部）特征在动作生成侧的融合权重，从而实现阶段感知的视觉信息利用。

## 数学形式

$$
c^{(k)} = \alpha_g^{(k)} \cdot c_g^{(k)} + \alpha_l^{(k)} \cdot c_l^{(k)}
$$

路由权重由意图概率线性映射生成，约束在 $(0.25, 0.75)$ 范围内（最大 3:1 比例）：

$$
\alpha_g^{(k)}, \alpha_l^{(k)} \in (0.25, 0.75)
$$

## 核心要点

1. **在动作侧融合**：相比在特征提取侧（BFA 等方法）融合，在动作生成侧融合可直接影响动作预测，避免早期融合的信息损失（消融实验：动作侧 79.5% vs. 特征侧 67.0%）
2. **逐步路由**：每个动作 chunk 的每一步均有独立路由权重，而非整个 chunk 共享一个权重
3. **有界权重**：路由权重限制在 $(0.25, 0.75)$，确保两个视角均有基础贡献，避免视角完全遮蔽
4. **意图驱动**：与随机遮蔽（RDT 风格）或学习静态权重不同，路由权重由前瞻性意图预测驱动

## 代表工作

- [[FocusVLA]]: 提出 Action-Side Dynamic Routing 模块，结合 [[Future Intent Prediction]] 实现阶段感知路由

## 相关概念

- [[Future Intent Prediction]]
- [[View-Dependency Inertia]]
- [[Cross-Attention]]
- [[Action Chunking]]
- [[VLA]]
