---
type: concept
aliases: [跨世界迁移, Cross-World Transfer]
---

# Transfer Across Worlds

## 定义

WICL 中的核心特性：在世界 $W$（如仿真器）中排练习得的 context 知识，在语义一致的情况下（$\sigma(W)=\sigma(W')$）可直接应用于世界 $W'$（如随机化场景或真实机器人），无需重新训练或重新排练。

## 数学形式

$$
\sigma(W) = \sigma(W') \implies C^* \text{ 在 } W' \text{ 有效}
$$

其中 $\sigma(e)$ 表示实例的语义内容（对象、角色、指令、相对几何）。

## 核心要点

1. **语义不绑定外观**：context 只记录角色和相对几何，不记录视觉特征
2. **权重 vs context 的不对称性**：权重会吸收 $W$ 中所有帮助拟合训练数据的规律（包括 $W'$ 不共享的），context 不会
3. **实验验证**：RoboScript 仅在干净场景排练，随机化保持率 95–101%

## 代表工作

- [[RoboScript]]: 跨世界迁移的核心实验证明

## 相关概念

- [[World In-Context Learning]]
- [[Sim-to-Real Transfer]]
- [[Task Memory]]
