---
type: concept
aliases: [World Prediction Token, 世界预测 Token, WPT]
---

# World Prediction Tokens

## 定义
嵌入 VLA 骨干前缀的可学习 Token 集合，专门用于编码短视野场景演化意图；训练时由 [[World Representation Head]] 提供未来 Gaussian 中心位移监督，推理时通过原生注意力将短视野动态先验传递给 Action Expert。

## 数学形式

$$Z^P \in \mathbb{R}^{N_p \times d}, \quad N_p = 4$$

预测 Token 参与位移计算：

$$\Delta\mu_i^h = H_\Delta(F_{t,i}^G,\, H_{t,h}^P,\, e_h)$$

## 核心要点
1. 共 4 个 Token，每个对应一个预测视野（horizon）槽
2. 与 [[World State Tokens]] 一起在 [[PaliGemma]] 中联合上下文化，获得视觉-语言上下文感知能力
3. 调制于**观测和任务上下文**（而非候选动作），学习短视野期望演化而非反事实仿真
4. 训练时驱动 [[耦合未来预测]] 中每个视野的 Gaussian 中心位移预测
5. 推理时保留在前缀中，传递动态先验但不执行任何在线推演

## 代表工作
- [[GaussianDream++]]: 首次提出，与 [[World State Tokens]] 合计 20 个 Token 替代 1024-token 前缀

## 相关概念
- [[World State Tokens]]
- [[World Representation Head]]
- [[耦合未来预测]]
- [[静态-动态分解]]
- [[GaussianDream++]]
