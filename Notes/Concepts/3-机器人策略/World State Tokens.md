---
type: concept
aliases: [World State Token, 世界状态 Token, WST]
---

# World State Tokens

## 定义
嵌入 VLA 骨干前缀的可学习 Token 集合，专门用于编码当前物理场景的几何-外观结构；训练时由 [[World Representation Head]] 提供 3D Gaussian 重建监督，推理时通过原生注意力直接约束动作生成。

## 数学形式

$$Z^S \in \mathbb{R}^{N_s \times d}, \quad N_s = 16$$

## 核心要点
1. 共 16 个 Token，以 4×4 空间排布并附有标定的空间嵌入和射线嵌入，维持粗粒度空间组织
2. 与视觉-语言 Token 在 [[PaliGemma]] 中**联合上下文化**，无需额外投影层
3. 训练时被 [[World Representation Head]] 解码为当前 Gaussian 世界 $G_t$，获得度量几何监督
4. 推理时 World Representation Head 和渲染器完全移除，World State Token 仍留在前缀中条件化 Action Expert
5. 与 [[World Prediction Tokens]] 共 20 个 Token，取代 [[GaussianDream]] 的 1024-token 前缀

## 代表工作
- [[GaussianDream++]]: 首次提出，实现 LIBERO 98.6%、LIBERO-Plus 87.8%，推理开销仅 +44ms

## 相关概念
- [[World Prediction Tokens]]
- [[World Representation Head]]
- [[GaussianDream++]]
- [[PaliGemma]]
- [[3DGS]]
- [[耦合未来预测]]
