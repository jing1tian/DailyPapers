---
type: concept
aliases: [RoleSub, Role-Conditioned Sub-Token Routing]
---

# RoleSub

## 定义
一种 VLA 推理效率方法，通过将 token 的 value vector 分解为 role-conditioned 子空间并按重要性路由，实现 sub-token 级别压缩，比 token dropping 信息损失更少。

## 核心要点
1. Orthogonal Value-Space Decomposition：将每个 token 的 value 向量分解为正交子空间
2. Learned Latent Role Representation：学习每种 token 类型（视觉/语言）对应的 role 表示
3. Token-Type-Dependent Value Budgets：视觉和语言 token 使用不同压缩 budget
4. 在 LIBERO-10 上验证，相比 FastV 等 token dropping 方法损失更少

## 代表工作
- Jiang & Wang, 2026 — (arXiv 2608.18410)

## 相关概念
- [[VisionZip]]
- [[SparseVLM]]
- [[FastV]]
- [[OpenVLA]]
