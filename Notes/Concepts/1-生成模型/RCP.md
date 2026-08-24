---
type: concept
aliases: [Reference Context Packing, 参考上下文压缩]
---

# RCP (Reference Context Packing)

## 定义
4DAnyone 提出的方法，将不断增长的 reference views 压缩为固定长度的 mixed-resolution context，把 reference 侧的注意力复杂度从 O(N) 降至 O(1)，解决多视角扩散生成中的 bounded-attention-context 问题。

## 数学形式
$$C_{\text{ref}} = \text{Compress}\left(\{v_1, v_2, \ldots, v_N\}\right) \in \mathbb{R}^{L \times d}$$

其中 $L$ 为固定压缩长度，$N$ 为已生成 reference views 数量，$L \ll N \cdot T$（$T$ 为每视角 token 数）。

## 核心要点
1. 随着已生成视角增多，reference context 不再线性增长，防止显存爆炸
2. 使用 mixed-resolution 编码保留不同细节层次的信息
3. 与 [[TCR]] 配合：RCP 解决 reference 侧复杂度，TCR 解决 target 视角间信息孤立问题

## 代表工作
- [[4DAnyone]]: RCP 与 TCR 共同实现单目视频到 4DGS 的高一致性多视角生成

## 相关概念
- [[TCR]]: Target Context Routing，4DAnyone 另一个互补设计
- [[DiT]]: RCP 在 DiT-based 视频扩散模型框架内应用
