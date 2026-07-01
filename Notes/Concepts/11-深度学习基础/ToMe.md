---
type: concept
aliases: [Token Merging, ToMe]
---

# ToMe（Token Merging）

## 定义
Vision Transformer 的 token 合并加速方法，通过在每层合并相似的 token 来降低序列长度，减少计算量。

## 核心要点
1. 用 token 间余弦相似度选择合并对
2. 合并后 token 数减少，加速自注意力计算（$O(n^2)$）
3. 训练无关，可后插 ViT / DiT 等架构

## 代表工作
- [[ST-Merge]]: 空间 + 时间双维度 ToMe 扩展

## 相关概念
- [[FastV]]
- [[TempMe]]
- [[视觉 Token 剪枝]]
