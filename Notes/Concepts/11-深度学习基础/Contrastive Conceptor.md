---
type: concept
aliases: [对比概念算子, Contrastive Conceptor Operator]
---

# Contrastive Conceptor（对比 Conceptor）

## 定义

通过布尔 AND-NOT 运算将成功 Conceptor 与失败 Conceptor 的补集合并，得到专门用于激活引导的判别子空间算子。

## 数学形式

$$
C_{\text{steer}} = C_{\text{success}} \land \neg C_{\text{failure}} = C_{\text{success}} \land (I - C_{\text{failure}})
$$

展开为：

$$
C_{\text{steer}} = (C_{\text{success}}^{-1} + (I - C_{\text{failure}})^{-1} - I)^{-1}
$$

## 核心要点

1. 隔离"成功特有"方向，同时排除与失败共享的低区分度方向
2. 结果子空间约为总隐藏维度的 1%（低秩但高判别性）
3. 与 Positive-only 变体（$C_{\text{steer}} = C_{\text{success}}$）对比：对比项通常带来更大提升
4. 成功-失败子空间重叠越大，对比运算越有效（相关系数 $\rho = 0.59$）

## 代表工作

- [[COAST]]: 提出 Contrastive Conceptor 概念，用于 VLA 推理时激活引导

## 相关概念

- [[Conceptor]]
- [[Conceptor Boolean Logic]]
- [[Subspace Similarity]]
- [[Activation Steering]]
