---
type: concept
aliases: [子空间相似度, Conceptor Similarity]
---

# Subspace Similarity（子空间相似度）

## 定义

量化两个子空间（或 Conceptor 矩阵）之间重叠程度的度量，用于预测激活引导效果或跨任务迁移能力。

## 数学形式

**成功-失败 Conceptor 重叠相似度**:

$$
\text{sim}(C^+, C^-) = \frac{\text{tr}(C^+ C^-)}{\sqrt{\text{tr}((C^+)^2)\,\text{tr}((C^-)^2)}}
$$

**跨任务失败子空间包容度**:

$$
\text{contain}(C_{\text{src}}^f, C_{\text{tgt}}^f) = \frac{\text{tr}(C_{\text{src}}^f C_{\text{tgt}}^f)}{\text{tr}((C_{\text{src}}^f)^2)}
$$

## 核心要点

1. 相似度高 → 成功/失败子空间重叠多 → AND-NOT 运算可有效隔离判别方向 → 引导增益大
2. COAST 中实证：$\rho = 0.59, p = 0.002$（相似度与引导增益正相关）
3. 失败子空间包容度预测跨任务迁移效果（$r = 0.30$-$0.49$，$p < 0.005$）
4. 成功子空间包容度与迁移无关（$|r| < 0.13$），体现失败几何的跨任务共享性

## 代表工作

- [[COAST]]: 提出并验证子空间相似度作为引导效果的几何预测指标

## 相关概念

- [[Conceptor]]
- [[Activation Steering]]
- [[Cross-Task Transfer]]
