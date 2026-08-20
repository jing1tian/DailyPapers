---
type: concept
aliases: [KL Divergence, Kullback-Leibler Divergence, 相对熵, KL 散度]
---

# KL 散度 (KL Divergence)

## 定义

衡量两个概率分布 $P$ 和 $Q$ 之间差异的非对称度量，表示"用分布 $Q$ 近似分布 $P$ 时损失的额外信息量"。

## 数学形式

对离散分布：

$$
\mathrm{KL}(P \| Q) = \sum_x P(x) \log \frac{P(x)}{Q(x)}
$$

对连续分布：

$$
\mathrm{KL}(P \| Q) = \int p(x) \log \frac{p(x)}{q(x)} \, dx
$$

性质：$\mathrm{KL}(P \| Q) \geq 0$，当且仅当 $P = Q$ 时取 0；**非对称**：$\mathrm{KL}(P \| Q) \neq \mathrm{KL}(Q \| P)$。

## 核心要点

1. **非对称性**: $\mathrm{KL}(P \| Q)$ 称为"前向 KL"，最小化它使 $Q$ 覆盖 $P$ 的所有模式（均值寻求）；$\mathrm{KL}(Q \| P)$ 称为"反向 KL"，使 $Q$ 集中在 $P$ 的某个模式（模式寻求）。
2. **与交叉熵的关系**: $\mathrm{KL}(P \| Q) = H(P, Q) - H(P)$，其中 $H(P,Q)$ 为交叉熵，$H(P)$ 为 $P$ 的熵。最小化 KL 散度等价于最小化交叉熵（固定 $P$ 时）。
3. **在变分推断中的作用**: VAE 的 ELBO 中包含 $\mathrm{KL}(q(z|x) \| p(z))$，将近似后验拉向先验。
4. **在知识蒸馏/分布对齐中的应用**: 用于让学生模型预测分布接近教师模型，或让模型的置信分布对齐质量导出分布（如 [[充分性校准]]）。

## 代表工作

- [[LoopVLA]]: Stage 2 充分性校准使用 $\mathrm{KL}(q \| p)$ 对齐置信分布与动作质量分布。
- VAE、GRPO、[[FlowGRPO]] 等多种生成/训练方法均使用 KL 散度。

## 相关概念

- [[充分性校准]]
- [[RLHF]]
- [[Flow Matching]]
