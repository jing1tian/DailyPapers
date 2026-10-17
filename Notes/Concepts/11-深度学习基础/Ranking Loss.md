---
type: concept
aliases: [排名损失, Hinge Loss, Margin Ranking Loss, 对比排名损失]
---

# Ranking Loss（排名损失）

## 定义

一类通过**相对排序约束**训练模型的损失函数，要求正样本对的相似度/质量高于负样本对至少 $\mu$ 的边距（margin），以 hinge 函数强制执行。

## 数学形式

标准 Margin Ranking Loss：

$$
\mathcal{L}_{\text{rank}} = \left[\mu + d(x, x^+) - d(x, x^-)\right]_+
$$

批次不匹配排名损失（TS-WAM [[delta-lwm]] 变体）：

$$
\mathcal{L}_{\text{dep}} = \left[\mu + \left\|D_\psi(F_t, \hat{z}_t) - \Delta F_t\right\|_2^2 - \left\|D_\psi(F_t, \tilde{z}_t) - \Delta F_t\right\|_2^2\right]_+
$$

其中 $\tilde{z}_t$ 为 batch 内随机置换的负样本潜在动作。

## 核心要点

1. **相对而非绝对**: 不要求预测值达到某个绝对阈值，只要求正样本比负样本"好" $\mu$ 以上
2. **Hinge 函数 $[\cdot]_+$**: 只有当约束被违反时才产生梯度，已满足约束的样本对不贡献损失（稀疏梯度）
3. **边距 $\mu$**: 决定正负样本之间要求的间隔大小，过大可能过于严格，过小则约束力弱
4. **负样本构造**: 负样本的质量至关重要
   - **Batch 内置换**（[[delta-lwm]] 方式）：将同 batch 内其他样本的条件作为负样本，简单高效
   - **Hard Negative Mining**：主动选择最接近正样本的负样本，训练更困难但效果更好
5. **与对比损失的关系**: Ranking Loss 是 Triplet Loss / Contrastive Loss 的一般化形式，专注于排序而非绝对距离

## 代表工作

- [[delta-lwm]] / [[TS-WAM]]: 用排名损失确保世界模型解码器对潜在动作的真实依赖
- Triplet Network: 利用 Triplet Loss（一种 Ranking Loss）学习嵌入空间
- Learning to Rank: 信息检索中的排序学习

## 相关概念

- [[Contrastive Learning]]: 对比学习，与 Ranking Loss 思想相近
- [[Triplet Loss]]: Ranking Loss 的经典三元组形式
- [[delta-lwm]]: 使用批次不匹配 Ranking Loss 的代表应用
- [[Latent Action]]: delta-lwm 中被约束排序的潜在动作
