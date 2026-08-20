---
type: concept
aliases: [Instruction Attention Boost, Additive Attention Bias]
---

# InstABoost

## 定义
一种推理时注意力干预方法：在 Transformer 的 softmax 前对指令相关的 key-query logit 施加常数加性偏置，使模型在生成时更强调指定的指令 token，同时兼容 FlashAttention。

## 数学形式
标准注意力：$\text{Attention}(Q, K, V) = \text{softmax}(QK^T / \sqrt{d}) V$

InstABoost 修改为：
$$\text{logit}_{ij}' = \text{logit}_{ij} + \alpha \cdot \mathbf{1}[j \in \text{instruction tokens}]$$

其中 $\alpha$ 为正偏置常数，偏置在 softmax 前加入，对规则竞争产生指数级影响。

## 核心要点
1. 加性偏置（additive bias）vs 乘法 upweight（PASTA）：前者保持 FlashAttention 的 kernel 兼容性
2. 无需梯度更新，推理时零额外训练
3. 适用于 VLA 等需要强化特定指令跟随的生成式策略

## 代表工作
- Inference-Time Attention Steering (2608.17095): 提出 InstABoost 并与 PASTA 对比

## 相关概念
- [[VLA]]
- [[FlashWorld]]
