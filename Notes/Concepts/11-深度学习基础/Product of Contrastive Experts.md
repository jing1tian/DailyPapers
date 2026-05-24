---
type: concept
aliases: [PoCE, Contrastive Product of Experts, Product of Contrastive Experts]
---

# Product of Contrastive Experts

## 定义
[[Product of Experts]] 的改进版本，为每个专家引入对比系数 $\alpha_i > 1$ 和无条件基线 $\bar{p}_i$，在放大条件信号的同时抑制虚假模式，而不改变主模式的形状。

## 数学形式

$$
p(x \mid \mathcal{M}) \propto \prod_{i=1}^{K} \tilde{p}_i(x), \quad \tilde{p}_i(x) \propto p_i(x)^{\alpha_i} \cdot \bar{p}_i(x)^{1-\alpha_i}
$$

采样时的分数函数合成：

$$
\nabla_{x_t} \log p_{\text{CoM}}(x_t) = \sum_k \left[ \alpha_k \nabla_{x_t} \log p_k(x_t) + (1-\alpha_k) \nabla_{x_t} \log \bar{p}_k(x_t) \right]
$$

## 核心要点
1. $\alpha_i > 1$ 时，与 [[Classifier-Free Guidance]] 等价（单专家情况下 $\alpha = 1 + w$）
2. PoCE 是 CFG 到多专家的自然推广
3. Proposition 1（论文理论结果）：在核密度估计框架下，对比混合重加权主导核而不扭曲其局部几何
4. 避免了朴素 PoE 的方差崩溃问题

## 代表工作
- [[CoME]]: PoCE 的提出者，将其用于三类记忆专家（STM/LTM/SLTM）的组合

## 相关概念
- [[Product of Experts]]
- [[Classifier-Free Guidance]]
- [[扩散模型]]
