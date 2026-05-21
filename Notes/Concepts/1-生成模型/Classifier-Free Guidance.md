---
type: concept
aliases: [CFG, Classifier Free Guidance, 无分类器引导]
---

# Classifier-Free Guidance

## 定义
在扩散模型采样时，用条件分数与无条件分数的线性外插来增强条件生成强度，无需额外分类器，仅需训练时随机丢弃条件信号。

## 数学形式

$$
\tilde{\varepsilon}_\theta(x_t, c) = (1+w)\,\varepsilon_\theta(x_t, c) - w\,\varepsilon_\theta(x_t, \emptyset)
$$

等价地用分数函数表示：

$$
\nabla_{x_t} \log \tilde{p}(x_t | c) = (1+w)\,\nabla_{x_t} \log p(x_t | c) - w\,\nabla_{x_t} \log p(x_t)
$$

## 核心要点
1. $w > 0$ 时增强条件一致性，代价是样本多样性略降
2. 训练时以概率 $p_\text{drop}$（通常 10-20%）将条件替换为空条件
3. Product of Contrastive Experts（PoCE）是 CFG 的推广——每个专家独立施加对比引导

## 代表工作
- [[CoME]]: PoCE 将 CFG 推广到多专家场景，$\alpha_i > 1$ 对应 CFG 的 $(1+w)$

## 相关概念
- [[扩散模型]]
- [[Product of Contrastive Experts]]
