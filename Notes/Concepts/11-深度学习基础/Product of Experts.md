---
type: concept
aliases: [PoE, Product-of-Experts]
---

# Product of Experts

## 定义
将多个专家模型的概率分布相乘（取交集）以生成联合条件分布，每个专家捕捉不同方面的约束，乘积结果为同时满足所有约束的分布。

## 数学形式

$$
p(x \mid \mathcal{M}) \propto p_\text{query}(x) \prod_i p_i(x)
$$

在扩散模型中等价为分数函数相加：

$$
\nabla_x \log p(x \mid \mathcal{M}) = \nabla_x \log p_\text{query}(x) + \sum_i \nabla_x \log p_i(x)
$$

## 核心要点
1. 各专家独立建模，组合无需重新训练
2. 缺陷：若专家分布支撑不重叠，乘积退化（模式崩溃）
3. 在扩散模型中，组合仅发生在采样阶段，即分数函数加权求和

## 代表工作
- [[CoME]]: 提出 Product of Contrastive Experts（PoCE）解决朴素 PoE 的模式崩溃问题

## 相关概念
- [[Product of Contrastive Experts]]
- [[Classifier-Free Guidance]]
- [[扩散模型]]
