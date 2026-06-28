---
type: concept
aliases: [模型汤, 权重平均, Weight Averaging]
---

# Model Soup

## 定义
一种模型融合（model merging）技术：将多个在不同数据/任务上独立微调、但共享同一预训练初始化的模型权重直接做平均（或加权平均），得到一个不增加推理开销、却能综合多个微调模型能力的单一模型，常用于把多个 domain-specific SFT 模型合并回一个通用模型。

## 数学形式
$$
\theta_{\text{soup}} = \frac{1}{N}\sum_{i=1}^{N} \theta_i
$$

其中 $\theta_i$ 是第 $i$ 个在相同初始化基础上、针对不同领域/任务微调得到的模型权重，$\theta_{\text{soup}}$ 是融合后的最终权重。

## 核心要点
1. 前提假设是所有待融合模型共享同一预训练初始化，使各自的权重在同一损失盆地（loss basin）附近，直接平均才不会破坏性能
2. 相比集成（ensemble），Model Soup 不增加推理时的参数量和计算量，融合后只有一份权重
3. 是模型融合家族中最简单的基线方法，更精细的变体（如 TIES、DARE、WUDI-Merging）通过稀疏化、符号一致性筛选等方式选择性合并参数，缓解任务间冲突、进一步提升融合效果
4. 常作为多个 domain-specific 微调阶段之后、上线前的最后一步，把分领域强化的能力重新汇聚到一个统一模型中

## 代表工作
- [[Kairos]]: 在 Domain-specific SFT 之后用 Model Soup、CART、TIES、DARE、WUDI-Merging 等多种模型融合策略组合，再接 DPO 强化学习收尾，得到最终部署模型

## 相关概念
- [[DPO]]
- [[Kairos]]
