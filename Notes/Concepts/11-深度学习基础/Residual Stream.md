---
type: concept
aliases: [残差流, Residual Pathway]
---

# Residual Stream

## 定义

Transformer 架构中贯穿所有层的主信息通道，每层的注意力和 MLP 子层通过残差连接将输出加回到该流，使信息得以积累和传递。

## 数学形式

$$
h_\ell = h_{\ell-1} + \text{Attn}_\ell(h_{\ell-1}) + \text{MLP}_\ell(h_{\ell-1} + \text{Attn}_\ell(h_{\ell-1}))
$$

## 核心要点

1. 残差流是 Mechanistic Interpretability 分析的核心对象
2. 不同层的残差流激活编码不同抽象层级的信息
3. 对残差流施加干预（加法或乘法）可影响后续所有层的计算
4. 在 VLA 中，动作专家（Action Expert）的残差流激活编码任务执行相关信息

## 代表工作

- [[COAST]]: 对 VLA 动作专家残差流施加对比 Conceptor 乘法门控，提升任务成功率

## 相关概念

- [[Activation Steering]]
- [[Conceptor]]
- [[Transformer]]
- [[VLA]]
