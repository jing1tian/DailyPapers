---
type: concept
aliases: [TAAM, 触觉非对称注意力机制]
---

# Tactile Asymmetric Attention Mechanism (TAAM)

## 定义
[[Tactile-WAM]] 提出的注意力层机制，组合 [[VideoClean Attention Mask|VideoClean 掩码]]与[[Touch-Aware Bias|触觉感知偏置]]，让触觉 token 在视觉预训练的 [[World Action Model|World Action Model]] 中被"选择性路由"：阻断对视频预测路径的扰动，同时按需增强对动作生成路径的影响。

## 数学形式

$$
B_{q,k}=\bar{B}_{q,k}+\mathbb{I}[G(q)=A,\;G(k)=\tau]\,b^{\tau}_{a(k)}, \qquad \bar{B}_{q,k}=B^{0}_{q,k}+M^{\mathrm{vc}}_{q,k}
$$

其中 $M^{vc}$ 为 VideoClean 掩码（阻断 video-query 对 tactile-key 的访问），$b^\tau$ 为仅作用于 action-query→tactile-key 路径的触觉感知偏置。

## 核心要点
1. **非对称性**是核心设计：视频对触觉不可见（被阻断），动作对触觉可见且可被加权（保留+偏置）
2. 不缩放触觉特征本身，不增加任何其他 token 组对触觉的注意力——只精确调制"动作关注触觉锚点"的强度
3. 解决两个问题：触觉污染（VideoClean 部分）+ 触觉感知不完整（触觉感知偏置部分）
4. 偏置由预测的触觉变化代理驱动，因此模型"知道何时该更关注触觉"，而非始终均匀关注

## 代表工作
- [[Tactile-WAM]]: 提出该机制，在 ManiFeel 上整体成功率提升 38.9%，接触为主任务提升 86%

## 相关概念
- [[VideoClean Attention Mask]]
- [[Touch-Aware Bias]]
- [[World Action Model]]
- [[Dream-Tac]]（同类思想：接触门控自注意力 CASA）
