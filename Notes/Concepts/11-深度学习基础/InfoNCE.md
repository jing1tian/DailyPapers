---
type: concept
aliases: [InfoNCE Loss, Noise Contrastive Estimation, 对比学习损失]
---

# InfoNCE

## 定义
Noise Contrastive Estimation 的信息论变体，用于自监督对比学习，最大化正样本对的互信息下界。

## 数学形式
$$\mathcal{L}_{\text{InfoNCE}} = -\mathbb{E}\left[\log \frac{\exp(f(x)^\top f(x^+)/\tau)}{\exp(f(x)^\top f(x^+)/\tau) + \sum_{j=1}^{N} \exp(f(x)^\top f(x_j^-)/\tau)}\right]$$

其中 $\tau$ 为温度系数，$x^+$ 为正样本，$x_j^-$ 为负样本。

## 核心要点
1. 等价于 N+1 分类问题的交叉熵损失，将正样本与 N 个负样本区分开
2. 温度 $\tau$ 控制分布尖锐程度：小 $\tau$ 使梯度集中在难负样本
3. 负样本数量越多，下界越紧（与真实互信息差距越小）
4. 常用于强制两个表示不相关（如 DWM 中分离 action/world effect）

## 代表工作
- [[CPC]]: 提出 InfoNCE，用于时序自监督预测
- [[SimCLR]]: 用 InfoNCE 做图像对比学习
- [[DWM]]: 用 InfoNCE 约束 action effect 和 world effect 的解耦

## 相关概念
- [[CEM]]（规划方法，独立于 InfoNCE）
- [[EMA]]（对比学习中常用 momentum encoder）
