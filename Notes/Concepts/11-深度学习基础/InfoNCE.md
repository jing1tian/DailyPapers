---
type: concept
aliases: [InfoNCE Loss, Noise Contrastive Estimation, 对比损失]
---

# InfoNCE

## 定义
一种对比学习损失函数，通过最大化正样本对的互信息下界来学习判别性表示。

## 数学形式
$$\mathcal{L}_{\text{InfoNCE}} = -\mathbb{E}\left[\log \frac{\exp(f(x)^\top f(x^+)/\tau)}{\exp(f(x)^\top f(x^+)/\tau) + \sum_{i=1}^N \exp(f(x)^\top f(x_i^-)/\tau)}\right]$$

其中 $x^+$ 是正样本，$x_i^-$ 是负样本，$\tau$ 是温度系数。

## 核心要点
1. 是 SimCLR、MoCo、CLIP 等对比学习方法的基础损失
2. 负样本数量 $N$ 越大，估计越精确，但计算开销越高
3. **RS-CL** 用 InfoNCE 以 proprioception 状态为锚点做 VLA 表示正则化
4. 温度 $\tau$ 控制分布的平滑度：小 $\tau$ 更 sharp，大 $\tau$ 更 uniform

## 代表工作
- [[RS-CL]]: 用 InfoNCE 构建 Robot State-aware Contrastive Loss
- [[CLIP]]: 用 InfoNCE 对齐视觉和语言表示

## 相关概念
- [[CLIP]] — 大规模对比学习的代表
- [[扩散策略]] — 表示学习的下游应用
