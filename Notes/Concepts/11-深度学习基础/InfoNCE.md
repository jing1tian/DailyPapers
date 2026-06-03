---
type: concept
aliases: [InfoNCE Loss, Noise Contrastive Estimation]
---

# InfoNCE

## 定义
对比学习中的核心损失函数，最大化正样本对之间的互信息，同时通过负样本区分不相似的表示。

## 数学形式
$$\mathcal{L}_{\text{InfoNCE}} = -\mathbb{E}\left[\log \frac{\exp(\text{sim}(z_i, z_j)/\tau)}{\sum_{k} \exp(\text{sim}(z_i, z_k)/\tau)}\right]$$

## 核心要点
1. 温度参数 $\tau$ 控制分布的锐度
2. RS-CL 用 InfoNCE 构建 Robot State-aware 对比损失
3. 正样本对通常是同一时刻的不同视角或同一任务的不同时刻

## 代表工作
- RS-CL (2510.01711): 基于 InfoNCE 的 VLA 表示正则化

## 相关概念
- [[对比学习]]
- [[SigLIP2]]
