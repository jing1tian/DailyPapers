---
type: concept
aliases: [Posterior-Prior Epistemic Gap]
---

# PPE-Gap

## 定义
The information gap between posterior (observation-updated) and prior (imagined) state distributions in world model rollouts; large gap means imagination loses task-relevant signals

## 数学形式
$$\text{Gap} = D_{KL}(q(z_t|o_t, z_{t-1}) \| p(z_t|z_{t-1}, a_{t-1}))$$

## 核心要点
1. 见论文摘要与来源

## 代表工作
- [[PPE-Gap]]: 参见论文

## 相关概念
- [[RSSM]]
- [[DreamerV3]]
