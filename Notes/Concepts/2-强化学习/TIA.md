---
type: concept
aliases: [Task-Irrelevant Attention, TIA]
---

# TIA (Task-Irrelevant Attention)

## 定义
一种用于 model-based RL 的表示分解方法，通过注意力机制将观测分离为任务相关和任务无关两部分，抑制 distractor 对 world model 预测的影响。

## 核心要点
1. 将 latent state 显式分解：$z = z_{task} \oplus z_{irr}$
2. World model 只对 $z_{task}$ 部分做 action-conditioned 前向预测
3. $z_{irr}$ 部分用独立的无 action 条件的模型建模（纯观察者）
4. 通过对抗训练或 mutual information 最小化强制两部分解耦

## 代表工作
- [[TIA]] (Deng et al., 2021): 原始论文，在 DMC 干扰物环境验证
- [[Dueling-WM]]: 用 advantage-style action channel 替代显式分解，避免了 TIA 的对抗训练不稳定问题

## 相关概念
- [[RSSM]]
- [[DreamerV3]]
- [[RepWAM]]
