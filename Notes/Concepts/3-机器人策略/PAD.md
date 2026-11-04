---
type: concept
aliases: [Policy Adaptation via Dreaming, PAD]
---

# PAD（Policy Adaptation via Dreaming）

## 定义
一种利用 World Model 生成的 "梦境" 数据来在线适应策略的方法，在真实交互受限时通过 WM rollout 进行策略自适应。

## 核心要点
1. 不需要大量真实环境交互，用 WM 想象未来来适应策略
2. 属于 test-time adaptation 范畴
3. 是 WISE 等 WAM test-time scaling 工作中对比的基线

## 代表工作
- [[WISE]]: 以 PAD 作为 test-time WAM 方法的对比基线

## 相关概念
- [[DreamerV3]]
- [[WISE]]
- [[RSSM]]
