---
type: concept
aliases: [AnchorDream]
---

# AnchorDream

## 定义

一种机器人视频生成方法，通过将渲染的机器人运动视频作为条件锚点（anchor）输入视频扩散模型，避免生成过程中的 embodiment hallucination（机器人形态幻觉）。

## 核心要点

1. **运动锚点**: 用仿真渲染的机器人视频条件化生成，确保生成视频中机器人形态与真实机器人一致
2. **局限性**: 需要针对每个具体任务或环境单独微调，泛化能力有限，无法零样本迁移到新场景
3. **与 RoboDream 的关系**: RoboDream 在此基础上引入了物体先验和场景先验，实现了真正的零样本组合泛化

## 代表工作

- [[RoboDream]]: 直接对比基线，RoboDream 在零样本泛化能力上超越 AnchorDream

## 相关概念

- [[视频扩散模型]]
- [[RoboDream]]
- [[DreamGen]]
