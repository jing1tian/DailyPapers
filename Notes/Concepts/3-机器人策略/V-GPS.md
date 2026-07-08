---
type: concept
aliases: [Value-Guided Policy Search, 价值引导策略搜索]
---

# V-GPS（Value-Guided Policy Search）

## 定义

通过训练一个价值函数对策略输出的候选动作进行单步重排序，从而无需修改策略权重即可提升策略质量的方法。是 [[Policy Steering]] 的早期代表。

## 核心要点

1. **单步重排**: 仅对当前时刻的候选动作评分，不做多步轨迹预测
2. **与 DreamSteer 的区别**: V-GPS 缺乏多步世界模型预测，无法评估动作的长期影响；DreamSteer 通过 H=10 步 rollout 提供更全面的轨迹评估

## 代表工作

- [[DreamSteer]] 中作为对比基线，DreamSteer 将其扩展到多步轨迹级别的引导

## 相关概念

- [[Policy Steering]]
- [[VLAC]]
- [[VLA（视觉-语言-动作模型）]]
