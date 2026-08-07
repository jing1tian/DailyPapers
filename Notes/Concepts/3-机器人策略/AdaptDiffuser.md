---
type: concept
aliases: [Adaptive Diffuser, AdaptDiffuser]
---

# AdaptDiffuser

## 定义
AdaptDiffuser 是一种自适应扩散策略规划方法，在测试时通过奖励引导的 diffusion 采样来适应新任务，无需任务特定的微调数据。

## 核心要点
1. 利用扩散过程的灵活性，在 denoising 过程中注入奖励信号引导轨迹生成
2. 测试时自适应：不需要重新训练，通过引导采样即可迁移到新任务
3. 常与 offline RL 方法（[[IQL]]、[[CQL]]）对比
4. 在 [[MuJoCo]] [[AntMaze]] 等 offline RL benchmark 上评估

## 代表工作
- [[Curiosity-Diffuser]]: 将 RND 好奇心信号与 diffusion 结合，作为 AdaptDiffuser 的比较基线

## 相关概念
- [[Diffusion Policy]]
- [[IQL]]
- [[RND]]
