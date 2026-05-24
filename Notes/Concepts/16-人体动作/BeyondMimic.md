---
type: concept
aliases: [BeyondMimic humanoid control]
---

# BeyondMimic

## 定义
人形机器人运动控制方法，超越直接动作模仿（motion imitation），通过学习任务级别的行为策略而非单纯跟踪参考运动，实现更自然、更鲁棒的全身控制，是 SONIC 的对比基线之一。

## 核心要点
1. 不直接最小化与参考运动的 MPJPE 误差
2. 以任务奖励为主要学习信号，运动自然性作为辅助约束
3. 在 OOD 运动内容上比纯 motion tracking 方法更鲁棒
4. SONIC 将其作为 humanoid scaling 研究的对比基线

## 代表工作
- [[BeyondMimic]]：humanoid control 基线，被 SONIC 对比引用

## 相关概念
- [[SMPL]]
- [[SONIC]]
- [[PPO]]
