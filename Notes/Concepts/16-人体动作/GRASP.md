---
type: concept
aliases: [Gradient Recovery from Archived Samples for Proposals, 梯度引导运动重定向]
---

# GRASP

## 定义
接触丰富的物理约束运动重定向方法：结合 [[MPPI]] 随机采样探索和可微分物理梯度精化，将人类示例动作 retarget 到机器人控制，同时满足 contact dynamics 约束。

## 数学形式
$$\text{GRASP} = \text{MCMC探索} + \nabla_\theta J(\theta)\text{梯度精化}$$

## 核心要点
1. 双阶段：MPPI sampling 做全局探索，梯度 refinement 做局部精化
2. 避免 pure sampling 效率低和 pure gradient 陷入 local minima
3. 适用于 dexterous manipulation 和 contact-rich retargeting
4. 对比 [[OmniRetarget]], [[GMR]]；数据集 [[OakInk2]], [[AMASS]]
5. 注意：当前版本仅 sim 验证，无真实机器人实验

## 相关概念
- [[MPPI]]
- [[OmniRetarget]]
- [[GMR]]
- [[运动重定向]]
