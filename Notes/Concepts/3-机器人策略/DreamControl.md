---
type: concept
aliases: [DreamControl]
---

# DreamControl

## 定义

用于场景交互的人形机器人全身控制方法，使用引导扩散生成人体启发的运动参考，结合强化学习实现全身操作，是 DreamVLA 流水线中参考生成和奖励设计的基础。

## 核心要点

1. 利用生成式人体运动模型（如 OmniControl）生成满足任务约束的参考轨迹
2. 结合强化学习在仿真中训练全身控制策略，使用参考运动提供密集监督信号
3. DreamVLA 直接采用 DreamControl 的轨迹生成流程和特权观测 / 奖励设计

## 代表工作

- DreamControl (Kalaria, Harithas, Katara, Kwak, Bhagat, Sastry, Sridhar, Vemprala, Kapoor, Huang, 2025, arXiv:2509.14353): 原始论文

## 相关概念

- [[OmniControl]]
- [[SONIC]]
- [[Whole-Body Control]]
- [[DreamVLA]]
- [[Human-to-Robot Retargeting]]
