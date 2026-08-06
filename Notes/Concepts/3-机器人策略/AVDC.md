---
type: concept
aliases: [Action-conditioned Video Diffusion Controller]
---

# AVDC

## 定义
Action-conditioned Video Diffusion Controller：用动作条件视频扩散模型预测未来帧，再从预测视频中提取机器人动作的方法。

## 核心要点
1. 视频扩散模型作为 world model 生成未来观测
2. 从生成的视频序列中逆向提取动作序列
3. 先预见后行动的"imagination-then-act"范式

## 代表工作
- [[DriftWorld]]: 与 AVDC 对比，单步快速 WM 替代迭代视频生成

## 相关概念
- [[DriftWorld]]
- [[WAM]]
