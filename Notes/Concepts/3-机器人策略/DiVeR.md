---
type: concept
aliases: [Decision-Critical Verifier]
---

# DiVeR

## 定义
Park 等（2026，arXiv 2610.04933，"DiVeR: Decision-Critical Verifier Learning for VLA Test-Time Scaling"）提出的 VLA 测试时扩展验证器，聚焦"决策关键时刻"（抓取接触点、放置精度帧）评分候选 rollout，而非全轨迹通用 reward model。

## 核心要点
1. 识别任务执行中影响成败的关键帧（决策关键时刻）
2. 用关键帧置信度而非全轨迹评分进行 Best-of-N 候选选择
3. 在 RoboCasa 和 LIBERO 演示数据上训练验证器

## 代表工作
- Park et al. (2026): NTU, Osaka, Tokyo, Wisconsin

## 相关概念
- [[VLA]]
- [[RoboMonkey]]
