---
type: concept
aliases: [World Model Gym]
---

# WorldGym

## 定义
WorldGym：世界模型评测基准，提供标准化的 rollout 环境用于比较不同 action-conditioned world model 的生成质量和推理速度。

## 核心要点
1. 标准化 world model 评测协议
2. 支持多样任务下的 long-horizon rollout 评测
3. FVD 等生成质量指标和 inference latency 共同评测

## 代表工作
- [[DriftWorld]]: 在 WorldGym 上测试 drifting WM 的速度和质量

## 相关概念
- [[DriftWorld]]
- [[MuJoCo]]
