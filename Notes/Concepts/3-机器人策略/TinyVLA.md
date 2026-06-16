---
type: concept
aliases: [TinyVLA, 轻量 VLA]
---

# TinyVLA

## 定义
轻量化 VLA（Vision-Language-Action）模型，通过模型压缩/蒸馏实现低延迟、低资源开销的机器人控制策略，是 SmolVLA 等小型 VLA 系列的代表之一。

## 核心要点
1. 减少参数量，适合边缘设备部署
2. [[PhysVLA]] 论文中与 OpenVLA、CogACT 等并列作为 baseline
3. 与 [[SmolVLA]]、[[ReactVLA]] 属于同一"轻量高效 VLA"方向
4. 牺牲部分性能换取推理速度，适合实时控制场景

## 代表工作
- [[PhysVLA]]：TinyVLA 作为对比 baseline

## 相关概念
- [[SmolVLA]]
- [[OpenVLA]]
- [[ReactVLA]]
