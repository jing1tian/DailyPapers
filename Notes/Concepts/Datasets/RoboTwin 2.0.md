---
type: concept
aliases: [RoboTwin 2.0, RoboTwin]
---

# RoboTwin 2.0

## 定义

基于 Sapien 物理引擎的双臂操作 benchmark，包含 10 个下游评估任务和 5 个具有可随机化背景（最多 10k 种）的能力提取任务，支持大规模视觉多样性测试。

## 核心要点

1. 双臂协作操作场景，任务难度高于单臂 benchmark
2. 支持背景随机化：单一背景（clean）和 10k+ 种随机背景（randomized）两种配置
3. 每个任务 100 次 rollout 评估，100 episodes 训练数据
4. 可作为 OOD 测试环境（与 LIBERO 场景完全不同）

## 代表工作

- [[CapVector]]: 在 RoboTwin 2.0 上验证 capability vector 的跨环境 OOD 迁移能力；随机化背景数据集比单一背景提取的向量性能显著更高
- [[SVP-IL]]: 在 RoboTwin 2.0 的 Clean/Randomized 两种协议上评测，验证显式空间提示对视觉分布偏移的鲁棒性优于纯数据增强

## 相关概念

- [[LIBERO]]
- [[VLA]]
