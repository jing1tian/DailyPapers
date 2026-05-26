---
type: concept
aliases: [DeepMind Control Suite, dm_control]
---

# DMC

## 定义
DeepMind Control Suite，基于 MuJoCo 的连续控制 benchmark 套件，包含 Cheetah、Walker、Hopper、Humanoid 等经典任务，是 model-based RL 和连续控制研究的标准测试平台。

## 核心要点
1. 所有任务基于 MuJoCo 物理引擎，提供像素和状态两种观测模式
2. 经典任务：cheetah-run、walker-walk、humanoid-stand、reacher-easy
3. 样本效率是 DMC 的主要评估指标（步数限制在 100K-500K）
4. 像素模式（DMC-Pixels）用于评估视觉表征学习
5. DreamerV3、GPLD、TD-MPC 等 world model 论文的主要 benchmark

## 代表工作
- [[DreamerV3]]: 在 DMC 上展示世界模型的样本效率
- [[GPLD]]: 在 DMC 上验证梯度惩罚潜动态的有效性

## 相关概念
- [[MuJoCo]]
- [[DreamerV3]]
- [[强化学习]]
