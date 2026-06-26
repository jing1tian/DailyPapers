---
type: concept
aliases: [Human-in-the-Loop SERL, HIL SERL, Precise Dexterous Robotic Manipulation]
---

# HIL-SERL

## 定义

HIL-SERL（Human-in-the-Loop Sample Efficient Robotic Reinforcement Learning）是 SERL 的扩展版本，通过允许人类操作员在训练过程中实时干预并提供正确动作，大幅加速 RL 机器人训练，实现从头高效学习精密操作技能。

## 核心要点

1. **从头 RL 训练**: 使用小型高斯策略从零开始训练，不依赖预训练模型先验
2. **人类干预机制**: 操作员可在任意时刻接管机器人，提供高质量示范动作
3. **干预数据融合**: 人类干预数据和自主探索数据共同存储在回放缓冲区
4. **高精度操作**: 在精密操作任务（如插针、旋转连接器）上取得极高成功率

## 局限性

- 不能利用预训练 VLA 的泛化能力，在新任务上需要从头训练
- 在 VLA 场景下效果有限，因其设计针对小型策略网络
- 需要大量人类监督时间

## 代表工作

- [[EXPO-FT]]: 与 HIL-SERL 对比，展示 VLA RL 微调的优越性
- [[FORCE]]: 旨在去除人类在环干预依赖，用分布式 Warm-up + VGPD 实现全自动化 RL 微调

## 相关概念

- [[SERL]]
- [[强化学习]]
- [[模仿学习]]
- [[行为克隆]]
