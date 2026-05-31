---
type: concept
aliases: [DeepMind Control Suite, dm_control, DMControl Suite]
---

# DMControl（DeepMind Control Suite）

## 定义

DeepMind Control Suite（dm_control）是 DeepMind 开发的连续控制强化学习基准环境套件，基于 [[MuJoCo]] 物理引擎，提供从简单摆锤到复杂人形机器人的标准化连续控制任务。

## 核心要点

1. **基于 MuJoCo**: 使用 MuJoCo 物理引擎，提供高精度刚体动力学仿真
2. **标准化 API**: 遵循统一的 `env.reset() / env.step(action)` 接口
3. **多样任务**: 包含 Cartpole、Cheetah、Humanoid、Walker、Finger、Reacher 等数十个任务
4. **连续动作空间**: 所有任务均为连续动作空间，适合测试策略梯度方法
5. **可扩展性**: 支持自定义 XML 模型定义新任务

## 常见任务

| 任务 | 描述 | 自由度 |
|------|------|--------|
| Cheetah-run | 半猎豹向前奔跑 | 6 |
| Humanoid-stand/walk | 人形机器人站立/行走 | 21 |
| Walker-walk/run | 双足行走者 | 6 |
| Reacher-easy/hard | 机械臂末端到达目标 | 2 |
| Cartpole-swingup | 倒立摆摆起保持 | 1 |

## 在 stable-worldmodel 中

[[stable-worldmodel]] 将 DMControl 任务整合进 SWM 环境套件，每个 DMControl 环境提供 6-10 个可控变异因子（FoV），支持对物理参数（质量、摩擦、重力）和视觉属性的系统性扰动。

## 代表工作

- [[stable-worldmodel]]: 将 DMControl 纳入标准化 WM 评估平台
- [[Dreamer]]: 在 DMControl 上验证基于世界模型的规划

## 相关概念

- [[MuJoCo]]: DMControl 的底层物理引擎
- [[Gymnasium]]: DMControl 与 Gymnasium 接口兼容
- [[世界模型]]: DMControl 是 WM 方法的常用评估环境
