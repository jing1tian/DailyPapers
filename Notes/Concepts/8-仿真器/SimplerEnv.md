---
type: concept
aliases: [SimplerEnv, SIMPLER]
---

# SimplerEnv

## 定义
UC Berkeley 提出的机器人操作评测框架，专为评估"sim-to-real"可信度设计——在标准化仿真环境中快速测试真实机器人策略的性能，与真实机器人结果高度相关。

## 核心要点
1. 基于 ManiSkill2 + SAPIEN 构建，覆盖 Google Robot 和 WidowX 两种机器人配置
2. 提供标准化的操作任务（pick & place、抽屉、旋钮等）
3. "Sim-to-real 可信度"是其核心 claim——sim 结果与真实机器人结果相关性高
4. 常用于快速筛选 VLA 模型，再在真实机器人上精细测试

## 代表工作
- Li et al. (2024): "Evaluating Real-World Robot Manipulation Policies in Simulation"

## 相关概念
- [[MuJoCo]]
- [[VLA]]
- [[RoboCasa]]
