---
type: concept
aliases: [CALVIN Benchmark, Language Conditioned Robot Manipulation]
---

# CALVIN

## 定义
用于评估语言条件机器人操作的长视程 benchmark，要求机器人连续完成多步任务（最长 5 步），是评估 VLA 策略泛化性的主流基准之一。

## 核心要点
1. 四个环境变体（ABC→D）测试跨场景泛化
2. 评估指标为连续任务链成功率（1/5 chain）
3. 支持语言指令条件的操作任务
4. 被 OpenVLA、Diffusion Policy 等主流工作用作评估基准

## 代表工作
- [[OpenVLA]]: 在 CALVIN 上验证 VLA 泛化能力
- [[StableVLA]]: 在 CALVIN 上评估鲁棒性

## 相关概念
- [[LIBERO]]
- [[MetaWorld]]
- [[RoboTwin]]
