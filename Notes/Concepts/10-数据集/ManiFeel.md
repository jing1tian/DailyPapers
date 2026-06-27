---
type: concept
aliases: [ManiFeel benchmark]
---

# ManiFeel

## 定义
可复现、可扩展的视触觉操作策略学习仿真基准（arXiv 2505.18472），涵盖插入、装配、重定向、搜索、分类等多种接触密集型机器人操作任务，支持对 RGB-only 策略、触觉条件非世界模型策略、物理感知 [[World Action Model|WAM]] 进行统一对比。

## 核心要点
1. 提供 9 个仿真任务：ball sorting、bolt-nut assembly、bulb insertion、gear insertion、object search、peg insertion、peg reorientation、power insertion、USB insertion
2. 系统研究"哪些任务类型最受益于触觉感知"，揭示触觉表征方法对策略学习的影响因素
3. 支持多种输入模态和触觉表征方法的对比实验
4. 已成为多个触觉 WAM/视触觉策略工作（如 [[Tactile-WAM]]）的标准评估基准

## 代表工作
- 原始论文: Luu, Zhou, Xu, Zhang, Qiu, She (2025) "ManiFeel: Benchmarking and Understanding Visuotactile Manipulation Policy Learning"
- [[Tactile-WAM]]: 使用 ManiFeel 评估，接触为主子集成功率达 87.5%（vs DreamZero 1.5%）

## 相关概念
- [[Tactile-WAM]]
- [[World Action Model]]
- [[UniT]]
