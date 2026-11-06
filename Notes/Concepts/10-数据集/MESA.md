---
type: concept
aliases: [MESA benchmark, Compositional Semantic Spatial benchmark]
---

# MESA

## 定义
机器人操作评估框架，针对 compositional、semantic、spatial 三个维度的泛化能力设计 70 个任务（MESA-70），基于 [[RoboCasa]] 环境，支持 OOD 和双臂（BiMESA）评估。

## 核心要点
1. 三维度：组合泛化（新指令组合）、语义泛化（新物体/场景）、空间泛化（新布局）
2. 区分 in-distribution 和 OOD 评估，更细粒度诊断
3. 整合 [[MimicGen]] 生成数据，对比 [[LIBERO]], [[CALVIN]], [[RLBench]]
4. Georgia Tech 出品

## 相关概念
- [[RoboCasa]]
- [[LIBERO]]
- [[MimicGen]]
- [[RLBench]]
