---
type: concept
aliases: [Benchmark for Autonomous Robot Navigation]
---

# BARN

## 定义
BARN（Benchmark for Autonomous Robot Navigation）是针对受限环境下移动机器人导航的 benchmark，包含 300 个随机生成的高度受限障碍场景，用于评估导航规划器在拥挤环境下的成功率和效率。

## 核心要点
1. **300 测试场景**：随机生成的网格世界，障碍密度高，通道极窄
2. **导航规划器标准测试台**：DWA、TEB、MPPI 等规划器的常用对比基准
3. **BARN Challenge**：每年举办竞赛，衡量自主导航系统的鲁棒性
4. **标准指标**：导航成功率（SR）、路径效率

## 代表工作
- [[APPLV]]: 在 BARN 上测试 VLA 自适应导航参数学习

## 相关概念
- [[DWA]]
- [[TEB]]
- [[MPPI]]
