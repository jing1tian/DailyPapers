---
type: concept
aliases: [RoboLab, RoboLab Benchmark, 开环策略评估 benchmark]
---

# RoboLab

## 定义
一个用于评估机器人操作策略的 benchmark，通过世界模型开环仿真成功率与真实机器人执行参考成功率的相关性，提供无需实机部署的策略比较方案。

## 核心要点
1. 包含 π₀、π₀.₅、GR00T N1.7、Cosmos-3 Edge、Cosmos-3 Nano 等主流策略的评估
2. 每个策略/任务点聚合 10 次试验
3. Hydra-0 在 RoboLab 上实现 **Pearson r = 0.96** 的仿真-真实相关性
4. 目标：用开环世界模型评估替代昂贵的真实机器人测试

## 代表工作
- [[Hydra-0]]: 在 RoboLab 上验证开环策略评估的有效性

## 相关概念
- [[Hydra-0]]: 提出/使用此 benchmark 的主要论文
