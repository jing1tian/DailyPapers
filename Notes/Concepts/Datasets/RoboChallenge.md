---
type: concept
aliases: [Robo Challenge]
---

# RoboChallenge

## 定义

RoboChallenge 是一个真实机器人操作评测 benchmark，采用服务式部署（中央评测站），支持 30 个操作任务，使用视频叠加实现高可复现性，但不支持分布式部署。

## 核心要点

1. 任务多样（30 个任务），覆盖范围广
2. 视频叠加实现高可复现性
3. 服务式（Service-based）设计：实验室需要将机器人送到评测站，不支持各实验室自主部署
4. 不提供专家演示数据集

## 代表工作

- [[VLA-REPLICA]]: 将 RoboChallenge 作为对比，指出其无法分布式评测的局限

## 相关概念

- [[VLA-REPLICA]]
- [[模仿学习]]
