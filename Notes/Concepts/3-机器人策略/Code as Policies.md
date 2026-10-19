---
type: concept
aliases: [CaP, Code-as-Policy, 代码即策略]
---

# Code as Policies

## 定义

一种机器人控制范式：用语言模型生成机器人控制代码（Python 程序），通过组合感知和控制 API 实现任务，并支持动态定义缺失函数。

## 核心要点

1. **角色转变**: 从选择预定义技能（如 SayCan），变为组合 API 生成完整程序
2. **扩展性**: 技能集成为词汇表而非菜单，可按需生成新函数
3. **演化路径**: Code as Policies → ProgPrompt（带断言）→ VoxPoser（代价图代码）→ Skill Libraries → Skill Evolution
4. **与 Skill 的区别**: 技能是跨任务复用的，而基本代码生成是为单任务生成后丢弃

## 代表工作

- Code as Policies (Liang et al., 2023): 原始论文
- [[VoxPoser]]: 生成代价图代码驱动运动规划器
- ProgPrompt: 带断言的任务计划代码
- [[AgenticRobots-Survey]]: 将此范式纳入 Skill 域演化脉络

## 相关概念

- [[Harness]]: 提供代码执行的基础设施层
- [[VoxPoser]]: 重要扩展
- [[Agentic Robot]]: 代码策略是 Agentic 系统的重要组件
