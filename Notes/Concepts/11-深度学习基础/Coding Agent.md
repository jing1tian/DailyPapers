---
type: concept
aliases: [代码智能体, Software Engineering Agent]
---

# Coding Agent

## 定义

通过在代码执行终端（terminal）中运行代码、读取输出来学习和解决软件工程任务的 LLM agent，以 terminal 作为"世界"提供反馈。

## 核心要点

1. **Terminal = World**：coding agent 的终端是其世界，运行结果是反馈
2. **无权重更新**：依靠上下文中的经验积累解决新任务
3. **技能库增长**：代表工作 Voyager 构建可复用代码技能库
4. **迁移类比**：RoboScript 将此范式迁移到机器人操作（simulator = terminal）

## 代表工作

- [[Voyager]]: 在 Minecraft 中构建技能库的 coding agent
- [[SWE-Agent]]: 解决真实 GitHub issue 的 coding agent
- [[RoboScript]]: 将 coding agent 范式映射到机器人操作

## 相关概念

- [[World In-Context Learning]]
- [[In-Context Learning]]
- [[Reflexion]]
