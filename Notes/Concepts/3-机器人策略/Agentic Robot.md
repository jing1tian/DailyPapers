---
type: concept
aliases: [Agentic Robotics, 智能体机器人, 自主机器人]
---

# Agentic Robot

## 定义

一种机器人系统，其中 Agent 自主解释人类指令、与机器人环境交互、自我纠正执行错误，并在任务过程中动态调整策略——不仅执行动作，还主动规划、采集信息、从失败中恢复。

## 核心要点

1. **四大功能域**: Model（模型决策）、Data（数据采集与精炼）、Environment（环境构建与测试）、Harness（能力编排与执行框架）
2. **与传统策略的区别**: 不只是动作预测，而是目标导向的多步骤编排，包括失败检测与恢复
3. **演化路径**: 传统策略 → VLA → World Model → Agentic Robot
4. **核心能力**: 长时序规划、工具调用、记忆管理、执行监控、自我纠错

## 代表工作

- [[AgenticRobots-Survey]]: 首个系统性综述，提出 4 域 17 类框架（2026）

## 相关概念

- [[VLA]]: 前置范式，提供视觉-语言-动作联合建模
- [[World Model]]: 提供想象能力，支持模型内规划
- [[Harness]]: Agentic Robot 的执行框架层
- [[Code as Policies]]: Skill 域的重要技术基础
