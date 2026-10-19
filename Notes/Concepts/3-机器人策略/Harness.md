---
type: concept
aliases: [Robot Harness, 机器人执行框架, Agentic Harness]
---

# Harness

## 定义

在 Agentic Robotics 中，Harness 是让 Agent 能力对机器人可用的软件框架——管理各组件的调用方式、接收信息、执行状态检查和可复用行为积累。由 [[AgenticRobots-Survey]] 正式提出。

## 核心要点

1. **六类组件**: Orchestration（编排）、Memory（记忆）、Monitoring & Recovery（监控恢复）、Skills（技能）、Infrastructure（基础设施）、Layered Systems（分层系统）
2. **编排**: 管理哪个模型/策略/工具何时运行，控制权如何传递
3. **记忆**: 不修改权重地为后续决策提供历史信息
4. **技能**: 可插拔的程序化知识包，可被加载或调用
5. **分层系统**: System 2（LLM/VLM 推理，低频）+ System 1（VLA，中频）+ System 0（解析控制，高频）

## 数学形式

无标准数学形式（系统架构概念）

## 代表工作

- [[AgenticRobots-Survey]]: 提出 Harness 框架（2026）
- Harness VLA: 将冻结 VLA 作为工具之一放在 dispatch 接口后
- HarnessWAM: 进度估计器触发 VLM 任务管理器

## 相关概念

- [[Agentic Robot]]: Harness 是其软件执行层
- [[Code as Policies]]: Skill 域的起点
- [[VLA]]: System 1 层的典型执行器
