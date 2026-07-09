---
type: concept
aliases: [BEHAVIOR-1K, iGibson, OmniGibson]
---

# BEHAVIOR

## 定义
BEHAVIOR-1K：一个大规模家庭活动具身AI基准，包含 1000 种日常家庭任务（如洗碗、叠衣物、清洁），基于 OmniGibson 仿真平台，测试具身智能体在真实感场景中的长程任务执行能力。

## 数学形式
$$\text{Task} = \langle \mathcal{O}, \text{Pre}, \text{Goal} \rangle, \quad \text{Success} = \mathbf{1}[\text{Goal} \text{ satisfied at } T]$$

任务用 BDDL（BEHAVIOR Domain Definition Language）形式化描述，语义上与 PDDL 兼容。

## 核心要点
1. 1000 个任务覆盖日常家务，比 BEHAVIOR-100 扩展 10 倍
2. 基于 [[OmniGibson]] 仿真，支持物理精确交互（关节对象、液体、软体）
3. 用 BDDL（基于 PDDL 的扩展语言）定义任务前提和目标状态
4. 长程任务执行（技能组合）是主要挑战，技能切换边界往往是失败点

## 代表工作
- Diagnosing Semantic Handoff Failures（今日论文，在 BEHAVIOR-1K 上诊断 VLA 技能切换失败）

## 相关概念
- [[PDDL]]（BDDL 的基础语言）
- [[VLA]]（在 BEHAVIOR 上测试的主要模型类型）
