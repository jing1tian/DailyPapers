---
type: concept
aliases: [Planning Domain Definition Language, 规划领域定义语言]
---

# PDDL

## 定义
Planning Domain Definition Language：一种描述规划问题的标准形式化语言，用于定义动作前提条件（preconditions）和效果（effects）、状态谓词，以及目标状态，供经典规划求解器（如 FastDownward）使用。

## 数学形式
$$\text{Plan} = \langle a_1, a_2, \ldots, a_n \rangle, \quad \text{s.t.} \; \text{Pre}(a_i) \subseteq s_{i-1}, \; s_i = \text{Eff}(a_i)(s_{i-1})$$

其中 $\text{Pre}(a)$ 为动作前提，$\text{Eff}(a)$ 为动作效果，状态满足封闭世界假设（Closed World Assumption, CWA）。

## 核心要点
1. 传统 PDDL 假设完全已知的封闭世界，这在 open-world robotics 场景下不成立
2. LLM 可生成 PDDL 动作描述，扩展经典规划到自然语言指令
3. 在具身AI中常与 TaskPlanner（LLM）结合：LLM 负责生成 PDDL，求解器负责搜索最优规划
4. 核心限制：对未预见物体/动作泛化性差，服务机器人的开放场景是主要挑战

## 代表工作
- [[FastDownard]]（主流 PDDL 规划求解器）
- Hypothesis-driven Model Expansion（今日论文，用 LLM 动态扩展 PDDL 知识库）

## 相关概念
- [[LLM]]（生成 PDDL 的语言模型）
- [[ReAct]]（结合推理和执行的 LLM Agent 框架）
