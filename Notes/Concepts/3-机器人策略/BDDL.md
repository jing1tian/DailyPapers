---
type: concept
aliases: [Behavior Domain Description Language, 行为域描述语言]
---

# BDDL

## 定义
行为域描述语言（Behavior Domain Description Language），一种基于谓词逻辑的任务规范语言，用于在家庭/实验室机器人场景中描述任务目标和状态转移条件，常用于 BEHAVIOR-1K 等基准。

## 数学形式
任务规范为一阶谓词公式：
$$\text{Goal} = \bigwedge_i P_i(o_1, o_2, \ldots)$$
其中 $P_i$ 是谓词（如 $\text{InReceptacle}$、$\text{OnTop}$），$o_i$ 是对象。

## 核心要点
1. **符号化任务描述**：用谓词定义目标状态，而非隐含在奖励函数中
2. **长程任务适用**：支持多步子目标分解和条件依赖
3. **生成局限**：谓词需要人工或模板定义，难以从自然语言自动转换

## 代表工作
- [[EA-WM]] (2606.13053): 用 BDDL 提供谓词级任务进度监督
- BEHAVIOR-1K 基准: BDDL 的主要使用场景

## 相关概念
- [[SERF]]
- [[OmniGibson]]
