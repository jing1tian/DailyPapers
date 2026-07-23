---
type: concept
aliases: [Programmatic Prompting, Program-based Task Planning]
---

# ProgPrompt

## 定义
一种基于程序化提示（programmatic prompt）的 LLM 任务规划方法，用代码风格的格式描述任务步骤，LLM 生成可执行的程序化计划。

## 核心要点
1. 以 Python 代码风格描述环境状态和可用动作，LLM 直接生成程序步骤
2. 比自然语言规划更精确，可以明确调用 assertion 和条件检查
3. 缺乏形式化的时序逻辑保证（与 [[STeP]] 的 STL 方法形成对比）
4. 成功的条件依赖 LLM 生成代码的正确性

## 代表工作
- Singh et al. 2022: ProgPrompt (SRI International + CMU)

## 相关概念
- [[VoxPoser]]（类似的语言引导机器人规划）
- [[ReKep]]（关键点约束规划）
- [[STeP]]（STL 形式化规范，形成对比）
- [[STL]]（时序逻辑形式化方法）
