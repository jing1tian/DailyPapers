---
type: concept
aliases: [Embodied Chain-of-Thought, 具身链式思维]
---

# ECoT

## 定义

ECoT（Embodied Chain-of-Thought）是将 [[Chain-of-Thought Reasoning|Chain-of-Thought 推理]] 引入机器人操控 [[VLA]] 的方法，通过在动作生成前显式生成自然语言推理链（包括任务目标、规划、空间定位、动作描述等）来提升策略的泛化能力。

## 数学形式

推理链作为动作的自回归前缀：

$$
p(\mathbf{a} \mid o, l) = p(\mathbf{a} \mid c, o, l) \cdot p(c \mid o, l)
$$

其中 $c$ 为自然语言推理链，$o$ 为观测，$l$ 为语言指令，$\mathbf{a}$ 为动作序列。

## 核心要点

1. **层级化推理**: 推理链分为任务目标（Goal）、规划（Planning）、空间定位（Grounding）、动作描述（Movement）等层级
2. **自回归前缀**: 推理链以 token 序列形式拼接在动作 token 之前，通过同一 VLM backbone 生成
3. **局限性**: 推理链中的噪声在测试时通过自回归传播到动作 token，导致复合误差；规模扩展时性能易降低
4. **ERVLA 改进**: 通过 Reasoning Dropout 让推理链成为可选训练信号而非强制推理前缀

## 代表工作

- [[ERVLA]]: 提出 Reasoning Dropout 解决 ECoT 的自回归误差传播问题，训练时内化 CoT 监督

## 相关概念

- [[Chain-of-Thought Reasoning]]
- [[VLA]]
- [[Embodied Forcing]]
- [[ERVLA]]
