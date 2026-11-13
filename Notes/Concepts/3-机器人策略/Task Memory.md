---
type: concept
aliases: [任务记忆, 任务存储]
---

# Task Memory

## 定义

WICL 中 agent 通过排练学到并冻结的结构化语义知识文档，以 Markdown 形式存储操作任务的对象角色、工作流、参数规格和经验教训，可直接部署到未见场景及不同机体。

## 数学形式

$$
C^* = \text{freeze}(C_K) \quad \text{when convergence}
$$

其中 $C_K$ 是经过 $K$ 轮排练更新后的 context。

## 核心要点

1. **五个部分**：指令与对象角色、工作流（逐阶段 API 调用）、参数表（GraspSpec/PlaceSpec）、Lessons
2. **语义表示**：规格用对象角色和相对几何表达，不含绝对坐标（linter 强制）
3. **Linter 保障**：拒绝坐标、真值标注词、实例标识符，确保跨场景可用
4. **可读写**：支持 Recursive Self-Improvement，反思 agent 可追加 Guidance

## 代表工作

- [[RoboScript]]: Task Memory 的具体实现，通过 S1（模仿）+ S2（排练）学习

## 相关概念

- [[World In-Context Learning]]
- [[Recursive Self-Improvement]]
- [[In-Context Learning]]
