---
type: concept
aliases: [Reasoning Anchor Token, 推理锚定Token, τ_R]
---

# 推理锚点 Token

## 定义

插入在任务指令与动作预测指令之间的特殊 token（$\tau_R$），通过同时参与教师分支（3D 推理提示）和学生分支（动作预测提示）的前向传播，为在线蒸馏提供共享的语义锚点，使动作预测路径能够继承 3D 空间推理表征。

## 数学形式

教师分支：
$$H^R_{teacher} = \text{sg}(f_\theta(I_t, L_{task}, L^{teacher}_{3D}, \tau_R))$$

学生分支：
$$H^R_{student} = f_\theta(I_t, L_{task}, \tau_R, L_{action}, \tau_A)$$

## 核心要点

1. **位置设计**：插入于任务指令之后、动作预测指令之前，吸收视觉和任务语义
2. **共享参数**：教师和学生分支使用同一个 token 嵌入和主干参数，仅输入提示不同
3. **桥接推理缺口**：解决标准动作预测提示导致模型绕过3D空间先验的"prompt-induced reasoning gap"

## 代表工作

- [[3DThinkVLA]]: 首次提出推理锚点 token 概念，用于在线3D推理蒸馏，无需推理时生成CoT文本

## 相关概念

- [[在线蒸馏]]
- [[推理适配器]]
- [[Chain-of-Thought Reasoning]]
- [[VLA（视觉-语言-动作模型）]]
