---
type: concept
aliases: [动作查询, Action Queries, 动作查询 token]
---

# Action Query

## 定义

Transformer 基 VLA 模型（如 Mind-VLA）中一组可学习查询 token，其最终层输出作为扩散动作头的条件输入，驱动未来多步动作预测。

## 核心要点

1. **功能**: 汇聚语言指令、视觉观测、本体感知的多模态信息，输出 $\mathbf{H}_t^a$ 作为扩散动作头的条件
2. **动作预测**: [[Diffusion Transformer]] 以 $\mathbf{H}_t^a$ 为条件，通过 DDIM 采样预测未来 $T_a$ 步 7 维动作
3. **注意力隔离**: 通过因果注意力掩码与 Scene Query、Object Query 隔离
4. **推理时保留**: 与辅助查询不同，Action Query 在推理时完整保留

## 代表工作

- [[Mind-VLA]]: 三类查询设计中的动作查询

## 相关概念

- [[Scene Query]]
- [[Object Query]]
- [[Diffusion Transformer]]
- [[Action Chunking]]
- [[Vision-Language-Action]]
