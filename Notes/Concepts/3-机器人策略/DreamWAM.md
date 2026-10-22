---
type: concept
aliases: [DreamWAM, DW-U, DW-J, DreamWAM-uncond, DreamWAM-joint]
---

# DreamWAM

## 定义

DreamWAM 是一种基于世界模型的机器人操纵策略，通过预测未来视觉潜变量（"梦境"）来指导动作生成，具有 unconditioned（DW-U）和 joint（DW-J）两个变体。

## 核心要点

1. **DreamWAM-uncond（DW-U）**：无条件预测未来视觉潜变量，动作生成与世界预测相对独立
2. **DreamWAM-joint（DW-J）**：联合建模未来视觉潜变量和动作序列的联合分布
3. **共享视觉编码器**：与 [[FastWAM]] 使用相同的公共预训练编码器，使其成为 [[TAPDreamer]] 的有效迁移目标
4. **迁移攻击脆弱性**：TAPDreamer 补丁将 DW-U 和 DW-J 成功率分别降至 1.45% 和 1.00%

## 代表工作

- [[DreamWAM]]（论文笔记）: 原始论文
- [[TAPDreamer]]: 将 DreamWAM 作为迁移攻击目标

## 相关概念

- [[WAM]]
- [[FastWAM]]
- [[Visual Encoder]]
- [[TAPDreamer]]
