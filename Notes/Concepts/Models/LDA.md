---
type: concept
aliases: [LDA-1B, Large Diffusion Action, LDA Policy]
---

# LDA（Large Diffusion Action Model）

## 定义

LDA 是一个大规模扩散动作模型（~10亿参数），在海量跨具身演示数据（EI-30k 语料库）上预训练，作为 [[World Action Model]] 的基础模型，预训练目标包括直接策略、视觉预测、逆动力学和前向动力学四个任务。

## 核心要点

1. **架构**: 基于 [[MM-DiT|多模态扩散 Transformer]]，包含视觉专家和动作专家
2. **预训练规模**: EI-30k 语料库（约 30,000 小时演示数据，跨多具身）
3. **预训练目标**: 四个任务——直接策略 / 视觉预测 / 逆动力学 / 前向动力学
4. **下游微调**: 可直接微调用于特定机器人平台（RoboCasa-GR1、LIBERO 等）
5. **RoboCasa-GR1 基准**: 发布版成功率约 52.1%

## 代表工作

- [[UniFlowWAM]]: 在 LDA-1B 基础上增加流专家，提出 UniFlowWAM，将成功率从 52.1% 提升至 56.4%

## 相关概念

- [[World Action Model]]: 所属模型类别
- [[MM-DiT]]: 架构基础
- [[Action Chunking]]: 动作表示方式
- [[Flow Matching]]: 去噪目标函数
