---
type: concept
aliases: [NEO-unify Architecture, NEO-unify框架]
---

# NEO-unify

## 定义
SenseNova-U1 提出的统一多模态架构，在单模型内同时处理多模态理解（VQA、图像理解）和生成（文生图、图像编辑），通过 [[MoT]]（Mixture of Transformers）做模态路由。

## 核心要点
1. 统一理解与生成：同一模型权重处理理解任务和生成任务
2. MoT 路由：不同模态激活不同 transformer expert，避免模态干扰
3. 两阶段训练：SFT 预训练 → RL 对齐
4. 评估含 VBVR（Video-Based Visual Reasoning）自有 benchmark

## 代表工作
- [[SenseNova-U1]]: 提出 NEO-unify 架构的原始论文

## 相关概念
- [[MoT]]
- [[BAGEL]]
- [[OmniGen2]]
- [[FLUX]]
