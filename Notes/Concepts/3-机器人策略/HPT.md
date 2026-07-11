---
type: concept
aliases: [Heterogeneous Pretrained Transformer, 异构预训练Transformer, HPT]
---

# HPT（Heterogeneous Pretrained Transformer）

## 定义
HPT 是一种用于机器人操作策略的统一骨干架构，通过模态独立的 stems 将异构传感器输入（视觉、本体感知、触觉等）映射到共享的 Transformer trunk，再由任务特定 head 输出动作或世界状态预测。

## 数学形式

$$
z_t = f_\phi(o_t) = \text{Trunk}\bigl(\text{Stem}_{ego}(o^{ego}_t),\; \text{Stem}_{wrist}(o^{wrist}_t),\; \text{Stem}_{prop}(q_t)\bigr)
$$

## 核心要点

1. **模态独立 Stems**: 每种传感器模态有独立编码器（视觉 ResNet-18 / DINO，本体感知 MLP），输出对齐到统一维度（256-d）
2. **共享 Trunk**: 多层 Transformer 处理拼接的 token 序列，学习跨模态表征
3. **可插拔 Heads**: Action Head（流匹配 CrossTransformer）和 World-Model Head（Pixel/DINO/3D Flow）可灵活替换，推理时仅保留 Action Head
4. **跨体态统一**: 人类数据和机器人数据通过共享 trunk 对齐，但 stems 可分别适配不同传感器配置

## 代表工作

- [[EgoWAM]]: 使用 HPT 作为 WAM co-training 骨干，trunk dim=256，16 层，8 头
- [[FAWAM]]: 基于 HPT 扩展的 Foundation WAM

## 相关概念

- [[World Action Model|WAM]]: HPT 常作为 WAM 类框架的骨干
- [[Cross-Embodiment]]: HPT 的 shared trunk 设计支持跨体态学习
- [[Action Chunking]]: HPT action head 的输出格式
- [[CFM|Conditional Flow Matching]]: action head 的训练目标
