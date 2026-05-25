---
type: concept
aliases: [Vision-Bridged Value Representation, 视觉桥接值表示]
---

# VBVR

## 定义
统一多模态理解与生成的表示空间技术，通过"视觉桥接"使理解侧和生成侧的 visual token 共享同一表征空间。

## 核心要点
1. 解决理解模型（encoder-based）和生成模型（decoder-based）visual token 表征不对齐的问题
2. 设计 bridge 机制，让同一个 value representation 服务于 cross-modal understanding 和 visual generation
3. 是 [[SenseNova-U1]] NEO-unify 架构的核心组件
4. 允许在单一模型内统一完成图像理解和图像生成，无需级联两个独立系统

## 代表工作
- [[SenseNova-U1]]（商汤, 2026）: 提出 VBVR，用于统一多模态理解和生成

## 相关概念
- [[MoT]]: Mixture of Transformers，SenseNova-U1 的另一架构组件
- [[VLM]]: 视觉语言模型，VBVR 的应用场景
- [[Cross-Attention]]: 多模态交互的基础机制
