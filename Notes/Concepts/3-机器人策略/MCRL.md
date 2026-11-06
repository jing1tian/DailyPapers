---
type: concept
aliases: [Motion-guided Contrastive Regularization for Latent World Models]
---

# MCRL

## 定义
针对 JEPA-style latent world model 的运动引导对比正则化方法，通过 optical flow mask 构造运动区域 vs 静态区域的对比损失，防止 static collapse。

## 核心要点
1. 识别 static collapse 问题：JEPA latent representation 丢失动态信息，导致控制失效
2. 用 [[RAFT]] 提取 optical flow，生成运动区域 mask
3. 以 [[InfoNCE]] 对比损失强制 latent 保留动作相关动态信息
4. 在 [[LeWorldModel]] 框架上实现，[[LIBERO]] 上验证

## 相关概念
- [[LeWM]]
- [[JEPA]]
- [[InfoNCE]]
- [[RAFT]]
- [[静态坍塌]]
