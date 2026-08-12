---
type: concept
aliases: [Mixture of Latent Prompts, Context-Routed Latent Prompts]
---

# MoLP (Mixture of Latent Prompts)

## 定义
一种基于上下文路由的潜变量提示混合机制，根据当前视觉-语言上下文选择/混合不同的潜变量提示，用于 VLA 的任务隔离和测试时训练（TTT）。

## 核心要点
1. 不同任务路由到不同的 latent prompt，避免任务间干扰
2. 上下文相关路由（context-routing）确保适配信号与任务对齐
3. 用于 [[VANE]] 中做可靠的 test-time training：防止多任务适配时的梯度冲突

## 代表工作
- [[VANE]]: Reliable Test-Time Training for Vision-Language-Action Models (2026)

## 相关概念
- [[VANE]]
- [[VLA]]
