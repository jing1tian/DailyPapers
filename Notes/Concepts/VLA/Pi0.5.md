---
type: concept
aliases: [π0.5, pi0.5]
---

# Pi0.5（π0.5）

## 定义

Physical Intelligence（物理智能）发布的 Flow Matching 视觉-语言-动作模型，在 π0 基础上改进，支持多任务操作并具备动作专家（Action Expert）架构。

## 核心要点

1. 采用 [[Flow Matching|流匹配]] 框架生成连续动作序列
2. 包含独立的动作专家（Action Expert）模块，与语言/视觉骨干分离
3. 在 MetaWorld、LIBERO、RoboCasa 等 benchmark 上广泛使用

## 代表工作

- [[COAST]]: 作为主要测试 VLA 之一，COAST 在 π0.5 上验证了激活引导的有效性（LIBERO-10: 0.43→0.80）

## 相关概念

- [[VLA]]
- [[Flow Matching]]
- [[Diffusion Policy]]
- [[Pi0-FAST]]
