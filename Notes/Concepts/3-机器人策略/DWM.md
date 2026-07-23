---
type: concept
aliases: [Disentangled World Model, World-Action Disentanglement]
---

# DWM

## 定义
Latent World Model 的解耦方法，在 latent 空间中显式分离"动作引起的状态变化"（action effect）和"环境自身动态"（world effect），通过双头预测 + InfoNCE 解耦约束实现。

## 数学形式
给定 latent state $z_t$ 和 action $a_t$，分解下一 latent：
$$\Delta z_t = \underbrace{f_w(z_t)}_{\text{world effect}} + \underbrace{f_a(z_t, a_t)}_{\text{action effect}}$$
$$z_{t+1} = z_t + \Delta z_t$$

解耦约束（InfoNCE）：
$$\mathcal{L}_{\text{decouple}} = \text{InfoNCE}(f_w(z_t), f_a(z_t, a_t))$$

规划时仅在 action effect 子空间用 CEM 搜索：
$$a^* = \arg\max_{a} \sum_t r(z_t + f_a(z_t, a))$$

## 核心要点
1. 现有 monolithic transition 监督的问题：action 和 world dynamics 混淆
2. 双头解耦：world effect head（不依赖 action）+ action effect head
3. InfoNCE loss 强制两头表示不相关
4. 规划效率提升：只在 action effect 子空间搜索，降低复杂度
5. OOD 泛化：新 gravity 条件下 CEM 规划仍有效

## 代表工作
- Zhang et al. 2026: DWM（Peking University）
- [[LeWM]]: 先前的 latent 分离尝试（基于 causal masking，不够干净）

## 相关概念
- [[LeWM]]（基线对比）
- [[InfoNCE]]（解耦训练 loss）
- [[CEM]]（规划方法）
- [[PushT]]（测试环境）
