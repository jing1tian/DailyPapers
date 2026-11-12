---
type: concept
aliases: [World Expert, 世界专家]
---

# 世界专家（World Expert）

## 定义
[[世界动作模型（WAM）]] 中专门负责预测未来视频帧的子模块，通常由预训练视频扩散模型初始化，提供全局场景物理动力学表征，作为动作生成的条件信号。

## 核心要点

1. 在 [[Mixture-of-Transformers]] 架构中，世界专家是第一个专家，其输出 Key-Value 被下游潜专家和动作专家条件化。
2. 通常保持较强预训练能力（Key-Value detach 阻止梯度回传），避免破坏物理先验。
3. 在 LaST0.5 中，世界专家使用 [[Cosmos-Predict-2.5|Cosmos-Predict2.5-2B]] DiT 初始化，以历史 5 帧预测未来 12 帧（$256\times256$）。
4. 世界专家的 DiT 隐状态也被复用于进度值估计（见 [[Progress Estimation]]）。

## 代表工作

- [[LaST0.5]]: 世界专家（Cosmos-Predict2.5-2B）+ 潜专家（Janus）+ 动作专家（Flow Matching）的三专家 MoT 架构

## 相关概念

- [[世界动作模型（WAM）]]
- [[Mixture-of-Transformers]]
- [[Cosmos-Predict-2.5]]
- [[Interaction-Centric Latent Reasoning]]
- [[Action Expert]]
