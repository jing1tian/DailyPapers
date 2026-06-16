---
type: concept
aliases: [Hunyuan World Model, 混元世界模型]
---

# HunyuanWorld

## 定义
腾讯混元团队发布的开源视频世界模型，基于 Autoregressive DiT，支持长序列交互式视频生成和游戏/仿真场景下的 neural world model 应用。

## 数学形式
AR DiT 在 token 空间按帧递归生成：
$$p(x_{1:T}) = \prod_{t=1}^T p_\theta(x_t \mid x_{<t})$$

## 核心要点
1. Autoregressive 生成允许流式/增量推理，适合 interactive world model
2. 支持动作条件化，可作为 embodied AI 的 learned simulator
3. [[TempCache]] 等加速工作以 HunyuanWorld 为主要测试平台
4. 已开源，是 video world model 领域的重要基础设施

## 代表工作
- [[TempCache]]：KV cache 压缩加速 HunyuanWorld 推理

## 相关概念
- [[Cosmos-Predict2]]
- [[TempCache]]
- [[Action-Conditioned World Model]]
