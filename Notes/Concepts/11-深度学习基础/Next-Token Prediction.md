---
type: concept
aliases: [NTP, 自回归语言建模, Autoregressive Language Modeling, 下一词预测]
---

# Next-Token Prediction

## 定义

**Next-Token Prediction**（NTP）是自回归语言模型（和多模态模型）的核心预训练目标：给定上下文序列，最大化每个 token 的条件概率。

## 数学形式

$$
\mathcal{L}_{NTP} = -\sum_{t=1}^{T} \log p_{\theta}(u_{t} \mid u_{<t}, \mathbf{a}_{\leq t}, \mathbf{c})
$$

**符号说明**:
- $u_t$: 时刻 $t$ 的 token（可以是文本、图像 patch、动作等）
- $u_{<t}$: 所有历史 token
- $\mathbf{a}_{\leq t}$: 条件动作序列（具身 AI 场景）
- $\mathbf{c}$: 可选条件（任务描述等）

**无条件 LM 形式**（标准 GPT 目标）:

$$
\mathcal{L}_{NTP} = -\sum_{t=1}^{T} \log p_{\theta}(u_{t} \mid u_{<t})
$$

## 核心要点

1. 在自回归框架下，**因果注意力掩码**保证 $u_t$ 只依赖 $u_{<t}$
2. NTP 是大语言模型（GPT 系列）和多模态世界模型（Cosmos、Genie）的主要预训练方式
3. 与 [[Masked Autoencoding]]（MAE）的区别：MAE 随机掩码并预测，NTP 单向自回归预测
4. 在世界模型中，token 可扩展为多模态（视频帧、动作 token、语言 token）

## 代表工作

- [[WorldModelRoadmap]]: 将 NTP 纳入世界模型自监督预训练范式
- GPT 系列 (OpenAI): 语言模型 NTP 的代表

## 相关概念

- [[Scaling Law]]
- [[Transformer]]
- [[World Model]]
- [[Masked Autoencoding]]
