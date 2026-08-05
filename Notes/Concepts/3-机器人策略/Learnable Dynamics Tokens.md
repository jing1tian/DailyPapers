---
type: concept
aliases: [动力学 Token, Dynamics Tokens, Latent Dynamics Tokens]
---

# Learnable Dynamics Tokens

## 定义

插入 VLM 序列的一组可学习 token，在动作条件下隐式表示机器人场景的动态状态变化，作为世界预测器的输入和目标锚点。

## 核心要点

1. 以随机初始化的参数形式存在，通过端到端训练学习动作相关的状态表示
2. 拼接在视觉和语言 token 之后进入 VLM 的自注意力层，获得对观测的完整上下文感知
3. 经 EMA 动量教师处理未来帧，得到稳定的预测目标潜态
4. 最优数量为 8 个，过少（1 个）或过多（16 个）均降低性能

## 数学形式

$$
\bm{H}_t = \mathcal{E}_\theta\left([\bm{f}_t^V, \bm{f}^L, \bm{Q}]\right), \quad \bm{H}_t = [\bm{H}_t^V, \bm{H}^L, \bm{H}_t^Q]
$$

其中 $\bm{Q}$ 为可学习动力学 token 参数，$\bm{H}_t^Q$ 为上下文化后的表示。

## 代表工作

- [[SG-WAM]]: 提出 Learnable Dynamics Tokens，与 SGWP 和几何监督联合使用，8 个 token 为最优配置

## 相关概念

- [[Self-Guided World Predictor]]
- [[World Action Model]]
- [[VLM]]
- [[EMA]]
