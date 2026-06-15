---
type: concept
aliases: [Causal Diffusion Transformer, Causal DiT, 因果扩散变换器]
---

# 因果扩散 Transformer

## 定义
将[[扩散变换器|扩散 Transformer (DiT)]] 与[[块因果注意力|Block Causal Attention]] 相结合的架构，用于自回归式地生成时序 token 序列（如视频帧序列或视觉-动作 chunk）。每个 chunk 只关注历史 chunk，支持流式/在线生成，同时保留扩散模型的多步去噪能力。

## 数学形式

训练目标（[[条件流匹配]]）：

$$
\mathcal{L}_\text{FM} = \mathbb{E}\!\left[\left\|F_\theta^v(x_\alpha, \alpha, s_{<t}) - \dot{x}_\alpha^v\right\|_2^2 + \lambda_a \left\|F_\theta^a(x_\alpha, \alpha, s_{<t}) - \dot{x}_\alpha^a\right\|_2^2\right]
$$

块因果掩码确保每个 chunk $u_{t:t+k}$ 只访问历史序列 $s_{<t}$。

## 核心要点
1. **块因果掩码**：以 chunk 为单位实施因果注意力，平衡并行训练效率与序列建模能力
2. **模态共享注意力**：视觉 token 和动作 token 共享注意力权重，但使用各自的前馈网络
3. **语言条件**：冻结文本编码器提供任务指令条件，通过 cross-attention 或 token concat 注入
4. **两阶段推理**：视频去噪 5 步，动作去噪 10 步，注意力窗口 32

## 代表工作
- [[RepWAM]]: 提出将 Causal DiT 用于视觉-动作 chunk 联合生成的世界动作模型
- [[Lingbot-VA]]: 类似的因果视频生成 WAM

## 相关概念
- [[扩散变换器]]
- [[Block Causal Attention]]
- [[条件流匹配]]
- [[世界动作模型]]
- [[Visual-Action Chunk]]
