---
type: concept
aliases: [Causal Self-Attention, 因果Transformer, 自回归Transformer]
---

# Causal Transformer（因果 Transformer）

## 定义

Causal Transformer 是带有因果（自回归）掩码的 Transformer 架构，每个位置的 token 只能关注其历史位置，保证序列的时序因果性，常用于时序建模和语言生成任务。

## 数学形式

标准自注意力的因果掩码：

$$
\text{Attn}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d}} + M\right) V
$$

其中掩码矩阵 $M_{ij} = -\infty$ 当 $j > i$（未来位置），$M_{ij} = 0$ 当 $j \leq i$（当前或历史位置）。

## 核心要点

1. **因果性保证**：位置 $t$ 的输出只依赖位置 $\leq t$ 的输入，适合顺序决策和时序预测
2. **与 GPT 一致**：GPT 系列语言模型的核心架构，也是许多 VLA 的自回归 decoder 基础
3. **对比双向 Transformer**：BERT 等双向模型可看到未来 token，适合理解任务；Causal Transformer 适合生成任务
4. **时序聚合**：在机器人策略中用于聚合多帧历史观测，天然维护时间因果性

## 在 VLA-RL 中的应用

- **WCM 世界预测器**：Causal Transformer Trunk 聚合 K 帧历史潜在状态，输出包含时序动态的历史摘要向量 $h_t$
- **VLA Actor**：OpenVLA 等自回归 VLA 使用 Causal LLM 生成动作 token 序列

## 代表工作

- [[WCM]]: 使用 Causal Transformer 作为世界预测器主干，聚合历史帧动态
- [[GPT]]: 经典因果语言模型
- [[OpenVLA-OFT]]: 基于 Causal LLM 的 VLA 策略

## 相关概念

- [[Cross-Attention]]：WCM 中与 Causal Transformer 配合使用的语言融合机制
- [[JEPA]]：与 Causal Transformer 结合的联合嵌入预测架构
- [[Transformer]]：Causal Transformer 的基础架构
