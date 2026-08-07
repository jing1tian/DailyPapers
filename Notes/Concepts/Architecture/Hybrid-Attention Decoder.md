---
type: concept
aliases: [混合注意力解码器, Hybrid Attention Decoder, Hybrid-Attention Decoder]
---

# Hybrid-Attention Decoder（混合注意力解码器）

## 定义

一种将自回归因果注意力（用于语言/推理文本生成）和双向并行注意力（用于连续动作序列解码）在同一解码器中串联使用的架构设计，通过动态切换 attention mask 实现模态特定的最优解码方式。

## 数学形式

解码过程分为两个阶段，注意力掩码在运行时动态切换：

$$
\text{Attention Mask} = \begin{cases}
M_{\text{causal}} & \text{CoT token 生成阶段} \\
M_{\text{bidirectional}} & \text{动作 token 并行解码阶段}
\end{cases}
$$

- $M_{\text{causal}}$: 下三角掩码，保证自回归顺序
- $M_{\text{bidirectional}}$: 全 1 掩码（无掩码），允许动作 token 互相关注

## 核心要点

1. **设计动机**: 解决 [[Decoding Alignment|解码对齐]] 问题——强行用单一 AR 解码器处理语言+动作，延迟增加 4× 且性能下降
2. **切换机制**: 在 token 序列中，CoT 文本 token 使用因果掩码自回归生成；完成后动作 token 切换为双向掩码并行解码
3. **性能对比**: AR+AR（85.5%→81.3%, 4× 延迟）< 无 CoT 基线（85.5%）< **混合解码器**（96.8%, ~1× 延迟）
4. **RL 兼容性**: 并行解码恢复了推理速度，使大规模仿真器 rollout 可行，进而支持 RL 阶段训练

## 代表工作

- [[DeepThinkVLA]]: 首次将混合注意力解码器用于 CoT+VLA 的解码对齐

## 相关概念

- [[Decoding Alignment]]: 混合注意力解码器解决的核心问题
- [[Autoregressive Generation]]: CoT 生成阶段使用的解码范式
- [[Chain-of-Thought Reasoning]]: 由因果注意力生成的推理文本
- [[动作分块]]: 动作并行解码的目标输出
- [[Causal Transformer]]: 相关的因果注意力架构
