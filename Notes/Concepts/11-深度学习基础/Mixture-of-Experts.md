---
type: concept
aliases: [MoE, Mixture of Experts]
---

# Mixture-of-Experts (MoE)

## 定义

MoE 是一种稀疏神经网络架构，将模型分解为多个"专家"子网络，通过路由器（Router/Gating Network）为每个输入 token 动态选择激活少数专家，在不增加推理计算量的前提下扩大模型容量。

## 数学形式

$$
\text{MoE}(x) = \sum_{i=1}^{N} G(x)_i \cdot E_i(x)
$$

其中 $G(x)$ 为稀疏门控函数（通常 Top-K 激活），$E_i$ 为第 $i$ 个专家网络。

## 核心要点

1. 每个 token 只激活 Top-K（通常 K=1 或 2）个专家，推理 FLOP 与单一 dense 模型相当
2. 总参数量可以远大于单一模型，提升模型容量
3. 需要负载均衡损失防止专家塌缩（collapse）
4. 在 LLM（Mixtral、GPT-4）和视觉模型中广泛使用

## 代表工作

- [[Mixture-of-Transformers]]: MoT 将 MoE 思想扩展到多分支 Transformer 架构

## 相关概念

- [[Mixture-of-Transformers]]
- [[Transformer]]
