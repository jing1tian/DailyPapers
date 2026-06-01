---
type: concept
aliases: [Infinite LLM, 无限上下文 LLM]
---

# InfLLM

## 定义
一种 LLM 长上下文推理框架，通过将远程 token 的 KV cache 存储到 CPU 内存并按需检索，在不修改模型参数的情况下支持近乎无限长度的上下文推理。

## 数学形式

分块注意力：仅保留局部窗口 $W$ 的 KV cache 在 GPU，远程 block 存 CPU：
$$\text{Attn}(q, K, V) = \text{Attn}(q, K_{\text{local}} \cup K_{\text{retrieved}}, V_{\text{local}} \cup V_{\text{retrieved}})$$

## 核心要点
1. KV cache 分为"热"（GPU 显存）和"冷"（CPU 内存）两级，按 attention score 动态检索
2. 不需要重新训练或微调，即插即用于任意已有 LLM
3. MiniCPM4 使用 InfLLM 实现超长文档处理能力
4. 推理延迟取决于 CPU-GPU 传输带宽

## 代表工作
- [[MiniCPM4]]: 集成 InfLLM 支持长上下文

## 相关概念
- [[FlashAttention]]
- [[RoPE]]
