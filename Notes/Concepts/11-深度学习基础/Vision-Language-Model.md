---
type: concept
aliases: [VLM, 视觉语言模型, Multimodal LLM, MLLM]
---

# Vision-Language-Model

## 定义

Vision-Language Model（VLM）是将视觉编码器与语言模型结合，能够同时理解图像和文本输入的多模态大模型，广泛用于图像描述、视觉问答（VQA）和机器人感知等任务。

## 数学形式

$$
p(y \mid v, x) = \prod_t p_\theta(y_t \mid f_{\mathrm{vis}}(v),\; x,\; y_{<t})
$$

其中 $v$ 为图像，$x$ 为文本输入，$f_{\mathrm{vis}}$ 为视觉编码器（通常冻结）。

## 核心要点

1. 通常由预训练 LLM + 视觉编码器（如 ViT）+ 视觉-语言对齐模块（MLP/cross-attn）组成；
2. 视觉塔（vision tower）在下游微调时常冻结；
3. VLM 可用于机器人感知（VQA 类），但不直接输出动作；
4. [[Vision-Language-Action]] 是在 VLM 基础上增加动作输出头的机器人策略模型；
5. CaPtain 用 VLM（Qwen3.5-4B）生成代码而非直接预测关节角度。

## 代表工作

- [[CaPtain]]: 将 Qwen3.5-4B VLM 训练为 Code-as-Policy 机器人智能体
- [[Qwen]]: Qwen-VL 系列 VLM
- LLaVA, InternVL, Phi-Vision: 主流开源 VLM

## 相关概念

- [[Vision-Language-Action]]
- [[Code-as-Policy]]
- [[Qwen]]
