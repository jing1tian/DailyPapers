---
type: concept
aliases: [Bidirectional Encoder Representations from Transformers, 双向编码器表示, bert]
---

# BERT

## 定义

BERT（Bidirectional Encoder Representations from Transformers）是 Google 于 2018 年提出的基于双向 Transformer 编码器的预训练语言模型，通过遮蔽语言建模（MLM）和下一句预测（NSP）任务学习深层上下文感知的文本表示。

## 数学形式

$$
H = \text{TransformerEncoder}(\text{Embed}(x_1, \ldots, x_N))
$$

其中 $H \in \mathbb{R}^{N \times d}$ 为完整 token 序列的双向上下文表示，$d=768$（BERT-Base）或 $d=1024$（BERT-Large）。

## 核心要点

1. **双向注意力**: 每个 token 同时关注左右上下文，不同于 GPT 的单向（左→右）自回归
2. **遮蔽语言建模（MLM）**: 随机遮蔽 15% 的 token，训练模型预测被遮蔽词，强制学习双向上下文
3. **完整 token 序列输出**: 保留所有 $N$ 个 token 的表示，适合需要细粒度语义定位的下游任务（如机器人指令理解）
4. **参数效率**: BERT-Base 仅 110M 参数，远小于现代 LLM，推理速度快

## 代表工作

- [[TurboVLA]]: 用 BERT 替代 LLM 作为轻量文本编码器，保留完整 token 序列进行细粒度视觉-语言融合，以 <1GB VRAM 实现 32 Hz 实时机器人操控

## 相关概念

- [[Cross-Attention]]: BERT 输出常作为 key/value 供视觉特征查询
- [[VLA|Vision-Language-Action Model]]: TurboVLA 等 VLA 中作为轻量文本骨干
- [[Vision Transformer]]: 同为 Transformer 架构，BERT 处理文本，ViT 处理图像
