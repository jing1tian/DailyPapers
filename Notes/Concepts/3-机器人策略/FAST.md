---
type: concept
aliases: [FAST, Frequency-based Action Tokenizer]
---

# FAST (Frequency-based Action Sequence Tokenizer)

## 定义
一种基于频域分解的机器人动作 tokenizer，将连续动作序列转换为离散 token，用于 VLA 的动作头设计。

## 核心要点
1. 通过频域分析（如 DCT）将动作序列压缩为紧凑的 token 表示，减少 token 数量
2. 相比 per-step tokenization，FAST 能更高效地表示平滑的机器人轨迹
3. 被 [[RS-CL]] 和 [[DUST]] 等 VLA 论文使用作为动作 tokenizer
4. 与 [[VQ-BeT]] 等 VQ-based tokenizer 是同类竞品

## 代表工作
- π₀ (Black et al.): 使用 FAST tokenizer 的代表 VLA
- [[RS-CL]], [[DUST]]: 采用 FAST 作为动作头

## 相关概念
- [[VLA（视觉-语言-动作模型）]]
- [[扩散策略]] — 另一类连续动作建模方法
- [[VQ-BeT]] — 基于 VQ 的离散动作 tokenizer
