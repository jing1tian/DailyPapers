---
type: concept
aliases: [输出分布不匹配, VLM-VLA分布鸿沟]
---

# Output-Distribution Mismatch

## 定义

将预训练 VLM 适配为 VLA 时，目标输出从自然语言 token 切换为高精度数值动作 token（如离散化的机器人关节角度），两者的 token 分布差异巨大，导致灾难性遗忘和 VLM 预训练知识丢失的问题。

## 数学形式

VLM 预训练分布：$p_{\text{pretrain}}(\text{natural language tokens})$

VLA 微调目标：$p_{\text{finetune}}(\underbrace{408, 980, 166, \ldots}_{\text{离散化数值 token}})$

两者分布的 KL 散度极大，直接微调会破坏 VLM 的语言表示能力。

## 核心要点

1. 数值动作 token（如 `408 980 166 319 505 491`）在 VLM 预训练语料中极少出现
2. 即使词表包含数字，其语义含义与机器人动作完全不同（数字 vs 离散化关节值）
3. 直接微调（VLA-0）会导致预训练 VLM 的泛化能力和知识转移受损
4. 解决方案：引入语言-动作接地（[[Language-Action Grounding]]），用语言计划桥接两个分布

## 代表工作

- [[CLAP]]: 通过 Language-Action Grounding 缓解输出分布不匹配

## 相关概念

- [[Language-Action Grounding]]
- [[VLA（视觉-语言-动作模型）]]
- [[Autoregressive Policy]]
- [[VLA-0]]
