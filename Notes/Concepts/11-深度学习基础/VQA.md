---
type: concept
aliases: [Visual Question Answering, 视觉问答]
---

# VQA

## 定义

视觉问答（Visual Question Answering）：给定图像或视频，模型回答关于视觉内容的自然语言问题，是衡量视觉-语言多模态理解能力的核心任务形式。

## 数学形式

$$
p_\Theta(a \mid v,\, q)
$$

其中 $v$ 为视觉输入，$q$ 为问题，$a$ 为答案。

## 核心要点

1. 需要模型同时理解视觉内容和语言语义，是视觉-语言对齐的核心 proxy 任务
2. 在世界模型预训练中作为语义理解监督信号，防止模型丢失语言对齐能力
3. 数据来源：人工标注（VQAv2）、合成生成（LLaVA 风格）等

## 代表工作

- [[Orca]]: 使用 11.5M VQA 样本作为有意识学习的语义强化信号

## 相关概念

- [[VLM]]
- [[Conscious Learning]]
- [[Language-Conditioned]]
