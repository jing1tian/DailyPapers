---
type: concept
aliases: [Vision-and-Language Navigation, VLN]
---

# VLN

## 定义
Vision-and-Language Navigation：智能体根据自然语言指令，在三维环境中导航到目标位置的任务。

## 数学形式
$$\pi(a_t | o_t, I) \rightarrow \text{target}$$

其中 $I$ 为语言指令，$o_t$ 为视觉观测。

## 核心要点
1. 结合语言理解和视觉感知进行导航决策
2. 常用环境：R2R（Room-to-Room）、REVERIE、R4R
3. 难点：长指令理解、视觉-语言对齐、未见环境泛化
4. 近年趋势：用 VLM 作为导航 backbone，zero-shot 能力显著提升

## 代表工作
- [[WorldVLN]]：结合世界模型的 VLN
- [[Uni-LaViRA]]：统一多种具身导航任务

## 相关概念
- [[EQA]]
- [[VLM]]
- [[VLA]]
- [[NAVSIM]]
