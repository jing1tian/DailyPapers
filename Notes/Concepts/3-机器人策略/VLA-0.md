---
type: concept
aliases: [裸Token VLA, Bare-Token VLA]
---

# VLA-0

## 定义

一种直接将 VLM 适配为 VLA 的基线方法，不引入任何语言中间表示，直接输出离散化的数值动作 token 序列（"裸 token" 方法），与 CLAP 等语言增强方法形成对比。

## 数学形式

$$
\pi_\theta(\boldsymbol{a} | o, l) = \prod_{i=1}^{7h} \pi_\theta(a^{(i)} | a^{(<i)}, o, l)
$$

直接对 $7h$ 个数值动作 token 进行自回归预测，无语言计划条件化。

## 核心要点

1. 输出格式：直接生成 `408 980 166 319 ...` 形式的离散化关节值
2. 存在 [[Output-Distribution Mismatch|输出分布不匹配]] 问题，与 VLM 预训练分布差距大
3. 在 LIBERO 上使用 Qwen3.5-2B 约 75.9% 均值（单 epoch），低于 CLAP 90.8%
4. 在 Qwen3.5（0.8B/2B/4B）和 Qwen2.5-VL（3B）等主干上有不同表现

## 代表工作

- [[CLAP]]: 在 VLA-0 基础上引入 Language-Action Grounding，全面超越 VLA-0

## 相关概念

- [[Output-Distribution Mismatch]]
- [[Language-Action Grounding]]
- [[Autoregressive Policy]]
- [[VLA（视觉-语言-动作模型）]]
