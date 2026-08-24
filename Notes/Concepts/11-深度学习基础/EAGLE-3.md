---
type: concept
aliases: [EAGLE3, Speculative Decoding EAGLE]
---

# EAGLE-3

## 定义
一种 autoregressive draft model 的 speculative decoding 方法，通过轻量级 draft 模型并行预测多个候选 token，经 target LLM 验证后加速推理，是 DFlash 的主要对比 baseline。

## 数学形式
$$\text{Speedup} = \frac{1}{1 - \alpha} \cdot \frac{1}{\tau_{\text{draft}}/\tau_{\text{target}} + \alpha}$$

其中 $\alpha$ 为平均接受率，$\tau$ 为单步推理时延。

## 核心要点
1. Draft 模型以自回归方式预测 token，仍受序列依赖约束，无法真正并行
2. 通过共享 target 模型的 feature 作为 draft 模型的上下文来提升接受率
3. EAGLE-3 是迭代改进版本，DFlash 在此基础上替换为并行 block diffusion draft，实现更高加速比

## 代表工作
- [[DFlash]]: 以 EAGLE-3 为主要对比，DFlash 在相同场景下实现最高 2.5x 的额外加速

## 相关概念
- [[DiT]]: 扩散 Transformer，DFlash 的 draft 模型基础架构
- [[GRPO]]: 强化学习算法，可与 speculative decoding 结合提升样本效率
