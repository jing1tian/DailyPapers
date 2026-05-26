---
type: concept
aliases: [Denoising Diffusion Implicit Models, DDIM采样]
---

# DDIM

## 定义
DDIM（Denoising Diffusion Implicit Models）是对 DDPM 的改进采样方法，通过非马尔可夫扩散过程将推理步数从 1000 步压缩到 10-50 步，同时保持样本质量。

## 数学形式
DDIM 确定性采样：
$$x_{t-1} = \sqrt{\bar\alpha_{t-1}} \underbrace{\frac{x_t - \sqrt{1-\bar\alpha_t}\epsilon_\theta(x_t)}{\sqrt{\bar\alpha_t}}}_{\text{predicted }x_0} + \sqrt{1-\bar\alpha_{t-1}} \cdot \epsilon_\theta(x_t)$$

## 核心要点
1. 原 DDPM 推理需要 1000 步，DDIM 引入跳步采样只需 10-50 步
2. DDIM 的"隐式"来源：扩散过程不再是随机的（无噪声注入），采样确定性
3. 提供平滑的 latent space interpolation（相比 DDPM 更平滑）
4. eta=0 为确定性 DDIM；eta=1 退化回 DDPM

## 代表工作
- [[JOPAT]]: 基于 DDIM 采样的扩散 Transformer WAM
- [[扩散模型]]: DDIM 是扩散模型推理加速的基础方法

## 相关概念
- [[扩散模型]]
- [[Diffusion Policy]]
- [[Flow Matching]]
