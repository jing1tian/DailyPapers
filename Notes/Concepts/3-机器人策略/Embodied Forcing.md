---
type: concept
aliases: [EBF, 形态强迫, 形态感知噪声调制]
---

# Embodied Forcing

## 定义

一种类 Classifier-Free Guidance 技术，通过在扩散前向过程中引入形态特定的噪声尺度，将动作生成隐式引导到机器人特定形态的功能组件，从而实现跨形态统一动作头的形态感知生成。

## 数学形式

全局形态感知前向过程：
$$\tau_{t}^{k}=\sqrt{1-\beta_{k}}\tau_{t}^{k-1}+\sqrt{\beta_{k}}\cdot\sigma_{e}\epsilon, \quad \epsilon\sim\mathcal{N}(0,I)$$

局部功能感知金字塔噪声：
$$\boldsymbol{\Sigma}_{e,f}=\text{diag}(\sigma_{e},\delta\sigma_{e},\delta^{2}\sigma_{e},\ldots)$$

## 核心要点

1. **全局层面**：为每种机器人形态分配形态特定的噪声尺度 $\sigma_e$，在前向加噪过程中编码形态差异
2. **局部层面**：基于关节与末端执行器的距离施加递减噪声（空间金字塔），越靠近末端的关节噪声越小，强化精细动作
3. **最优超参数**：衰减系数 δ=0.5，窗口大小 $W_n=4$
4. **头部兼容性**：对扩散头有显著提升效果，但与流匹配头不兼容（推测因少步采样无法捕捉形态方差）

## 代表工作

- [[X-DiffVLA]]: 提出 EBF，在 RoboCasa 基准中使扩散头从 41.2% 提升到 56.9%（+15.7%）

## 相关概念

- [[Classifier-Free Guidance]]
- [[扩散模型]]
- [[扩散策略]]
- [[Morphological Tree Diffusion]]
