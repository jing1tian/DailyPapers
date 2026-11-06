---
type: concept
aliases: [Diffusion Policy, 扩散策略]
---

# Diffusion-Policy

## 定义
Chi et al. 2023 提出的机器人操作策略学习方法，将 [[扩散模型]] 应用于动作生成，通过条件去噪过程输出动作序列。

## 数学形式
$$a_0 = \text{DDPM}(a_T; c), \quad c = \text{encoder}(o)$$

## 核心要点
1. 以观测 $o$ 为条件，从高斯噪声去噪生成动作序列
2. 支持 DDPM 和 DDIM 两种推理模式（速度/质量权衡）
3. 多模态动作分布建模优于 MSE 策略
4. [[PHIL]] 以 Diffusion Policy 类架构做主动人在回路干预

## 代表工作
- [[PHIL]]: 在 Diffusion Policy 上加 proactive HitL 预览机制
- [[BESO]]: 类似 score function 方法但推理更快

## 相关概念
- [[扩散模型]]
- [[BESO]]
- [[ACT]]
