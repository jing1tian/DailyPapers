---
type: concept
aliases: [Diffusion Policy, Chi Diffusion Policy]
---

# Diffusion Policy（DP）

## 定义
Chi 等（2023）提出的基于扩散模型的机器人操作策略，将动作生成建模为条件扩散过程，通过去噪预测动作序列。

## 数学形式
$$\mathbf{a}_{t-1} = \frac{1}{\sqrt{\alpha_t}}\left(\mathbf{a}_t - \frac{1-\alpha_t}{\sqrt{1-\bar\alpha_t}}\epsilon_\theta(\mathbf{a}_t, \mathbf{o}, t)\right) + \sigma_t \mathbf{z}$$

## 核心要点
1. 多模态动作分布：扩散模型天然处理多峰分布，解决行为克隆的均值退化问题
2. 动作块（Action Chunk）预测：一次预测多步动作，降低频率错误
3. CNN/Transformer 两种骨干变体
4. 是现代 VLA（π₀、GR00T 等）的动作头基础

## 代表工作
- [[π₀]]: 用流匹配（flow matching）替代扩散的改进版本
- [[RoboFlow4D]]: 用 3D flow 条件化 Diffusion Policy

## 相关概念
- [[VLA]]
- [[流匹配]]
- [[Action Chunking]]
