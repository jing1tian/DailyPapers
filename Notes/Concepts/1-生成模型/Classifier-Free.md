---
type: concept
aliases: [CFG, Classifier-Free Guidance, 无分类器引导]
---

# Classifier-Free Guidance (CFG)

## 定义
训练时随机丢弃条件（用空条件替代），推理时将条件预测与无条件预测线性外推以增强条件对生成的影响。

## 数学形式
$$\hat{\epsilon}_\theta(x_t, c) = \epsilon_\theta(x_t, \varnothing) + w \cdot [\epsilon_\theta(x_t, c) - \epsilon_\theta(x_t, \varnothing)]$$

其中 $w > 1$ 为引导强度，$c$ 为条件，$\varnothing$ 为空条件。

## 核心要点
1. 单模型同时学习条件和无条件分布，无需额外分类器
2. 引导强度 $w$ 控制生成多样性与条件一致性的 tradeoff
3. 推理时额外一次前向传播（计算无条件输出），开销翻倍
4. 已成为 diffusion policy 和 VLA 里标配的测试时干预工具

## 代表工作
- [[SDN]]（Selected Diffusion Noise）：用 CFG 做动作平滑，提升 VLA 鲁棒性
- Ho & Salimans 2022: 提出 CFG 的原始论文

## 相关概念
- [[Diffusion Policy]]
- [[DDPM]]
- [[DDIM]]
