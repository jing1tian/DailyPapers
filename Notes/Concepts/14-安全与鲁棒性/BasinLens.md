---
type: concept
aliases: [BasinLens, failure basin discovery, WM failure discovery]
---

# BasinLens

## 定义
自动发现世界模型在自然输入下灾难性预测失败点的框架，用 UCB 探索 + 高斯过程建模 failure landscape，找到导致 WM 崩溃的"失败盆地"（failure basin）。

## 数学形式
$$x^* = \argmax_{x \in \mathcal{X}} \mu(x) + \kappa \sigma(x)$$

UCB 策略在 GP 建模的 failure score 上优化，$\mu, \sigma$ 为 GP 的均值和标准差。

## 核心要点
1. 不依赖 adversarial 扰动，专门找自然输入下的失败
2. GP 建模 failure landscape 使搜索高效，避免全空间暴力搜索
3. 在 DIAMOND、IRIS、LeWorldModel 等多个 WM 上发现了真实失败模式

## 代表工作
- [[Where World Models Break]]: 提出 BasinLens 框架的论文

## 相关概念
- [[World Model]]
- [[Gaussian Process]]
- [[UCB]]
