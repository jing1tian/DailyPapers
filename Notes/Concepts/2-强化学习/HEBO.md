---
type: concept
aliases: [Heteroscedastic Evolutionary Bayesian Optimization, 异方差进化贝叶斯优化]
---

# HEBO

## 定义
一种贝叶斯优化算法，通过异方差噪声建模（不同输入点有不同的观测噪声水平）和进化策略的结合，提升在复杂黑盒优化问题上的搜索效率。

## 数学形式
$$\alpha(\mathbf{x}) = \mu(\mathbf{x}) + \kappa \sigma(\mathbf{x})$$

其中 $\mu, \sigma$ 来自异方差 GP，$\kappa$ 控制探索-利用权衡。

## 核心要点
1. 用异方差高斯过程（GP）建模目标函数，允许不同区域的噪声方差不同
2. 结合进化算法优化采集函数（Acquisition Function），避免局部最优
3. 在 NeurIPS 2020 Black-Box Optimization Challenge 中获得第一名

## 代表工作
- [[Ace]]: 用 HEBO 优化乒乓球发球策略参数

## 相关概念
- [[PPO]]
- [[RL]]
