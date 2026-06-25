---
type: concept
aliases: [Bootstrap Your Own Latent]
---

# BYOL

## 定义
自监督表征学习方法，用在线网络和目标网络（target network，参数为在线网络的滑动平均/momentum 更新）互相预测彼此对同一输入不同增强视图的表征，无需负样本对即可避免表征坍缩。

## 核心要点
1. 核心是 momentum target encoder：目标网络参数是在线网络的 EMA，提供稳定但缓慢演化的监督目标
2. 不依赖负样本对比，区别于 SimCLR 等方法
3. 这一"momentum target 防坍缩"思路被广泛借用到强化学习蒸馏、世界模型表征学习等场景

## 相关概念
- [[CTS]]
