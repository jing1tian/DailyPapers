---
type: concept
aliases: ["Intervention Weighted Regression"]
---

# IWR

## 定义
交互式模仿学习变体，对人类干预样本加权以提升样本效率，减少所需干预次数。

## 数学形式
$$\mathcal{L}_{IWR} = \mathbb{E}_{(s,a)\sim\mathcal{D}}[w(s) \cdot ||a - \pi_\theta(s)||^2]$$

## 核心要点
1. 比 DAgger 更高效地利用干预数据
2. 适用于 VLA 恢复学习

## 代表工作
- 见关联笔记

## 相关概念
- [[DAgger]]
- [[HG-DAgger]]
