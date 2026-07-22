---
type: concept
aliases: [TRADES, TRadeoff-inspired Adversarial DEfense via Surrogate-loss minimization]
---

# TRADES

## 定义
一种对抗训练框架，通过 surrogate loss 显式平衡自然精度（natural accuracy）和鲁棒性（robust accuracy）之间的 tradeoff。

## 数学形式
$$\min_f \mathbb{E}_{(x,y)} \left[ \mathcal{L}(f(x), y) + \beta \cdot \max_{\|x'-x\|\leq\epsilon} \mathcal{L}(f(x'), f(x)) \right]$$

其中第一项是自然损失，第二项是对抗 surrogate 损失，$\beta$ 控制 tradeoff 程度。

## 核心要点
1. 用 KL 散度作为 surrogate loss，避免标签硬化导致的过拟合
2. 内层 max 用 PGD 近似求解
3. $\beta$ 越大越鲁棒但自然精度越低

## 代表工作
- [[RobustVLA]]: 将 TRADES 扩展到多模态 VLA 鲁棒性训练

## 相关概念
- [[PGD]]
- [[对抗训练]]
