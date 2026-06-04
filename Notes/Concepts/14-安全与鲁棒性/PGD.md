---
type: concept
aliases: [Projected Gradient Descent]
---

# PGD

## 定义
投影梯度下降（Projected Gradient Descent）对抗攻击方法，通过多步梯度更新生成强对抗样本。

## 数学形式
$$x^{t+1} = \Pi_{\mathcal{S}}(x^t + lpha \cdot 	ext{sign}(
abla_x \mathcal{L}(f(x^t), y)))$$

## 核心要点
1. 多步迭代更新，比 FGSM 更强
2. 投影操作确保扰动在 $\ell_\infty$ 球约束内
3. 是生成对抗补丁的常用优化方法

## 代表工作
- Towards Deep Learning Models Resistant to Adversarial Attacks (Madry et al., 2018)

## 相关概念
- [[TRAP]]
- [[POPA-VLA]]
- [[DIP]]
