---
type: concept
aliases: [Expectation Over Transformation]
---

# EOT

## 定义
Expectation Over Transformation：一种对抗训练技术，通过在多种物理变换下优化对抗扰动，使对抗补丁在物理场景中更鲁棒。

## 数学形式
$$\delta^* = \arg\max_{\delta} \mathbb{E}_{t \sim T}[\mathcal{L}(f(t(x + \delta)), y_{adv})]$$

## 核心要点
1. 模拟物理世界中的视角变化、光照变化等变换 $T$
2. 优化在期望意义下有效的对抗扰动
3. 是物理对抗补丁研究的标准训练范式

## 代表工作
- [[SARF]]: 在 VLA 鲁棒性评测中使用 EOT 训练攻击

## 相关概念
- [[对抗补丁攻击]]
- [[PGD]]
