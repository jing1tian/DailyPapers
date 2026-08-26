---
type: concept
aliases: [counterfactual training, 反事实监督]
---

# Counterfactual Supervision

## 定义
通过构造"假如执行了错误动作会怎样"的反事实样本作为负监督信号，弥补行为克隆只有正样本的不足。

## 数学形式
$$\mathcal{L} = \mathcal{L}_\text{BC}(a^+) + \lambda \cdot \mathcal{L}_\text{neg}(a^-)$$

其中 $a^+$ 是专家动作，$a^-$ 是构造的 counterfactual 动作。

## 核心要点
1. BC 只提供"应该做什么"的正向监督，没有"不应该做什么"的约束
2. Counterfactual 动作通过偏离指令方向或语义不一致的动作生成
3. 类似对比学习中的负样本对的作用

## 代表工作
- [[CounterAlign]]: 为 VLA 引入 counterfactual 监督

## 相关概念
- [[Behavior Cloning]]
- [[Advantage-Weighted Regression]]
- [[VLA]]
