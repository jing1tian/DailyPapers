---
type: concept
aliases: [策略引导, Deployment-time Steering, 部署时引导]
---

# Policy Steering（策略引导）

## 定义

在不修改预训练策略权重的前提下，通过外部评估机制（价值函数、世界模型等）对策略输出的候选动作进行重排或筛选，从而在部署时提升策略鲁棒性的框架。

## 核心要点

1. **冻结策略**: 原始策略权重保持不变，避免任务特定微调的高成本
2. **候选生成与评估分离**: 策略负责生成候选，外部评估器负责排序
3. **即插即用**: 可以与任意预训练 VLA 策略组合使用
4. **对比微调**: 与 finetuning 形成互补，适用于数据稀缺的 OOD 部署场景

## 数学形式

$$
k^* = \operatorname{argmax}_{k}\ \text{Evaluator}\left(o_t,\, a^{(k)}_{t:t+H-1},\, \ell\right)
$$

其中评估器可以是价值函数、世界模型 + 价值函数组合等。

## 代表工作

- [[DreamSteer]]: 用潜在世界模型 + 冻结 VLAC 评分，无需任何微调实现部署时引导
- [[V-GPS]]: 单步动作重排的早期策略引导方法

## 相关概念

- [[VLA（视觉-语言-动作模型）]]
- [[World Model]]
- [[VLAC]]
- [[Distribution Shift]]
