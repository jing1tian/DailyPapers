---
type: concept
aliases: [First-Order MAML, 一阶元学习]
---

# FOMAML

## 定义
MAML 的一阶近似版本，在计算元梯度时忽略二阶导数（Hessian 项），将外循环梯度近似为内循环最终参数处的一阶梯度，大幅降低计算成本。

## 数学形式
$$\nabla_\theta \mathcal{L}_{\mathcal{T}_i}(\theta') \approx \nabla_{\theta'} \mathcal{L}_{\mathcal{T}_i}(\theta'), \quad \theta' = \theta - \alpha \nabla_\theta \mathcal{L}_{\mathcal{T}_i}(\theta)$$

## 核心要点
1. 忽略内循环梯度对外循环梯度的影响（stop gradient through inner loop）
2. 计算复杂度与标准梯度相同，无需二阶导数
3. 实践中效果与 MAML 接近，但计算量仅为其一小部分
4. ForeSplat 用 FOMAML 做 meta-training，让前馈模型优化感知前置

## 代表工作
- [[FOMAML]]：Nichol et al. 2018 / Finn et al. 2017 附录
- [[ForeSplat]]：Optimization-Aware 前馈 3DGS

## 相关概念
- [[MAML]]
