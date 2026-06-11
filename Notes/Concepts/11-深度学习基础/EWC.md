---
type: concept
aliases: [Elastic Weight Consolidation, 弹性权重巩固]
---

# EWC（Elastic Weight Consolidation）

## 定义
EWC 是一种防止神经网络灾难性遗忘的方法，通过在损失函数中添加基于 Fisher 信息矩阵的正则项，约束对旧任务重要的参数不发生大幅偏移。

## 数学形式
$$\mathcal{L}(\theta) = \mathcal{L}_B(\theta) + \sum_i \frac{\lambda}{2} F_i (\theta_i - \theta_{A,i}^*)^2$$

其中 $F_i$ 为 Fisher 信息矩阵对角元素，$\theta_{A,i}^*$ 为旧任务的最优参数，$\lambda$ 控制正则化强度。

## 核心要点
1. Fisher 信息矩阵衡量每个参数对旧任务的重要性，重要参数被"弹性"固定
2. 适用于持续学习场景：在新任务微调 VLA 时防止旧场景性能下降
3. 计算成本：需要在旧任务数据上估计 Fisher 矩阵

## 代表工作
- Kirkpatrick et al. (2017)：EWC 原始论文
- [[SafeNavVLA]]：用 EWC 防止社会导航微调时遗忘原始 VLA 能力

## 相关概念
- [[灾难性遗忘]]
- [[持续学习]]
- [[LoRA]]
