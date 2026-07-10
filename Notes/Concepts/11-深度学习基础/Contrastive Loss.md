---
type: concept
aliases: [对比损失, 对比学习损失, InfoNCE, Supervised Contrastive Loss]
---

# Contrastive Loss

## 定义

一种通过拉近同类样本（正样本对）、推开异类样本（负样本对）的嵌入距离来学习表示的损失函数；监督版本（Supervised Contrastive Loss）允许同一类别内多个正样本同时参与对比。

## 数学形式

**标准 InfoNCE（无监督/自监督）**:

$$
\mathcal{L}_{\text{NCE}} = -\log \frac{\exp(\mathbf{z}_i \cdot \mathbf{z}_j / \tau)}{\sum_{k \neq i} \exp(\mathbf{z}_i \cdot \mathbf{z}_k / \tau)}
$$

**有监督对比损失（SupCon）**:

$$
\mathcal{L}_{\text{SupCon}} = \frac{1}{|P|} \sum_{p \in P} -\log \frac{\exp(\mathbf{z} \cdot \mathbf{z}_p / \tau)}{\sum_{a \in A} \exp(\mathbf{z} \cdot \mathbf{z}_a / \tau)}
$$

## 核心要点

1. **温度系数 $\tau$**: 控制分布的尖锐程度，$\tau$ 越小分布越集中，对难负样本越敏感
2. **正样本对**: 自监督场景为同一样本的不同数据增强视图；有监督场景为同一类别的不同样本
3. **负样本**: 同 batch 内的异类样本，batch size 越大负样本越丰富，性能越好

## 代表工作

- [[HiMoE-VLA]]: 将 Supervised Contrastive Loss 应用于 MoE 路由分布对齐（AS-Reg），使同一动作空间的路由向量相互靠近

## 相关概念

- [[Mixture-of-Experts]]
- [[Load Balancing Loss]]
