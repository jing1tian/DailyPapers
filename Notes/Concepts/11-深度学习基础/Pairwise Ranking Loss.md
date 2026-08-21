---
type: concept
aliases: [成对排序损失, Pairwise Ranking, Listwise Ranking, BPR Loss]
---

# Pairwise Ranking Loss（成对排序损失）

## 定义

一类监督排序学习的损失函数，通过构造样本对的偏好关系（哪个更好），训练模型正确排序候选项，而非直接预测绝对分值。

## 数学形式

经典 sigmoid 形式（Bradley-Terry 模型）：

$$
\mathcal{L}_\text{rank} = -\sum_{(i,j)} \Bigl[y_{ij}\log\sigma(\hat{s}_i - \hat{s}_j) + (1-y_{ij})\log\sigma(\hat{s}_j - \hat{s}_i)\Bigr]
$$

其中偏好标签：

$$
y_{ij} = \mathbb{1}[s_i > s_j]
$$

## 核心要点

1. **对比对构造**: 从候选集中采样正负对，核心在于如何选择"有信息量"的对（如硬负样本对）
2. **梯度特性**: 预测分差越小（难以区分），梯度越大，自动聚焦于决策边界
3. **与 Contrastive Loss 关系**: 概念相似，均通过对比监督，但 Contrastive Loss 通常在表示空间操作，Ranking Loss 在分值空间操作
4. **应用场景**: 推荐系统、搜索排序、轨迹规划候选选择

## 代表工作

- [[DA-WAM]]: 在自动驾驶轨迹评分中使用，重点构造安全关键硬负样本对

## 相关概念

- [[Contrastive Loss]]
- [[Stop-Gradient]]
