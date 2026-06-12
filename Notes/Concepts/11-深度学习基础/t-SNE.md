---
type: concept
aliases: [t-SNE, t-distributed stochastic neighbor embedding, t分布随机邻域嵌入]
---

# t-SNE

## 定义

t-SNE（t-distributed Stochastic Neighbor Embedding）是一种非线性降维可视化方法，将高维数据映射到 2D/3D 空间，同时尽可能保留局部邻域结构，常用于聚类结果的直观展示。

## 数学形式

$$
KL(P \| Q) = \sum_{i \neq j} p_{ij} \log \frac{p_{ij}}{q_{ij}}
$$

高维相似度用高斯核计算，低维相似度用 t 分布（自由度 1）计算：

$$
q_{ij} = \frac{(1 + \|y_i - y_j\|^2)^{-1}}{\sum_{k \neq l}(1 + \|y_k - y_l\|^2)^{-1}}
$$

## 核心要点

1. 高维空间用高斯核计算 $p_{ij}$，低维空间用 t 分布（重尾）计算 $q_{ij}$，缓解"拥挤问题"
2. 通过最小化两个分布的 KL 散度，迭代优化低维坐标
3. t-SNE 结果受 perplexity 超参数影响，仅保留局部结构，全局距离不可直接比较
4. 常用于验证表征学习质量：聚类分离 = 模型学到了判别性特征

## 代表工作

- [[Dream-Tac]]: 用 t-SNE 可视化触觉表征，验证不同操作动作形成独立聚类

## 相关概念

- [[对比学习]]
- [[多头自注意力]]
- [[VAE]]
