---
type: concept
aliases: [Dirichlet Process Mixture Model, 狄利克雷过程混合模型]
---

# DPMM

## 定义
Dirichlet Process Mixture Model：一种非参数贝叶斯混合模型，使用狄利克雷过程作为先验，允许混合成分数量随数据规模自适应增长，无需预先指定聚类数量。

## 数学形式
$$G \sim \text{DP}(\alpha, G_0), \quad \theta_i \sim G, \quad x_i \sim F(\theta_i)$$

其中 $\alpha$ 为浓度参数（控制新聚类产生概率），$G_0$ 为基分布，$F(\theta)$ 为数据似然。等价于 Chinese Restaurant Process 的无限混合。

## 核心要点
1. 核心优势：不需要预先指定成分数量，数据驱动地发现聚类数
2. 推断方法：MCMC（Gibbs 采样）或变分推断（CAVI）
3. 在 3DGS 中的应用：用 DPMM 控制 Gaussian 基元数量，自适应场景复杂度
4. 计算代价较高（相比 k-means 等判别方法），在大规模问题中需近似推断

## 代表工作
- [[BayesianGS]]（今日论文，3DGS + DPMM 自适应复杂度控制）

## 相关概念
- [[NIW]]（Normal-Inverse-Wishart，常用于 DPMM 的共轭先验）
- [[MCMC]]（Gibbs 采样推断方法）
