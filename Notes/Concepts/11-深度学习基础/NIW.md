---
type: concept
aliases: [Normal-Inverse-Wishart, 正态逆威沙特分布]
---

# NIW

## 定义
Normal-Inverse-Wishart（NIW）分布：多元正态分布参数（均值 $\mu$ 和协方差矩阵 $\Sigma$）的共轭先验，常用于贝叶斯混合模型中的参数后验推断。

## 数学形式
$$(\mu, \Sigma) \sim \text{NIW}(\mu_0, \lambda, \Psi, \nu)$$
$$p(\mu, \Sigma) \propto |\Sigma|^{-(\nu+d/2+1)} \exp\!\left(-\frac{1}{2}\text{tr}(\Psi \Sigma^{-1}) - \frac{\lambda}{2}(\mu-\mu_0)^T\Sigma^{-1}(\mu-\mu_0)\right)$$

其中 $\mu_0$ 为先验均值，$\lambda$ 为精度缩放，$\Psi$ 为尺度矩阵，$\nu$ 为自由度。

## 核心要点
1. 是多元正态参数的**共轭先验**，后验仍是 NIW，解析可解
2. 在 DPMM 中作为基分布 $G_0$，对每个混合成分的参数建模
3. 在 3DGS 中：对每个 Gaussian 基元的均值和协方差建模后验不确定性
4. 相比点估计，NIW 先验提供原生不确定性量化（uncertainty quantification）

## 代表工作
- [[BayesianGS]]（今日论文，NIW 先验用于 3DGS 贝叶斯参数建模）
- [[DPMM]]（NIW 作为 DPMM 的共轭基分布）

## 相关概念
- [[DPMM]]（使用 NIW 先验的非参数贝叶斯混合）
- [[UQ]]（不确定性量化，NIW 提供的核心能力）
