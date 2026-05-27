---
type: concept
aliases: [Koopman Operator, Koopman Theory, 科普曼算子]
---

# Koopman 算子

## 定义
**Koopman 算子**（Koopman Operator）是将非线性动力系统嵌入无限维线性空间中的全局线性化工具，由 B.O. Koopman 于 1931 年提出，近年在控制和机器人中广泛应用。

## 数学形式
对于非线性系统 $x_{t+1} = f(x_t)$，Koopman 算子 $\mathcal{K}$ 作用在观测函数空间上：

$$\mathcal{K} \phi(x) = \phi(f(x))$$

其中 $\phi: \mathcal{X} \to \mathbb{R}^N$ 是从状态空间到高维特征空间的嵌入。在 Koopman 空间中：

$$z_{t+1} = K z_t, \quad z_t = \phi(x_t)$$

即在嵌入空间中变为线性传播。

## 核心要点
1. 全局线性化：任何非线性系统在 Koopman 嵌入下变为线性（代价是维度增加）
2. 实际使用 EDMD（Extended Dynamic Mode Decomposition）从数据中估计有限维 Koopman 矩阵
3. 神经网络版（Deep Koopman）用网络学习嵌入函数 $\phi$
4. 适合与 MPC 结合（线性约束优化更高效）

## 代表工作
- Budisic et al. (2012): Koopman 算子理论综述
- [[DNK]]: Dynamic Neural Koopman，将扩散策略蒸馏为单步 Koopman 线性化

## 相关概念
- [[MPC]]
- [[扩散策略]]
