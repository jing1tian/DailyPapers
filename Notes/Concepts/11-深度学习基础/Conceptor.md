---
type: concept
aliases: [概念算子, Conceptors]
---

# Conceptor

## 定义

Conceptor 是一种正半定线性算子，通过软投影将激活数据映射到其主成分子空间，兼顾重构精度与算子复杂度的正则化权衡。

## 数学形式

**优化目标**:

$$
\min_C \frac{1}{N}\|\tilde{X} - \tilde{X}C\|_F^2 + \alpha^{-2}\|C\|_F^2
$$

**闭式解**:

$$
C = R(R + \alpha^{-2}I)^{-1}
$$

其中 $R = \frac{1}{N}\tilde{X}^\top\tilde{X}$ 为激活协方差矩阵。

**特征值结构**:

$$
\mu_i = \frac{\lambda_i}{\lambda_i + \alpha^{-2}}
$$

高方差方向 $\mu_i \approx 1$（通过），低方差方向 $\mu_i \approx 0$（抑制）。

## 核心要点

1. Conceptor 是 PCA 的软化版本，孔径参数 $\alpha$ 控制覆盖范围
2. 支持布尔代数运算（AND、OR、NOT），可进行子空间的逻辑组合
3. 布尔 NOT：$\neg C = I - C$；布尔 AND：$A \land B = (A^{-1} + B^{-1} - I)^{-1}$
4. 由 Jaeger (2014) 提出，原用于循环神经网络的任务切换

## 代表工作

- [[COAST]]: 将 Conceptor 迁移至 VLA 推理时激活引导，通过对比 Conceptor 提升机器人操作成功率
- Postmus & Abreu (2024, 2025): 将 Conceptor 应用于 LLM 激活引导

## 相关概念

- [[Activation Steering]]
- [[Residual Stream]]
- [[Principal Component Analysis]]
- [[Subspace Similarity]]
