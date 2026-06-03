---
type: concept
aliases: [Feature-wise Linear Modulation, 特征线性调制]
---

# FiLM

## 定义

FiLM（Feature-wise Linear Modulation）是一种条件特征调制机制，通过从条件输入生成逐通道的缩放参数 $\gamma$ 和偏移参数 $\beta$，对中间特征图进行仿射变换，从而将外部条件信息注入网络中间层。

## 数学形式

$$
F'_\ell = \gamma_\ell \odot F_\ell + \beta_\ell
$$

其中 $[\gamma_\ell, \beta_\ell] = \mathrm{MLP}(\mathbf{c})$，$\mathbf{c}$ 为条件向量。

## 核心要点

1. **轻量条件注入**: 仅需在每层添加一个生成 $\gamma, \beta$ 的小型 MLP，参数开销极低
2. **层级调制**: 可在网络各层独立施加不同强度的调制
3. **广泛适用性**: 适用于语言条件、动作条件、风格迁移等多种场景

## 代表工作

- [[SKIP]]: AC-FILM 使用 FiLM 机制将动作序列注入插帧网络各金字塔层级

## 相关概念

- [[AdaLN]]（自适应层归一化，类似思路）
- [[注意力机制]]
