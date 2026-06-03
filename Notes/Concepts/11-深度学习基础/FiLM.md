---
type: concept
aliases: [Feature-wise Linear Modulation, 特征线性调制, FiLM Layer]
---

# FiLM（Feature-wise Linear Modulation）

## 定义

通过对中间特征图施加逐通道的仿射变换（缩放 + 偏移），将条件信号（如动作、语言、类别）注入神经网络的一种轻量级条件机制。

## 数学形式

$$
F'_\ell = \gamma_\ell \odot F_\ell + \beta_\ell
$$

其中 $\gamma_\ell$ 和 $\beta_\ell$ 由条件输入通过 MLP 预测，$\odot$ 为逐元素乘法。

## 核心要点

1. **条件注入无需修改主干**: 通过预测缩放/偏移参数插入条件信息，对原始网络结构改动最小
2. **逐层调制**: 通常在多个网络层分别注入，实现不同层次的条件控制
3. **与 LayerNorm 关系**: 当 $\gamma$ 和 $\beta$ 为常数时等价于条件 LayerNorm，FiLM 更通用（参数由输入动态预测）
4. **Zero-init 技巧**: 初始化 $\gamma=1, \beta=0$ 使训练初期为恒等变换，稳定训练

## 代表工作

- [[SKIP]]: AC-FILM 中用于将动作序列注入插值网络，每层特征图通过 FiLM 对齐运动信息
- [[FiLM: Visual Reasoning with a General Conditioning Layer]]: 原始提出，用于视觉问答中的语言条件注入

## 相关概念

- [[Cross-Attention]]: 另一种条件注入机制，通过注意力机制融合条件
- [[AdaLN]]: Adaptive LayerNorm，与 FiLM 思想类似，常见于扩散 Transformer
- [[AC-FILM]]: 将 FiLM 与动作条件和门控机制结合的 SKIP 变体
