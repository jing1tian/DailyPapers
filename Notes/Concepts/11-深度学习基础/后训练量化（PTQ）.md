---
type: concept
aliases: [PTQ, Post-Training Quantization, 训练后量化, 后处理量化]
---

# 后训练量化（PTQ）

## 定义

在模型训练完成后，不进行任何梯度更新，直接将浮点模型的权重和/或激活值压缩到低比特整数表示（如 INT8、INT4）的模型压缩技术。

## 数学形式

对称均匀量化：

$$
\Delta_Z = \frac{\max(|Z|)}{q_{\max}}, \quad Q_Z = \operatorname{clamp}\!\left(\left\lfloor \frac{Z}{\Delta_Z} \right\rceil, -q_{\max}, q_{\max}\right)
$$

其中 $q_{\max} = 2^{k-1} - 1$（$k$ 位量化，INT4 时为 7）。

## 核心要点

1. **无需重训练**: 仅需少量校准数据（通常几十到几百个样本），大幅降低部署成本
2. **权重量化 vs 激活量化**: W4A16（只量化权重）精度损失小；W4A4（同时量化激活）压缩更激进但更难
3. **主要挑战**: 激活异常值（少数通道幅值极大，破坏量化精度）和动态范围差异
4. **常用技术**: 旋转矩阵（消除异常值）、通道缩放（SmoothQuant）、二阶优化（GPTQ）

## 代表工作

- [[GPTQ]]: 基于二阶近似的 LLM 权重量化，W4A16
- [[SmoothQuant]]: 通道平滑激活异常值，实现 W8A8
- [[Omega-QVLA]]: VLA 模型的 W4A4 全量化，复合旋转 + 逐步激活缩放

## 相关概念

- [[GPTQ]]
- [[SmoothQuant]]
- [[Hadamard 矩阵]]
- [[SVD]]
- [[量化感知训练]]
