---
type: concept
aliases: [Post-Training Quantization, 训练后量化]
---

# PTQ (Post-Training Quantization)

## 定义
在模型训练完成后，不修改训练流程，直接将浮点权重和/或激活值量化为低精度（INT8、INT4、FP8 等）表示，以降低推理时的内存占用和计算开销。

## 数学形式

线性量化：
$$x_q = \text{round}\left(\frac{x}{s}\right) + z, \quad x \approx s(x_q - z)$$

其中 $s$ 为缩放因子（scale），$z$ 为零点偏移（zero point）。

## 核心要点
1. 无需重新训练，仅需少量校准数据（calibration set）来估计激活值的范围
2. 权重量化（W-only）对精度损失最小；激活量化（W+A）可进一步降低计算量
3. 常见格式：INT8、INT4、FP8（E4M3/E5M2）、NF4（NormalFloat）
4. 与 QAT 相比精度通常略低，但无需训练成本

## 代表工作
- [[GPTQ]]: 基于二阶信息的 LLM 权重量化（INT4）
- [[AWQ]]: 激活感知的权重量化

## 相关概念
- [[QAT]]
- [[GPTQ]]
- [[HiF8]]
