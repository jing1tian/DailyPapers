---
type: concept
aliases: [Quantization-Aware Training, 量化感知训练]
---

# QAT (Quantization-Aware Training)

## 定义
在训练过程中模拟量化误差（插入 fake quantization 节点），使模型在训练时就适应低精度表示，从而在推理部署时获得比 PTQ 更高的量化精度。

## 数学形式

前向传播中模拟量化（straight-through estimator 反传）：
$$x_q = \text{round}\left(\text{clamp}\left(\frac{x}{s}, -2^{b-1}, 2^{b-1}-1\right)\right) \cdot s$$

反向传播使用 STE（Straight-Through Estimator）近似量化操作的梯度：
$$\frac{\partial \mathcal{L}}{\partial x} \approx \frac{\partial \mathcal{L}}{\partial x_q}$$

## 核心要点
1. 在浮点训练图中插入 fake-quant 节点模拟量化噪声
2. 量化参数（scale/zero-point）可以固定或与模型参数一同学习
3. 比 PTQ 精度损失更小，但需要额外的训练成本（通常微调几个 epoch）
4. 适合 INT4/INT8 等激进量化精度场景

## 代表工作
- [[BitCPM4]]: MiniCPM4 的 INT4 量化版本，使用 QAT
- [[GPTQ]]: 属于 PTQ 方向的对比方案

## 相关概念
- [[PTQ]]
- [[GPTQ]]
