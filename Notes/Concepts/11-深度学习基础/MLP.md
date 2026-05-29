---
type: concept
aliases: [MLP, Multi-Layer Perceptron, 多层感知机, 全连接网络, Feedforward Network]
---

# MLP（多层感知机）

## 定义
多层感知机（Multi-Layer Perceptron, MLP）是最基础的前馈神经网络，由输入层、若干隐藏层和输出层组成，每层通过仿射变换 + 非线性激活函数堆叠。

## 数学形式

$$
h^{(l)} = \sigma\left(W^{(l)} h^{(l-1)} + b^{(l)}\right)
$$

其中 $\sigma$ 为激活函数（ReLU、GELU、SiLU 等），$W^{(l)}, b^{(l)}$ 为第 $l$ 层的权重矩阵和偏置。

## 核心要点
1. **通用逼近**: 单隐藏层 MLP 理论上可逼近任意连续函数（Universal Approximation Theorem）
2. **轻量调度器**: 在大模型中常用 MLP 作为轻量决策头，参数量远小于主干网络
3. **瓶颈结构**: 常见设计为先升维再降维（或先降维形成瓶颈），增加表达能力并控制参数量

## 代表工作
- [[SANTS]]: 用 2 层 MLP（512→256）作为轻量调度器，预测去噪停止点和噪声进展比

## 相关概念
- [[Transformer]]: 现代深度学习主干，MLP 是其 FFN 模块的基础
- [[扩散策略]]: 扩散策略中常用 MLP 头预测动作
