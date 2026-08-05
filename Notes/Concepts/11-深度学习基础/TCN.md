---
type: concept
aliases: [Temporal Convolutional Network, 时序卷积网络]
---

# TCN (Temporal Convolutional Network)

## 定义
用因果扩张卷积（causal dilated convolution）处理时序信号的网络架构，每层感受野随深度指数增长，在时序建模上可替代 RNN。

## 数学形式
$$y_t = \sum_{k=0}^{K-1} w_k \cdot x_{t - d \cdot k}$$
其中 $d$ 为扩张因子（dilation），$K$ 为卷积核大小，因果性由只访问 $t$ 及之前时步保证。

## 核心要点
1. 因果卷积保证不使用未来信息（适合在线控制）
2. 扩张卷积使感受野达到 $O(2^L)$（$L$ 层），参数量远小于 LSTM
3. 训练可完全并行（不同于 RNN 的序列依赖）
4. 在 FACT 论文中用于从视频序列估计接触力时序特征

## 代表工作
- [[FACT]]: 用 TCN 做时序感知力估计（Time-Aware TCN）

## 相关概念
- [[Autoregressive Policy]]
