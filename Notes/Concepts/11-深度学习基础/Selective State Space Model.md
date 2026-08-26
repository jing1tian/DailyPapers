---
type: concept
aliases: [SSM, Mamba, selective SSM, 状态空间模型]
---

# Selective State Space Model

## 定义
一类序列建模架构（以 Mamba 为代表），通过输入依赖的选择性门控机制过滤信息，在线性时间复杂度下处理长序列，无需注意力机制的二次开销。

## 数学形式
$$h_t = A_t h_{t-1} + B_t x_t, \quad y_t = C_t h_t$$

其中矩阵 $A_t, B_t, C_t$ 由输入 $x_t$ 动态决定（选择性）。

## 核心要点
1. 时间复杂度 $O(L)$，Transformer 注意力为 $O(L^2)$
2. 选择性机制允许模型忽略无关输入，类比 LSTM 的门控但更高效
3. 在长序列 VLA action chunk 推理中可改善速度-精度权衡

## 代表工作
- [[MambaSmolVLA]]: 用 SSM 替换部分 Transformer 层改善 VLA 效率
- [[RoboMamba]]: Mamba 用于机器人控制
- [[MaIL]]: Mamba-based 模仿学习

## 相关概念
- [[Mamba]]
- [[Transformer]]
- [[Action Chunking]]
