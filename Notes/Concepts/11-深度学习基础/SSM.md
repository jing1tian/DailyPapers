---
type: concept
aliases: [State Space Model, 状态空间模型, S4, Mamba]
---

# SSM（State Space Model）

## 定义
用连续或离散状态方程 $\dot{h}(t) = Ah(t) + Bx(t)$ 对序列进行线性递推建模的架构，以线性时间复杂度处理长序列，是 Transformer 的高效替代。

## 数学形式

$$
h_t = \bar{A} h_{t-1} + \bar{B} x_t, \quad y_t = C h_t
$$

其中 $\bar{A}, \bar{B}$ 为离散化后的状态矩阵。

## 核心要点
1. 时间/空间复杂度 $O(n)$，相比 Transformer 的 $O(n^2)$ 大幅节省
2. 历史信息被压缩进固定维度的隐状态 $h_t$，长距细节可能丢失
3. [[Mamba]] 引入输入依赖的选择机制，使 SSM 更具表达能力
4. 与 Transformer 混合使用（Interleaved）可兼顾效率和质量

## 代表工作
- [[CoME]]: 将 SSM 作为对比 baseline，CoME 的记忆专家组合优于纯 SSM

## 相关概念
- [[Mamba]]
- [[Transformer]]
