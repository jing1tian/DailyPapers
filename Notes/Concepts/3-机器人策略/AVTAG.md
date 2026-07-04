---
type: concept
aliases: [Action-Visual-Tactile Attention Guidance, Contact-Gated Tactile Attention]
---

# AVTAG

## 定义
在 VLA/WAM 的 flow matching 推理中，通过门控机制在接触阶段强制动作查询（action query）优先关注触觉 token 而非视觉 token 的注意力引导策略。

## 数学形式
$$
\text{Attn}(Q_\text{act}, K, V) = \text{softmax}\left(\frac{Q_\text{act} K^\top}{\sqrt{d}} + \lambda(t) \cdot M_\text{tac}\right) V
$$

其中 $\lambda(t)$ 为接触阶段门控系数，$M_\text{tac}$ 为触觉 token 掩码（接触阶段置 1，非接触阶段置 0）。

## 核心要点
1. 触觉信号在接触发生时才有意义，非接触阶段对动作决策贡献极小
2. 简单地把触觉 token 拼入输入序列会导致视觉与触觉权重竞争，AVTAG 用门控解决这个问题
3. 与 [[Asymmetric MoT|非对称 MoT 注意力]] 配合使用：MoT 负责视觉-触觉时序融合，AVTAG 负责动作查询阶段的接触聚焦
4. 接触阶段检测可基于触觉传感器读数阈值，也可学习得到

## 代表工作
- [[VT-WAM]]: 首次提出 AVTAG，在六个接触密集任务上验证，成功率比 Fast-WAM 高 26.67%

## 相关概念
- [[World Action Model]]
- [[Flow Matching]]
- [[CBF]]
- [[MoT]]
