---
type: concept
aliases: [动作损失, Action Prediction Loss, 动作监督损失]
---

# Action Loss（动作损失）

## 定义

在机器人学习策略中，衡量模型预测动作序列与 Ground-Truth 动作之间差异的损失函数，直接监督动作预测质量。

## 数学形式

通用形式：

$$
\mathcal{L}_{\text{action}} = \ell(A_{\text{pred}}, \hat{A})
$$

在循环框架（如 [[LoopVLA]]）中，对所有 $N$ 次中间预测求和：

$$
\mathcal{L}_{\text{action}} = \sum_{n=1}^{N} \ell\!\left(A^{(n)}, \hat{A}\right)
$$

常用的 $\ell(\cdot)$ 包括：
- **L2 损失**: $\|\hat{A} - A_{\text{pred}}\|_2^2$（用于回归型动作头）
- **[[Flow Matching]] 损失**: 基于流匹配的连续动作预测
- **交叉熵**: 用于离散动作（如 [[OpenFAST|FAST Token]] 化动作）

## 核心要点

1. **动作块监督**: 通常监督未来 $c$ 步的动作序列（[[Action Chunking]]），而非单步动作。
2. **中间监督**: 在循环/深度可变模型中，对每个中间预测施加 Action Loss 有助于保证各精炼深度的输出质量（Stage 1 训练）。
3. **与充分性校准的关系**: [[充分性校准]] 的目标分布 $q^{(n)} \propto \exp(-\mathcal{L}_{\text{action}}^{(n)} / \tau)$ 直接由 Action Loss 导出——损失越低，该迭代被选中的概率越大。

## 代表工作

- [[LoopVLA]]: 在 Stage 1 对所有 $N$ 次中间预测求和 Action Loss，强迫每个精炼深度均有动作预测能力。

## 相关概念

- [[Action Chunking]]
- [[Flow Matching]]
- [[充分性校准]]
- [[Vision-Language-Action Model]]
