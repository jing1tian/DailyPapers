---
type: concept
aliases: [CFG, 分类器自由引导, Classifier-Free Guidance]
---

# Classifier-Free Guidance (CFG)

## 定义

在扩散模型推理阶段，通过插值无条件预测与条件预测来增强条件信号强度的引导策略，无需训练独立分类器。

## 数学形式

标准 CFG：

$$
\hat{\mathbf{z}}_{\text{cfg}} = \hat{\mathbf{z}}_u + s \cdot (\hat{\mathbf{z}}_c - \hat{\mathbf{z}}_u)
$$

Per-modality CFG（用于联合扩散，如 A2World-policy）：

$$
\hat{\mathbf{z}}_{\text{cfg}}^m = \hat{\mathbf{z}}_u^m + s_m \left(\hat{\mathbf{z}}_c^m - \hat{\mathbf{z}}_u^m\right)
$$

**符号**:
- $\hat{\mathbf{z}}_u$: 无条件预测（训练时随机丢弃条件）
- $\hat{\mathbf{z}}_c$: 条件预测
- $s$（或 $s_m$）: 引导尺度，$s>1$ 时增强条件效果

## 核心要点

1. **训练**: 以一定概率（通常 10-20%）将条件置零，使模型同时学习有/无条件预测
2. **推理**: 分别计算条件和无条件预测，按引导尺度 $s$ 做外推
3. **Per-modality**: 多模态联合扩散中对每个模态独立应用 CFG，允许差异化引导强度

## 代表工作

- Ho & Salimans (2022): 提出 CFG
- [[A2World]]: 对视频模态和动作模态分别应用 Per-modality CFG

## 相关概念

- [[Diffusion Model]]
- [[EDM (Elucidated Diffusion Model)|EDM]]
- [[Joint Diffusion]]
