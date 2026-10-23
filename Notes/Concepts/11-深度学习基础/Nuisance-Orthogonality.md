---
type: concept
aliases: [噪声正交性, Nuisance Orthogonality, 干扰正交性]
---

# Nuisance Orthogonality

## 定义
Nuisance Orthogonality 是一种衡量视觉表征泛化能力的几何指标：量化动作无关的视觉变化（风格、光照、背景等干扰因素）是否被投影到表征流形之外，而非嵌入在动作相关的子空间中。

## 数学形式
设 $\mathbf{z}$ 为模型的视觉表征，$\mathcal{M}$ 为训练分布的表征流形，$\delta \mathbf{z}_\text{nuisance}$ 为由干扰因素引起的表征偏移：
$$\text{NO} = 1 - \frac{\|\text{proj}_{\mathcal{M}} \delta \mathbf{z}_\text{nuisance}\|}{\|\delta \mathbf{z}_\text{nuisance}\|}$$

NO 越高，说明干扰变化越正交于动作子空间，泛化性越好。

## 核心要点
1. 高 NO 意味着 OOD 视觉输入（换背景、换光照）不会影响动作预测
2. 可在训练前通过 backbone 评估进行 WAM 预选型
3. 由 generative backbone 的预训练表征先验决定，不只是 WAM 训练后的结果
4. 是 WAM 零样本泛化能力的预测因子

## 代表工作
- [[ImageWAM]]/How Representational Priors: 提出 NO 指标并验证其与 WAM 泛化的相关性

## 相关概念
- [[World Action Model]]
- [[OOD Generalization]]
- [[Representation Learning]]
