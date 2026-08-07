---
type: concept
aliases: [CCR, 反事实协方差消减]
---

# Counterfactual Covariance Reduction

## 定义
CofactVLA 提出的特征级因果干预方法：通过计算反事实与事实分支特征协方差之差 $\Delta\Sigma$，利用广义特征值分解提取混淆变量子空间 $\mathcal{S}_C$，再将 Transformer 中间层特征在该子空间上的投影归零，实现根源层面的视觉偏差消除。

## 数学形式

$$
\Delta\Sigma = \Sigma_\text{cf} - \Sigma_f
$$

$$
F_\text{causal} = F - \beta (F U_\text{bias}) U_\text{bias}^\top
$$

其中 $U_\text{bias}$ 为 $\Delta\Sigma$ 正谱的主成分（混淆子空间基向量）。

## 核心要点
1. **核心假设（对比特征间隙）**: 混淆变量子空间 $\mathcal{S}_C$ 的特征值严格大于观测子空间 $\mathcal{S}_O$，保证可分离
2. **干预位置**: Transformer 第 15、16 层（实验验证最优）
3. **干预强度**: $\beta = 0.15$，过大则过度消除语义信息
4. **与 OPG 的互补**: OPG 在动作解码时干预，CCR 在特征提取时干预，双层协同

## 代表工作
- [[CofactVLA]]: 提出 CCR，消融实验验证其对 Long-horizon 任务的额外提升

## 相关概念
- [[Dual-path Deconfounding Graph]]
- [[广义特征值分解]]
- [[对比特征间隙假设]]
- [[Orthogonal Projection Guidance]]
