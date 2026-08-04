---
type: concept
aliases: [Sketched Isotropic Gaussian Regularizer]
---

# SIGReg

## 定义
通过对 latent 在 $M$ 个随机方向上做一维投影并施加正态性检验，迫使 latent 边缘分布逼近各向同性标准高斯，从而避免 [[Representation Collapse]]。

## 数学形式
$$
\mathrm{SIGReg}(Z) = \frac{1}{M}\sum_{m=1}^M T\big(\langle u^{(m)}, Z\rangle\big),\quad u^{(m)}\sim \mathcal{U}(S^{d-1})
$$
其中 $T$ 为 [[Epps-Pulley Test]] 单变量正态性统计量。

## 核心要点
1. 由 [[Cramér-Wold 定理]]：所有 1D marginal 均高斯 $\Leftrightarrow$ 联合高斯。
2. 单一标量项替代 [[VICReg]] 的 variance + invariance + covariance 三项。
3. 计算只需若干随机投影 + 闭式检验，常用 $M=1024$。
4. 在 [[LeWM]] 中是仅有两项 loss 之一。

## 代表工作
- [[LeWM]]：首次将 SIGReg 用于端到端 JEPA 世界模型
- [[WCM]]：将 SIGReg 引入 VLA-RL critic 训练，防止世界预测目标导致的特征塌缩

## 相关概念
- [[Representation Collapse]]
- [[VICReg]]
- [[JEPA]]
- [[Cramér-Wold 定理]]
- [[Epps-Pulley Test]]
