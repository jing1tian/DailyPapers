---
type: concept
aliases: [低层Tokenizer损失, Stage I Loss]
---

# Tokenizer Loss

## 定义
HiMem-WAM Stage I 中低层动作 tokenizer 的训练目标，结合光流重建损失、可选动作对齐损失和 KL 散度正则化，驱动编码器学习运动先验感知的紧凑潜变量。

## 数学形式

$$
\mathcal{L}_l = \|\hat{\Phi}_t - \Phi_t\|_1 + \lambda_a \mathbb{I}^{\text{act}}_t \|\hat{a}_t - a_t\|^2_2 + \beta \, D_{\text{KL}}\!\left(q_\phi(z^l_t | c_t) \,\|\, \mathcal{N}(0, I)\right)
$$

## 核心要点
1. 光流重建是主要监督信号，保证潜变量捕获运动语义
2. 动作对齐为可选项，由指示变量控制（有动作标注时激活）
3. KL 散度正则化约束潜空间接近标准正态分布
4. 推理时无需光流，仅训练时作为监督

## 代表工作
- [[HiMem-WAM]]: Stage I 训练目标

## 相关概念
- [[Variational Autoencoder]]
- [[DPFlow]]
- [[Optical Flow]]
