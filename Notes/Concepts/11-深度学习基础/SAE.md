---
type: concept
aliases: [Sparse Autoencoder, Sparse Auto-Encoder]
---

# SAE（Sparse Autoencoder）

## 定义
稀疏自编码器，一种通过强制激活稀疏性来学习可解释特征表示的神经网络，近年被广泛用于 LLM/VLM 的 mechanistic interpretability 研究。

## 数学形式
$$\hat{x} = W_d \cdot \text{ReLU}(W_e x + b_e) + b_d, \quad \mathcal{L} = \|x - \hat{x}\|^2 + \lambda \|\text{hidden}\|_1$$

## 核心要点
1. 稀疏约束（L1 正则）迫使每个输入只激活少数特征，增强可解释性
2. 训练在冻结的大模型激活上进行，不影响原模型
3. 可结合因果干预（zero-out）验证特征的因果作用
4. 从 LLM interpretability 迁移到 VLA 的前沿应用（见 [[EventSAE]]）

## 代表工作
- [[EventSAE]]: 将 SAE 用于 VLA 策略的可解释性分析与因果干预
- [[COAST]]: 以 SAE Steering 作为对比基线，证明 Conceptor 乘法门控在 VLA 引导中优于稀疏特征干预

## 相关概念
- [[VLA]]
- [[JEPA]]
- [[信息瓶颈]]
