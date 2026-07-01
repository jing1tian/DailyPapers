---
type: concept
aliases: [EDM, Elucidated Diffusion Model, 阐明扩散模型]
---

# EDM (Elucidated Diffusion Model)

## 定义

Karras et al. (2022) 提出的统一扩散模型框架，通过对噪声调度、预训练目标和采样过程进行数学阐明，在统一框架下分析和改进各类扩散模型。

## 数学形式

EDM 预训练目标：

$$
\mathcal{L}_{\text{EDM}}(\sigma) = \mathbb{E}_{\mathbf{z},\mathbf{n}}\left[\left\|D_\theta(\mathbf{z}+\mathbf{n};\sigma) - \mathbf{z}\right\|_2^2\right]
$$

噪声权重函数 $\mathbf{w}(\sigma)$：对不同噪声级别加权，高噪声和低噪声段权重不同。

去噪网络 $D_\theta$ 的预条件化：

$$
D_\theta(\mathbf{x};\sigma) = c_{\text{skip}}(\sigma)\mathbf{x} + c_{\text{out}}(\sigma) F_\theta(c_{\text{in}}(\sigma)\mathbf{x}; c_{\text{noise}}(\sigma))
$$

## 核心要点

1. **统一视角**: 将 DDPM、NCSN、Score SDE 等方法统一在同一框架下分析
2. **数值稳定训练**: 通过预条件化 ($c_{\text{skip}}, c_{\text{out}}, c_{\text{in}}$) 保证训练数值稳定
3. **采样改进**: 提出二阶 Heun 采样器，在少步数下质量更优
4. **噪声范围**: 明确定义 $t_{\min}$ 和 $t_{\max}$（如 A2World：0.01 和 200.0）

## 代表工作

- Karras et al. (2022) "Elucidating the Design Space of Diffusion-Based Generative Models"
- [[Cosmos]]: 大规模机器人世界模型采用 EDM 框架
- [[A2World]]: 动作条件世界模型预训练采用 EDM 目标

## 相关概念

- [[Diffusion Model]]
- [[Diffusion Transformer (DiT)|DiT]]
- [[Classifier-Free Guidance (CFG)|CFG]]
