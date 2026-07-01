---
type: concept
aliases: [联合扩散, 视频-动作联合扩散, Joint Video-Action Diffusion]
---

# Joint Diffusion

## 定义

在单一扩散模型框架内，对多个异质模态（如视频帧和动作序列）同时进行噪声预测和去噪，通过共享骨干网络实现模态间的隐式对齐与互补。

## 数学形式

联合训练目标（以 A2World-policy 为例）：

$$
\mathcal{L}(\sigma_v, \sigma_a) = \mathbb{E}\left[\mathbf{w}(\sigma_v)\|\hat{\mathbf{z}}^v - \mathbf{z}^v\|_2^2 + \lambda_a \mathbf{w}(\sigma_a)\|\hat{\mathbf{z}}^a - \mathbf{z}^a\|_2^2\right]
$$

各模态独立噪声调度：$\sigma_v = \alpha_v \sigma_{\text{base}}$，$\sigma_a = \alpha_a \sigma_{\text{base}}$

## 核心要点

1. **模态共享 + 独立处理**: 通常采用共享 Self-Attention，但各模态有独立的归一化（AdaLN）和 MLP（即 MoE-like 块）
2. **独立噪声调度**: 不同模态的噪声级别可独立设置，允许对视频和动作施加不同的噪声强度
3. **正向耦合**: 视频生成质量与动作预测质量在训练中正相关，二者互相促进
4. **Per-modality CFG**: 推理时对每个模态分别应用不同强度的引导

## 代表工作

- [[A2World]]: 通过 MoE 式视频-动作联合扩散块实现策略与世界模型的统一建模

## 相关概念

- [[Diffusion Model]]
- [[Classifier-Free Guidance (CFG)|CFG]]
- [[AdaLN]]
- [[Action Conditioning]]
