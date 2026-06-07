---
type: concept
aliases: [VFM, 视频基础模型, Video FM]
---

# Video Foundation Model

## 定义
在大规模视频数据上预训练的生成式大模型，能够根据文本/图像/动作等条件生成高质量视频序列，广泛用作下游任务的 prior 或数据生成引擎。

## 数学形式
$$p_\theta(x_{1:T} \mid c) = \prod_{t=1}^{T} p_\theta(x_t \mid x_{<t}, c)$$

其中 $x_{1:T}$ 为视频帧序列，$c$ 为条件输入（文本/图像/动作）。现代 VFM 通常用扩散模型或流匹配建模，而非显式自回归。

## 核心要点
1. 通常基于 [[DiT]] 或 U-Net 架构，在潜在空间（[[VAE]] 压缩后）建模视频扩散
2. 支持多种条件：T2V（文本），I2V（图像），action-conditioned（动作条件）
3. 作为 prior 用于数据生成时（如 [[GRAIL]]），可在已知 3D 配置下生成高质量交互视频，降低深度歧义
4. 代表性开源模型：[[HunyuanVideo]]、Wan2.1、CogVideoX 等

## 代表工作
- [[GRAIL]]: 在已知 3D 配置下调用 VFM 生成人机交互视频，作为 humanoid loco-manipulation 的数据生成引擎
- [[minWM]]: 将 VFM（HunyuanVideo、Wan2.1）蒸馏为实时交互式世界模型

## 相关概念
- [[扩散世界模型]]
- [[DiT]]
- [[条件视频生成]]
- [[Rectified Flow]]
