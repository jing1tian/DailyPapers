---
type: concept
aliases: [OpenSora, Open Sora]
---

# Open-Sora

## 定义
一个完全开源的视频生成框架，旨在复现并超越 OpenAI Sora 的视频生成能力，提供完整的训练/推理代码、数据管线和预训练模型权重。

## 数学形式

基于 DiT 架构的扩散模型，时空注意力分解：
$$\epsilon_\theta(z_t, t, c) = \text{DiT}\left(\text{SpAttn}(z_t) + \text{TempAttn}(z_t), t, c\right)$$

其中 $z_t$ 为 VAE latent，$c$ 为文本条件。

## 核心要点
1. 完全开源，包含训练代码、数据集处理脚本和模型权重
2. 使用 Spatial-Temporal DiT 架构处理视频时空建模
3. 支持多分辨率、多时长的灵活视频生成
4. OSP-Next 在 Open-Sora 基础上扩展了稀疏注意力和量化

## 代表工作
- [[OSP-Next]]: 在 Open-Sora 基础上的工程优化扩展

## 相关概念
- [[DiT]]
- [[SANA-Video]]
- [[LongCat]]
