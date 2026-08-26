---
type: concept
aliases: [modality dropout, 模态掩码, modality masking regularization]
---

# Modality Masking

## 定义
训练时随机掩盖（dropout）一部分模态输入（如某个视角的图像或语言 token），防止模型过度依赖单一模态，提高多模态融合的鲁棒性。

## 数学形式
$$\mathcal{L} = \mathbb{E}_{m \sim p_\text{mask}} \left[ \mathcal{L}_\text{policy}(f(x_{\bar{m}})) \right]$$

其中 $x_{\bar{m}}$ 为掩盖部分模态后的输入，$p_\text{mask}$ 为掩码概率分布。

## 核心要点
1. 类似 Dropout 的正则化效果，但在模态层面操作
2. 强迫模型从不完整输入中恢复完整行为，增强鲁棒性
3. 推理时使用完整多模态输入，无额外开销

## 代表工作
- [[BimanualMM]]: 双臂 VLA 中用模态掩码解决多视角和语言融合不稳定

## 相关概念
- [[Dropout]]
- [[Multi-View Fusion]]
- [[Vision-Language-Action Model]]
