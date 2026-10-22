---
type: concept
aliases: [WiSE-FT, Weight-Space Ensembles, 权重空间融合]
---

# WiSE

## 定义
Weight-Space Ensembles（WiSE-FT）：通过在权重空间中对 zero-shot 预训练模型和 fine-tuned 模型进行线性插值，平衡分布内性能和分布外鲁棒性。

## 数学形式
$$\theta_{WiSE} = (1-\alpha)\theta_{pretrain} + \alpha\theta_{finetune}$$
其中 $\alpha \in [0,1]$ 控制向 fine-tuned 端的倾斜程度。

## 核心要点
1. 不需要额外训练，只是权重线性插值
2. 插值后模型通常比 fine-tuned 模型在 OOD 上更鲁棒，同时保留大部分 fine-tuned 性能
3. 在 VLA 视觉微调场景中用于缓解跨维度鲁棒性下降（VLA-Retain 引用）
4. 前提：模型权重在线性路径上功能平滑变化（loss landscape 相对平坦）

## 代表工作
- WiSE-FT: Wortsman et al., 2022 (CVPR)

## 相关概念
- [[LoRA]]
- [[OpenVLA-OFT]]
