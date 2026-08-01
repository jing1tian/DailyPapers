---
type: concept
aliases: [DINOv3, DINO v3]
---

# DINOv3

## 定义

DINOv3 是 Meta 提出的自监督视觉基础模型（self-supervised vision foundation model），通过大规模无标签图像数据上的自蒸馏训练得到通用视觉特征，是 DINO/DINOv2 系列的后续迭代，常作为机器人策略（VLA）的视觉编码器候选方案。

## 核心要点

1. **自监督预训练**: 不依赖图文配对（区别于 [[CLIP]]），仅通过自蒸馏（teacher-student）在纯图像数据上学习特征，具有更强的细粒度空间/几何表征能力
2. **与 CLIP 的对比**: 在机器人操作任务中，DINOv3 的局部空间特征通常优于 CLIP 的语言对齐全局特征，但缺乏天然的视觉-语言对齐能力
3. **作为视觉编码器**: 常与 Cross-Attention 等条件机制配合，将其特征注入扩散/VLA 动作头

## 代表工作

- [[ABC]]: 系统比较了 CLIP（AdaLN/Cross-Attention 两种条件方式）与 DINOv3（Cross-Attention 条件）三种 ABC-DiT 架构变体的下游性能，发现视觉编码器选择对策略性能有显著影响
- [[TurboVLA]]: 用 DINOv3（ViT-B/ViT-L）作为视觉骨干，结合 BERT 文本编码器通过双向跨注意力融合，以 0.2B 参数/32 Hz 在 LIBERO 上实现 97.7% 成功率

## 相关概念

- [[CLIP]]
- [[SigLIP]]
- [[Cross-Attention]]
- [[AdaLN]]
- [[Diffusion Transformer (DiT)|DiT]]
