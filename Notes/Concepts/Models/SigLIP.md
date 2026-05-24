---
type: concept
aliases: [SigLIP, Sigmoid Loss for Language Image Pre-Training, SigLIP 2]
---

# SigLIP

## 定义

Google 提出的视觉-语言预训练模型，用 Sigmoid 损失替代 CLIP 的 Softmax 对比损失，在视觉-语言对齐任务上取得更好效果，是多个 VLA 模型的视觉编码器基础。

## 核心要点

1. 使用 sigmoid 二元分类损失替代 CLIP 的 softmax N-way 分类损失
2. 每个图像-文本对独立计算，不依赖 batch 内负样本，可使用更大 batch
3. 性能优于同等参数量的 CLIP 模型，尤其在 zero-shot 分类任务上
4. [[SigLIP2]] 是改进版本，融合了 CLIP + 自监督目标

## 代表工作

- [[SigLIP2]]: 更新版本，增加了自监督预训练目标
- [[DyGRO-VLA]]: 以 SigLIP 作为视觉-语言对齐的视觉编码器

## 相关概念

- [[CLIP]]: 前身，使用 softmax 对比损失
- [[DINOv2]]: 常与 SigLIP 联合使用提取视觉特征
- [[VLA]]: SigLIP 是 VLA 模型的视觉骨干之一
