---
type: concept
aliases: [Vision Foundation Model, 视觉基础模型]
---

# VFM (Vision Foundation Model)

## 定义
在大规模视觉数据上预训练、具备通用视觉理解能力的基础模型，可用于 zero-shot 或 few-shot 迁移到下游视觉任务。

## 核心要点
1. 典型代表：CLIP、DINOv2、SAM、SigLIP2 等
2. 用于视频生成时作为先验：提供人体姿态、深度、光流等结构信息
3. GRAIL 中将 VFM 用于条件化视频生成（robot-centric HOI generation）
4. 与 VLM 区别：VFM 专注纯视觉特征，VLM 则联合语言-视觉

## 代表工作
- [[GRAIL]]: 用 VFM 提供视频生成先验，合成 HOI 训练数据
- [[CLIP]]: 文本-图像对比学习
- [[DINOv2]]: 自监督视觉特征

## 相关概念
- [[CLIP]]
- [[VLM]]
- [[DINO]]
- [[SigLIP2]]
