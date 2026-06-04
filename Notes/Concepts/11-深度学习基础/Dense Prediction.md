---
type: concept
aliases: [密集预测, 像素级预测]
---

# Dense Prediction（密集预测）

## 定义

密集预测是指对输入图像/视频中的每个像素（或空间位置）输出预测结果的任务，包括语义分割、深度估计、光流估计、表面法线估计等。

## 核心要点

1. 与图像级预测（分类）相对，密集预测需要保留空间分辨率信息
2. 关键挑战：下采样提取特征 vs. 上采样恢复分辨率（编码器-解码器结构）
3. Transformer 时代的代表架构：DPT（Dense Prediction Transformer）
4. 在机器人领域用于场景理解：深度图、语义图作为辅助监督或观测输入

## 代表工作

- [[GeoSem-WAM]]: 将几何（深度）和语义密集预测作为训练辅助监督
- [[Dense Prediction Transformer]]: 密集预测任务的 Transformer 基础架构

## 相关概念

- [[Dense Prediction Transformer]]
- [[Semantic Segmentation]]
