---
type: concept
aliases: [ArcFace Loss, Additive Angular Margin Loss]
---

# ArcFace

## 定义
一种基于加性角度边距（Additive Angular Margin）的人脸识别损失函数，通过在超球面上优化特征角度间隔，提升人脸嵌入的判别能力。

## 数学形式
$$\mathcal{L} = -\frac{1}{N}\sum_{i=1}^{N} \log \frac{e^{s\cos(\theta_{y_i}+m)}}{e^{s\cos(\theta_{y_i}+m)} + \sum_{j\neq y_i} e^{s\cos\theta_j}}$$

## 核心要点
1. 在角度空间引入固定边距 $m$，迫使同类特征更紧凑、类间特征更分散
2. 特征和权重均归一化到超球面，通过缩放因子 $s$ 控制梯度尺度
3. 输入分辨率通常为 112×112，低于此分辨率时需上采样，影响匹配准确性

## 代表工作
- [[WithEveryone]]: 用 ArcFace 计算身份相似度（identity similarity），评估生成图像与参考人脸的一致性
- [[4DAnyone]]: 用于 4DGS 人体重建中的身份验证指标

## 相关概念
- [[CLIP]]: 另一种对比学习嵌入方法，ArcFace 专门针对人脸
- [[SMPL]]: 人体参数化模型，与 ArcFace 常配合用于人体重建评估
