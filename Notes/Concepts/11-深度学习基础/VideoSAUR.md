---
type: concept
aliases: [Video Slot Attention with Unsupervised Representation]
---

# VideoSAUR

## 定义
VideoSAUR：面向视频的 object-centric 表示学习方法，将 slot attention 扩展到时序视频域，通过无监督方式学习视频中每个对象的时序 slot 表示，无需实例分割标注。

## 数学形式
Slot Attention 核心迭代：
$$A = \text{Softmax}\left(\frac{QK^T}{\sqrt{d}}\right), \quad \text{slots} \leftarrow \text{slots} + \text{GRU}(A \cdot V)$$

VideoSAUR 在此基础上引入跨帧时序一致性约束，确保同一对象的 slot 在时间上稳定。

## 核心要点
1. 基于 Slot Attention 框架，扩展到视频序列（逐帧处理 + 时序传播）
2. 利用光流或 DINO 特征作为弱监督信号，引导对象发现
3. 在 Object-Centric World Models（OCWM）评估中常作为基线 slot encoder
4. 与 [[SlotContrast]] 等增强方法组合使用可提升鲁棒性

## 代表工作
- VideoSAUR 原文（Zadaianchuk et al., 2023/2024）
- 用于 Better Slots, Better Worlds 中的 baseline 对比

## 相关概念
- [[SlotContrast]]: 基于对比学习的 slot 增强
- [[JEPA]]: 另一种预测性视觉表示框架
- [[DINO]]: 提供初始化特征
