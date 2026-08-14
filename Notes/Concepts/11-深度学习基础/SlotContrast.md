---
type: concept
aliases: [Slot Contrast, slot-based contrastive learning]
---

# SlotContrast

## 定义
SlotContrast：基于 slot 的对比学习方法，通过对比损失改进 object-centric 表示的质量，使 slot encoder 学到更具区分性和鲁棒性的对象表示，用于增强 Object-Centric World Models（OCWM）在 OOD 场景下的泛化能力。

## 数学形式
对比损失基本形式：
$$\mathcal{L}_\text{contrast} = -\log \frac{\exp(\text{sim}(z_i, z_j)/\tau)}{\sum_{k \neq i} \exp(\text{sim}(z_i, z_k)/\tau)}$$

其中 $z_i$ 是 slot $i$ 的表示，正例对来自同一对象的不同视角/时间步。

## 核心要点
1. 标准 slot attention（如 [[SAVi]]）的 slot encoder 只优化重建损失，缺乏判别性
2. 引入对比损失后，slot 更能区分不同对象，不易崩塌到重建相同的槽
3. 与 [[DINO]] 预训练特征结合可进一步提升：用 DINO 特征初始化 slot attention 的 key/value
4. 在 [[PushT]] 等控制任务上，更好的 slot 表征直接改善规划性能

## 代表工作
- 用于 Better Slots, Better Worlds（OCWM 表征质量研究，2608.12078）

## 相关概念
- [[JEPA]]: 另一种预测性表示学习方法
- [[DINO]]: 自监督视觉表示，常用于初始化 slot features
- [[VideoSAUR]]: 另一种 object-centric 视频表示方法
