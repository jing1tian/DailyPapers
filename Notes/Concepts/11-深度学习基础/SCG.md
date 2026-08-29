---
type: concept
aliases: [Successive Capacity Growth, 逐步容量增长, 自适应网络扩展]
---

# SCG

## 定义
Successive Capacity Growth：从最小网络开始，根据任务复杂度信号动态增加模型宽度（注意力头）或深度（Transformer 层）的自适应扩展方法，用于 JEPA 类世界模型训练。

## 数学形式
$$\text{添加 head 条件}: \text{attention-head entropy variance} > \tau_w$$
$$\text{添加 layer 条件}: \text{prediction loss plateau} > \tau_d$$

## 核心要点
1. 从极小模型（1头2层，283K参数）开始训练
2. 宽度增长（Width）：检测注意力头多样性不足时添加新头
3. 深度增长（Depth）：预测损失平台时添加新层
4. Function-Preserving 初始化：扩展时不破坏已学到的知识
5. 目标：简单任务不过度配置，复杂任务不欠配置

## 代表工作
- [[Successive Capacity Growth]] (2608.27367): 原始方法

## 相关概念
- [[JEPA]]
- [[LeWM]]
- [[ViT]]
