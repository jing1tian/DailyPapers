---
type: concept
aliases: [gym-miniworld, MiniWorld-v0]
---

# MiniWorld

## 定义
基于 OpenGL 的轻量级 3D 导航/操作环境，提供第一人称视角观测，常用于验证世界模型和 model-based RL 方法的原型。

## 核心要点
1. 3D 网格世界，比 MiniGrid 更真实但仍高度简化
2. 支持自定义房间布局、物体放置和任务目标
3. 计算开销低，适合快速原型验证（toy-scale benchmark）
4. 常与 [[MiniGrid]] 一起用于神经符号 / MBRL 方法评估

## 代表工作
- NeSy-WM (2608.17959): 在 MiniWorld + MiniGrid 上验证零样本任务迁移

## 相关概念
- [[MiniGrid]]
- [[DreamerV3]]
