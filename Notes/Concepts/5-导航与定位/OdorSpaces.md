---
type: concept
aliases: [Visual-Olfactory Embodied Navigation]
---

# OdorSpaces

## 定义
视觉-嗅觉 embodied navigation 仿真平台和 benchmark：在 [[Habitat-Sim]] 中集成 physics-based 气体扩散仿真，训练 RL agent 通过视觉 + 嗅觉信号导航到气味源。

## 核心要点
1. 多感知融合：视觉 + 嗅觉（类 [[SoundSpaces]] 的多感知框架）
2. CFD (Computational Fluid Dynamics) 模拟气体扩散，物理真实
3. RL 训练 visual-olfactory navigation agent
4. 局限：气体传感器 sim2real gap 目前无解，实用性受限

## 相关概念
- [[Habitat-Sim]]
- [[SoundSpaces]]
- [[具身AI]]
