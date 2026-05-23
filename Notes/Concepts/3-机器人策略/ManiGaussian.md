---
type: concept
aliases: [Manipulation with Gaussian Splatting]
---

# ManiGaussian

## 定义
将 3D Gaussian Splatting 用于机器人操控任务，通过构建和更新操控场景的 Gaussian 表示，为策略提供丰富的 3D 空间信息，支持遮挡下的操控推理。

## 核心要点
1. 用 3DGS 建立场景的 3D 表示，跟踪物体状态变化
2. Gaussian 表示随操控动作实时更新，提供遮挡后的场景预测
3. 作为 EvoScene-VLA 等工作的对比基线

## 代表工作
- [[ManiGaussian]]：3DGS-based 操控策略对比基线
- [[EvoScene-VLA]]：以 ManiGaussian 为对比，提出场景信念解码器方案

## 相关概念
- [[3D Gaussian Splatting]]
- [[VLA]]
- [[Diffusion Policy]]
