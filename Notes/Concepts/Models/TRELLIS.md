---
type: concept
aliases: [TRELLIS 3D, TRELLIS Generation]
---

# TRELLIS

## 定义
一种大规模 3D 生成模型，能够从文字或图片描述生成高质量的 3D 资产（mesh + texture），是当前 3D 内容生成的代表性方法之一。

## 核心要点
1. 基于扩散模型生成结构化 3D 表示（如 triplane 或 voxel）
2. 支持从单张图像或文字 prompt 生成完整 3D 资产
3. 以静态几何生成为主，不包含运动学/物理属性
4. [[PhysForge]] 将 TRELLIS 作为对比基线，展示其缺乏物理感知的局限

## 代表工作
- [[PhysForge]]: 与 TRELLIS 对比，显示在物理感知 3D 生成上的优势
- [[PhysX-Omni]]: 使用 TRELLIS 作为体素到高质量网格的解码器模块

## 相关概念
- [[3D Gaussian Splatting]]
- [[Diffusion Model]]
- [[PhysForge]]
