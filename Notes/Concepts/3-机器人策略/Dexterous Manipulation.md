---
type: concept
aliases: [Dexterous Manipulation, 灵巧操作, 灵巧手操作, Dexterous Grasping]
---

# Dexterous Manipulation

## 定义
**灵巧操作**：利用多指灵巧手（通常 15~24 DoF）完成精细物体操控任务的机器人能力，涵盖拾取、旋转、重构、工具使用等复杂手内操作（in-hand manipulation）。

## 核心要点
1. **高维动作空间**: 典型系统 17~23 个关节，远超平行夹爪（1-2 DoF）
2. **接触密集**: 需要建模多指与物体的接触动力学，对仿真精度和感知要求高
3. **数据稀缺**: 与简单夹爪操作相比，高质量灵巧手演示数据难以大规模采集
4. **Sim-to-Real 挑战**: 仿真与真实的接触/摩擦差异使策略迁移困难

## 代表工作
- [[Mask2Real-WM]]: 通过掩码中间表示缩减 Sim-to-Real Gap，为灵巧手构建 23 维可控世界模型
- [[DexWM]]: 基于 3D 关键点和 DINOv2 特征空间的灵巧手世界模型

## 相关概念
- [[ORCA Hand]]
- [[Sim-to-Real Gap]]
- [[MimicGen]]
- [[Video Diffusion Model]]
