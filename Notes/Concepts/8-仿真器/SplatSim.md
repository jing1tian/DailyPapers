---
type: concept
aliases: [SplatSim, Gaussian Splatting Simulator]
---

# SplatSim

## 定义
SplatSim 是一个基于 3D Gaussian Splatting 的仿真框架，将从真实场景重建的 3DGS 模型接入物理仿真器，实现逼真外观与物理交互的结合，用于减小 sim-to-real gap。

## 核心要点
1. 用真实场景的 3DGS 模型替代传统 CAD 渲染，大幅提升视觉真实度
2. 物理交互（碰撞、抓取）仍依赖传统物理引擎（如 [[MuJoCo]]）
3. 可以直接导入 [[URDF]] 描述的机器人模型
4. 特别适合关节物体仿真（配合 [[ScrewSplat]] 等方法重建）

## 代表工作
- [[RORA]]: 用 SplatSim 接入 3DGS 重建的关节物体，验证操控策略

## 相关概念
- [[3DGS]]
- [[URDF]]
- [[MuJoCo]]
- [[IsaacLab]]
