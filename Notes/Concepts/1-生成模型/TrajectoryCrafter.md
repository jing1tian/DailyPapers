---
type: concept
aliases: [TrajectoryCrafter]
---

# TrajectoryCrafter

## 定义
一种基于轨迹控制的视频生成/重定向模型，通过给定 3D 相机轨迹来控制视频中场景的视角变换，是 camera-controlled video diffusion 领域的重要工作。

## 数学形式
轨迹条件视频生成：$p(V_{\text{novel}} | V_{\text{ref}}, \{\mathbf{T}_t\}_{t=1}^{T})$，其中 $\mathbf{T}_t$ 为各帧相机位姿。

## 核心要点
1. 将相机轨迹作为几何先验注入视频扩散模型，实现对新视角视频的合成
2. 在 4DAnyone 中作为对比 baseline
3. 与 [[ReCamMaster]]、[[CameraCtrl]] 共同构成主流 camera-controlled 视频生成方法

## 代表工作
- [[4DAnyone]]: 以 TrajectoryCrafter 为对比 baseline

## 相关概念
- [[ReCamMaster]]: 同类方法，相机重定向
- [[CameraCtrl]]: Plücker 光线注入的相机控制
- [[RCP]]: 4DAnyone 用来解决 TrajectoryCrafter 等方法跨视角不一致问题的新方案
