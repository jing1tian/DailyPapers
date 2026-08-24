---
type: concept
aliases: [ReCamMaster]
---

# ReCamMaster

## 定义
一种 camera-controlled 视频生成/重定向模型，允许通过指定相机轨迹控制视频中的视角变化，是多视角一致视频生成领域的代表工作之一。

## 数学形式
条件视频生成：$p(V | c_{\text{cam}})$，其中 $c_{\text{cam}}$ 为指定的相机参数序列（外参/内参）。

## 核心要点
1. 以相机参数序列为条件，实现对已有视频的相机轨迹重编辑
2. 与 [[TrajectoryCrafter]]、[[CameraCtrl]] 共同构成 camera-controlled video diffusion 主要方法族
3. 在 4DAnyone 中作为对比 baseline，4DAnyone 在多视角一致性上超越了它

## 代表工作
- [[4DAnyone]]: 以 ReCamMaster 为对比 baseline 之一

## 相关概念
- [[TrajectoryCrafter]]: 同类 camera-controlled 视频生成方法
- [[CameraCtrl]]: 基于 Plücker 光线注入的相机控制方法
- [[DiT]]: 扩散 Transformer 基础架构
