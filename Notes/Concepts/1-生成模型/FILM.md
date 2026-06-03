---
type: concept
aliases: [Frame Interpolation for Large Motion, 帧插值]
---

# FILM

## 定义

FILM（Frame Interpolation for Large Motion）是一种基于特征金字塔和双向光流的视频帧插值算法，通过递归中点协议生成高质量中间帧，尤其擅长处理大幅运动场景。

## 数学形式

递归中点协议：给定首尾帧 $I_0, I_1$，先生成中点帧 $I_{0.5}$，再递归生成 $I_{0.25}, I_{0.75}$，直到所需时间分辨率。

$$
I_{t} = \mathcal{F}(I_0, I_1, t), \quad t \in [0, 1]
$$

## 核心要点

1. **特征金字塔**: 在多个分辨率尺度上提取特征，分别计算双向流
2. **递归中点插值**: 二分法逐步细化时间插值，稳定性优于直接多帧预测
3. **大运动鲁棒性**: 多尺度流估计使其对位移较大的运动保持稳定

## 代表工作

- [[SKIP]]: AC-FILM 基于 FILM 递归协议并加入动作条件调制

## 相关概念

- [[FiLM]]（注意：不同于同名 Feature-wise Linear Modulation）
- [[光流 (Optical Flow)]]
