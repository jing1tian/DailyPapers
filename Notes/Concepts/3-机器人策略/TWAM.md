---
type: concept
aliases: ["Tactile World Action Model"]
---

# TWAM

## 定义
融合触觉传感的世界动作模型，在视频世界模型预测中显式包含触觉未来帧，用于接触丰富操作。

## 数学形式
$$\hat{o}_{t+1} = [\hat{I}_{t+1}, \hat{T}_{t+1}], \quad a_t = \pi(o_t, \hat{o}_{t+1})$$

## 核心要点
1. 比 VTAM 更早的触觉 WAM 工作，推理较慢（约3.4s）
2. VTAM 以此为对比基线

## 代表工作
- 见关联笔记

## 相关概念
- [[VTAM]]
- [[Tactile-WAM]]
- [[GelSight]]
