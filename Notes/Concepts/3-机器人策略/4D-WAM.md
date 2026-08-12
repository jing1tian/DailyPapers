---
type: concept
aliases: [4D World Action Model]
---

# 4D-WAM

## 定义
一种 model-agnostic 训练策略，通过注入 3D 轨迹场的时空监督信号，给现有 World Action Model 加入四维（3D 空间 + 时间）感知能力，无需修改模型架构。

## 核心要点
1. **Motion Alignment via Token-wise Frame Differences**：用相邻帧 token 差分对齐运动信息，让模型学到显式运动语义
2. **Source-Conditioned Destination Constraint**：以起始帧为条件约束预测目标帧，减少长时序预测漂移
3. 基于 [[FastWAM]]-Joint，在 RoboTwin 2.0 和 LIBERO 上验证，有真实机器人实验

## 核心区别
相比 [[JEPA-WAM]]（从特征空间引入时空感知）和 [[GWM-VLA]]（从几何编码引入），4D-WAM 从训练监督信号侧入手，是 plug-in 策略而非架构变更。

## 代表工作
- [[4D-WAM]]: Infusing Spatiotemporal Awareness into World Action Models through Trajectory Fields (2026)

## 相关概念
- [[FastWAM]]
- [[WAM]]
- [[JEPA-WAM]]
