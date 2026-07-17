---
type: concept
aliases: [Tactile FiLM, TacFiLM]
---

# TacFiLM

## 定义
将触觉传感器信号通过 FiLM（Feature-wise Linear Modulation）注入 VLA 模型的轻量多模态融合方法。

## 数学形式
$$\text{FiLM}(x; \tau) = \gamma(\tau) \odot x + \beta(\tau)$$

其中 $\tau$ 为触觉特征，$\gamma, \beta$ 为由触觉网络生成的逐通道缩放和平移参数。

## 核心要点
1. 触觉特征（DIGIT 传感器）经 DINO 提取后作为 FiLM 条件
2. 支持三种注入位置：EarlyFiLM、MiddleFiLM、AllFiLM
3. 基于 [[OpenVLA-OFT]] 骨干，添加触觉分支
4. 适用于精确接触任务（圆插销、USB 插入、HDMI 插拔）

## 代表工作
- [[TacFiLM]]（2607.13960, McGill + NVIDIA）: 提出方法并系统对比注入位置

## 相关概念
- [[FiLM]]
- [[OpenVLA-OFT]]
- [[VLA]]
