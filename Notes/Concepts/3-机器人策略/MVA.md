---
type: concept
aliases: [Masked Visual Actions, 掩码视觉动作]
---

# MVA

## 定义
一种将机器人动作编码为视觉空间掩码的方法，用 masked image patches 替代低维 action token，使机器人控制与视频模型的视觉预训练对齐。

## 数学形式
给定末端执行器轨迹 $\{e_t\}$，在图像 $I_t$ 中 mask 覆盖轨迹区域 $\mathcal{M}_t$：
$$\tilde{I}_t = I_t \odot (1 - \mathbf{1}_{\mathcal{M}_t})$$

视频模型需预测 masked region：
$$\mathcal{L} = \mathbb{E}[\|f_\theta(\tilde{I}_t, \text{context}) - I_t[\mathcal{M}_t]\|^2]$$

动作提取通过 inverse dynamics model $g_\phi$：
$$a_t = g_\phi(I_t, I_{t+1})$$

## 核心要点
1. 动作在视觉空间内表达，消除 action token 与 image token 的对齐鸿沟
2. 掩码区域由 [[AprilTag]] 或 [[CoTracker3]] 追踪末端执行器确定
3. 推理时用 inverse dynamics model 从预测图像还原动作
4. 在 [[DROID]] 和 [[RoboCasa]] 上验证，有 real world 测试

## 代表工作
- Alzayer et al. 2026: MVA 原始论文（Stanford+UMD+Harvard）

## 相关概念
- [[VLA]]（传统 action token 方法）
- [[DreamGen]]（基于 text conditioning 的 action，不同思路）
- [[CoTracker3]]（末端执行器追踪）
- [[DROID]]（测试数据集）
