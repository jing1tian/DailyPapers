---
type: concept
aliases: [潜在预见, Visual Foresight, Latent Visual Foresight]
---

# Latent Foresight（潜在预见）

## 定义
在压缩的潜在空间（而非像素空间）中预测未来视觉状态，用于为机器人策略提供前瞻性规划能力，同时避免像素级世界模型的高计算开销。

## 数学形式
$$\hat{z}_{t+k} = f_\theta(z_t, a_t, \ldots, a_{t+k-1})$$

其中 $z_t$ 为当前潜在状态，$f_\theta$ 为动作条件的潜在预测网络，$\hat{z}_{t+k}$ 为预测的未来潜在状态。

## 核心要点
1. 潜在空间预测比 pixel-level 预测计算开销低 1-2 个量级
2. 预见目标作为辅助训练信号，为动作预测提供物理先验
3. 可利用预训练视频模型的动力学知识，无需从头学习物理规律

## 代表工作
- [[InternVLA-A1.5]]：MoT 架构中 foresight expert 在潜在空间做未来帧预测，与动作 expert 并联
- [[DREAMSTEER]]：用潜在 WM 预测未来状态，引导 VLA 动作选择

## 相关概念
- [[World Model]]
- [[VLAW]]
- [[Mixture-of-Transformers]]
