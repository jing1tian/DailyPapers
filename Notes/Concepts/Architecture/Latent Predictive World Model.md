---
type: concept
aliases: [Latent Predictive WAM, 潜在预测世界模型, LPWM]
---

# Latent Predictive World Model

## 定义

**Latent Predictive World Model** 是 [[World Action Model]] 的一个变体：不重建未来完整视频帧（像素级预测），而是预测**紧凑的未来观测嵌入**（潜变量），作为动作生成的轻量级辅助目标，在保留任务进度感知的同时大幅降低计算成本。

## 数学形式

未来视觉潜变量预测损失：

$$
\mathcal{L}_{video} = \left\| h^v - y_{t+1:t+K}^v \right\|_2^2
$$

其中 $h^v$ 为模型预测的未来潜变量，$y_{t+1:t+K}^v$ 为目标编码器（如 [[Wan]] 编码器）对未来 $K$ 帧的编码结果。

## 核心要点

1. **轻量化**: 避免高分辨率视频解码器，预测成本远低于像素级重建
2. **任务感知**: 通过未来潜变量预测保留"世界将如何演变"的语义信息，帮助策略做长时域规划
3. **与动作生成耦合**: 视频预测与动作扩散共享同一 Transformer 主干，通过双查询机制（运动查询 + 视频查询）联合优化
4. **区别于视频重建 WAM**: Fast-WAM、DiT4DiT 等需要视频→动作反演，Latent Predictive WAM 直接生成动作潜变量，无反演瓶颈

## 代表工作

- [[Omega0]]: ω-0 的 Stage 2 采用此范式，使用 [[Wan]] 编码器作为冻结目标，联合预测未来潜变量与 SONIC 全身动作潜变量

## 相关概念

- [[World Action Model]]
- [[Action Diffusion]]
- [[V-JEPA 2.1]]
- [[Wan]]
- [[SONIC]]
