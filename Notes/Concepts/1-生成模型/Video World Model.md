---
type: concept
aliases: [视频世界模型, 视频生成世界模型, video-world-model]
---

# Video World Model（视频世界模型）

## 定义

**视频世界模型**是以视频为主要输出模态的世界模型，通过学习物理世界的视觉动态规律，能够根据文本或图像条件生成符合物理规律的视频序列。通常基于大规模视频数据预训练，权重在下游使用时保持冻结。

## 数学形式

$$
v_{1:T} = G_\theta(c, z), \quad z \sim \mathcal{N}(0, I)
$$

其中 $c$ 为条件（文本/图像），$z$ 为潜变量，$G_\theta$ 为冻结的生成网络，$v_{1:T}$ 为生成视频帧序列。

## 核心要点

1. **预训练冻结**: 通常在大规模视频数据上预训练后权重冻结，通过推理时自适应（[[Inference-Time Adaptation]]）提升特定任务性能
2. **物理推理能力**: 高质量视频世界模型应能正确建模物体间的力学、碰撞、流体等物理交互
3. **推理时可控性**: 通过文本条件、噪声调度、采样策略等外部控制调整生成行为
4. **评估挑战**: 物理合理性难以用标准视频质量指标衡量，需专用基准如 [[Physics-IQ Verified|Physics-IQ]]

## 代表工作

- [[Wan2.2]]: Wanx 系列视频世界模型
- [[CogVideoX]]: 认知视频生成模型
- [[HunyuanVideo]]: 混元视频世界模型
- [[SCOPE]]: 提出对冻结视频世界模型进行推理时自适应的框架

## 相关概念

- [[Inference-Time Adaptation]]
- [[Score Isolation]]
- [[Physics-IQ Verified|Physics-IQ]]
- [[Video Diffusion Model]]
