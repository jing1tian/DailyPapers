---
type: concept
aliases: [视频预测, Video Prediction, Future Frame Prediction, 未来帧预测]
---

# Video Prediction

## 定义

视频预测（Video Prediction）是给定当前观测帧，生成/预测未来视频帧序列的任务，是世界模型的核心能力之一，提供物理因果推理的归纳偏置。

## 数学形式

给定历史帧 $o_{0:t}$ 和动作序列 $a_{0:t}$，预测未来帧分布：

$$
p(o_{t+1:t+H} \mid o_{0:t}, a_{0:t})
$$

## 核心要点

1. **因果关系建模**: 视频预测学习"动作→视觉后果"的因果映射
2. **世界知识**: 即使动作质量低，视频预测仍可提供真实物理结果作为监督
3. **策略辅助**: 联合视频预测与动作生成（WAM 框架）可提升策略泛化性
4. **评估指标**: FID、FVD、SSIM、LPIPS 等

## 代表工作

- [[LoopWAM]]: 联合视频预测和动作生成，用低质量轨迹的视频预测损失提供世界知识，即使对失败轨迹也计算视频损失

## 相关概念

- [[World Action Model]]
- [[Flow Matching]]
- [[Diffusion Transformer]]
