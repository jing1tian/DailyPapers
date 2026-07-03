---
type: concept
aliases: [进度增强VLA, Progress-Augmented VLA]
---

# Progress-Enhanced VLA

## 定义

在标准 VLA（Vision-Language-Action Model）的输出端附加连续进度预测头，使模型在预测动作的同时估计当前子任务的完成进度，实现无外部状态估计器的长时域任务执行。

## 核心要点

1. **联合预测**: 在 $n$ 维动作输出基础上增加 1 维标量进度信号，总维度 $n+1$
2. **连续优于离散**: 连续进度（$[0,1]$ 内线性插值）相比离散 one-hot 标签学习更稳定，可避免梯度稀疏问题
3. **训练时自动标注**: 进度标签由动作原语自动生成，无需额外人工标注
4. **推理时触发切换**: 进度预测超过阈值 $\tau_p$ 时自动切换语言指令至下一子任务

## 代表工作

- [[FurnitureVLA]]: 首次将进度增强机制应用于 VLA，在 IVAR 椅子 7 子任务 / 1550 步任务中达 80% 平均成功率

## 相关概念

- [[π0.5]]: FurnitureVLA 的骨干 VLA 模型
- [[Progress Estimation]]: 进度预测的通用概念
- [[子任务切换]]: 进度信号的实际应用
- [[Action Chunking]]: 联合输出中的动作部分
