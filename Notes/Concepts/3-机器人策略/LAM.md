---
type: concept
aliases: [Latent Action Model, 潜动作模型]
---

# LAM

## 定义
Latent Action Model，从无动作视频中学习结构化潜动作表征的模型，将视频帧之间的变化压缩到低维潜空间作为隐式动作。

## 核心要点
1. 输入为视频帧序列，输出为潜在动作 $z_t$ 和预测的下一帧
2. 潜动作捕捉了环境变化的本质，但不直接对应机器人 joint angles
3. 下游需要动作解码器将潜动作映射到真实控制命令
4. [[CLAW]] 用对抗性正则化改进了 LAM 的训练稳定性

## 代表工作
- [[CLAW]]: 基于 LAM 框架，引入对抗潜变量正则化

## 相关概念
- [[LAPO]]
- [[LAPA]]
- [[CLAW]]
- [[action-conditioned world model]]
