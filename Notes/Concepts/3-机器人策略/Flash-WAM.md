---
type: concept
aliases: [Flash World Action Model, Flash-WAM Distillation]
---

# Flash-WAM

## 定义

Flash-WAM 是一种面向实时部署的 [[World Action Model]]，通过**模态感知一致性蒸馏**将联合视频-动作 WAM 的推理延迟从 8.1 秒压缩至 348 毫秒（约 23× 加速），同时保持高成功率。核心创新在于针对视频流和动作流分别设计不同的一致性函数参数化，解决了标准 LCM 在动作流上梯度消失的问题。

## 核心要点

1. **梯度消失诊断**: 证明标准 LCM 参数化在动作流（集中于低噪声区）存在二次消失梯度（$|b(\sigma)| \propto \sigma^2$），导致朴素联合蒸馏成功率从 91.25% 崩溃至 23.97%
2. **模态感知参数化**: 为动作流设计线性参数化（$f^a = \mathbf{x}_\sigma - \sigma v_\theta$），为视频流保留方差保持 LCM 参数化
3. **实时验证**: 在 RoboTwin 2.0 达到 85.54%（19× 加速）、LIBERO 达到 95.7%（13.7× 加速），并在真实 Unitree G1 人形机器人（60% 成功率）上部署

## 代表工作

- [[Flash-WAM]]: 本概念对应的论文（arXiv 2606.05254）

## 相关概念

- [[World Action Model]]
- [[Flow Matching]]
- [[LCM|一致性蒸馏（LCM）]]
- [[LingBot-VA]]
- [[AdaWAM]]
