---
type: concept
aliases: [Real DPO, 真实视频 DPO, Real or Not Real DPO]
---

# RealDPO

## 定义
视频/图像生成的偏好优化方法，将真实视频帧作为 chosen 样本、模型生成结果作为 rejected 样本构建偏好对，利用真实数据的分布信号直接对齐生成质量，无需额外奖励模型。

## 核心要点
1. **真实 vs. 生成对比**：正样本来自真实数据集，负样本为模型当前生成结果，偏好信号来自"真实性"本身
2. **避免奖励 hacking**：相比基于奖励模型的 RLHF/GRPO，真实视频提供更直接的监督，减少奖励模型拟合偏差
3. **与 DiffusionNFT 互补**：RealDPO 关注真实数据偏好，DiffusionNFT 提供前向过程轻量优化框架；[[LingBot-Video]] 将两者结合为 RealNFT

## 代表工作
- RealDPO (Cheng et al., arXiv 2510.14955)
- [[LingBot-Video]]: 以 RealNFT = RealDPO 思路 + DiffusionNFT 框架实现真实视频偏好对比

## 相关概念
- [[DPO]]
- [[DiffusionNFT]]
- [[GRPO]]
- [[RLHF]]
