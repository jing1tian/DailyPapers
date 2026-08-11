---
type: concept
aliases: [FLOWER, Flow-based reward]
---

# FLOWER

## 定义
用于 VLA RL post-training 的基于 optical flow 的奖励函数，衡量预测动作与观测帧间视觉流的一致性，无需真实 reward 标注。

## 核心要点
1. 用预训练 optical flow 网络估计连续帧之间的像素运动场
2. 将 VLA 预测的 action 转换为预期的视觉流，与实际流做对比
3. 一致性高 → 正奖励；一致性低 → 负奖励
4. 可作为 dense reward 信号，解决稀疏奖励下 VLA RL 训练困难的问题

## 代表工作
- [[TEMPO]]: 在语义-动作解耦 RL post-training 中使用 FLOWER 作为主要奖励信号

## 相关概念
- [[VLA-RL]]
- [[ConRFT]]
- [[Diffusion Policy]]
