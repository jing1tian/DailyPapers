---
type: concept
aliases: [Generative Video Latent, Generative Video Learning reward]
---

# GVL

## 定义
使用视频生成模型的潜在表示或生成质量分数作为机器人强化学习中的奖励信号，代替手工设计的 reward 函数。

## 核心要点
1. 用预训练视频生成模型（如 diffusion video model）评估当前轨迹与目标视频的相似度
2. 生成质量分数（FID、LPIPS、DreamSim 等）作为 dense reward
3. 无需人工标注 reward，但依赖视频生成模型的质量和对任务的理解
4. 风险：高质量生成的错误场景可能产生误导性高分奖励

## 代表工作
- [[ViVa]]: 用视频生成价值函数驱动机器人 RL

## 相关概念
- [[强化学习]]
- [[VLM Critic]]
- [[视频扩散模型]]
