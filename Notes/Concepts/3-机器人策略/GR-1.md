---
type: concept
aliases: [Goal-conditioned Robot-1, Unleashing Large-Scale Video Generative Pre-training for Visual Robot Manipulation]
---

# GR-1

## 定义
基于视频生成预训练的目标条件机器人操控策略，用 GPT 风格的自回归 Transformer 联合预测视频帧和动作，是 SOMA 等工作对比的主要 VLA 基线之一。

## 数学形式
$$p(a_t, o_{t+1} | o_{1:t}, g) = \prod_t p(a_t | o_{1:t}, g) \cdot p(o_{t+1} | o_{1:t}, a_t, g)$$

## 核心要点
1. 用大规模视频数据做预训练，学习视觉动态先验
2. 联合预测未来图像帧和动作，实现视觉-动作对齐
3. 条件输入包含视觉目标（goal image），支持目标条件控制
4. 在 LIBERO、FurnitureBench 等 benchmark 上竞争力强

## 代表工作
- [[GR-1]]：Wu et al. 2023，视频预训练 + 目标条件操控
- [[SOMA]]：在空间记忆 VLA 中以 GR-1 作为基线对比

## 相关概念
- [[VLA]]
- [[World Model]]
- [[目标条件策略]]
