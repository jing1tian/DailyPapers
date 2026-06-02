---
type: concept
aliases: [TeaCache]
---

# TeaCache Inference Acceleration

## 定义
视频扩散/世界模型推理加速方法，通过缓存和复用相邻时间步的 KV（Key-Value）特征来减少冗余计算。

## 核心要点
1. 1. 相邻帧 KV 高度相似可以复用
2. 2. 免训练，接在已有模型上直接用
3. 3. Light Interaction 在此基础上扩展

## 代表工作
- [[Light-Interaction]]

## 相关概念
- [[DiT]]
- [[World Model]]
