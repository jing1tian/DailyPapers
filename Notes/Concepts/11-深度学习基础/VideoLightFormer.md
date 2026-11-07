---
type: concept
---

# VideoLightFormer

## 定义
轻量级时序视频 Transformer，高效处理连续帧的时序关系，适合实时视频理解任务。

## 数学形式
$$h_t = \text{Attn}(q_t, K_{t-k:t}, V_{t-k:t})$$

## 核心要点
1. AffectiveVLA 中与 TinyViT 配合做实时人类意图识别
2. 强调低延迟实时推理

## 代表工作
- 见关联笔记

## 相关概念
- [[TinyViT]]
