---
type: concept
---

# iVideoGPT

## 定义
用离散 token GPT 建模的交互式视频生成模型，将动作条件化的视频世界模型建模为自回归序列预测。

## 数学形式
$$p(v_{1:T}) = \prod_t p(v_t | v_{<t}, a_{<t})$$

## 核心要点
1. RoboImagine 的对比基线
2. GPT 式的视频世界模型先驱

## 代表工作
- 见关联笔记

## 相关概念
- [[IRASim]]
