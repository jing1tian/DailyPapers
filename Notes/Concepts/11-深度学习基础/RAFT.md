---
type: concept
aliases: [RAFT, Recurrent All-Pairs Field Transforms]
---

# RAFT

## 定义
基于循环网络和全局相关体积的光流估计方法，AHEAD 用其估计操作场景中运动物体的像素级速度场。

## 数学形式
$$\mathbf{f}^{t+1} = \text{GRU}(\mathbf{h}^t, \text{Corr}(\mathbf{f}^t))$$

## 核心要点
1. 构建全对相关矩阵（All-Pairs Correlation）捕获跨帧对应关系
2. GRU 迭代更新光流场，精度高于单次前向推理方法
3. AHEAD 用 RAFT 辅助动态物体运动预测

## 代表工作
- [[AHEAD]]: 用 RAFT 估计移动物体的光流辅助拦截时机预测

## 相关概念
- [[CoTracker3]]
- [[AHEAD]]
