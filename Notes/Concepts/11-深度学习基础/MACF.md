---
type: concept
aliases: [Motion-Aware Cross-attention Feature Forecasting]
---

# MACF (Motion-Aware Cross-attention Feature Forecasting)

## 定义
DF³ 提出的无解码器特征预测机制：以 motion query 在 DINO/ViT 特征空间上做跨帧注意力，直接预测下一帧特征，无需 pixel decoder。

## 核心要点
1. 预测对象为 ViT/DINO 特征向量，而非 pixel 或 latent code → 跳过 decoder 的计算瓶颈
2. Motion query 显式编码运动趋势，使预测关注动态区域
3. 基于 EoMT（decoder-free query architecture）实现高效跨帧注意力
4. 下游策略（如 ViPlanner）直接消费预测特征，无需额外解码步骤

## 数学形式
$$\hat{f}_{t+1} = \text{CrossAttn}(q_{\text{motion}}, f_t, f_{t-k:t})$$

## 代表工作
- [[DF3]]: 提出 MACF，在腿足 + 轮式机器人导航上验证

## 相关概念
- [[DINO]]
- [[ViPlanner]]
