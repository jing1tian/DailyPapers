---
type: concept
aliases: [MaxSemDist, Maximum Semantic Distance]
---

# MaxSemDist

## 定义
SKIP 提出的关键帧选择准则，在视频序列中选取语义变化最大的帧作为稀疏关键帧，减少 World Model 推理次数。

## 数学形式
$$K^* = \arg\max_{K} \sum_{i} \|f_\theta(F_{k_i}) - f_\theta(F_{k_{i-1}})\|_2$$

其中 $f_\theta$ 为视觉编码器，$F_{k_i}$ 为第 $i$ 个关键帧。

## 核心要点
1. 用预训练视觉编码器提取语义特征，计算相邻帧特征距离
2. 贪心或动态规划选取最大化总语义距离的关键帧子集
3. 物理接触瞬间、物体状态改变等通常语义距离大，自然成为关键帧

## 代表工作
- [[SKIP]]: MaxSemDist 是 KeyWorld 关键帧选择的核心组件

## 相关概念
- [[KeyWorld]]
- [[SKIP]]
