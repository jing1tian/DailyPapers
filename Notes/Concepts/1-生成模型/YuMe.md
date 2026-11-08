---
type: concept
aliases: [YuMe]
---

# YuMe

## 定义
YuMe 是一种长视频自回归生成方法，作为 FrameMorrow 的对比 baseline，代表了基于历史帧条件生成的视频扩展范式。

## 数学形式
$$p(x_{t+1:T} \mid x_{1:t}) = \prod_{i=t+1}^{T} p(x_i \mid x_{1:i-1})$$

## 核心要点
1. 自回归生成：每一帧以已生成历史帧为条件续写视频
2. 属于长视频生成方向的代表工作

## 代表工作
- [[FrameMorrow]]: 以 YuMe 为对比 baseline，提出前瞻性帧选择改进历史帧利用效率

## 相关概念
- [[LongLive]]
- [[CausVid]]
- [[Self-Forcing]]
