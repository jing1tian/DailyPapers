---
type: concept
aliases: [力特征编码, 接触特征编码, Contact Feature Encoding]
---

# Force Feature Encoding

## 定义
将历史力/力矩传感器信号通过轻量级 MLP 编码为紧凑的接触特征向量，用于调制机器人策略的动作生成过程。

## 数学形式

$$
\mathbf{c}_t = \text{Enc}_f(\mathbf{w}_{t-H_f+1:t})
$$

其中 $\mathbf{w}_t \in \mathbb{R}^6$ 为 6 轴力/力矩测量值，$H_f$ 为历史窗口长度。

## 核心要点

1. 使用轻量级 MLP 对时序力信号进行非线性压缩，提取接触状态的紧凑表示
2. 编码后的特征可用于 AdaLN 调制、残差修正等下游模块
3. 通常与二阶低通滤波器配合使用，去除高频噪声后再编码

## 代表工作

- [[FAWAM]]: 将历史 $H_f$ 步 6 轴力信号编码后用于 AdaLN 调制动作生成

## 相关概念

- [[AdaLN]]
- [[ForceVLA]]
- [[Wrench Tracking Error]]
