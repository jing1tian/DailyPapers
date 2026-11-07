---
type: concept
aliases: [Tube-Motion Encoding, Tube Motion, 管状运动编码, Tubelet Encoding]
---

# Tube-Motion Encoding

## 定义
一种用于视频/时序视觉特征的位置编码方式，在空间 patch 维度和时间维度同时编码，并携带局部时间运动先验，使模型能够感知帧间的运动信息，优于静态位置编码（Static Positional Encoding）。

## 核心要点
1. **时空联合编码**: 相比仅对时间维度和空间维度分别编码，Tube-Motion 在时空管道（tube）粒度上编码，捕捉像素块的时间轨迹信息
2. **运动先验注入**: 通过在编码中嵌入相邻帧间的运动信息，让注意力机制更容易关注动态变化区域
3. **与 V-JEPA 结合**: 在 [[V-JEPA]] 2.1 类模型中，视频帧被划分为时空 tubelet，Tube-Motion 编码附加到每个 tubelet 的特征上
4. **性能提升**: 在 La-JEPA 消融研究中，相比 Static Encoding，Tube-Motion 在 RoboTwin 2.0 上提升 +3.4pp（clean）/ +3.7pp（randomized）

## 数学形式
$$
h_i = f_{\text{patch}}(x_i) + \text{PE}_{\text{tube-motion}}(t_i, s_i, \Delta m_i)
$$

其中 $t_i$ 为时间索引，$s_i$ 为空间位置，$\Delta m_i$ 为局部运动量。

## 相关概念
- [[RoPE|旋转位置编码]]: 时序序列中另一种常用位置编码方式
- [[V-JEPA]]: Tube-Motion Encoding 的主要应用场景
- [[Vision Transformer]]: 基础架构，接受 tubelet 作为输入 token

## 代表工作
- [[La-JEPA]]: 在设计研究（Section 4, Block C）中验证 Tube-Motion 优于 Static 编码，用于 V-JEPA 2.1 视觉特征的时序位置编码
