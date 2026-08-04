---
type: concept
aliases: [CTE, 通道时序编码, 通道时序编码模块]
---

# Channel-wise Temporal Encoding（通道时序编码）

## 定义

将稀疏历史帧的时序运动信息压缩为单帧 RGB 表示的编码模块：通过帧差检测运动区域，以衰减热度图表达运动持续性，并将近（Near）、中（Mid）、远（Far）三档时间范围的特征分别映射到 R、G、B 通道，形成单张可供 VLA 视觉编码器处理的"回顾特征帧"。

## 数学形式

**帧差运动图**:

$$
D(\cdot, i) = |I(\cdot, t-k_i) - I(\cdot, t-k_{i+1})|
$$

**二值运动掩码**（阈值 $\xi$ 过滤背景噪声）:

$$
\Psi(\cdot, i) = \begin{cases} 1, & \text{if } D(\cdot, i) > \xi \\ 0, & \text{otherwise} \end{cases}
$$

**衰减时序热度编码**:

$$
H(\cdot, i) = \begin{cases} \tau, & \text{if } \Psi(\cdot, i) = 1 \\ \max(0,\, H(\cdot, i+1) - \delta), & \text{otherwise} \end{cases}
$$

其中 $\tau$ 为最大强度，$\delta$ 为衰减率。

## 核心要点

1. **背景过滤**: 运动掩码 $\Psi$ 抑制静态背景噪声，保留有意义的物体运动轨迹
2. **持久性表达**: 衰减编码 $H$ 保留运动历史记忆（物体消失后仍有热度残留）
3. **三通道压缩**: Near/Mid/Far 三档映射到 RGB，以单帧代替多帧历史，不增加 VLA 序列长度
4. **即插即用**: 输出与当前帧拼接，不修改视觉编码器结构

## 代表工作

- [[FibVLA]]: 提出 CTE，与 [[Logarithmic Hindsight Sampling|对数回顾采样]] 和 [[Fibonacci Recurrent Inference]] 协同使用

## 相关概念

- [[Logarithmic Hindsight Sampling]]: 提供 CTE 输入的历史帧采样索引
- [[Fibonacci Recurrent Inference]]: 与 CTE 协同的推理效率优化
- [[Vision-Language-Action Model]]: CTE 面向 VLA 的时序增强模块
