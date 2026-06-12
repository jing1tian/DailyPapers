---
type: concept
aliases: [Action-to-Control Projection, 动作视觉控制投影, Γ投影]
---

# Action-to-Control 投影

## 定义

将机器人关节空间动作序列通过正向运动学和相机投影，转化为头部相机视角的 2D 夹爪控制标记序列，用于条件化视频生成模型的视觉控制信号。

## 数学形式

$$
C_t = \Gamma(A_t^{\Delta,K}) = \{c_{t+\Delta}, c_{t+2\Delta}, \ldots, c_{t+K\Delta}\}
$$

其中 $c_{t+k\Delta}$ 为在图像平面叠加夹爪位置标记后的控制帧。

## 核心要点

1. 利用确定性**正向运动学**将关节角度映射为 3D 夹爪位置，无需端到端学习
2. 通过相机内参/外参投影矩阵将 3D 位置映射到 2D 图像平面
3. **标记编码**：圆点位置表示夹爪 xy 坐标，圆点大小表示夹爪开合状态（大=开，小=闭）
4. 主要作用于**头部相机**视角，在严重遮挡场景下效果有限

## 代表工作

- [[PiL-World]]: 提出该投影模块用于 VLA Policy-in-the-Loop 闭环 World Model 评估

## 相关概念

- [[正向运动学]]
- [[World Model]]
- [[Action Chunking]]
- [[潜在历史记忆]]
