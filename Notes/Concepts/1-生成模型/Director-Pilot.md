---
type: concept
aliases: [Director-Pilot Architecture, 导演-飞行员架构]
---

# Director-Pilot

## 定义
LingBot-World-Infinity 提出的双角色世界模型控制架构：Director 负责高层场景规划与意图表达，Pilot 负责低层帧生成与视觉连续性维持。

## 数学形式
$$z_t^{dir} = \text{Director}(c_t, h_{t-1})$$
$$x_t = \text{Pilot}(z_t^{dir}, x_{t-1}, a_t)$$

其中 $c_t$ 为控制信号，$h_{t-1}$ 为历史状态，$a_t$ 为动作输入。

## 核心要点
1. 将世界模型的"想去哪"（语义规划）与"怎么动"（视觉生成）解耦
2. Director 运行在低频率，Pilot 运行在高频率，减少计算冗余
3. 与 hierarchical planner 思想类似，但在视频生成 latent space 中实现

## 代表工作
- [[LingBot-World-Infinity]]: 首次提出，用于无界交互世界模型

## 相关概念
- [[World Model]]
- [[Hierarchical Planning]]
- [[LingBot-Video]]
