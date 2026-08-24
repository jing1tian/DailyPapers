---
type: concept
aliases: [Target Context Routing, 目标上下文路由]
---

# TCR (Target Context Routing)

## 定义
4DAnyone 提出的方法，在去噪过程中轮换 target view 的分组方式，使不同时间步的 target view 组之间能够跨组共享 context，解决分组去噪导致的全局结构漂移问题。

## 数学形式
去噪时间步 $t$ 处，分组 $G_t$ 通过旋转策略选取：
$$G_t = \text{Rotate}(\{v_1, \ldots, v_M\}, t \mod K)$$

高噪声阶段（$t$ 大）：跨组共享 context，保全局一致性；低噪声阶段（$t$ 小）：固定分组，稳定局部细节。

## 核心要点
1. 分组去噪是 DiT 前向受限时的必要工程手段，但不同组之间信息断裂导致视角间不一致
2. TCR 通过轮换分组而非固定分组，让每个 target view 在不同步骤与不同邻居交换信息
3. switching-time 超参数控制从跨组共享切换到固定分组的时间点，需要消融调整

## 代表工作
- [[4DAnyone]]: RCP + TCR 联合设计，实现高多视角一致性的 4D 人体重建

## 相关概念
- [[RCP]]: Reference Context Packing，与 TCR 互补的另一个设计
- [[DiT]]: 基础模型架构
