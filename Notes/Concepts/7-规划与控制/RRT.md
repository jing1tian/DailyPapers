---
type: concept
aliases: [Rapidly-exploring Random Tree, 快速探索随机树]
---

# RRT

## 定义
RRT（Rapidly-exploring Random Tree）是基于随机采样的运动规划算法，通过在状态空间随机生长树来高效探索高维配置空间，找到无碰撞路径。

## 数学形式
$$
x_\text{new} = \text{Steer}(x_\text{near}, x_\text{rand}, \Delta), \quad \text{if } \text{CollisionFree}(x_\text{near}, x_\text{new})
$$

随机采样 $x_\text{rand}$，找最近邻 $x_\text{near}$，向随机点延伸步长 $\Delta$。

## 核心要点
1. **概率完备**：采样无限多时必然找到路径（若存在）
2. **高维适用**：计算复杂度随维度线性增长，而非指数增长
3. **RRT***：添加 rewiring 步骤保证渐进最优，路径代价收敛到最优
4. **RRT-Connect**：双向生长，速度更快
5. **局限**：路径不光滑，需后处理；对狭窄通道的效率较低

## 代表工作
- [[MDOC]] 等多机器人规划文章的对比基线

## 相关概念
- [[CBS]]
- [[CEM]]
- [[MPC]]
