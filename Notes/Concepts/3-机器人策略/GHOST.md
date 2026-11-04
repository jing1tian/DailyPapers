---
type: concept
aliases: [GHOST, 对象中心隐场景重建]
---

# GHOST

## 定义
SCOUT 中的对象中心隐场景重建模块：从稀疏正面 RGB-D 视图融合可见槽位，并以学习型隐藏槽位重建遮挡区域，输出含三维中心、尺寸和存在概率的完整槽位集合。

## 数学形式

$$
z_t = \{s_i^t\}, \quad s_i^t = (\mu_i, \sigma_i, p_i)
$$

其中 $\mu_i \in \mathbb{R}^3$ 为三维中心，$\sigma_i \in \mathbb{R}^3$ 为三维尺寸，$p_i \in [0,1]$ 为存在概率。

## 核心要点
1. 可见槽位由多视角 [[SAM]] SAM3D 先验融合产生
2. 固定数量的**学习型隐藏槽位**代表遮挡对象（默认 16 个）
3. [[Slot Attention|Slot Transformer]]（4 层、128-dim）细化全部槽位
4. 高斯细化渲染占据体积，并导出关系规划所需图结构
5. 两阶段训练：先仿真，再在 GHOST 重建上微调；幻觉槽位监督为零运动

## 代表工作
- [[SCOUT]]: GHOST 作为 SCOUT 的核心场景重建组件

## 相关概念
- [[Slot Attention]]
- [[SAM]]
- [[SCOUT]]
- [[MCTS]]
