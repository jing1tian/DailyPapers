---
type: concept
aliases: [CMA-ES, Covariance Matrix Adaptation Evolution Strategy, 协方差矩阵自适应进化策略]
---

# CMA-ES

## 定义
**协方差矩阵自适应进化策略**（Covariance Matrix Adaptation Evolution Strategy）：一种无梯度黑箱优化算法，通过迭代更新多元高斯分布的均值与协方差矩阵来搜索最优解，特别适合连续参数、非凸、多模态优化问题。

## 数学形式

每代从当前高斯分布采样 $\lambda$ 个候选解：

$$
x_i^{(g+1)} \sim \mathcal{N}\!\left(m^{(g)},\, \sigma^{(g)2} C^{(g)}\right), \quad i = 1,\ldots,\lambda
$$

根据适应度排序的前 $\mu$ 个精英更新均值、步长 $\sigma$ 和协方差 $C$。

## 核心要点
1. **无梯度**: 无需目标函数可微，适用于物理仿真等黑箱场景
2. **自适应协方差**: 捕获参数间的相关性，搜索方向随优化过程自适应旋转和缩放
3. **中等维度高效**: 通常在 10~1000 维参数空间表现良好
4. **系统辨识常用**: 在机器人仿真校准中用于匹配 PD 增益、摩擦系数等物理参数

## 代表工作
- [[Mask2Real-WM]]: 用 CMA-ES 优化 [[IsaacLab]] 仿真中的 PD 增益与摩擦系数，使仿真轨迹匹配真实机器人

## 相关概念
- [[IsaacLab]]
- [[Sim-to-Real Gap]]
