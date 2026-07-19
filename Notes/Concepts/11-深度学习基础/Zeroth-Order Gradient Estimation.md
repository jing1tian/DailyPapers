---
type: concept
aliases: [零阶梯度估计, 有限差分梯度估计, ZO-SGD, Zeroth-Order Optimization]
---

# Zeroth-Order Gradient Estimation（零阶梯度估计）

## 定义
零阶梯度估计是在只能查询函数值（无梯度访问）的黑盒设定下，通过有限差分法对目标函数梯度进行近似估计的方法，是黑盒对抗攻击和黑盒优化的核心工具。

## 数学形式

$$
\hat{\nabla}J(\delta) = \frac{1}{m} \sum_{i=1}^{m} \frac{J(\delta + c u_i) - J(\delta - c u_i)}{2c} \cdot u_i
$$

配合投影梯度上升更新：

$$
\delta \leftarrow \Pi_{[-\varepsilon, \varepsilon]}\bigl(\delta + \eta\, \hat{\nabla}J(\delta)\bigr)
$$

## 核心要点
1. **无梯度需求**：每步只需查询目标函数值 $J(\cdot)$，不需要反向传播或模型内部结构
2. **Rademacher 随机方向**：使用随机 $\pm 1$ 向量 $u_i$ 降低估计方差
3. **计算代价**：每次梯度估计需 $2m$ 次函数查询，$m$ 越大估计越准确但代价越高；实践中 $m=1$ 单向量估计在低维时已足够
4. **有限差分半径 $c$**：需要足够小以近似梯度，但不能太小导致数值不稳定

## 代表工作
- [[BadWAM]]: 以每步 8 次迭代（每步 2 次查询 = 17 次总查询）对冻结 WAM 进行黑盒攻击，约 16 次迭代后收益递减

## 相关概念
- [[PGD]]: 白盒投影梯度下降（需要真实梯度），零阶梯度估计的黑盒替代
- [[Imagination-Preserving Attack]]: 使用零阶梯度估计实现的保形 WAM 攻击
- [[World-Action Drift]]: 零阶优化的攻击目标漏洞
