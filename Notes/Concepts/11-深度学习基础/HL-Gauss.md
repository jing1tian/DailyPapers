---
type: concept
aliases: [HL-Gauss Distribution, Histogram Loss Gaussian, 直方图高斯分布]
---

# HL-Gauss

## 定义

一种将标量值（如奖励回报）转换为固定 bin 数（如 101 个 bin）的**软高斯分布**表示方法，使得原本的回归问题转化为分类问题，通过交叉熵损失训练，同时保留连续值的分布感知。

## 数学形式

$$
\rho_{t,b} \propto \exp\!\left[-\frac{(z_b - \tilde{y}_t)^2}{2\sigma^2}\right]
$$

$$
\mathcal{L}_{\mathrm{value}} = -\mathbb{E}_{\mathcal{D}}\!\left[\sum_{b=1}^{101} \rho_{t,b} \log p_\theta(b \mid x_t, A_t)\right]
$$

期望值从分布中恢复：

$$
Q_\theta(x) = \sum_{b=1}^{101} z_b \cdot p_\theta(b \mid x)
$$

## 核心要点

1. **避免回归退化**：分类 + 软标签比直接 MSE 回归在价值学习中更稳定
2. **bin 中心 $z_b$**：均匀分布在值域范围内（通常需归一化后的回报范围）
3. **扩散宽度 $\sigma$**：控制软标签的平滑程度，过窄趋近 one-hot，过宽丢失精度
4. **常用 101 bins**：经验上足够表达连续回报，同时计算开销可控

## 代表工作

- [[LAW]]: 用于子任务局部价值头的 101-bin 软分布训练

## 相关概念

- [[Value Function]]
- [[子任务局部价值学习]]
- [[Action Chunking]]
