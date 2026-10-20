---
type: concept
aliases: [Drop-and-Rescale压缩, 稀疏保留缩放, Top-K稀疏压缩]
---

# Drop-and-Rescale

## 定义

Drop-and-Rescale 是一种用于压缩神经网络权重增量（如任务向量）的稀疏化方法：对权重张量**按幅值保留 Top-K 位置**（其余置零），再**乘以全局重缩放系数 $\alpha$** 补偿截断引入的幅度损失，实现以极低存储代价保留最关键参数信息。

## 数学形式

对权重模块 $p$ 的任务向量 $\tau_m^{(p)}$：

$$
z_m^{(p)} = \alpha \cdot \tau_m^{(p)} \odot M_m^{(p)}
$$

其中 $M_m^{(p)}$ 为 Top-K 二值掩码：

$$
M_m^{(p)}[i] = \begin{cases} 1 & \text{if } |\tau_m^{(p)}[i]| \in \text{Top-}K\% \\ 0 & \text{otherwise} \end{cases}
$$

## 核心要点

1. **Top-K 保留**: 仅保留每个模块中幅值最大的 K% 参数，保留最"重要"的权重变化；K 通常取 10%，即压缩至原大小的 10%（加上 1-bit 掩码开销）。
2. **全局重缩放 $\alpha$**: 截断会导致整体幅值下降，乘以 $\alpha > 1$（通常 1.1）补偿此损失，类似于 Dropout 训练时的缩放操作。
3. **与 SVD 残差对比**: Drop-and-Rescale 无需矩阵分解，计算成本低；实验中在相同存储预算下匹配或优于 SVD 残差（[[NestedVLA]] Table 7）。
4. **与 1-bit 量化对比**: 1-bit 量化对权重增量的压缩效果极差（严重失效），Drop-and-Rescale 稳健得多。
5. **掩码存储**: 保留位置需额外存储 1-bit 掩码（$\frac{1}{32}$ 的 32-bit float 开销），对整体存储影响较小。

## 代表工作

- [[NestedVLA]]: 在 VLA 技能整合中用于压缩各任务的 VLM 权重残差，$K=10\%$，$\alpha=1.1$，实现约 10× VLM 残差压缩

## 相关概念

- [[任务向量]]: Drop-and-Rescale 的直接操作对象
- [[模型合并]]: 压缩后的任务向量可视为更节省存储的"技能表示"
- [[Hypernetwork|超网络]]: NestedVLA 中超网络重建的也是 Top-K 掩码内的稀疏目标
