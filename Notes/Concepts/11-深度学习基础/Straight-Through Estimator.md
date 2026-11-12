---
type: concept
aliases: [STE, 直通梯度估计器, Straight-Through Gradient]
---

# Straight-Through Estimator（直通梯度估计器）

## 定义
一种用于训练包含离散/不可微操作（如 argmax、硬 top-K 选择）的神经网络的梯度估计方法：前向传播使用离散硬操作，反向传播直接"穿透"该操作，将梯度传递到其输入。

## 数学形式

设离散操作为 $f_{\mathrm{hard}}(x)$，软近似为 $f_{\mathrm{soft}}(x)$，直通估计器构造：

$$
M^{\mathrm{ST}} = f_{\mathrm{hard}}(x) + f_{\mathrm{soft}}(x) - \mathrm{stopgrad}[f_{\mathrm{soft}}(x)]
$$

前向传播：$M^{\mathrm{ST}} \approx f_{\mathrm{hard}}(x)$（软硬相减抵消，只剩硬值）

反向传播：梯度通过 $f_{\mathrm{soft}}(x)$ 流动（硬值无梯度，软值有梯度，$\mathrm{stopgrad}$ 切断重复计数）

等价地，原始 STE（Bengio et al., 2013）直接令：

$$
\frac{\partial f_{\mathrm{hard}}}{\partial x} \approx \frac{\partial f_{\mathrm{soft}}}{\partial x} \quad \text{or} \quad \frac{\partial f_{\mathrm{hard}}}{\partial x} \approx 1
$$

## 核心要点
1. **解决离散不可微问题**：Gumbel-Softmax、VQ-VAE、稀疏 token 选择等场景的标准梯度估计方法
2. **硬软分离**：前向使用硬离散操作保证确定性输出，反向通过软近似传递有意义的梯度
3. **偏差-方差权衡**：梯度是有偏估计（忽略了离散函数本身的不连续性），但方差远低于 REINFORCE 等无偏估计

## 代表工作
- Bengio et al. (2013): "Estimating or Propagating Gradients Through Stochastic Neurons for Conditional Computation"
- [[ElasticWAM]]: 用于 WHERE 阶段可微分 top-K token 选择训练

## 相关概念
- [[Stop-Gradient]]: STE 的实现依赖 stopgrad 操作
- [[Refinement Demand Field]]: 使用 STE 训练 token 空间选择掩码
