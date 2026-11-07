---
type: concept
aliases: [能量引导, 基于能量的引导, Energy-Based Guidance]
---

# Energy Guidance

## 定义

在扩散模型或 flow matching 采样过程中，通过在采样步注入可微能量函数 $\mathcal{E}(\hat{x}_0)$ 的梯度，使生成样本向低能量区域偏移，从而在不重训模型的情况下施加新的生成约束或目标。

## 数学形式

**目标倾斜分布**:

$$
p_\gamma(x_0 \mid c) \propto p(x_0 \mid c) \cdot e^{-\gamma \mathcal{E}(x_0)}
$$

**Jacobian-free 速度更新**（flow matching 下）:

$$
v' = v_\theta + \lambda(t) \cdot \nabla_{\hat{x}_0} \mathcal{E}(\hat{x}_0)
$$

其中 $\hat{x}_0$ 为单步干净估计，$\lambda(t)$ 为引导强度调度函数。

## 核心要点

1. **无需重训**: 能量函数在采样阶段注入，冻结模型权重不变
2. **可微性要求**: 能量函数 $\mathcal{E}$ 必须对生成样本可微，以便计算梯度
3. **与 Classifier Guidance 的区别**: Classifier Guidance 需要在加噪数据上训练分类器，Energy Guidance 直接在干净估计上定义能量
4. **稳定性挑战**: 规范调度 $\lambda^*(t) = \gamma \cdot t/(1-t)$ 在高噪声时发散，需要额外稳定化

## 相关变体

- **Classifier Guidance**: 使用 $p(c|x_t)$ 的梯度引导采样
- **[[Classifier-Free Guidance]]**: 使用条件/无条件模型差值引导
- **Score Distillation Sampling (SDS)**: 在 3D 生成中使用扩散分数引导

## 代表工作

- [[GuidedWAM]]: 在自动驾驶 flow matching WAM 上使用能量引导，提出 Jacobian-free 更新 + 三重稳定化机制

## 相关概念

- [[Flow-Matching]]
- [[Classifier-Free Guidance]]
- [[Diffusion Policy]]
