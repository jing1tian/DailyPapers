---
type: concept
aliases: [GAF, 高斯动作场, 4D Gaussian Representation]
---

# Gaussian Action Field

## 定义
将 [[3D Gaussian Splatting]] 扩展为含可学习运动属性的 4D 场景表示，每个高斯原语附加位移向量 $\Delta\mu$，从单一前向传播中同时输出当前场景重建、未来状态预测和初始动作估计。

## 数学形式

$$
\mathcal{F}_\Theta: \{g(x), t\} \mapsto \{\mu, \Delta\mu, f\}
$$

其中 $\mu \in \mathbb{R}^3$ 为当前高斯位置，$\Delta\mu \in \mathbb{R}^3$ 为运动位移，$f = \{c, \sigma, r, s\}$ 为外观参数。

## 核心要点

1. 在标准 3DGS 基础上增加 Motion Prediction Head，预测每点位移 $\Delta\mu$
2. 当前高斯 $\{\mu, f\}$ 用于重建当前帧，未来高斯 $\{\mu + \Delta\mu, f\}$ 预测 $t+\Delta t$ 时刻场景
3. 通过 [[Iterative Closest Point|ICP]] 点云配准，从运动场直接估计末端执行器的刚体变换动作
4. 属于 V-4D-A（Vision-to-4D-to-Action）范式，相比 V-3D-A 增加了时序动态建模

## 代表工作

- [[GAF]]: 提出 GAF 表示并用于机器人操作，RLBench 60.4% 成功率，PSNR +11.54 dB

## 相关概念

- [[3D Gaussian Splatting]]
- [[Iterative Closest Point]]
- [[Alpha-Blending 渲染]]
- [[ManiGaussian]]
