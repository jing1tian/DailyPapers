---
type: concept
aliases: [ThermoRes, 热属性分解编码器, Thermal Residual Encoder]
---

# ThermoRes Encoder

## 定义

Thermo-VLA 提出的热感知编码模块，将热图像分解为冷/热双分支属性视图，通过[[Gated Residual Modulation|语言条件门控残差]]注入 RGB token，使 VLA 模型具备温度感知能力而不破坏原始 RGB 表示。

## 数学形式

**背景估计与属性分解**：

$$
\begin{aligned}
k &= \text{clip}(x_t - b + \mu,\ 0,\ I_\text{max}) \\
x_\text{cold} &= I_\text{max} \cdot \frac{\min(k, \mu)}{\mu} \\
x_\text{hot} &= I_\text{max} \cdot \frac{I_\text{max} - \max(k, \mu)}{I_\text{max} - \mu}
\end{aligned}
$$

**token 生成与残差注入**：

$$
Z_\text{cold} = P_r(E_t(x_\text{cold})), \quad Z_\text{hot} = P_r(E_t(x_\text{hot}))
$$

$$
Z_\text{rgb}^\text{cold} = Z_\text{rgb}^\text{head} + \alpha \cdot Z_\text{cold}, \quad
Z_\text{rgb}^\text{hot} = Z_\text{rgb}^\text{head} + \beta \cdot Z_\text{hot}
$$

## 核心要点

1. **背景参考估计**：从热图上边界 patch 均值众数估计背景亮度 $b$，消除环境背景的绝对温度影响
2. **双分支分解**：$x_\text{cold}$ 保留低于背景的冷区域，$x_\text{hot}$ 保留高于背景的热区域，各自归一化到全量程
3. **共享编码器**：冷/热两路共用同一 [[SigLIP]] 结构热编码器 $E_t$，参数高效
4. **语言门控残差**：权重 $[\alpha, \beta]$ 由语言指令条件动态计算（见[[Gated Residual Modulation]]），不破坏 RGB 恒等路径
5. **可泛化性**：同一残差注入框架可推广到颜色属性，见[[RGBRes Encoder]]

## 代表工作

- [[Thermo-VLA]]: 提出该模块，在 [[Unitree G1]] 双臂机器人上验证温度感知操控

## 相关概念

- [[Gated Residual Modulation]]: 门控权重计算模块
- [[RGBRes Encoder]]: 颜色属性残差扩展版本
- [[Thermal Camera]]: 提供热图像输入 $x_t$
- [[Residual Adapter]]: 结构同构的通用残差注入范式
- [[VLA（视觉-语言-动作模型）]]: 应用场景
