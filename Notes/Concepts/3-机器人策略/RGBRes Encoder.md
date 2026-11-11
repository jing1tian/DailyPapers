---
type: concept
aliases: [RGBRes, 颜色属性残差编码器, Color Residual Encoder]
---

# RGBRes Encoder

## 定义

[[ThermoRes Encoder]] 机制在颜色属性上的泛化扩展：将 RGB 图像分解为红/绿/蓝三路颜色属性视图，通过[[Gated Residual Modulation|语言条件门控残差]]分别注入 RGB head-camera token，提升 VLA 模型在颜色条件任务上的性能。

## 数学形式

**颜色属性 token 生成**：

$$
Z_j = P_\text{col}(E_\text{col}(x_j^\text{col})), \quad j \in \{\text{red, green, blue}\}
$$

**三路门控残差注入**：

$$
\begin{aligned}
Z_\text{rgb}^\text{red} &= Z_\text{rgb}^\text{head} + \alpha' \cdot Z_\text{red} \\
Z_\text{rgb}^\text{green} &= Z_\text{rgb}^\text{head} + \beta' \cdot Z_\text{green} \\
Z_\text{rgb}^\text{blue} &= Z_\text{rgb}^\text{head} + \gamma \cdot Z_\text{blue}
\end{aligned}
$$

## 核心要点

1. **与 ThermoRes 结构对称**：将冷/热二分支替换为红/绿/蓝三分支，机制完全一致
2. **泛化验证**：证明[[Gated Residual Modulation]]不依赖热模态特殊性，可迁移到颜色属性
3. **仿真+真实双验证**：在 LIBERO 仿真和真实机器人颜色条件任务上均有提升
4. **性能提升**：四项颜色条件任务平均成功率从 0.572 提升到 0.775（+20.3pp）

## 代表工作

- [[Thermo-VLA]]: 提出并验证 RGBRes Encoder，作为对 ThermoRes 泛化性的证明

## 相关概念

- [[ThermoRes Encoder]]: 热属性版本，原始提出的残差注入方案
- [[Gated Residual Modulation]]: 共享的门控残差注入机制
- [[VLA（视觉-语言-动作模型）]]: 应用场景
