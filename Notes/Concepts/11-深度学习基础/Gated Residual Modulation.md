---
type: concept
aliases: [门控残差调制, Language-Conditioned Gated Residual, 语言条件门控残差]
---

# Gated Residual Modulation

## 定义

以语言指令为条件，通过 MLP + softmax 动态计算属性 token 的注入权重，再将加权属性 token 以残差形式叠加到主流（RGB）表示上；主流保留完整恒等路径，属性信息以"可选加成"而非"强制替换"的方式注入。

## 数学形式

**门控权重计算**：

$$
[\alpha, \beta] = \text{softmax}\!\left(\text{MLP}\!\left(\text{LN}(u_t)\right)\right)
$$

$$
u_t = [\bar{Z}_\text{text};\ \bar{O}_c;\ \bar{O}_h;\ \bar{O}_c - \bar{Z}_\text{text};\ \bar{O}_h - \bar{Z}_\text{text}]
$$

**残差注入（通用形式）**：

$$
y = x + f(a, l)
$$

其中 $x$ 为原始主流表示，$a$ 为属性观测，$l$ 为语言条件，$f(\cdot)$ 为属性条件残差函数。

## 核心要点

1. **语言条件性**：权重 $[\alpha, \beta]$ 由语言 token 均值 $\bar{Z}_\text{text}$ 与属性 token 的差值共同决定，使不同任务指令激活不同属性分支
2. **softmax 归一化**：$\alpha + \beta = 1$，确保总注入量可控，避免表示空间过度扰动
3. **恒等路径保留**：残差形式 $y = x + f(a, l)$ 保证在无相关属性时（$f \approx 0$）原始表示不受影响
4. **低秩近似**：作者指出残差更新近似低秩，与 [[Residual Adapter]] 和 LoRA 的低秩分解思路一致
5. **可推广性**：通用公式适用于任何"主流表示 + 条件属性修正"的场景（热、颜色、深度等）

## 代表工作

- [[Thermo-VLA]]: 在 VLA 热感知任务中提出并验证该机制（[[ThermoRes Encoder]] + [[RGBRes Encoder]]）

## 相关概念

- [[ThermoRes Encoder]]: 在热属性注入中使用该机制
- [[RGBRes Encoder]]: 在颜色属性注入中使用该机制
- [[Gated Cross-Attention]]: 门控权重计算中使用的注意力变体
- [[Residual Adapter]]: 结构同构的通用残差注入范式
- [[Residual Stream]]: 变换器中残差流的基础概念
