---
type: concept
aliases: [SNR偏移调度, Signal-to-Noise Ratio Shift, SNR-Shifted Schedule]
---

# SNR Shift（SNR 偏移噪声调度）

## 定义

通过偏移系数 $s$ 对均匀噪声调度进行重参数化，将训练质量集中到特定噪声水平范围，常用于[[流匹配]]中调整视频/动作不同模态的训练分布。

## 数学形式

$$
\sigma = \frac{s\tilde{\sigma}}{1 + (s-1)\tilde{\sigma}}, \quad \tilde{\sigma} \sim \mathcal{U}[0, 1]
$$

- 当 $s = 1$ 时退化为均匀调度 $\sigma = \tilde{\sigma}$
- 当 $s > 1$ 时，训练质量集中于高噪声区（适合视频）
- 当 $s = 1$（均匀）时，低噪声区有大量质量（适合动作）

## 核心要点

1. **视频流**: 使用大偏移（$s^v = 5.0$），使模型专注于去除结构级噪声，容忍细节冗余
2. **动作流**: 使用小偏移（$s^a = 1.0$，均匀），要求模型在全噪声范围（尤其低噪声区）保持精确
3. **不对称性**: $s^v > s^a$ 造成两流训练分布的根本差异，是 [[模态感知蒸馏]] 设计的物理动机

## 代表工作

- [[Flash-WAM]]: 分析 SNR Shift 不对称性对一致性蒸馏的影响，设计模态感知参数化解决由此产生的梯度消失问题

## 相关概念

- [[流匹配]]: SNR Shift 在流匹配框架中的应用
- [[模态感知蒸馏]]: 利用 SNR Shift 不对称性进行分模态蒸馏
- [[一致性蒸馏]]: SNR Shift 影响一致性蒸馏梯度的关键机制
- [[World Action Model]]: 使用非对称 SNR Shift 的多模态场景
