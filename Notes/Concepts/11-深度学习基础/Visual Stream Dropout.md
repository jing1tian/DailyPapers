---
type: concept
aliases: [流Dropout, Stream Dropout, 视觉流随机丢弃]
---

# Visual Stream Dropout

## 定义

一种多模态模型训练技术：在训练时以一定概率独立地随机丢弃各视觉输入流（如 RGB、Pointmap、DINO），使单一模型 checkpoint 在推理时能够灵活适配任意输入/输出流组合，实现计算-精度权衡的部署灵活性。

## 数学形式

设视觉流集合为 $\{z^o, z^p, d\}$，输入在场掩码：

$$
\mathbf{m}^{in} \in \{0, 1\}^3
$$

每个流独立以概率 $p = 0.5$ 丢弃，约束至少一个流被保留。输出注意力掩码 $\mathbf{m}^{out}$ 控制联合生成时哪些流参与推理。支持组合数：

$$
|\{\mathbf{m}^{in}, \mathbf{m}^{out}\}| = 2^3 \times 2^3 - \text{无效组合} = 56 \text{ 种有效配置}
$$

## 核心要点

1. **单 checkpoint 多配置**: 通过 Dropout 训练，同一模型在推理时可选择任意流组合，无需针对不同传感器配置重新训练。
2. **计算-精度权衡**: 例如 Flex-π 中动作专一模式（60ms）vs 全联合生成（193ms），可根据实时要求选择。
3. **与 [[Cross-Modality Forcing|跨模态强迫]] 协同**: 流 Dropout 提供输入多样性，跨模态强迫提供输出监督，共同强化鲁棒表征。
4. **传感器鲁棒性**: 训练时的随机丢弃使模型对推理时传感器缺失或噪声天然鲁棒。

## 代表工作

- [[Flex-Pi|Flex-π]]: 首次系统化引入 Visual Stream Dropout 实现 56 种部署配置，动作专一模式延迟 60ms 仍超越所有基线

## 相关概念

- [[Cross-Modality Forcing]]: 与流 Dropout 配合的跨模态预测训练策略
- [[Mixture-of-Transformers]]: 多流并行处理的主干架构基础
- [[WAM]]: Visual Stream Dropout 常用于 World Action Model 的灵活部署设计
