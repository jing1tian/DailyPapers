---
type: concept
aliases: [跨模态强迫, Cross-Modal Forcing, Modality Forcing]
---

# Cross-Modality Forcing

## 定义

一种多模态模型训练技术：在训练时对部分输入模态进行 dropout，同时要求模型仍能从保留的模态中生成被丢弃模态的预测，迫使模型在共享表征空间中建立各模态之间的互相预测关系。

## 数学形式

给定多模态输入 $\{z^o, z^p, d\}$，通过输入掩码 $\mathbf{m}^{in} \in \{0,1\}^3$ 随机丢弃部分流；输出掩码 $\mathbf{m}^{out}$ 要求模型预测包含被丢弃流在内的所有流的未来状态：

$$
\mathcal{L}_{CMF} = \sum_{i \notin \text{observed}} \mathcal{L}^{FM}_i(i_{t+1})
$$

训练目标包含对**未观测**模态的流匹配重建损失。

## 核心要点

1. **与 Teacher Forcing 的类比**: 类似于 NLP 中的 Teacher Forcing，但作用于跨模态预测而非时序自回归；区别在于这里是空间模态而非时序步骤的"强迫"。
2. **表征互预测性**: 迫使模型学习一个表征空间，在该空间中外观、几何和语义特征可以互相预测，提升多模态表征的内聚性。
3. **泛化性**: 训练时经历多种模态缺失组合，使模型在推理时对传感器缺失具有鲁棒性。
4. **关键效果**: 相比不使用该技术，性能提升 21%（Flex-π 在 RoboTwin 上的消融）；相比 baseline 提升 47%（相对）。

## 代表工作

- [[Flex-Pi|Flex-π]]: 首次在多流 World Action Model 中提出并系统评估，应用于 RGB + Pointmap + DINO 三流预测

## 相关概念

- [[Visual Stream Dropout]]: 配合 Cross-Modality Forcing 使用的输入随机丢弃机制
- [[Flow Matching]]: Cross-Modality Forcing 损失通常基于流匹配目标
- [[Mixture-of-Transformers]]: 常用于承载多流表征的主干架构
- [[WAM]]: Cross-Modality Forcing 常用于 World Action Model 的多流训练
