---
type: concept
aliases: [RoboPoint Dataset, RoboPoint预训练数据集]
---

# RoboPoint

## 定义

面向机器人操纵的大规模语言条件 2D 目标定位数据集，包含 120K 样本，用于在 VLM 上预训练空间定位能力，使模型在微调到具体机器人任务时保留强大的视觉-语言对齐能力。

## 数学形式

预训练任务：给定图像 $I$ 和语言描述 $l$，预测目标位置的热图 $\hat{H}$，监督信号为截断高斯：

$$
H^{\text{gt}}(\mathbf{x}) \propto \sum_{i} \exp\!\left(-\frac{\|\mathbf{x} - \hat{\mathbf{x}}_i\|_2^2}{2\sigma^2}\right) \cdot \mathbf{1}[p_i(\mathbf{x}) \geq p_{\min}]
$$

## 核心要点

1. **规模**：120K 语言-图像-位置三元组
2. **作用**：激活 VLM（如 PaliGemma）的空间定位能力，弥补其在精确坐标预测上的不足
3. **关键性**：消融实验显示跳过此预训练后 RLBench SR 从 90.5% 跌至 56.2%

## 代表工作

- [[BridgeVLA++]]: 使用 RoboPoint 作为预训练数据集，两阶段训练的第一阶段

## 相关概念

- [[热图动作解码]]
- [[PaliGemma]]
- [[RLBench]]
