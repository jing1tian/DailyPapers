---
type: concept
aliases: [MCP, 多块预测, Multi-Token Prediction for Video]
---

# Multi-Chunk Prediction

## 定义

Multi-Chunk Prediction（MCP）是一种针对自回归视频/序列模型的训练增强框架，通过轻量辅助模块同时预测多个未来时序块（chunk），提供密集多尺度时序监督，解决标准 [[teacher forcing]] 的近视监督（myopic supervision）问题。

## 数学形式

时序块偏移目标：

$$
\mathbf{x}_0^{[k][i]} = \mathbf{x}_0[\min(i+k, F)]
$$

总训练损失（以 [[流匹配]] 为底层框架）：

$$
\mathcal{L} = \mathcal{L}_{\text{video}} + \mathcal{L}_{\text{action}} + \sum_{k=1}^{3} w_k \cdot \mathcal{L}_k^{\text{MCP}}
$$

其中 $w_1=0.5,\, w_2=0.2,\, w_3=0.1$。

## 核心要点

1. **时序块偏移**: 将训练目标向未来偏移 k 块，强制模型学习长程时序依赖，防止"外观捷径"
2. **独立噪声注入**: 各预测深度使用独立噪声，MCP 时间步偏移 $s_{\text{mcp}} > s_{\text{main}}$，保证辅助模块依赖主模型语义特征而非简单复制
3. **因果链**: 深度 k 的 MCP 模块以深度 k-1 的输出为输入，近未来预测辅助远未来预测
4. **多层特征融合**: 收集主干中间层 {4, 12, 20, 30} 的隐状态融合后输入 MCP，使时序监督深度穿透骨干
5. **推理模式**: Zero-overhead 模式（训练后丢弃 MCP）或 2× 并行加速模式（保留 Depth-1 MCP）

## 代表工作

- [[NextForcing]]: 提出 MCP 框架，应用于自回归 WAM，RoboTwin SOTA 94.1%，2.3× 训练加速

## 相关概念

- [[teacher forcing]]: MCP 所改进的标准训练范式
- [[自回归视频生成]]: MCP 的应用场景
- [[流匹配]]: MCP 的底层生成框架
- [[Timestep Shift]]: MCP 中控制噪声水平的关键技术
