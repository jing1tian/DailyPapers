---
type: concept
aliases: [ICL, 上下文学习, in-context learning]
---

# In-Context Learning

## 定义

在推理时将少量示范样例（examples）直接拼接到输入上下文中，使模型在不更新参数的情况下从示例中"学习"并泛化到新任务的能力。

## 核心要点

1. **无梯度更新**: ICL 在推理时完成，不修改模型权重，与微调（fine-tuning）本质不同。
2. **示例格式**: 通常为 (输入, 输出) 对序列，最后接待预测的查询输入。
3. **涌现能力**: ICL 在大型语言模型（GPT-3 等）中涌现，规模越大效果越好。
4. **多模态扩展**: 除文本外，ICL 已扩展到图像、视频、机器人动作等模态，见 [[In-Context Imitation Learning]]。

## 数学形式

给定 $k$ 个示范样例 $(x_1, y_1), \ldots, (x_k, y_k)$ 和查询 $x_{k+1}$，模型计算：

$$
p_{\theta}(y_{k+1} \mid x_1, y_1, \ldots, x_k, y_k, x_{k+1})
$$

其中 $\theta$ 保持固定，模型通过注意力机制从示范样例中提取模式。

## 代表工作

- [[Zero-WAM]]: 以人类视频为 ICL prompt，实现机器人零样本操作任务泛化
- [[In-Context Imitation Learning]]: 将 ICL 扩展到机器人模仿学习
- [[In-Context Policy Adaptation]]: 策略层面的上下文适应

## 相关概念

- [[In-Context Imitation Learning]]
- [[In-Context Policy Adaptation]]
- [[Flow Matching]]
- [[In-Context Future Chunk Prediction]]
