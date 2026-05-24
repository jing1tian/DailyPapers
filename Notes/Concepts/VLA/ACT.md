---
type: concept
aliases: [ACT, Action Chunking with Transformers]
---

# ACT (Action Chunking with Transformers)

## 定义
用 Transformer 对机器人操作动作进行"分块"预测（action chunking）的模仿学习方法，由 Zhao et al. (2023) 提出，专为双臂操作设计。

## 数学形式
$$
\hat{a}_{t:t+k} = \pi_\theta(o_t)
$$
每步预测 $k$ 个连续动作（chunk），通过时序集成（temporal ensembling）平滑执行。

## 核心要点
1. Action Chunking：预测一段动作序列而非单步，减少复合误差
2. CVAE 编码器：将演示动作编码为风格潜变量，注入生成网络
3. 时序集成：对重叠 chunk 的预测加权平均，提高平滑性
4. 专为精密双臂操作（ALOHA 平台）设计

## 代表工作
- [[ACT]]: "Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware"
- [[Diffusion Policy]]: 类似的 action chunking + 扩散生成策略

## 相关概念
- [[Diffusion Policy]]
- [[Action Chunking]]
- [[VLA]]
