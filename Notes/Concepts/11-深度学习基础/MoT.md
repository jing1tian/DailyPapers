---
type: concept
aliases: [Mixture of Transformers]
---

# MoT

## 定义
Mixture of Transformers，在单一模型内混合多个专门化 Transformer 分支，不同类型的 token（文本、图像、视频等）由不同分支处理。

## 核心要点
1. 与 [[MoE]]（Mixture of Experts）类似，但在 Transformer 块级别而非 FFN 层级别进行专门化
2. 不同模态的 token 路由到对应的 Transformer 分支，避免不同模态互相干扰
3. 参数利用率高于单一 unified Transformer，推理时激活参数量受控
4. 被 [[SenseNova-U1]] 用于统一多模态理解和生成任务

## 数学形式
$$\text{output}_i = \text{Transformer}_{m(i)}(\text{token}_i)$$

其中 $m(i)$ 是 token $i$ 对应的模态分支。

## 代表工作
- [[SenseNova-U1]]: 使用 MoT 统一图像理解和生成
- [[SimWAM]]: 使用 MoT 共享视频专家与动作专家的注意力，配合 [[Isolated Attention Mask]] 实现训练-推理解耦

## 相关概念
- [[MoE]]: Mixture of Experts，相关的稀疏激活思路
- [[VBVR]]: SenseNova-U1 的另一核心组件
- [[Transformer]]: MoT 的基础架构
