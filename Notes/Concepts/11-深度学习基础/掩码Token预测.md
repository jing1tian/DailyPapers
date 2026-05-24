---
type: concept
aliases: [Masked Token Prediction, Masked Token Generation, 掩码生成, MTG]
---

# 掩码 Token 预测

## 定义
一种生成式自回归方法，通过迭代地预测并填充被掩码（mask）的 token 位置来生成序列，常用于离散 token 空间中的运动/视频/图像生成。与 BERT 的双向掩码不同，生成时按置信度从高到低逐步揭示 token。

## 核心要点

1. **迭代细化**：每轮预测置信度最高的若干 token 并固定，多轮后完成全序列生成
2. **双向上下文**：与自回归 next-token 预测不同，可利用序列两端的已知信息（如起止帧）
3. **可并行性**：每步可并行填充多个 token，比逐 token 自回归更快
4. **条件生成**：常以起止姿态/帧为条件，生成满足约束的中间序列

## 数学形式

$$
p(\boldsymbol{z}) = \prod_{t} p(z_t | \boldsymbol{z}_{\text{unmasked}})
$$

其中每步固定置信度最高的 $\lfloor T \cdot r \rfloor$ 个 token，$r$ 为迭代比例。

## 代表工作

- [[MAGVIT]]：视频生成中用掩码 token 预测替代自回归
- [[SONIC]]：运动学规划器中用掩码 token 预测生成运动序列，以起止4帧为条件

## 相关概念

- [[FSQ]]：提供离散 token 的量化方法
- [[Autoregressive Policy]]：对比方法，逐 token 预测
