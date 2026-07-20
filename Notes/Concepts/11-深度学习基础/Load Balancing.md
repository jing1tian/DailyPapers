---
type: concept
aliases: [负载均衡, Expert Load Balancing, MoE Load Balancing]
---

# Load Balancing（MoE 负载均衡）

## 定义

在 [[Sparse MoE|Mixture-of-Experts]] 架构中，确保各专家接收近似均等 token 数量的技术，防止少数专家被过度激活（routing collapse）而其他专家闲置。

## 核心要点

1. **问题来源**: top-K 路由天然倾向于将 token 集中分配给少数热门专家，造成容量浪费和梯度不均
2. **辅助损失法**: 在训练目标中加入显式均衡惩罚项（如 GShard、Switch Transformer），但会引入与主损失的权衡
3. **无损失法（Auxiliary-Loss-Free）**: 通过动态偏置修正（[[DeepSeekMoE]]）在 top-K 选择阶段调整各专家的亲和度分数
4. **序列级 vs. 批级**: 批级统计可能掩盖单个视频/序列内部的不均衡，[[LingBot-Video]] 提出序列级辅助损失

## 数学形式

**在线偏置校正（LingBot-Video）**：

$$
b_j \leftarrow b_j - \eta \cdot \text{sign}(n_j - \bar{n})
$$

$$
b_j \leftarrow b_j - \frac{1}{N_r} \sum_{k=1}^{N_r} b_k
$$

**序列级辅助损失**：

$$
\mathcal{L}_{\text{seq}} = \frac{1}{S} \sum_{s=1}^{S} \sum_{j=1}^{N_r} f_j^{(s)} P_j^{(s)}
$$

## 代表工作

- [[LingBot-Video]]: 在线偏置校正 + 序列级辅助损失，解决视频 MoE 中的批统计掩盖问题
- [[DeepSeekMoE]]: 提出 auxiliary-loss-free 偏置修正策略

## 相关概念

- [[Mixture-of-Experts]]
- [[Sparse MoE]]
- [[Sigmoid Router]]
