---
type: concept
aliases: [信息论损失掩码, Turn-Level Loss Masking, 逐轮损失掩码]
---

# Information-Theoretic Loss Masking

## 定义

基于信息论统计信号的逐轮损失掩码技术，通过计算动作-观测对之间的词级重叠度、新颖性、Jaccard 系数和长度比，自动识别并保留高信息量的训练轮次，过滤低价值的重复/样板性观测。

## 数学形式

$$
OL = \frac{|W_{act} \cap W_{obs}|}{|W_{act}|}, \quad Nov = \frac{|W_{obs} \setminus W_{act}|}{|W_{obs}|}
$$

$$
Jac = \frac{|W_{act} \cap W_{obs}|}{|W_{act} \cup W_{obs}|}, \quad R = \frac{|obs|}{|act|}
$$

## 核心要点

1. 七类轮次（Retrieval/Expansion/Action/Transform/Boilerplate/Echo/Other）根据信号值分配不同保留率（5%～100%）
2. Echo 类（Think(x)→{thought:x}）保留率仅 5%，Boilerplate 类 10%，避免低价值样本主导训练
3. 不依赖域特定规则，通用于文本域和 GUI 域的多轮智能体轨迹
4. 通过不同的 keep ratio 实现软性过滤而非硬删除

## 代表工作

- [[QwenAgentWorld]]: 首次在多域语言世界模型训练中提出并应用该技术

## 相关概念

- [[Language World Model]]: 应用场景
- [[GSPO]]: 配合使用的 RL 算法
