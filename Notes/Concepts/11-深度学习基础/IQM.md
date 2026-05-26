---
type: concept
aliases: [Interquartile Mean, 四分位均值]
---

# IQM

## 定义
Interquartile Mean（四分位均值），一种评估 RL 算法性能的统计指标，取所有运行结果的中间 50%（去掉最低和最高各 25%）的均值，对异常值鲁棒。

## 数学形式
$$\text{IQM} = \frac{1}{|Q_{50}|} \sum_{x \in Q_{50}} x$$

其中 $Q_{50}$ 是数据集的中间四分位区间（第 25 到 75 百分位）。

## 核心要点
1. 相比均值，IQM 对异常运行结果不敏感（抗极端值）
2. 相比中位数，IQM 利用了更多数据，统计效率更高
3. Agarwal et al. (2021) 推荐 IQM 作为 RL benchmark 的标准报告指标
4. 通常搭配 confidence interval（bootstrap 方法）一起报告

## 代表工作
- [[Dream-MPC]]: 用 IQM 作为主要评估指标
- [[DreamerV3]]: IQM 是当前 model-based RL 论文的常用指标

## 相关概念
- [[强化学习]]
- [[DMC]]
